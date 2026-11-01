# q27v.jl: Qwen3.8-27B (qwen35: 48 Gated DeltaNet + 16 gated-attention layers, hidden 5120) on sentry's four Radeon
# Pro V340 dies (gfx900, wave64, 56 CUs, 8 GB HBM2 at 484 GB/s each), AMDGPU.jl. nm-z/flashnext#7.
# Model: mradermacher/Qwen3.8-27B-GGUF Q5_K_M (static K-quants: Q5_K, Q6_K, F32; its MTP layer is blk.64).
# Reports only: prefill tok/s, decode tok/s, and each request's time split into instruction, transfer and idle.
#   julia -t 24 q27v.jl --model FILE --bench-gemv      the quant GEMV kernels against a host dequant, and their GB/s
using AMDGPU, Mmap, Printf

# ---------------------------------------------------------------------------------------------
# GGUF (v3): metadata, tensor table, the file mapped read-only
# ---------------------------------------------------------------------------------------------
const GF32, GF16, GQ8_0, GQ4_K, GQ5_K, GQ6_K, GBF16 = 0, 1, 8, 12, 13, 14, 30
const GBLK = Dict(GF32 => (1, 4), GF16 => (1, 2), GQ8_0 => (32, 34), GQ4_K => (256, 144), GQ5_K => (256, 176),
                  GQ6_K => (256, 210), GBF16 => (1, 2))
struct GT
    typ::Int
    dims::Vector{Int}          # dims[1] = K (the row length), dims[2] = rows
    off::Int                   # byte offset of the data in the file
end
struct GGUF
    kv::Dict{String,Any}
    t::Dict{String,GT}
    map::Vector{UInt8}
end
rowbytes(t::GT) = (b = GBLK[t.typ]; t.dims[1] ÷ b[1] * b[2])
nrows(t::GT) = prod(t.dims[2:end])
function gguf(path)
    io = open(path); rd(T) = read(io, T); rs() = String(read(io, Int(rd(UInt64))))
    String(read(io, 4)) == "GGUF" || error("$path: not GGUF"); rd(UInt32) == 3 || error("$path: not GGUF v3")
    nt = Int(rd(UInt64)); nkv = Int(rd(UInt64))
    S = Dict(0=>UInt8, 1=>Int8, 2=>UInt16, 3=>Int16, 4=>UInt32, 5=>Int32, 6=>Float32, 7=>Bool, 10=>UInt64, 11=>Int64, 12=>Float64)
    function val(t)
        t == 8 && return rs()
        if t == 9
            et = Int(rd(UInt32)); n = Int(rd(UInt64))
            return et == 8 ? [rs() for _ in 1:n] : [rd(S[et]) for _ in 1:n]
        end
        rd(S[t])
    end
    kv = Dict{String,Any}(); align = 32
    for _ in 1:nkv
        k = rs(); v = val(Int(rd(UInt32))); kv[k] = v; k == "general.alignment" && (align = Int(v))
    end
    ts = Dict{String,GT}(); raw = Tuple{String,Int,Vector{Int},Int}[]
    for _ in 1:nt
        n = rs(); nd = Int(rd(UInt32)); dims = [Int(rd(UInt64)) for _ in 1:nd]; typ = Int(rd(UInt32)); o = Int(rd(UInt64))
        push!(raw, (n, typ, dims, o))
    end
    data = cld(position(io), align) * align
    close(io)
    for (n, typ, dims, o) in raw
        haskey(GBLK, typ) || error("$n: ggml type $typ is not handled (no importance-quantized formats)")
        ts[n] = GT(typ, dims, data + o)
    end
    GGUF(kv, ts, Mmap.mmap(path))
end
rowview(g::GGUF, t::GT, r0, n) = view(g.map, t.off + r0 * rowbytes(t) + 1:t.off + (r0 + n) * rowbytes(t))

# ---------------------------------------------------------------------------------------------
# Host dequantization (ggml's K-quant layouts), the reference the device kernels are checked against
# ---------------------------------------------------------------------------------------------
hf(lo, hi) = Float32(reinterpret(Float16, UInt16(lo) | UInt16(hi) << 8))
function scale_min_k4(j, q)                      # q: the 12 scale bytes, 1-based view
    if j < 4
        return Int(q[j+1] & 63), Int(q[j+5] & 63)
    end
    (Int(q[j+5] & 0x0f) | Int(q[j-3] >> 6) << 4, Int(q[j+5] >> 4) | Int(q[j+1] >> 6) << 4)
end
function dq_q5k!(y, b)                           # one 176-byte block into 256 floats
    d = hf(b[1], b[2]); dmin = hf(b[3], b[4]); sc = view(b, 5:16); qh = view(b, 17:48); qs = view(b, 49:176)
    u1, u2 = 0x01, 0x02
    for j in 0:3
        s1, m1 = scale_min_k4(2j, sc); s2, m2 = scale_min_k4(2j + 1, sc)
        for l in 0:31
            q = qs[32j+l+1]
            y[64j+l+1] = d * s1 * ((q & 0x0f) + ((qh[l+1] & u1) != 0 ? 16 : 0)) - dmin * m1
            y[64j+l+33] = d * s2 * ((q >> 4) + ((qh[l+1] & u2) != 0 ? 16 : 0)) - dmin * m2
        end
        u1 <<= 2; u2 <<= 2
    end
    y
end
function dq_q6k!(y, b)                           # one 210-byte block into 256 floats
    d = hf(b[209], b[210])
    for h in 0:1
        ql = view(b, 64h+1:64h+64); qh = view(b, 128+32h+1:128+32h+32); sc = reinterpret(Int8, view(b, 193+8h:200+8h))
        for l in 0:31
            is = l >> 4
            q1 = Int((ql[l+1] & 0x0f) | ((qh[l+1] >> 0) & 3) << 4) - 32
            q2 = Int((ql[l+33] & 0x0f) | ((qh[l+1] >> 2) & 3) << 4) - 32
            q3 = Int((ql[l+1] >> 4) | ((qh[l+1] >> 4) & 3) << 4) - 32
            q4 = Int((ql[l+33] >> 4) | ((qh[l+1] >> 6) & 3) << 4) - 32
            y[128h+l+1] = d * sc[is+1] * q1; y[128h+l+33] = d * sc[is+3] * q2
            y[128h+l+65] = d * sc[is+5] * q3; y[128h+l+97] = d * sc[is+7] * q4
        end
    end
    y
end
function dq_row(t::GT, bytes)
    K = t.dims[1]; y = Vector{Float32}(undef, K); nb, bb = GBLK[t.typ]
    t.typ == GF32 && return copy(reinterpret(Float32, bytes))
    t.typ == GQ5_K || t.typ == GQ6_K || error("dq_row: type $(t.typ)")
    for b in 0:K÷nb-1
        blk = view(bytes, b*bb+1:b*bb+bb)
        t.typ == GQ5_K ? dq_q5k!(view(y, 256b+1:256b+256), blk) : dq_q6k!(view(y, 256b+1:256b+256), blk)
    end
    y
end

# ---------------------------------------------------------------------------------------------
# Device GEMV for decode and verify widths: y[p][r] = sum_k W[r][k] x[p][k] for P positions (1..8), the weights read
# once for all P. One lane per row: a wavefront takes 64 rows, a workgroup 256, and K is cut into slices of KS
# elements, one workgroup per (256 rows, slice). The weights are repacked once at load so a wavefront's 16-byte loads
# of its 64 rows are one contiguous 1 KB run: row r = 64 G + lane, block b, 16-byte chunk c of the block at
# ((G nb + b) nchunk + c) 1024 + 16 lane (Q6_K's 2-byte d goes to its own array, so its 208 bytes are 13 chunks).
# The slice of x sits in shared memory as xs[k P + p] and every lane reads the same address (a broadcast). A lane
# sums q x over each scale group and the group's sum of x (one per group, made once when x is staged) is applied
# per group: Q5_K group value = d sc Σqx - dmin mn Σx, Q6_K = d sc (Σqx - 32 Σx). So a weight element costs a
# byte extract and a convert before its P multiply-adds. Each slice writes partial sums part[slice][p][row]; the
# consumer adds the slices (gemv_sum! below for the bench).
# ---------------------------------------------------------------------------------------------
const KS = 512
const NCH = Dict(GQ5_K => 11, GQ6_K => 13)                 # 16-byte chunks a block
const V4U = NTuple{4,VecElement{UInt32}}
const V4F = NTuple{4,VecElement{Float32}}
@inline function ld128(W, off)
    v = unsafe_load(reinterpret(Core.LLVMPtr{V4U,1}, pointer(W) + (off % UInt32)), 1, Val(16))
    (v[1].value, v[2].value, v[3].value, v[4].value)
end
@inline function lds4(xs, i)                              # shared floats i .. i + 3 (i a multiple of 4)
    v = unsafe_load(reinterpret(Core.LLVMPtr{V4F,3}, pointer(xs) + Int32(4) * i), 1, Val(16))
    (v[1].value, v[2].value, v[3].value, v[4].value)
end
@inline function xg(xs, i, ::Val{N}) where {N}            # shared floats i .. i + N - 1 (N, i multiples of 4)
    q = ntuple(j -> lds4(xs, i + Int32(4 * (j - 1))), Val(N ÷ 4))
    ntuple(j -> q[(j-1)>>2+1][(j-1)&3+1], Val(N))
end
@inline f16(u) = Float32(reinterpret(Float16, UInt16(u)))
@inline fmav(a::NTuple{P,Float32}, w, x, o) where {P} = ntuple(p -> fma(w, x[o+p], a[p]), Val(P))
@inline p4(P) = (P + 3) & ~3
# i, made to depend on dep (an empty asm the compiler can't see through): a word's x reads wait for the word
# LA(P) words back to finish, so about LA(P) words' x (4 P floats each) are in registers
@inline gate(i::Int32, dep::Float32) = Base.llvmcall("""
    %r = call i32 asm "; gate \$0 \$2", "=v,0,v"(i32 %0, float %1)
    ret i32 %r""", Int32, Tuple{Int32,Float32}, i, dep)
@inline LA(P) = clamp(8 ÷ P, 2, 8)
# stage x[k0 .. k0 + n) for all P into xs[k P + p], then the sums of each GS-element group into xs[KS P + g P4 + p]
@inline function stage!(xs, x, t, k0, n, ldx, ::Val{P}, ::Val{GS}) where {P,GS}
    i = t
    @inbounds while i < n * Int32(P)
        p = i ÷ n; k = i - p * n
        xs[k*Int32(P)+p+Int32(1)] = x[p*ldx+k0+k+Int32(1)]
        i += Int32(256)
    end
    sync_workgroup()
    i = t
    @inbounds while i < (n ÷ Int32(GS)) * Int32(P)
        gi = i ÷ Int32(P); p = i - gi * Int32(P)
        s = 0.0f0
        for e in Int32(0):Int32(GS - 1)
            s += xs[(gi*Int32(GS)+e)*Int32(P)+p+Int32(1)]
        end
        xs[Int32(KS * P)+gi*Int32(p4(P))+p+Int32(1)] = s
        i += Int32(256)
    end
    sync_workgroup()
end
# Q5_K (scale, min) of sub-block s from the three scale words
@inline function scmn(u0, u1, u2, ::Val{s}) where {s}
    if s < 4
        return Float32((u0 >> (8s)) & 0x3f), Float32((u1 >> (8s)) & 0x3f)
    end
    h = 8 * (s - 4)
    Float32(((u2 >> h) & 0x0f) | ((u0 >> (h + 6)) & 0x03) << 4), Float32(((u2 >> (h + 4)) & 0x0f) | ((u1 >> (h + 6)) & 0x03) << 4)
end
@inline xsum(xs, g, ::Val{P}) where {P} = xg(xs, Int32(KS * P) + g * Int32(p4(P)), Val(p4(P)))
function k_gemv_q5k!(part, W, x, nb::Int32, ldx::Int32, rows::Int32, ::Val{P}) where {P}
    xs = @ROCStaticLocalArray(Float32, KS * P + (KS ÷ 32) * p4(P), false)
    t = Int32(workitemIdx().x) - Int32(1); lane = t & Int32(63); G = (Int32(workgroupIdx().x) - Int32(1)) * Int32(4) + (t >> Int32(6))
    sl = Int32(workgroupIdx().y) - Int32(1); b0 = sl * Int32(KS ÷ 256); nbs = min(Int32(KS ÷ 256), nb - b0)
    stage!(xs, x, t, b0 * Int32(256), nbs * Int32(256), ldx, Val(P), Val(32))
    acc = ntuple(_ -> 0.0f0, Val(P))
    o = (G * nb + b0) * Int32(11 * 1024) + lane * Int32(16)
    tks = ntuple(_ -> 0.0f0, Val(LA(P)))
    for b in Int32(0):nbs-Int32(1)
        c = let o = o; ntuple(i -> ld128(W, o + Int32(1024 * (i - 1))), Val(11)) end; o += Int32(11 * 1024)
        d = f16(c[1][1] & 0xffff); dmin = f16(c[1][1] >> 16); u0, u1, u2 = c[1][2], c[1][3], c[1][4]
        qh = (c[2]..., c[3]...)
        Base.Cartesian.@nexprs 4 j -> begin
            qs = (c[2j+2]..., c[2j+3]...)
            # sub-block 2j - 2 (low nibbles), then 2j - 1 (high), so one group's x is live at a time
            sa = ntuple(_ -> 0.0f0, Val(P))
            Base.Cartesian.@nexprs 8 w -> begin
                qa = (qs[w] & 0x0f0f0f0f) | ((qh[w] >> (2j - 2)) & 0x01010101) << 4
                xa = xg(xs, gate(Int32(256 * P) * b, tks[1]) + Int32((64 * (j - 1) + 4 * (w - 1)) * P), Val(4P))
                Base.Cartesian.@nexprs 4 i -> (sa = fmav(sa, Float32((qa >> (8i - 8)) & 0xff), xa, (i - 1) * P))
                tks = (Base.tail(tks)..., sa[1])
            end
            sb = ntuple(_ -> 0.0f0, Val(P))
            Base.Cartesian.@nexprs 8 w -> begin
                qb = ((qs[w] >> 4) & 0x0f0f0f0f) | ((qh[w] >> (2j - 1)) & 0x01010101) << 4
                xv = xg(xs, gate(Int32(256 * P) * b, tks[1]) + Int32((64 * (j - 1) + 32 + 4 * (w - 1)) * P), Val(4P))
                Base.Cartesian.@nexprs 4 i -> (sb = fmav(sb, Float32((qb >> (8i - 8)) & 0xff), xv, (i - 1) * P))
                tks = (Base.tail(tks)..., sb[1])
            end
            sca, mna = scmn(u0, u1, u2, Val(2j - 2)); scb, mnb = scmn(u0, u1, u2, Val(2j - 1))
            ga = Int32(8) * b + Int32(2j - 2)
            xsa = xsum(xs, ga, Val(P)); xsb = xsum(xs, ga + Int32(1), Val(P))
            da = d * sca; ma = -dmin * mna; db = d * scb; mb = -dmin * mnb
            acc = let a = acc, sa = sa, sb = sb, xsa = xsa, xsb = xsb, da = da, ma = ma, db = db, mb = mb
                ntuple(p -> fma(mb, xsb[p], fma(db, sb[p], fma(ma, xsa[p], fma(da, sa[p], a[p])))), Val(P))
            end
        end
    end
    for p in 1:P
        @inbounds part[(sl*Int32(P)+Int32(p-1))*rows+G*Int32(64)+lane+Int32(1)] = acc[p]
    end
    return
end
function k_gemv_q6k!(part, W, D, x, nb::Int32, ldx::Int32, rows::Int32, ::Val{P}) where {P}
    xs = @ROCStaticLocalArray(Float32, KS * P + (KS ÷ 16) * p4(P), false)
    t = Int32(workitemIdx().x) - Int32(1); lane = t & Int32(63); G = (Int32(workgroupIdx().x) - Int32(1)) * Int32(4) + (t >> Int32(6))
    sl = Int32(workgroupIdx().y) - Int32(1); b0 = sl * Int32(KS ÷ 256); nbs = min(Int32(KS ÷ 256), nb - b0)
    stage!(xs, x, t, b0 * Int32(256), nbs * Int32(256), ldx, Val(P), Val(16))
    acc = ntuple(_ -> 0.0f0, Val(P))
    o = (G * nb + b0) * Int32(13 * 1024) + lane * Int32(16); od = (G * nb + b0) * Int32(64) + lane + Int32(1)
    tks = ntuple(_ -> 0.0f0, Val(LA(P)))
    for b in Int32(0):nbs-Int32(1)
        d = f16(@inbounds D[od]); od += Int32(64)
        sc = ld128(W, o + Int32(12 * 1024))
        Base.Cartesian.@nexprs 2 h -> begin                                # each half loads its own 6 chunks
            ch = let o = o; ntuple(i -> ld128(W, o + Int32(1024 * (i - 1))), Val(12)) end
            qla = (ch[4h-3]..., ch[4h-2]...); qlb = (ch[4h-1]..., ch[4h]...); qh = (ch[2h+7]..., ch[2h+8]...)
            Base.Cartesian.@nexprs 4 m -> begin
                # group m of the half: quant nibble m (low of ql a, low of ql b, high of a, high of b), high bits 2m - 2
                Base.Cartesian.@nexprs 2 is -> begin
                    sm = ntuple(_ -> 0.0f0, Val(P))
                    Base.Cartesian.@nexprs 4 ww -> begin
                        w = 4 * (is - 1) + ww
                        ql = m & 1 == 1 ? qla[w] : qlb[w]
                        q = ((ql >> (m > 2 ? 4 : 0)) & 0x0f0f0f0f) | ((qh[w] >> (2m - 2)) & 0x03030303) << 4
                        xm = xg(xs, gate(Int32(256 * P) * b, tks[1]) + Int32((128 * (h - 1) + 32 * (m - 1) + 4 * (w - 1)) * P), Val(4P))
                        Base.Cartesian.@nexprs 4 i -> (sm = fmav(sm, Float32((q >> (8i - 8)) & 0xff), xm, (i - 1) * P))
                        tks = (Base.tail(tks)..., sm[1])
                    end
                    gi = 8 * (h - 1) + (is - 1) + 2 * (m - 1)               # the group (and scale) index in the block
                    ds = d * Float32(reinterpret(Int8, UInt8((sc[gi>>2+1] >> (8 * (gi & 3))) & 0xff)))
                    xsm = xsum(xs, Int32(16) * b + Int32(gi), Val(P))
                    acc = let a = acc, ds = ds, xsm = xsm, sm = sm
                        ntuple(p -> fma(-32.0f0 * ds, xsm[p], fma(ds, sm[p], a[p])), Val(P))
                    end
                end
            end
        end
        o += Int32(13 * 1024)
    end
    for p in 1:P
        @inbounds part[(sl*Int32(P)+Int32(p-1))*rows+G*Int32(64)+lane+Int32(1)] = acc[p]
    end
    return
end
function k_gemv_sum!(y, part, nsl::Int32, rows::Int32, ldy::Int32, P::Int32)
    i = (Int32(workgroupIdx().x) - Int32(1)) * Int32(256) + Int32(workitemIdx().x) - Int32(1)
    i < rows * P || return
    p = i ÷ rows; r = i - p * rows
    s = 0.0f0
    for q in Int32(0):nsl-Int32(1)
        s += @inbounds part[(q*P+p)*rows+r+Int32(1)]
    end
    @inbounds y[p*ldy+r+Int32(1)] = s
    return
end
# a weight matrix on the device, repacked: rows r0 .. r0 + n of a GGUF tensor
struct DMat
    typ::Int; K::Int; rows::Int
    W::ROCVector{UInt8}
    D::ROCVector{UInt16}                                   # Q6_K d values (empty for Q5_K)
end
function repack(typ, K, rows, src::AbstractVector{UInt8})
    nb = K ÷ 256; bb = GBLK[typ][2]; nch = NCH[typ]; rb = nb * bb
    rows % 256 == 0 || error("repack: $rows rows, not a multiple of 256")
    W = Vector{UInt8}(undef, rows * nb * nch * 16); D = Vector{UInt16}(undef, typ == GQ6_K ? rows * nb : 0)
    Threads.@threads for G in 0:rows÷64-1
        for l in 0:63, b in 0:nb-1
            s = (64G + l) * rb + b * bb
            for c in 0:nch-1
                copyto!(W, ((G * nb + b) * nch + c) * 1024 + 16l + 1, src, s + 16c + 1, 16)
            end
            typ == GQ6_K && (D[(G*nb+b)*64+l+1] = UInt16(src[s+209]) | UInt16(src[s+210]) << 8)
        end
    end
    W, D
end
function DMat(g::GGUF, name::String, r0 = 0, n = -1)
    t = g.t[name]; n < 0 && (n = nrows(t))
    W, D = repack(t.typ, t.dims[1], n, rowview(g, t, r0, n))
    DMat(t.typ, t.dims[1], n, ROCArray(W), ROCArray(D))
end
nslices(m::DMat) = cld(m.K, KS)
# y[p ldy + r] = Σ_k W[r][k] x[p ldx + k], through part (nslices(m) P rows floats)
function gemv!(y, part, m::DMat, x, P::Int; ldx = m.K, ldy = m.rows)
    m.K % 256 == 0 || error("gemv!: K = $(m.K)")
    nb = Int32(m.K ÷ 256); g = (m.rows ÷ 256, nslices(m))
    if m.typ == GQ5_K
        @roc groupsize = 256 maxthreads = 256 gridsize = g k_gemv_q5k!(part, m.W, x, nb, Int32(ldx), Int32(m.rows), Val(P))
    elseif m.typ == GQ6_K
        @roc groupsize = 256 maxthreads = 256 gridsize = g k_gemv_q6k!(part, m.W, m.D, x, nb, Int32(ldx), Int32(m.rows), Val(P))
    else
        error("gemv!: type $(m.typ)")
    end
    @roc groupsize = 256 gridsize = cld(m.rows * P, 256) k_gemv_sum!(y, part, Int32(nslices(m)), Int32(m.rows), Int32(ldy), Int32(P))
    y
end

# ---------------------------------------------------------------------------------------------
# --bench-gemv: each format against the host dequant on a few rows, then GB/s of weight streamed at P = 1, 4, 8
# ---------------------------------------------------------------------------------------------
function bench_gemv(g::GGUF)
    for name in ("blk.0.ffn_up.weight", "blk.0.ffn_down.weight", "blk.3.attn_output.weight")
        t = g.t[name]; K = t.dims[1]; R = nrows(t)
        m = DMat(g, name)
        for P in (1, 2, 4, 8)
            xh = randn(Float32, K * P); x = ROCArray(xh); y = AMDGPU.zeros(Float32, R * P); part = AMDGPU.zeros(Float32, nslices(m) * P * R)
            for _ in 1:5; gemv!(y, part, m, x, P); end; AMDGPU.synchronize()          # warm (clocks ramp)
            yh = Array(y); err = 0.0; ref = 0.0
            for r in (0, 1, 63, 64, R ÷ 2 + 5, R - 1)
                wr = dq_row(t, rowview(g, t, r, 1))
                for p in 1:P
                    e = sum(Float64.(wr) .* Float64.(view(xh, (p-1)*K+1:p*K)))
                    err = max(err, abs(yh[(p-1)*R+r+1] - e)); ref = max(ref, abs(e))
                end
            end
            n = 50; tm = @elapsed (for _ in 1:n; gemv!(y, part, m, x, P); end; AMDGPU.synchronize())
            bytes = R * rowbytes(t)
            @printf("%-26s %-5s %5d x %5d  P=%d  %.3f ms  %6.1f GB/s  max err %.2e of %.2e\n", name, t.typ == GQ5_K ? "Q5_K" : "Q6_K",
                R, K, P, 1e3 * tm / n, n * bytes / tm / 1e9, err, ref)
        end
    end
end

function main(args)
    k = findfirst(==("--model"), args); k === nothing && error("--model FILE")
    g = gguf(args[k+1])
    "--bench-gemv" in args && return bench_gemv(g)
    error("nothing to do: --bench-gemv")
end
if abspath(PROGRAM_FILE) == @__FILE__()
    main(ARGS)
end
