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
# once for all P. A wavefront (64 lanes) per row, 4 rows a workgroup; each lane takes 4 consecutive elements of every
# 256-element block, dequantizes them in registers and multiplies them into the P accumulators; a wavefront sum ends it.
# The blocks stay in their GGUF layout, rows contiguous.
# ---------------------------------------------------------------------------------------------
@inline ldb(W, i) = @inbounds W[i+1]                                     # byte at 0-based offset i
@inline ldw(W, i) = @inbounds (UInt32(W[i+1]) | UInt32(W[i+2]) << 8 | UInt32(W[i+3]) << 16 | UInt32(W[i+4]) << 24)
@inline function ldh(W, i)
    u = @inbounds UInt16(W[i+1]) | UInt16(W[i+2]) << 8
    Float32(reinterpret(Float16, u))
end
@inline function smk4(W, o, s)                   # (scale, min) of sub-block s from the 12 scale bytes at offset o
    if s < 4
        return Float32(ldb(W, o + s) & 0x3f), Float32(ldb(W, o + s + 4) & 0x3f)
    end
    a = ldb(W, o + s + 4); lo = ldb(W, o + s - 4); hi = ldb(W, o + s)
    (Float32((a & 0x0f) | (lo >> 6) << 4), Float32((a >> 4) | (hi >> 6) << 4))
end
# Each workgroup (4 wavefronts) takes 16 rows, 4 a wavefront; rows must be a multiple of 16, K of 512. A step covers
# two 256-element blocks: half the lanes each, a lane 8 elements of its block (one 32-bit word of Q5_K quants, or two
# 16-bit words of Q6_K quants). The step's x (512 elements, all P positions) sits in shared memory in read order,
# xs[buf][p][64 slot + lane], so every read is 64 consecutive floats. x is double-buffered and the next step's
# weight words are loaded before the current step computes, so each step has one barrier and its global loads in
# flight under the arithmetic. A lane reads its x four slots at a time and uses them for its 4 rows.
const KC = 512
@inline ld32(W, off) = unsafe_load(reinterpret(Core.LLVMPtr{UInt32,1}, pointer(W) + off))
@inline ld16(W, off) = unsafe_load(reinterpret(Core.LLVMPtr{UInt16,1}, pointer(W) + off))
@inline f16(u::UInt16) = Float32(reinterpret(Float16, u))
@inline function xslot(k, ::Val{F}) where {F}         # plane position of step element k (0..511), format F (5 or 6)
    bl = k >> Int32(8); r = k & Int32(255)
    lane, slot = F == 5 ?
        (Int32(32) * bl + Int32(8) * (r >> Int32(6)) + ((r >> Int32(2)) & Int32(7)), Int32(4) * ((r >> Int32(5)) & Int32(1)) + (r & Int32(3))) :
        (Int32(32) * bl + Int32(16) * (r >> Int32(7)) + ((r >> Int32(1)) & Int32(15)), Int32(2) * ((r >> Int32(5)) & Int32(3)) + (r & Int32(1)))
    Int32(64) * slot + lane
end
# this thread's 2P x values of step g (thread t takes flat indices t + 256 j of the P x 512 step)
@inline xload(x, t, g, ldx, ::Val{P}) where {P} =
    ntuple(j -> (i = t + Int32(256 * (j - 1)); p = i >> Int32(9); @inbounds x[p*ldx+Int32(KC)*g+(i&Int32(511))+Int32(1)]), Val(2P))
@inline function xstore!(xs, xr::NTuple{N,Float32}, t, buf, ::Val{P}, ::Val{F}) where {N,P,F}
    for j in 1:N
        i = t + Int32(256 * (j - 1)); p = i >> Int32(9)
        @inbounds xs[Int32(KC * P)*buf+Int32(KC)*p+xslot(i & Int32(511), Val(F))+Int32(1)] = xr[j]
    end
end
@inline xv(xs, buf, slot, lane, ::Val{P}) where {P} = ntuple(p -> @inbounds(xs[Int32(KC * P)*buf+Int32(KC*(p-1))+Int32(64)*slot+lane+Int32(1)]), Val(P))
@inline fmav(a::NTuple{P,Float32}, w, x::NTuple{P,Float32}) where {P} = ntuple(p -> fma(w, x[p], a[p]), Val(P))
@inline function wsum!(y, acc::NTuple{P,Float32}, lane, row, ldy) where {P}
    for p in 1:P
        v = AMDGPU.Device.wfred(+, acc[p])
        lane == Int32(0) && @inbounds(y[(p-1)*ldy+row+1] = v)
    end
end
# Q5_K (scale, min) of sub-blocks s0 and s0 + 1 (s0 even) from the three scale words
@inline function smk4w(u0, u1, u2, s0)
    sh = Int32(8) * (s0 & Int32(3))
    pick(sh) = s0 < Int32(4) ?
        ((u0 >> sh) & 0x3f, (u1 >> sh) & 0x3f) :
        (((u2 >> sh) & 0x0f) | ((u0 >> (sh + Int32(6))) & 0x03) << 4, ((u2 >> (sh + Int32(4))) & 0x0f) | ((u1 >> (sh + Int32(6))) & 0x03) << 4)
    (a, b), (c, e) = pick(sh), pick(sh + Int32(8))
    Float32(a), Float32(b), Float32(c), Float32(e)
end
# a lane's words of one row and block: Q5_K (d dmin, 3 scale words, quant word, high-bit word)
@inline function w5(W, o, jj, e0)
    (ld32(W, o), ld32(W, o + Int32(4)), ld32(W, o + Int32(8)), ld32(W, o + Int32(12)), ld32(W, o + Int32(48) + Int32(32) * jj + e0), ld32(W, o + Int32(16) + e0))
end
function k_gemv_q5k!(y, W, x, nb::Int32, ldx::Int32, ldy::Int32, ::Val{P}) where {P}
    xs = @ROCStaticLocalArray(Float32, 2 * KC * P, false)
    t = Int32(workitemIdx().x) - Int32(1); lane = t & Int32(63); wave = t >> Int32(6)
    r0 = (Int32(workgroupIdx().x) - Int32(1)) * Int32(16) + wave * Int32(4)
    bs = lane >> Int32(5); jj = (lane >> Int32(3)) & Int32(3); e0 = (lane & Int32(7)) * Int32(4); s0 = Int32(2) * jj
    rbytes = nb * Int32(176); ns = nb >> Int32(1)
    Base.Cartesian.@nexprs 4 r -> (acc_r = ntuple(_ -> 0.0f0, Val(P)); ob_r = (r0 + Int32(r - 1)) * rbytes + bs * Int32(176))
    xstore!(xs, xload(x, t, Int32(0), ldx, Val(P)), t, Int32(0), Val(P), Val(5))
    Base.Cartesian.@nexprs 4 r -> (w_r = w5(W, ob_r, jj, e0))
    sync_workgroup()
    for g in Int32(0):ns-Int32(1)
        buf = g & Int32(1); gn = min(g + Int32(1), ns - Int32(1))
        xr = xload(x, t, gn, ldx, Val(P))
        Base.Cartesian.@nexprs 4 r -> (wn_r = w5(W, ob_r + gn * Int32(352), jj, e0))
        Base.Cartesian.@nexprs 4 r -> begin
            dd, u0, u1, u2, qs, qh = w_r
            d = f16(UInt16(dd & 0xffff)); dmin = f16(UInt16(dd >> 16))
            sc0, mn0, sc1, mn1 = smk4w(u0, u1, u2, s0)
            a0_r = d * sc0; m0_r = dmin * mn0; a1_r = d * sc1; m1_r = dmin * mn1
            ql_r = (qs & 0x0f0f0f0f) | ((qh >> s0) & 0x01010101) << 4
            qu_r = ((qs >> 4) & 0x0f0f0f0f) | ((qh >> (s0 + Int32(1))) & 0x01010101) << 4
        end
        Base.Cartesian.@nexprs 4 i -> (xi_i = xv(xs, buf, Int32(i - 1), lane, Val(P)))
        Base.Cartesian.@nexprs 4 r -> Base.Cartesian.@nexprs 4 i -> (acc_r = fmav(acc_r, a0_r * Float32((ql_r >> (8 * (i - 1))) & 0xff) - m0_r, xi_i))
        Base.Cartesian.@nexprs 4 i -> (xi_i = xv(xs, buf, Int32(i + 3), lane, Val(P)))
        Base.Cartesian.@nexprs 4 r -> Base.Cartesian.@nexprs 4 i -> (acc_r = fmav(acc_r, a1_r * Float32((qu_r >> (8 * (i - 1))) & 0xff) - m1_r, xi_i))
        xstore!(xs, xr, t, buf ⊻ Int32(1), Val(P), Val(5))
        sync_workgroup()
        Base.Cartesian.@nexprs 4 r -> (w_r = wn_r)
    end
    Base.Cartesian.@nexprs 4 r -> wsum!(y, acc_r, lane, r0 + Int32(r - 1), ldy)
    return
end
# a lane's words of one row and block: Q6_K (d, low quants of groups 1 2 and 3 4, high bits, the 4 scale bytes)
@inline function w6(W, o, h, l2, is)
    so = o + Int32(192) + Int32(8) * h + is
    (ld16(W, o + Int32(208)), ld16(W, o + Int32(64) * h + l2), ld16(W, o + Int32(64) * h + l2 + Int32(32)), ld16(W, o + Int32(128) + Int32(32) * h + l2),
     ldb(W, so), ldb(W, so + Int32(2)), ldb(W, so + Int32(4)), ldb(W, so + Int32(6)))
end
@inline q6(b0, bh) = Float32(Int32((b0 & 0x0f) | (bh & 0x03) << 4) - Int32(32))
function k_gemv_q6k!(y, W, x, nb::Int32, ldx::Int32, ldy::Int32, ::Val{P}) where {P}
    xs = @ROCStaticLocalArray(Float32, 2 * KC * P, false)
    t = Int32(workitemIdx().x) - Int32(1); lane = t & Int32(63); wave = t >> Int32(6)
    r0 = (Int32(workgroupIdx().x) - Int32(1)) * Int32(16) + wave * Int32(4)
    # a lane: elements l2, l2 + 1 of half h in each of the half's four 32-element groups
    bs = lane >> Int32(5); h = (lane >> Int32(4)) & Int32(1); l2 = (lane & Int32(15)) * Int32(2); is = l2 >> Int32(4)
    rbytes = nb * Int32(210); ns = nb >> Int32(1)
    Base.Cartesian.@nexprs 4 r -> (acc_r = ntuple(_ -> 0.0f0, Val(P)); ob_r = (r0 + Int32(r - 1)) * rbytes + bs * Int32(210))
    xstore!(xs, xload(x, t, Int32(0), ldx, Val(P)), t, Int32(0), Val(P), Val(6))
    Base.Cartesian.@nexprs 4 r -> (w_r = w6(W, ob_r, h, l2, is))
    sync_workgroup()
    for g in Int32(0):ns-Int32(1)
        buf = g & Int32(1); gn = min(g + Int32(1), ns - Int32(1))
        xr = xload(x, t, gn, ldx, Val(P))
        Base.Cartesian.@nexprs 4 r -> (wn_r = w6(W, ob_r + gn * Int32(420), h, l2, is))
        Base.Cartesian.@nexprs 4 r -> begin
            dh, q0_r, q1_r, qh_r, s1, s2, s3, s4 = w_r
            d = f16(dh)
            c1_r = d * Float32(reinterpret(Int8, s1)); c2_r = d * Float32(reinterpret(Int8, s2))
            c3_r = d * Float32(reinterpret(Int8, s3)); c4_r = d * Float32(reinterpret(Int8, s4))
        end
        Base.Cartesian.@nexprs 2 e -> begin
            Base.Cartesian.@nexprs 4 gg -> (xg_gg = xv(xs, buf, Int32(2 * (gg - 1) + e - 1), lane, Val(P)))
            Base.Cartesian.@nexprs 4 r -> begin
                b0 = q0_r >> (8 * (e - 1)); b1 = q1_r >> (8 * (e - 1)); bh = qh_r >> (8 * (e - 1))
                acc_r = fmav(acc_r, c1_r * q6(b0, bh), xg_1)
                acc_r = fmav(acc_r, c2_r * q6(b1, bh >> 2), xg_2)
                acc_r = fmav(acc_r, c3_r * q6(b0 >> 4, bh >> 4), xg_3)
                acc_r = fmav(acc_r, c4_r * q6(b1 >> 4, bh >> 6), xg_4)
            end
        end
        xstore!(xs, xr, t, buf ⊻ Int32(1), Val(P), Val(6))
        sync_workgroup()
        Base.Cartesian.@nexprs 4 r -> (w_r = wn_r)
    end
    Base.Cartesian.@nexprs 4 r -> wsum!(y, acc_r, lane, r0 + Int32(r - 1), ldy)
    return
end
# a weight matrix on the device: raw blocks of rows r0 .. r0 + n
struct DMat
    typ::Int; K::Int; rows::Int
    W::ROCVector{UInt8}
end
DMat(g::GGUF, name::String, r0 = 0, n = -1) = (t = g.t[name]; n < 0 && (n = nrows(t));
    DMat(t.typ, t.dims[1], n, ROCArray(collect(rowview(g, t, r0, n)))))
function gemv!(y, m::DMat, x, P::Int; ldx = m.K, ldy = m.rows)
    (m.K % KC == 0 && m.rows % 16 == 0) || error("gemv!: $(m.rows) x $(m.K): K must be a multiple of $KC, rows of 16"); nb = Int32(m.K ÷ 256); g = cld(m.rows, 16)
    if m.typ == GQ5_K
        @roc groupsize = 256 maxthreads = 256 gridsize = g k_gemv_q5k!(y, m.W, x, nb, Int32(ldx), Int32(ldy), Val(P))
    elseif m.typ == GQ6_K
        @roc groupsize = 256 maxthreads = 256 gridsize = g k_gemv_q6k!(y, m.W, x, nb, Int32(ldx), Int32(ldy), Val(P))
    else
        error("gemv!: type $(m.typ)")
    end
    y
end

# ---------------------------------------------------------------------------------------------
# --bench-gemv: each format against the host dequant on a few rows, then GB/s of weight streamed at P = 1, 4, 8
# ---------------------------------------------------------------------------------------------
function bench_gemv(g::GGUF)
    for name in ("blk.0.ffn_up.weight", "blk.0.ffn_down.weight", "blk.3.attn_output.weight")
        t = g.t[name]; K = t.dims[1]; R = nrows(t)
        m = DMat(g, name)
        for P in (1, 4, 8)
            xh = randn(Float32, K * P); x = ROCArray(xh); y = AMDGPU.zeros(Float32, R * P)
            gemv!(y, m, x, P); AMDGPU.synchronize()
            yh = Array(y); err = 0.0; ref = 0.0
            for r in (0, 1, R ÷ 2, R - 1)
                wr = dq_row(t, rowview(g, t, r, 1))
                for p in 1:P
                    e = sum(Float64.(wr) .* Float64.(view(xh, (p-1)*K+1:p*K)))
                    err = max(err, abs(yh[(p-1)*R+r+1] - e)); ref = max(ref, abs(e))
                end
            end
            n = 20; tm = @elapsed (for _ in 1:n; gemv!(y, m, x, P); end; AMDGPU.synchronize())
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
