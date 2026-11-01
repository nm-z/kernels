# q27v.jl: Qwen3.8-27B (qwen35: 48 Gated DeltaNet + 16 gated-attention layers, hidden 5120) on sentry's four Radeon
# Pro V340 dies (gfx900, wave64, 56 CUs, 8 GB HBM2 at 484 GB/s each), AMDGPU.jl. nm-z/flashnext#7.
# Model: mradermacher/Qwen3.8-27B-GGUF Q5_K_M (static K-quants: Q5_K, Q6_K, F32; its MTP layer is blk.64).
# Reports only: prefill tok/s, decode tok/s, and each request's time split into instruction, transfer and idle.
#   julia -t 24 q27v.jl --model FILE --bench-gemv      the quant GEMV kernels against a host dequant, and their GB/s
using AMDGPU, Mmap, Printf, Sockets

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
# elements, one workgroup per (256 rows, slice); a wavefront needs nothing from the others (no shared memory, no
# barrier). The weights are repacked once at load so a wavefront's 16-byte loads of its 64 rows are one contiguous
# 1 KB run: row r = 64 G + lane, block b, 16-byte chunk c of the block at ((G nb + b) nchunk + c) 1024 + 16 lane
# (Q6_K's 2-byte d goes to its own array, so its other 208 bytes are 13 chunks). x is the same for every lane, so it
# is read with scalar loads into SGPRs, which the multiply-adds take as an operand; it is laid out [k][p]
# (xt[k P + p]) with the sums of each scale group beside it (xsm[g P + p]), both made by k_xprep!. A lane sums q x
# over each scale group and applies the group's sum of x once: Q5_K group value = d sc Σqx - dmin mn Σx, Q6_K
# = d sc (Σqx - 32 Σx), so a weight element costs a byte extract and a convert before its P multiply-adds. Each
# slice writes partial sums part[slice][p][row]; the consumer adds the slices (k_gemv_sum! here).
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# Time accounting of a request, die by die: every kernel (instruction time) and every copy (transfer time) is
# bracketed by timing events on its die's stream; idle time is the rest of the request's wall time on that die. A
# copy overlapping a kernel counts as instruction time.
# ---------------------------------------------------------------------------------------------
if !isdefined(Main, :TR)
    const TR = Ref{Any}(nothing)                           # nothing, or device => [(kind, start, stop)]
end
function trace_op(kind, f)
    tr = TR[]
    tr === nothing && return f()
    v = get!(tr, AMDGPU.device(), Any[]); e0, e1 = pooled(2length(v) + 1), pooled(2length(v) + 2)
    AMDGPU.HIP.record(e0); r = f(); AMDGPU.HIP.record(e1)
    push!(v, (kind, e0, e1))
    r
end
# timing events made ahead of time (making one costs the host more than recording it), per device, reused
if !isdefined(Main, :EVP)
    const EVP = Dict{Any,Vector{Any}}()
end
function pooled(i)
    p = get!(EVP, AMDGPU.device(), Any[])
    while length(p) < i; push!(p, AMDGPU.HIP.HIPEvent(AMDGPU.stream(); do_record = false, timing = true)); end
    p[i]
end
macro k(ex) :(trace_op(:inst, () -> $(esc(ex)))) end
macro x(ex) :(trace_op(:xfer, () -> $(esc(ex)))) end
function union_len(iv)                                     # total length of a set of intervals
    isempty(iv) && return 0.0
    sort!(iv); tot = 0.0; (a, b) = iv[1]
    for (c, d) in iv[2:end]
        c > b ? (tot += b - a; (a, b) = (c, d)) : (b = max(b, d))
    end
    tot + b - a
end
# run f() with the accounting on; per die (instruction, transfer, idle) seconds and the request's wall time
function account(f, E)
    TR[] = Dict{Any,Any}(); st = Dict{Any,Any}()
    for D in E.dies
        AMDGPU.device!(D.dev); AMDGPU.synchronize(); st[D.dev] = AMDGPU.HIP.HIPEvent(AMDGPU.stream(); timing = true)
    end
    r = try f() finally end
    res = NTuple{4,Float64}[]
    for D in E.dies
        AMDGPU.device!(D.dev); en = AMDGPU.HIP.HIPEvent(AMDGPU.stream(); timing = true); AMDGPU.HIP.synchronize(en)
        t0 = st[D.dev]; wall = AMDGPU.HIP.elapsed(t0, en)
        ops = get(TR[], D.dev, Any[])
        iv(k) = [(AMDGPU.HIP.elapsed(t0, a), AMDGPU.HIP.elapsed(t0, b)) for (kk, a, b) in ops if kk == k]
        ins = iv(:inst); xf = iv(:xfer)
        ti = union_len(copy(ins)); tall = union_len(vcat(ins, xf))
        push!(res, (ti, tall - ti, wall - tall, wall))
    end
    TR[] = nothing
    r, res
end
function print_account(io, res)                           # idle share of the request, all dies
    @printf(io, "idle time %.1f%%\n", 100sum(r[3] for r in res) / sum(r[4] for r in res))
end
const KS = 512
const NCH = Dict(GQ5_K => 11, GQ6_K => 13)                 # 16-byte chunks a block
const GSZ = Dict(GQ5_K => 32, GQ6_K => 16)                 # elements a scale group
const V4U = NTuple{4,VecElement{UInt32}}
# memory either as a device array or a raw pointer (the persistent kernel takes one buffer of everything)
@inline wptr(a) = pointer(a)
@inline wptr(a::Core.LLVMPtr) = a
@inline pst!(a, v, i) = unsafe_store!(wptr(a), v, i)
@inline dld(a, i) = unsafe_load(wptr(a), i)
@inline function ld128(W, off)
    v = unsafe_load(reinterpret(Core.LLVMPtr{V4U,1}, wptr(W) + (off % UInt32)), 1, Val(16))
    (v[1].value, v[2].value, v[3].value, v[4].value)
end
# x[i] as an invariant load: with a uniform i it becomes a scalar load
@inline ldc(x, i) = Base.llvmcall(("""
    define float @ldc(ptr addrspace(1) %p, i32 %i) #0 {
        %e = sext i32 %i to i64
        %q = getelementptr inbounds float, ptr addrspace(1) %p, i64 %e
        %v = load float, ptr addrspace(1) %q, align 4, !invariant.load !0
        ret float %v
    }
    attributes #0 = { alwaysinline }
    !0 = !{}
    """, "ldc"), Float32, Tuple{Core.LLVMPtr{Float32,1},Int32}, pointer(x), Int32(i))
@inline xc(x, i, ::Val{N}) where {N} = ntuple(j -> ldc(x, i + Int32(j - 1)), Val(N))
# q, made to depend on dep (an empty asm the compiler can't see through): a word's quants are converted only after
# the word LA(P) back has been multiplied in, so the scheduler can't hold a whole block of converted floats
@inline vgate(q::UInt32, dep::Float32) = Base.llvmcall("""
    %r = call i32 asm "; vgate \$0 \$2", "=v,0,v"(i32 %0, float %1)
    ret i32 %r""", UInt32, Tuple{UInt32,Float32}, q, dep)
@inline LA(P) = clamp(8 ÷ P, 1, 8)
@inline f16(u) = Float32(reinterpret(Float16, UInt16(u)))
@inline fmav(a::NTuple{P,Float32}, w, x, o) where {P} = ntuple(p -> fma(w, x[o+p], a[p]), Val(P))
# x[p ldx + k] into xt[k P + p], and the sum of each GS-element group into xsm[g P + p]
function k_xprep!(xt, xsm, x, K::Int32, ldx::Int32, ::Val{P}, ::Val{GS}) where {P,GS}
    i = (Int32(workgroupIdx().x) - Int32(1)) * Int32(256) + Int32(workitemIdx().x) - Int32(1)
    if i < K * Int32(P)
        k = i ÷ Int32(P); p = i - k * Int32(P)
        @inbounds xt[i+Int32(1)] = x[p*ldx+k+Int32(1)]
    end
    if i < (K ÷ Int32(GS)) * Int32(P)
        g = i ÷ Int32(P); p = i - g * Int32(P); s = 0.0f0
        for e in Int32(0):Int32(GS - 1)
            s += @inbounds x[p*ldx+g*Int32(GS)+e+Int32(1)]
        end
        @inbounds xsm[i+Int32(1)] = s
    end
    return
end
# Q5_K (scale, min) of sub-block s from the three scale words
@inline function scmn(u0, u1, u2, ::Val{s}) where {s}
    if s < 4
        return Float32((u0 >> (8s)) & 0x3f), Float32((u1 >> (8s)) & 0x3f)
    end
    h = 8 * (s - 4)
    Float32(((u2 >> h) & 0x0f) | ((u0 >> (h + 6)) & 0x03) << 4), Float32(((u2 >> (h + 4)) & 0x0f) | ((u1 >> (h + 6)) & 0x03) << 4)
end
function k_gemv_q5k!(part, W, xt, xsm, nb::Int32, rows::Int32, ::Val{P}) where {P}
    t = Int32(workitemIdx().x) - Int32(1); lane = t & Int32(63); G = (Int32(workgroupIdx().x) - Int32(1)) * Int32(4) + (t >> Int32(6))
    G * Int32(64) < rows || return
    gemv5!(part, W, xt, xsm, nb, rows, G, Int32(workgroupIdx().y) - Int32(1), lane, Val(P))
    return
end
# one wavefront: rows 64 G .. 64 G + 63, K slice sl, into part
@noinline function gemv5!(part, W, xt, xsm, nb::Int32, rows::Int32, G::Int32, sl::Int32, lane::Int32, ::Val{P}) where {P}
    b0 = sl * Int32(KS ÷ 256); nbs = min(Int32(KS ÷ 256), nb - b0)
    acc = ntuple(_ -> 0.0f0, Val(P))
    o = (G * nb + b0) * Int32(11 * 1024) + lane * Int32(16)
    for b in b0:b0+nbs-Int32(1)
        c = let o = o; ntuple(i -> ld128(W, o + Int32(1024 * (i - 1))), Val(11)) end; o += Int32(11 * 1024)
        d = f16(c[1][1] & 0xffff); dmin = f16(c[1][1] >> 16); u0, u1, u2 = c[1][2], c[1][3], c[1][4]
        qh = (c[2]..., c[3]...)
        Base.Cartesian.@nexprs 4 j -> begin
            qs = (c[2j+2]..., c[2j+3]...)
            # sub-block 2j - 2 (low nibbles), then 2j - 1 (high nibbles)
            sa = ntuple(_ -> 0.0f0, Val(P))
            Base.Cartesian.@nexprs 8 w -> begin
                qa = (qs[w] & 0x0f0f0f0f) | ((qh[w] >> (2j - 2)) & 0x01010101) << 4
                xa = xc(xt, (Int32(256) * b + Int32(64 * (j - 1) + 4 * (w - 1))) * Int32(P), Val(4P))
                Base.Cartesian.@nexprs 4 i -> (sa = fmav(sa, Float32((qa >> (8i - 8)) & 0xff), xa, (i - 1) * P))
            end
            sb = ntuple(_ -> 0.0f0, Val(P))
            Base.Cartesian.@nexprs 8 w -> begin
                qb = ((qs[w] >> 4) & 0x0f0f0f0f) | ((qh[w] >> (2j - 1)) & 0x01010101) << 4
                xv = xc(xt, (Int32(256) * b + Int32(64 * (j - 1) + 32 + 4 * (w - 1))) * Int32(P), Val(4P))
                Base.Cartesian.@nexprs 4 i -> (sb = fmav(sb, Float32((qb >> (8i - 8)) & 0xff), xv, (i - 1) * P))
            end
            sca, mna = scmn(u0, u1, u2, Val(2j - 2)); scb, mnb = scmn(u0, u1, u2, Val(2j - 1))
            ga = Int32(8) * b + Int32(2j - 2)
            xsa = xc(xsm, ga * Int32(P), Val(P)); xsb = xc(xsm, (ga + Int32(1)) * Int32(P), Val(P))
            da = d * sca; ma = -dmin * mna; db = d * scb; mb = -dmin * mnb
            acc = let a = acc, sa = sa, sb = sb, xsa = xsa, xsb = xsb, da = da, ma = ma, db = db, mb = mb
                ntuple(p -> fma(mb, xsb[p], fma(db, sb[p], fma(ma, xsa[p], fma(da, sa[p], a[p])))), Val(P))
            end
        end
    end
    for p in 1:P
        pst!(part, acc[p], (sl*Int32(P)+Int32(p-1))*rows+G*Int32(64)+lane+Int32(1))
    end
    return
end
function k_gemv_q6k!(part, W, D, xt, xsm, nb::Int32, rows::Int32, ::Val{P}) where {P}
    t = Int32(workitemIdx().x) - Int32(1); lane = t & Int32(63); G = (Int32(workgroupIdx().x) - Int32(1)) * Int32(4) + (t >> Int32(6))
    G * Int32(64) < rows || return
    gemv6!(part, W, D, xt, xsm, nb, rows, G, Int32(workgroupIdx().y) - Int32(1), lane, Val(P))
    return
end
# one wavefront: rows 64 G .. 64 G + 63, K slice sl, into part
@noinline function gemv6!(part, W, D, xt, xsm, nb::Int32, rows::Int32, G::Int32, sl::Int32, lane::Int32, ::Val{P}) where {P}
    b0 = sl * Int32(KS ÷ 256); nbs = min(Int32(KS ÷ 256), nb - b0)
    acc = ntuple(_ -> 0.0f0, Val(P)); tks = ntuple(_ -> 0.0f0, Val(LA(P)))
    o = (G * nb + b0) * Int32(13 * 1024) + lane * Int32(16); od = (G * nb + b0) * Int32(64) + lane + Int32(1)
    for b in b0:b0+nbs-Int32(1)
        d = f16(dld(D, od)); od += Int32(64)
        c = let o = o; ntuple(i -> ld128(W, o + Int32(1024 * (i - 1))), Val(13)) end; o += Int32(13 * 1024)
        sc = c[13]
        Base.Cartesian.@nexprs 2 h -> begin
            qla = (c[4h-3]..., c[4h-2]...); qlb = (c[4h-1]..., c[4h]...); qh = (c[2h+7]..., c[2h+8]...)
            # group m of the half: nibble m (low of ql a, low of ql b, high of a, high of b), high bits 2m - 2
            Base.Cartesian.@nexprs 4 m -> Base.Cartesian.@nexprs 2 is -> begin
                sm = ntuple(_ -> 0.0f0, Val(P))
                Base.Cartesian.@nexprs 4 ww -> begin
                    w = 4 * (is - 1) + ww
                    ql = m & 1 == 1 ? qla[w] : qlb[w]
                    q = vgate(((ql >> (m > 2 ? 4 : 0)) & 0x0f0f0f0f) | ((qh[w] >> (2m - 2)) & 0x03030303) << 4, tks[1])
                    xm = xc(xt, (Int32(256) * b + Int32(128 * (h - 1) + 32 * (m - 1) + 4 * (w - 1))) * Int32(P), Val(4P))
                    Base.Cartesian.@nexprs 4 i -> (sm = fmav(sm, Float32((q >> (8i - 8)) & 0xff), xm, (i - 1) * P))
                    tks = (Base.tail(tks)..., sm[1])
                end
                gi = 8 * (h - 1) + (is - 1) + 2 * (m - 1)                   # the group (and scale) index in the block
                ds = d * Float32(reinterpret(Int8, UInt8((sc[gi>>2+1] >> (8 * (gi & 3))) & 0xff)))
                xsg = xc(xsm, (Int32(16) * b + Int32(gi)) * Int32(P), Val(P))
                acc = let a = acc, ds = ds, xsg = xsg, sm = sm
                    ntuple(p -> fma(-32.0f0 * ds, xsg[p], fma(ds, sm[p], a[p])), Val(P))
                end
            end
        end
    end
    for p in 1:P
        pst!(part, acc[p], (sl*Int32(P)+Int32(p-1))*rows+G*Int32(64)+lane+Int32(1))
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
    rows % 64 == 0 || error("repack: $rows rows, not a multiple of 64")
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
    DMat(g, name, [(r0, n)], [(0, t.dims[1] ÷ 256)])
end
# rows: (first, count) ranges, taken in order; blocks: (first, count) ranges of 256-column blocks (a K slice). The
# rows are padded with zero rows to a multiple of 64.
function DMat(g::GGUF, name::String, rows::Vector, blocks::Vector)
    t = g.t[name]; bb = GBLK[t.typ][2]; rb = rowbytes(t)
    nr = sum(last, rows); nbk = sum(last, blocks); np = cld(nr, 64) * 64
    src = zeros(UInt8, np * nbk * bb); i = 0
    for (r0, n) in rows, r in r0:r0+n-1
        row = rowview(g, t, r, 1)
        for (b0, nb) in blocks
            copyto!(src, i + 1, row, b0 * bb + 1, nb * bb); i += nb * bb
        end
    end
    W, D = repack(t.typ, 256nbk, np, src)
    DMat(t.typ, 256nbk, np, ROCArray(W), ROCArray(D))
end
nslices(m::DMat) = cld(m.K, KS)
# scratch of a GEMV at P positions: partial sums, x in [k][p] order, its group sums
struct GBuf
    part::ROCVector{Float32}; xt::ROCVector{Float32}; xsm::ROCVector{Float32}
end
GBuf(m::DMat, P) = GBuf(nslices(m) * P * m.rows, m.K, P)
GBuf(npart::Int, K::Int, P) = GBuf(AMDGPU.zeros(Float32, npart), AMDGPU.zeros(Float32, K * P), AMDGPU.zeros(Float32, K ÷ 16 * P))
# y[p ldy + r] = Σ_k W[r][k] x[p ldx + k]
function gemv!(y, gb::GBuf, m::DMat, x, P::Int; ldx = m.K, ldy = m.rows)
    (m.K % 256 == 0 && length(gb.part) >= nslices(m) * P * m.rows && length(gb.xt) >= m.K * P) || error("gemv!: K = $(m.K), scratch")
    nb = Int32(m.K ÷ 256); g = (cld(m.rows ÷ 64, 4), nslices(m))
    @k @roc groupsize = 256 gridsize = cld(m.K * P, 256) k_xprep!(gb.xt, gb.xsm, x, Int32(m.K), Int32(ldx), Val(P), Val(GSZ[m.typ]))
    if m.typ == GQ5_K
        @k @roc groupsize = 256 maxthreads = 256 gridsize = g k_gemv_q5k!(gb.part, m.W, gb.xt, gb.xsm, nb, Int32(m.rows), Val(P))
    elseif m.typ == GQ6_K
        @k @roc groupsize = 256 maxthreads = 256 gridsize = g k_gemv_q6k!(gb.part, m.W, m.D, gb.xt, gb.xsm, nb, Int32(m.rows), Val(P))
    else
        error("gemv!: type $(m.typ)")
    end
    @k @roc groupsize = 256 gridsize = cld(m.rows * P, 256) k_gemv_sum!(y, gb.part, Int32(nslices(m)), Int32(m.rows), Int32(ldy), Int32(P))
    y
end

# ---------------------------------------------------------------------------------------------
# The model on four dies. Die d holds DeltaNet key heads 4d .. 4d + 3 and the 12 value heads that read them
# (h = 16j + 4d + i, j = 0..2, i = 0..3; value head h reads key head h mod 16), attention query heads 6d .. 6d + 5 and
# KV head d, FFN rows (and down-projection columns) 4352d .., and a quarter of the vocabulary rows of the embedding
# and the head. The residual stream x is on every die; each mixer and FFN ends in a partial sum on each die, added
# across the dies (allreduce_add!).
# ---------------------------------------------------------------------------------------------
const NE, NFF, NV, NL = 5120, 17408, 248320, 64
const DK, NKH, NVH, HD, NH, NKV, NROT = 128, 16, 48, 256, 24, 4, 64
const EPS, THETA = 1.0f-6, 1.0e7
const ND, PMAX, VPD = 4, 8, 62080                          # dies, positions a forward, vocabulary rows a die
const QKVD, VHD = 2560, 12                                 # DeltaNet qkv channels and value heads a die
const CH = 256                                             # attention keys a workgroup
isrec(l) = (l + 1) % 4 != 0
ridx(l) = 3 * (l ÷ 4) + l % 4                              # index among the DeltaNet layers
aidx(l) = l ÷ 4                                            # index among the attention layers
@inline silu(x) = x / (1.0f0 + exp(-x))
const RED = 4                                              # wavefronts a 256-thread workgroup
@inline function wgsum(v, red)                             # sum over a 256-thread workgroup (red: 4 shared floats)
    s = AMDGPU.Device.wfred(+, v); t = Int32(workitemIdx().x) - Int32(1)
    sync_workgroup()
    t & Int32(63) == Int32(0) && @inbounds(red[(t>>6)+1] = s)
    sync_workgroup()
    @inbounds red[1] + red[2] + red[3] + red[4]
end
@inline function wgmax(v, red)
    s = AMDGPU.Device.wfred(max, v); t = Int32(workitemIdx().x) - Int32(1)
    sync_workgroup()
    t & Int32(63) == Int32(0) && @inbounds(red[(t>>6)+1] = s)
    sync_workgroup()
    @inbounds max(max(red[1], red[2]), max(red[3], red[4]))
end
function k_rms!(h, x, w, n::Int32)                         # h = x / rms(x) * w, one workgroup a position
    red = @ROCStaticLocalArray(Float32, RED, false)
    t = Int32(workitemIdx().x) - Int32(1); o = (Int32(workgroupIdx().x) - Int32(1)) * n
    s = 0.0f0; i = t
    @inbounds while i < n; v = x[o+i+1]; s += v * v; i += Int32(256); end
    r = 1.0f0 / sqrt(wgsum(s, red) / Float32(n) + EPS); i = t
    @inbounds while i < n; h[o+i+1] = x[o+i+1] * r * w[i+1]; i += Int32(256); end
    return
end
function k_add!(x, y, n::Int32)
    i = (Int32(workgroupIdx().x) - Int32(1)) * Int32(256) + Int32(workitemIdx().x) - Int32(1)
    i < n && @inbounds(x[i+1] += y[i+1])
    return
end
function k_silu_mul!(g, u, n::Int32)
    i = (Int32(workgroupIdx().x) - Int32(1)) * Int32(256) + Int32(workitemIdx().x) - Int32(1)
    i < n && @inbounds(g[i+1] = silu(g[i+1]) * u[i+1])
    return
end
# the embedding rows of tok[p] that this die holds (rows r0 .., repacked Q5_K), zeros for the others: y[p NE + k]
function k_embed!(y, W, tok, r0::Int32, rows::Int32)
    p = Int32(workgroupIdx().y) - Int32(1); b = Int32(workgroupIdx().x) - Int32(1); e = Int32(workitemIdx().x) - Int32(1)
    r = @inbounds(tok[p+1]) - r0; v = 0.0f0
    if r >= Int32(0) && r < rows
        G = r >> Int32(6); lane = r & Int32(63); base = (G * Int32(20) + b) * Int32(11 * 1024) + Int32(16) * lane
        byte(j) = @inbounds W[base+(j>>Int32(4))*Int32(1024)+(j&Int32(15))+Int32(1)]
        d = f16(UInt16(byte(Int32(0))) | UInt16(byte(Int32(1))) << 8); dmin = f16(UInt16(byte(Int32(2))) | UInt16(byte(Int32(3))) << 8)
        s = e >> Int32(5); l = e & Int32(31)
        sc, mn = if s < Int32(4)
            byte(Int32(4) + s) & 0x3f, byte(Int32(8) + s) & 0x3f
        else
            (byte(Int32(8) + s) & 0x0f) | (byte(s) >> 6) << 4, (byte(Int32(8) + s) >> 4) | (byte(Int32(4) + s) >> 6) << 4
        end
        qs = byte(Int32(48) + Int32(32) * (s >> Int32(1)) + l); qh = byte(Int32(16) + l)
        q = ((s & Int32(1)) == Int32(0) ? qs & 0x0f : qs >> 4) | ((qh >> s) & 0x01) << 4
        v = d * Float32(sc) * Float32(q) - dmin * Float32(mn)
    end
    @inbounds y[p*Int32(NE)+Int32(256)*b+e+Int32(1)] = v
    return
end
# DeltaNet conv (kernel 4) and silu over P positions, one thread a channel; ring holds the previous 3 inputs
function k_dnconv!(u, qkv, ring, cw, P::Int32, ldq::Int32)
    c = (Int32(workgroupIdx().x) - Int32(1)) * Int32(256) + Int32(workitemIdx().x) - Int32(1)
    c < Int32(QKVD) || return
    @inbounds begin
        r1 = ring[3c+1]; r2 = ring[3c+2]; r3 = ring[3c+3]
        w1 = cw[4c+1]; w2 = cw[4c+2]; w3 = cw[4c+3]; w4 = cw[4c+4]
        for p in Int32(0):P-Int32(1)
            x = qkv[p*ldq+c+Int32(1)]
            u[p*Int32(QKVD)+c+Int32(1)] = silu(w1 * r1 + w2 * r2 + w3 * r3 + w4 * x)
            r1, r2, r3 = r2, r3, x
        end
        ring[3c+1] = r1; ring[3c+2] = r2; ring[3c+3] = r3
    end
    return
end
# the gated delta rule over P positions, one workgroup a local value head: thread t holds row i = t ÷ 2 of the head's
# 128 x 128 state, columns 64 (t mod 2) .. + 63; out = rmsnorm(S q / sqrt(128)) * ssm_norm * silu(z)
function k_dnrec!(on, S, u, zb, al, be, dtb, ssa, sn, P::Int32)
    qs = @ROCStaticLocalArray(Float32, DK, false); ks = @ROCStaticLocalArray(Float32, DK, false)
    red = @ROCStaticLocalArray(Float32, RED, false)
    lv = Int32(workgroupIdx().x) - Int32(1); kh = lv & Int32(3); t = Int32(workitemIdx().x) - Int32(1)
    i = t >> Int32(1); jh = t & Int32(1); so = (lv * Int32(DK) + i) * Int32(DK) + Int32(64) * jh
    s = ntuple(j -> @inbounds(S[so+j]), Val(64))
    isc = 1.0f0 / sqrt(Float32(DK))
    for p in Int32(0):P-Int32(1)
        ub = p * Int32(QKVD)
        qv = t < Int32(DK) ? @inbounds(u[ub+Int32(DK)*kh+t+Int32(1)]) : 0.0f0
        kv = t < Int32(DK) ? @inbounds(u[ub+Int32(512)+Int32(DK)*kh+t+Int32(1)]) : 0.0f0
        nq = sqrt(wgsum(qv * qv, red)); nk = sqrt(wgsum(kv * kv, red))
        if t < Int32(DK)
            @inbounds qs[t+1] = qv / max(nq, EPS); @inbounds ks[t+1] = kv / max(nk, EPS)
        end
        sync_workgroup()
        x = @inbounds(al[p*Int32(64)+lv+Int32(1)]) + @inbounds(dtb[lv+1]); sp = x > 20.0f0 ? x : log1p(exp(x))
        a = exp(@inbounds(ssa[lv+1]) * sp); bt = 1.0f0 / (1.0f0 + exp(-@inbounds(be[p*Int32(64)+lv+Int32(1)])))
        s = let s = s; ntuple(j -> s[j] * a, Val(64)) end
        e = 0.0f0
        for j in 1:64; e = fma(s[j], @inbounds(ks[Int32(64)*jh+Int32(j)]), e); end
        e += AMDGPU.Device.shfl_xor(e, Cint(1))
        e = bt * (@inbounds(u[ub+Int32(1024)+Int32(DK)*lv+i+Int32(1)]) - e)
        s = let s = s, e = e; ntuple(j -> fma(e, @inbounds(ks[Int32(64)*jh+Int32(j)]), s[j]), Val(64)) end
        o = 0.0f0
        for j in 1:64; o = fma(s[j], @inbounds(qs[Int32(64)*jh+Int32(j)]), o); end
        o = (o + AMDGPU.Device.shfl_xor(o, Cint(1))) * isc
        r = 1.0f0 / sqrt(wgsum(jh == Int32(0) ? o * o : 0.0f0, red) / Float32(DK) + EPS)
        if jh == Int32(0)
            z = @inbounds zb[p*Int32(DK*VHD)+Int32(DK)*lv+i+Int32(1)]
            @inbounds on[p*Int32(DK*VHD)+Int32(DK)*lv+i+Int32(1)] = o * r * sn[i+1] * silu(z)
        end
        sync_workgroup()
    end
    for j in 1:64; @inbounds S[so+j] = s[j]; end
    return
end
# attention input: rmsnorm and neox rope (first 64 dims) of the 6 query heads and the KV head at P positions; the
# rotated key and the value go into the f16 cache at their positions (koff: this layer's cache)
function k_attn_prep!(qh, gh, kc, vc, qg, kb, vb, qn, kn, pos0::Int32, koff::Int32)
    L = @ROCStaticLocalArray(Float32, HD, false); red = @ROCStaticLocalArray(Float32, RED, false)
    hx = Int32(workgroupIdx().x) - Int32(1); p = Int32(workgroupIdx().y) - Int32(1); t = Int32(workitemIdx().x) - Int32(1)
    val = hx < Int32(6) ? @inbounds(qg[p*Int32(3072)+Int32(512)*hx+t+Int32(1)]) : @inbounds(kb[p*Int32(HD)+t+Int32(1)])
    w = hx < Int32(6) ? @inbounds(qn[t+1]) : @inbounds(kn[t+1])
    v = val * (1.0f0 / sqrt(wgsum(val * val, red) / Float32(HD) + EPS)) * w
    @inbounds L[t+1] = v
    sync_workgroup()
    if t < Int32(NROT)
        ii = t & Int32(31); th = Float64(pos0 + p) * THETA^(-2.0 * Float64(ii) / NROT); c = Float32(cos(th)); sn = Float32(sin(th))
        a = @inbounds L[ii+1]; b = @inbounds L[ii+Int32(33)]
        v = t < Int32(32) ? a * c - b * sn : b * c + a * sn
    end
    if hx < Int32(6)
        @inbounds qh[p*Int32(1536)+Int32(HD)*hx+t+Int32(1)] = v
        @inbounds gh[p*Int32(1536)+Int32(HD)*hx+t+Int32(1)] = qg[p*Int32(3072)+Int32(512)*hx+Int32(HD)+t+Int32(1)]
    else
        @inbounds kc[koff+(pos0+p)*Int32(HD)+t+Int32(1)] = Float16(v)
        @inbounds vc[koff+(pos0+p)*Int32(HD)+t+Int32(1)] = Float16(vb[p*Int32(HD)+t+Int32(1)])
    end
    return
end
# one (query head, position) over CH keys: part = (max, sum, Σ e^(s - max) v)
function k_attn!(part, qh, kc, vc, koff::Int32, pos0::Int32, nch::Int32)
    q = @ROCStaticLocalArray(Float32, HD, false); ps = @ROCStaticLocalArray(Float32, CH, false)
    red = @ROCStaticLocalArray(Float32, RED, false)
    x = Int32(workgroupIdx().x) - Int32(1); hx = x % Int32(6); p = x ÷ Int32(6); c = Int32(workgroupIdx().y) - Int32(1)
    t = Int32(workitemIdx().x) - Int32(1)
    @inbounds q[t+1] = qh[p*Int32(1536)+Int32(HD)*hx+t+Int32(1)]
    sync_workgroup()
    k0 = c * Int32(CH); nk = pos0 + p + Int32(1)               # keys 0 .. pos0 + p
    kk = k0 + t; sc = -Inf32
    if kk < nk
        s = 0.0f0; kb = koff + kk * Int32(HD)
        for d in Int32(1):Int32(HD); s = fma(q[d], Float32(@inbounds kc[kb+d]), s); end
        sc = s * (1.0f0 / sqrt(Float32(HD)))
    end
    m = wgmax(sc, red)
    e = kk < nk ? exp(sc - m) : 0.0f0
    l = wgsum(e, red)
    @inbounds ps[t+1] = e
    sync_workgroup()
    acc = 0.0f0
    for j in Int32(0):Int32(CH - 1)
        k0 + j < nk || break
        acc = fma(@inbounds(ps[j+1]), Float32(@inbounds vc[koff+(k0+j)*Int32(HD)+t+Int32(1)]), acc)
    end
    o = (x * nch + c) * Int32(HD + 2)
    if t == Int32(0)
        @inbounds part[o+1] = m; @inbounds part[o+2] = l
    end
    @inbounds part[o+Int32(2)+t+Int32(1)] = acc
    return
end
function k_attn_comb!(ao, part, gh, nch::Int32)
    x = Int32(workgroupIdx().x) - Int32(1); hx = x % Int32(6); p = x ÷ Int32(6); t = Int32(workitemIdx().x) - Int32(1)
    M = -Inf32
    for c in Int32(0):nch-Int32(1); M = max(M, @inbounds part[(x*nch+c)*Int32(HD+2)+1]); end
    L = 0.0f0; acc = 0.0f0
    for c in Int32(0):nch-Int32(1)
        o = (x * nch + c) * Int32(HD + 2); m = @inbounds part[o+1]
        if m > -Inf32
            f = exp(m - M); L += @inbounds(part[o+2]) * f; acc += @inbounds(part[o+Int32(3)+t]) * f
        end
    end
    g = @inbounds gh[p*Int32(1536)+Int32(HD)*hx+t+Int32(1)]
    @inbounds ao[p*Int32(1536)+Int32(HD)*hx+t+Int32(1)] = acc / L / (1.0f0 + exp(-g))
    return
end

# ---------------------------------------------------------------------------------------------
# The cooperative persistent kernel: one dispatch a die runs a whole request (every forward of the prefill, then
# every decode step, the token choice included). All its workgroups are resident at once and step through the
# same sequence of phases with a grid barrier (gbar!) between phases. Between dies, each mixer and FFN output, the
# embedding and the argmax go through an exchange: each die writes its share into page-locked host memory that
# every die maps, raises its counter, waits for the other three, and reads theirs.
# Time accounting: for each phase, the first start and the last end over the workgroups that worked in it
# (memrealtime, a 100 MHz clock) go into acct; their union is instruction time, the rest of the die's request time
# is idle (the exchanges compute while they move data, so their time is instruction time).
# ---------------------------------------------------------------------------------------------
const LP{T} = Core.LLVMPtr{T,1}
@inline memtime() = ccall("llvm.amdgcn.s.memrealtime", llvmcall, UInt64, ())
@inline dcinv() = ccall("llvm.amdgcn.s.dcache.inv", llvmcall, Cvoid, ())
@inline ssleep() = ccall("llvm.amdgcn.s.sleep", llvmcall, Cvoid, (Int32,), Int32(2))
@inline atadd(p::LP{UInt32}, v::UInt32) = Base.llvmcall("%r = atomicrmw add ptr addrspace(1) %0, i32 %1 syncscope(\"agent\") acq_rel, align 4\nret i32 %r", UInt32, Tuple{LP{UInt32},UInt32}, p, v)
@inline ldacq(p::LP{UInt32}) = Base.llvmcall("%r = load atomic i32, ptr addrspace(1) %0 syncscope(\"agent\") acquire, align 4\nret i32 %r", UInt32, Tuple{LP{UInt32}}, p)
@inline stmono(p::LP{UInt32}, v::UInt32) = Base.llvmcall("store atomic i32 %1, ptr addrspace(1) %0 syncscope(\"agent\") monotonic, align 4\nret void", Cvoid, Tuple{LP{UInt32},UInt32}, p, v)
@inline ldacq_sys(p::LP{UInt32}) = Base.llvmcall("%r = load atomic volatile i32, ptr addrspace(1) %0 acquire, align 4\nret i32 %r", UInt32, Tuple{LP{UInt32}}, p)
@inline strel_sys(p::LP{UInt32}, v::UInt32) = Base.llvmcall("store atomic volatile i32 %1, ptr addrspace(1) %0 release, align 4\nret void", Cvoid, Tuple{LP{UInt32},UInt32}, p, v)
@inline umin64(p::LP{UInt64}, v::UInt64) = Base.llvmcall("%r = atomicrmw umin ptr addrspace(1) %0, i64 %1 syncscope(\"agent\") monotonic, align 8\nret i64 %r", UInt64, Tuple{LP{UInt64},UInt64}, p, v)
@inline umax64(p::LP{UInt64}, v::UInt64) = Base.llvmcall("%r = atomicrmw umax ptr addrspace(1) %0, i64 %1 syncscope(\"agent\") monotonic, align 8\nret i64 %r", UInt64, Tuple{LP{UInt64},UInt64}, p, v)
@inline fence_rel() = Base.llvmcall("fence syncscope(\"agent\") release\nret void", Cvoid, Tuple{})
@inline fence_acq() = Base.llvmcall("fence syncscope(\"agent\") acquire\nret void", Cvoid, Tuple{})
@inline fence_rel_sys() = Base.llvmcall("fence release\nret void", Cvoid, Tuple{})
@inline fence_acq_sys() = Base.llvmcall("fence acquire\nret void", Cvoid, Tuple{})
@inline opq(v::Int32) = Base.llvmcall("%r = call i32 asm sideeffect \"; opq \$0\", \"=s,0\"(i32 %0)\nret i32 %r", Int32, Tuple{Int32}, v)
# x for the GEMV through scalar buffer loads: the resource's base goes through an opaque (side-effecting) copy at
# each phase, so the loads can't be hoisted above the barrier that orders them after x was written
struct SBuf
    rs::NTuple{4,VecElement{Int32}}
end
@inline function sbuf(p::LP{Float32}, nbytes)
    u = reinterpret(UInt64, p)
    SBuf((VecElement(opq(u % UInt32 % Int32)), VecElement(Int32((u >> 32) & 0xffff)), VecElement(Int32(nbytes)), VecElement(Int32(0x00027000))))
end
@inline sload(rs, off::Int32) = Base.llvmcall(("""
    declare float @llvm.amdgcn.s.buffer.load.f32(<4 x i32>, i32, i32)
    define float @sbl(<4 x i32> %r, i32 %o) #0 {
        %v = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %r, i32 %o, i32 0)
        ret float %v
    }
    attributes #0 = { alwaysinline }
    """, "sbl"), Float32, Tuple{NTuple{4,VecElement{Int32}},Int32}, rs, off)
@inline xc(x::SBuf, i, ::Val{N}) where {N} = ntuple(j -> sload(x.rs, Int32(4) * (i + Int32(j - 1))), Val(N))

# everything a die's kernel reads and writes
struct KA
    tab::LP{Int64}                                         # (address, rows, blocks, type, d address) a matrix; vector addresses
    x::LP{Float32}; xt::LP{Float32}; x16::LP{Float32}; x32::LP{Float32}; part::LP{Float32}; red::LP{Float32}
    u::LP{Float32}; qh::LP{Float32}; gh::LP{Float32}; ap::LP{Float32}; lg::LP{Float32}
    S::LP{Float32}; ring::LP{Float32}; kc::LP{Float16}; vc::LP{Float16}; maxctx::Int32
    hx::LP{Float32}; hf::LP{UInt32}; d::Int32
    bar::LP{UInt32}; acct::LP{UInt64}
    inp::LP{Int32}; ninp::Int32; ngen::Int32; out::LP{Int32}; mode::Int32; nwg::Int32; tok::LP{Int32}
end
const NMAT = 8                                             # matrix (and vector) slots a layer in the table
const TABF = 65 * NMAT * 5                                 # where the f32-vector offsets start in the table
const XSZ = 2 * ND * PMAX * NE                             # host exchange words (2 slots), then the argmax exchange
const TOUT = Int64(1_000_000_000)                          # 10 s of the 100 MHz clock: a wait longer than this aborts
@inline mat(A::KA, l, j) = (e = (Int64(l) * NMAT + j) * 5; (unsafe_load(A.tab, e + 1), Int32(unsafe_load(A.tab, e + 2)), Int32(unsafe_load(A.tab, e + 3)), Int32(unsafe_load(A.tab, e + 4)), unsafe_load(A.tab, e + 5)))
@inline fvec(A::KA, l, j) = reinterpret(LP{Float32}, unsafe_load(A.tab, TABF + Int64(l) * NMAT + j + 1))
@inline ld(p, i) = unsafe_load(p, i + 1)                   # 0-based
@inline st!(p, v, i) = unsafe_store!(p, v, i + 1)
@inline ld(a::AbstractArray, i) = @inbounds a[i+1]           # shared-memory arrays
@inline st!(a::AbstractArray, v, i) = (@inbounds a[i+1] = v; nothing)
@inline aborted(A::KA) = ldacq(A.bar + 8) != UInt32(0)
# grid barrier (count, generation, abort); a wait past TOUT sets abort, after which every barrier passes at once
@inline function gbar!(A::KA)
    fence_rel(); sync_workgroup()
    if workitemIdx().x == 1 && !aborted(A)
        g = ldacq(A.bar + 4)
        if atadd(A.bar, UInt32(1)) == UInt32(A.nwg - 1)
            stmono(A.bar, UInt32(0)); atadd(A.bar + 4, UInt32(1))
        else
            t0 = memtime()
            while ldacq(A.bar + 4) == g
                (memtime() - t0 > TOUT) && (atadd(A.bar + 8, UInt32(1)); break)
                ssleep()
            end
        end
    end
    sync_workgroup(); fence_acq(); dcinv()
end
@inline function acct!(A::KA, seq, t0)                     # thread 0 of a workgroup that worked in phase seq
    workitemIdx().x == 1 || return
    p = A.acct + 8 * (3 + 2Int64(seq)); umin64(p, t0); umax64(p + 8, memtime())
end
@inline wgid() = Int32(workgroupIdx().x) - Int32(1)
@inline tid() = Int32(workitemIdx().x) - Int32(1)

# ---- phases (every workgroup calls each one; the work is spread over workgroups or wavefronts) ----
@inline nsl(nb) = (nb + Int32(1)) >> Int32(1)              # K slices (KS = 512, two blocks) of a matrix
@inline gtid() = Int32(256) * wgid() + tid()
@inline nthr(A::KA) = Int32(256) * A.nwg
# partial sums of matrix slot js[m]: floats after the ones of js[1 .. m - 1]
@inline function pof(A::KA, l, js::NTuple{N,Int32}, m, P) where {N}
    o = Int64(0)
    for i in 1:m-1
        _, rows, nb, _, _ = mat(A, l, js[i]); o += Int64(nsl(nb)) * P * rows
    end
    o
end
@inline function psum(A::KA, o::Int64, ns, rows, P, p, r)  # Σ over the K slices of output row r at position p
    s = 0.0f0; b = A.part + 4o
    for q in Int32(0):ns-Int32(1); s += ld(b, (Int64(q) * P + p) * rows + r); end
    s
end
# the items (64 rows, one K slice) of the matrices js of layer l, over every wavefront of the die
function gemv_ph!(A::KA, l::Int32, js::NTuple{N,Int32}, seq::Int32, ::Val{P}) where {N,P}
    t = tid(); lane = t & Int32(63); gw = Int32(4) * wgid() + (t >> Int32(6)); NW = Int32(4) * A.nwg
    t0 = memtime(); did = false
    xr = sbuf(A.xt, NE * PMAX * 4); x16 = sbuf(A.x16, NE ÷ 16 * PMAX * 4); x32 = sbuf(A.x32, NE ÷ 32 * PMAX * 4)
    base = Int32(0); o = Int64(0)
    for m in 1:N
        woff, rows, nb, typ, doff = mat(A, l, js[m]); ns = nsl(nb); n = (rows >> Int32(6)) * ns
        g = gw + max(Int32(0), cld(base - gw, NW)) * NW
        while g < base + n
            it = g - base; G = it ÷ ns; sl = it - G * ns
            if typ == Int32(GQ5_K)
                gemv5!(A.part + 4o, reinterpret(LP{UInt8}, woff), xr, x32, nb, rows, G, sl, lane, Val(P))
            else
                gemv6!(A.part + 4o, reinterpret(LP{UInt8}, woff), reinterpret(LP{UInt16}, doff), xr, x16, nb, rows, G, sl, lane, Val(P))
            end
            did = true; g += NW
        end
        base += n; o += Int64(ns) * P * rows
    end
    did && lane == Int32(0) && (p = A.acct + 8 * (3 + 2Int64(seq)); umin64(p, t0); umax64(p + 8, memtime()))
    return
end
@inline function done!(A::KA, seq, t0, did)                # a workgroup's part of phase seq into the accounting
    did && tid() == Int32(0) && (p = A.acct + 8 * (3 + 2Int64(seq)); umin64(p, t0); umax64(p + 8, memtime()))
end
# the exchange words: (value bits, exchange number + 1), so a reader polls the data itself (the host memory is zeroed
# before each request)
@inline hxs(A::KA, slot, d) = reinterpret(LP{UInt64}, A.hx + 8 * ((Int64(slot) * ND + d) * PMAX * NE))
@inline st64_sys(p::LP{UInt64}, v::UInt64) = Base.llvmcall("store atomic volatile i64 %1, ptr addrspace(1) %0 monotonic, align 8\nret void", Cvoid, Tuple{LP{UInt64},UInt64}, p, v)
@inline ld64_sys(p::LP{UInt64}) = Base.llvmcall("%r = load atomic volatile i64, ptr addrspace(1) %0 monotonic, align 8\nret i64 %r", UInt64, Tuple{LP{UInt64}}, p)
@inline ldsadd(p, v::Float32) = Base.llvmcall("%r = atomicrmw fadd ptr addrspace(3) %0, float %1 monotonic, align 4\nret float %r", Float32, Tuple{Core.LLVMPtr{Float32,3},Float32}, p, v)
# the embedding value of row r (this die's vocabulary rows, repacked Q5_K), element k
@inline function embval(W::LP{UInt8}, r::Int32, k::Int32)
    b = k >> Int32(8); e = k & Int32(255)
    G = r >> Int32(6); lane = r & Int32(63); base = (G * Int32(20) + b) * Int32(11 * 1024) + Int32(16) * lane
    byte(j) = ld(W, base + (j >> Int32(4)) * Int32(1024) + (j & Int32(15)))
    d = f16(UInt16(byte(Int32(0))) | UInt16(byte(Int32(1))) << 8); dmin = f16(UInt16(byte(Int32(2))) | UInt16(byte(Int32(3))) << 8)
    s = e >> Int32(5); l = e & Int32(31)
    sc, mn = if s < Int32(4)
        byte(Int32(4) + s) & 0x3f, byte(Int32(8) + s) & 0x3f
    else
        (byte(Int32(8) + s) & 0x0f) | (byte(s) >> 6) << 4, (byte(Int32(8) + s) >> 4) | (byte(Int32(4) + s) >> 6) << 4
    end
    qs = byte(Int32(48) + Int32(32) * (s >> Int32(1)) + l); qh = byte(Int32(16) + l)
    q = ((s & Int32(1)) == Int32(0) ? qs & 0x0f : qs >> 4) | ((qh >> s) & 0x01) << 4
    d * Float32(sc) * Float32(q) - dmin * Float32(mn)
end
# The exchange, one thread an element of x: this die's share (the K-slice sums of matrix (l, j), or with j < 0 its
# embedding rows of tok[p]) goes to host memory; then the other dies' shares of the same element are read (polled
# until their exchange numbers match), x (+)= the sum, and the next norm's input is prepared: x w into xt with its
# 16- and 32-sums, and each workgroup's sum of squares a position into red (the consumers apply 1 / rms). Phase seq is
# the computing and writing; seq + 1 is the reading from when the first share arrived (the wait before it is idle).
function xwr_ph!(A::KA, l::Int32, j::Int32, w::LP{Float32}, set::Bool, xs::Int32, seq::Int32, P::Int32)
    sq = @ROCStaticLocalArray(Float32, PMAX, false)
    t = tid(); t < P && st!(sq, 0.0f0, t)
    t0 = memtime()
    sync_workgroup()
    did = false; slot = xs & Int32(1); tag = UInt64(xs + Int32(1)) << 32
    rows = Int32(0); ns = Int32(0); W = reinterpret(LP{UInt8}, Int64(0))
    if j >= Int32(0)
        _, rows, nb, _, _ = mat(A, l, j); ns = nsl(nb)
    else
        woff, _, _, _, _ = mat(A, Int32(64), Int32(0)); W = reinterpret(LP{UInt8}, woff)
    end
    # write
    i = gtid(); vals = 0.0f0
    while i < Int32(NE) * P
        p = i ÷ Int32(NE); k = i - p * Int32(NE)
        v = if j >= Int32(0)
            psum(A, Int64(0), ns, rows, P, p, k)
        else
            r = ld(A.tok, p) - Int32(VPD) * A.d
            r >= Int32(0) && r < Int32(VPD) ? embval(W, r, k) : 0.0f0
        end
        st64_sys(hxs(A, slot, A.d) + 8 * (Int64(p) * NE + k), UInt64(reinterpret(UInt32, v)) | tag)
        did = true; i += nthr(A)
    end
    fence_rel_sys()
    t1 = memtime()
    done!(A, seq, t0, did)
    # read (one element a thread a pass: the same elements as written)
    ta = UInt64(0); i = gtid()
    while i < Int32(NE) * P
        p = i ÷ Int32(NE); k = i - p * Int32(NE); o = Int64(p) * NE + k
        v = set ? 0.0f0 : ld(A.x, o)
        for d in Int32(0):Int32(ND - 1)
            q = hxs(A, slot, d) + 8o; u = ld64_sys(q)
            if u & 0xffffffff00000000 != tag
                tw = memtime()
                while u & 0xffffffff00000000 != tag
                    (memtime() - tw > TOUT || aborted(A)) && (atadd(A.bar + 8, UInt32(1)); break)
                    u = ld64_sys(q)
                end
            end
            v += reinterpret(Float32, UInt32(u & 0xffffffff))
        end
        ta == UInt64(0) && (ta = memtime())
        st!(A.x, v, o); xw = v * ld(w, k); st!(A.xt, xw, Int64(k) * P + p)
        ldsadd(pointer(sq) + 4p, v * v)
        a = xw
        for sh in (Cint(1), Cint(2), Cint(4), Cint(8)); a += AMDGPU.Device.shfl_xor(a, sh); end
        b = a + AMDGPU.Device.shfl_xor(a, Cint(16))
        k & Int32(15) == Int32(0) && st!(A.x16, a, Int64(k >> Int32(4)) * P + p)
        k & Int32(31) == Int32(0) && st!(A.x32, b, Int64(k >> Int32(5)) * P + p)
        i += nthr(A)
    end
    sync_workgroup()
    t < P && st!(A.red, ld(sq, t), Int64(wgid()) * PMAX + t)
    did && tid() == Int32(0) && (pa = A.acct + 8 * (3 + 2Int64(seq + Int32(1))); umin64(pa, ta); umax64(pa + 8, memtime()))
end
# raise this die's exchange counter and wait for the other dies' (workgroup 0; the rest wait at the next barrier):
# the argmax exchange
function xsig!(A::KA, xs::Int32)
    if wgid() == Int32(0) && tid() == Int32(0) && !aborted(A)
        fence_rel_sys(); strel_sys(A.hf + 4A.d, UInt32(xs + Int32(1)))
        t0 = memtime()
        for d in Int32(0):Int32(ND - 1)
            while ldacq_sys(A.hf + 4d) < UInt32(xs + Int32(1))
                (memtime() - t0 > TOUT) && (atadd(A.bar + 8, UInt32(1)); break)
                ssleep()
            end
        end
        fence_acq_sys()
    end
end
# 1 / rms of position p of x (from the workgroups' sums of squares), for each p < P, into the shared rs
@inline function rscale!(A::KA, rs, P)
    t = tid()
    if t < P
        ss = 0.0f0
        for q in Int32(0):A.nwg-Int32(1); ss += ld(A.red, Int64(q) * PMAX + t); end
        st!(rs, 1.0f0 / sqrt(ss / Float32(NE) + EPS), t)
    end
    sync_workgroup()
end
const JQKV = (Int32(0), Int32(1), Int32(2), Int32(3))      # qkv z alpha beta  |  q k v
const JQKVA = (Int32(0), Int32(1), Int32(2))
const JFF = (Int32(5), Int32(6))                            # gate up
# DeltaNet conv + silu, one thread a channel
function conv_ph!(A::KA, l::Int32, seq::Int32, P::Int32)
    t0 = memtime(); rs = @ROCStaticLocalArray(Float32, PMAX, false); rscale!(A, rs, P)
    c = gtid(); did = c < Int32(QKVD)
    if did
        _, rows, nb, _, _ = mat(A, l, Int32(0)); ns = nsl(nb)
        ring = A.ring + 4 * (Int64(ridx(l)) * 3 * QKVD); cw = fvec(A, l, 2)
        r1 = ld(ring, 3c); r2 = ld(ring, 3c + 1); r3 = ld(ring, 3c + 2)
        w1 = ld(cw, 4c); w2 = ld(cw, 4c + 1); w3 = ld(cw, 4c + 2); w4 = ld(cw, 4c + 3)
        for p in Int32(0):P-Int32(1)
            x = psum(A, Int64(0), ns, rows, P, p, c) * ld(rs, p)
            st!(A.u, silu(w1 * r1 + w2 * r2 + w3 * r3 + w4 * x), Int64(p) * QKVD + c)
            r1, r2, r3 = r2, r3, x
        end
        st!(ring, r1, 3c); st!(ring, r2, 3c + 1); st!(ring, r3, 3c + 2)
    end
    done!(A, seq, t0, did)
end
# the gated delta rule, one workgroup a local value head (thread t: row t ÷ 2, columns 64 (t mod 2) ..); output
# rmsnorm(S q / sqrt(128)) ssm_norm silu(z) into xt (the out-projection's x) with its 16- and 32-sums
function rec_ph!(A::KA, l::Int32, seq::Int32, ::Val{P}) where {P}
    qs = @ROCStaticLocalArray(Float32, DK, false); ks = @ROCStaticLocalArray(Float32, DK, false)
    ob = @ROCStaticLocalArray(Float32, DK, false); red = @ROCStaticLocalArray(Float32, RED, false)
    rs = @ROCStaticLocalArray(Float32, PMAX, false)
    lv = wgid(); lv < Int32(VHD) || return
    t0 = memtime(); rscale!(A, rs, Int32(P))
    kh = lv & Int32(3); t = tid(); i = t >> Int32(1); jh = t & Int32(1)
    ri = ridx(l); Sp = A.S + 4 * (Int64(ri) * VHD * DK * DK); so = (Int64(lv) * DK + i) * DK + 64jh
    _, _, nbz, _, _ = mat(A, l, Int32(1)); nsz = nsl(nbz); oz = pof(A, l, JQKV, 2, P)
    _, _, nba, _, _ = mat(A, l, Int32(2)); nsa = nsl(nba); oa = pof(A, l, JQKV, 3, P); ob4 = pof(A, l, JQKV, 4, P)
    dtb = fvec(A, l, 3); ssa = fvec(A, l, 4); sn = fvec(A, l, 5)
    s = ntuple(j -> ld(Sp, so + j - 1), Val(64))
    isc = 1.0f0 / sqrt(Float32(DK))
    for p in Int32(0):Int32(P - 1)
        ub = Int64(p) * QKVD
        qv = t < Int32(DK) ? ld(A.u, ub + DK * kh + t) : 0.0f0
        kv = t < Int32(DK) ? ld(A.u, ub + 512 + DK * kh + t) : 0.0f0
        nq = sqrt(wgsum(qv * qv, red)); nk = sqrt(wgsum(kv * kv, red))
        if t < Int32(DK)
            st!(qs, qv / max(nq, EPS), t); st!(ks, kv / max(nk, EPS), t)
        end
        sync_workgroup()
        x = psum(A, oa, nsa, Int32(64), P, p, lv) * ld(rs, p) + ld(dtb, lv); sp = x > 20.0f0 ? x : log1p(exp(x))
        a = exp(ld(ssa, lv) * sp); bt = 1.0f0 / (1.0f0 + exp(-psum(A, ob4, nsa, Int32(64), P, p, lv) * ld(rs, p)))
        s = let s = s; ntuple(j -> s[j] * a, Val(64)) end
        e = 0.0f0
        for j in 1:64; e = fma(s[j], ld(ks, Int32(64) * jh + Int32(j - 1)), e); end
        e += AMDGPU.Device.shfl_xor(e, Cint(1))
        e = bt * (ld(A.u, ub + 1024 + DK * lv + i) - e)
        s = let s = s, e = e; ntuple(j -> fma(e, ld(ks, Int32(64) * jh + Int32(j - 1)), s[j]), Val(64)) end
        o = 0.0f0
        for j in 1:64; o = fma(s[j], ld(qs, Int32(64) * jh + Int32(j - 1)), o); end
        o = (o + AMDGPU.Device.shfl_xor(o, Cint(1))) * isc
        r = 1.0f0 / sqrt(wgsum(jh == Int32(0) ? o * o : 0.0f0, red) / Float32(DK) + EPS)
        if jh == Int32(0)
            z = psum(A, oz, nsz, Int32(DK * VHD), P, p, DK * lv + i) * ld(rs, p)
            v = o * r * ld(sn, i) * silu(z); k = DK * lv + i
            st!(A.xt, v, Int64(k) * P + p); st!(ob, v, i)
        end
        sync_workgroup()
        if t < Int32(4)
            a2 = 0.0f0; b2 = 0.0f0
            for e2 in Int32(0):Int32(15); a2 += ld(ob, Int32(32) * t + e2); b2 += ld(ob, Int32(32) * t + Int32(16) + e2); end
            st!(A.x32, a2 + b2, Int64(4lv + t) * P + p); st!(A.x16, a2, Int64(8lv + 2t) * P + p); st!(A.x16, b2, Int64(8lv + 2t + 1) * P + p)
        end
        sync_workgroup()
    end
    for j in 1:64; st!(Sp, s[j], so + j - 1); end
    done!(A, seq, t0, true)
end
# attention input: rmsnorm and neox rope of the 6 query heads and the KV head; the key and value into the cache
function aprep_ph!(A::KA, l::Int32, pos0::Int32, seq::Int32, P::Int32)
    L = @ROCStaticLocalArray(Float32, HD, false); red = @ROCStaticLocalArray(Float32, RED, false)
    t0 = memtime(); rs = @ROCStaticLocalArray(Float32, PMAX, false); rscale!(A, rs, P)
    did = false; t = tid()
    _, rq, nbq, _, _ = mat(A, l, Int32(0)); nsq = nsl(nbq)
    _, rk, nbk, _, _ = mat(A, l, Int32(1)); nsk = nsl(nbk); ok = pof(A, l, JQKVA, 2, P)
    _, rv, nbv, _, _ = mat(A, l, Int32(2)); nsv = nsl(nbv); ov = pof(A, l, JQKVA, 3, P)
    qn = fvec(A, l, 2); kn = fvec(A, l, 3); koff = (Int64(aidx(l)) * A.maxctx) * HD
    it = wgid()
    while it < Int32(7) * P
        hx = it % Int32(7); p = it ÷ Int32(7)
        val = (hx < Int32(6) ? psum(A, Int64(0), nsq, rq, P, p, Int32(512) * hx + t) : psum(A, ok, nsk, rk, P, p, t)) * ld(rs, p)
        w = hx < Int32(6) ? ld(qn, t) : ld(kn, t)
        v = val * (1.0f0 / sqrt(wgsum(val * val, red) / Float32(HD) + EPS)) * w
        st!(L, v, t)
        sync_workgroup()
        if t < Int32(NROT)
            ii = t & Int32(31); th = Float64(pos0 + p) * THETA^(-2.0 * Float64(ii) / NROT); c = Float32(cos(th)); sn = Float32(sin(th))
            a = ld(L, ii); b = ld(L, ii + Int32(32))
            v = t < Int32(32) ? a * c - b * sn : b * c + a * sn
        end
        if hx < Int32(6)
            st!(A.qh, v, Int64(p) * 1536 + HD * hx + t)
            st!(A.gh, psum(A, Int64(0), nsq, rq, P, p, Int32(512) * hx + Int32(HD) + t) * ld(rs, p), Int64(p) * 1536 + HD * hx + t)
        else
            st!(A.kc, Float16(v), koff + Int64(pos0 + p) * HD + t)
            st!(A.vc, Float16(psum(A, ov, nsv, rv, P, p, t) * ld(rs, p)), koff + Int64(pos0 + p) * HD + t)
        end
        sync_workgroup()
        did = true; it += A.nwg
    end
    done!(A, seq, t0, did)
end
# one (query head, position, CH keys) a workgroup step: (max, sum, Σ e^(s - max) v) into ap
function attn_ph!(A::KA, l::Int32, pos0::Int32, seq::Int32, P::Int32)
    q = @ROCStaticLocalArray(Float32, HD, false); ps = @ROCStaticLocalArray(Float32, CH, false)
    red = @ROCStaticLocalArray(Float32, RED, false)
    t0 = memtime(); did = false; t = tid(); nch = cld(pos0 + P, Int32(CH)); koff = (Int64(aidx(l)) * A.maxctx) * HD
    it = wgid()
    while it < Int32(6) * P * nch
        x = it ÷ nch; c = it - x * nch; hx = x % Int32(6); p = x ÷ Int32(6)
        st!(q, ld(A.qh, Int64(p) * 1536 + HD * hx + t), t)
        sync_workgroup()
        k0 = c * Int32(CH); nk = pos0 + p + Int32(1); kk = k0 + t; sc = -Inf32
        if kk < nk
            s = 0.0f0; kb = koff + Int64(kk) * HD
            for d in Int32(0):Int32(HD - 1); s = fma(ld(q, d), Float32(ld(A.kc, kb + d)), s); end
            sc = s * (1.0f0 / sqrt(Float32(HD)))
        end
        m = wgmax(sc, red); e = kk < nk ? exp(sc - m) : 0.0f0; lsum = wgsum(e, red)
        st!(ps, e, t)
        sync_workgroup()
        acc = 0.0f0
        for j in Int32(0):Int32(CH - 1)
            k0 + j < nk || break
            acc = fma(ld(ps, j), Float32(ld(A.vc, koff + Int64(k0 + j) * HD + t)), acc)
        end
        o = (Int64(x) * nch + c) * (HD + 2)
        t == Int32(0) && (st!(A.ap, m, o); st!(A.ap, lsum, o + 1))
        st!(A.ap, acc, o + 2 + t)
        sync_workgroup()
        did = true; it += A.nwg
    end
    done!(A, seq, t0, did)
end
# combine the key chunks, times sigmoid(gate): into xt (the out-projection's x) with its 16- and 32-sums
function acomb_ph!(A::KA, pos0::Int32, seq::Int32, P::Int32)
    ob = @ROCStaticLocalArray(Float32, HD, false)
    t0 = memtime(); did = false; t = tid(); nch = cld(pos0 + P, Int32(CH)); it = wgid()
    while it < Int32(6) * P
        hx = it % Int32(6); p = it ÷ Int32(6); x = Int32(6) * p + hx
        M = -Inf32
        for c in Int32(0):nch-Int32(1); M = max(M, ld(A.ap, (Int64(x) * nch + c) * (HD + 2))); end
        L = 0.0f0; acc = 0.0f0
        for c in Int32(0):nch-Int32(1)
            o = (Int64(x) * nch + c) * (HD + 2); m = ld(A.ap, o)
            if m > -Inf32
                f = exp(m - M); L += ld(A.ap, o + 1) * f; acc += ld(A.ap, o + 2 + t) * f
            end
        end
        g = ld(A.gh, Int64(p) * 1536 + HD * hx + t); v = acc / L / (1.0f0 + exp(-g)); k = HD * hx + t
        st!(A.xt, v, Int64(k) * P + p); st!(ob, v, t)
        sync_workgroup()
        if t < Int32(8)
            a2 = 0.0f0; b2 = 0.0f0
            for e in Int32(0):Int32(15); a2 += ld(ob, Int32(32) * t + e); b2 += ld(ob, Int32(32) * t + Int32(16) + e); end
            st!(A.x32, a2 + b2, Int64(8hx + t) * P + p); st!(A.x16, a2, Int64(16hx + 2t) * P + p); st!(A.x16, b2, Int64(16hx + 2t + 1) * P + p)
        end
        sync_workgroup()
        did = true; it += A.nwg
    end
    done!(A, seq, t0, did)
end
# FFN: silu(gate) up into xt (the down-projection's x) with its 16- and 32-sums (Q6_K or Q5_K)
function act_ph!(A::KA, l::Int32, seq::Int32, P::Int32)
    t0 = memtime(); rs = @ROCStaticLocalArray(Float32, PMAX, false); rscale!(A, rs, P)
    did = false; i = gtid()
    _, rg, nbg, _, _ = mat(A, l, Int32(5)); ns = nsl(nbg); ou = pof(A, l, JFF, 2, P)
    while i < Int32(4352 ÷ 32) * P
        g = i ÷ P; p = i - g * P; a = 0.0f0; b = 0.0f0
        for e in Int32(0):Int32(31)
            k = Int32(32) * g + e
            v = silu(psum(A, Int64(0), ns, rg, P, p, k) * ld(rs, p)) * (psum(A, ou, ns, rg, P, p, k) * ld(rs, p))
            st!(A.xt, v, Int64(k) * P + p); e < Int32(16) ? (a += v) : (b += v)
        end
        st!(A.x16, a, Int64(2g) * P + p); st!(A.x16, b, Int64(2g + 1) * P + p); st!(A.x32, a + b, Int64(g) * P + p)
        did = true; i += nthr(A)
    end
    done!(A, seq, t0, did)
end
# the head: K-slice sums (logits into lg in score mode) and each workgroup's best (logit, row) a position
function hmax_ph!(A::KA, pos0::Int32, seq::Int32, P::Int32, pfirst::Int32)
    bv = @ROCStaticLocalArray(Float32, RED, false); bi = @ROCStaticLocalArray(Int32, RED, false)
    t0 = memtime(); rs = @ROCStaticLocalArray(Float32, PMAX, false); rscale!(A, rs, P)
    did = false; t = tid()
    _, rows, nb, _, _ = mat(A, Int32(64), Int32(1)); ns = nsl(nb)
    for p in pfirst:P-Int32(1)
        v = -Inf32; ix = Int32(0); r = gtid()
        while r < Int32(VPD)
            lg = psum(A, Int64(0), ns, rows, P, p, r) * ld(rs, p)
            A.mode == Int32(1) && st!(A.lg, lg, Int64(pos0 + p) * VPD + r)
            lg > v && (v = lg; ix = r); did = true; r += nthr(A)
        end
        for sh in (Cint(32), Cint(16), Cint(8), Cint(4), Cint(2), Cint(1))
            v2 = AMDGPU.Device.shfl_xor(v, sh); i2 = AMDGPU.Device.shfl_xor(ix, sh)
            (v2 > v || (v2 == v && i2 < ix)) && (v = v2; ix = i2)
        end
        sync_workgroup()
        t & Int32(63) == Int32(0) && (st!(bv, v, t >> Int32(6)); st!(bi, ix, t >> Int32(6)))
        sync_workgroup()
        if t == Int32(0)
            for w in Int32(1):Int32(3)
                (ld(bv, w) > v || (ld(bv, w) == v && ld(bi, w) < ix)) && (v = ld(bv, w); ix = ld(bi, w))
            end
            o = Int64(A.nwg) * PMAX + 2 * (Int64(wgid()) * PMAX + p)
            st!(A.red, v, o); st!(A.red, reinterpret(Float32, ix), o + 1)
        end
    end
    done!(A, seq, t0, did)
end
@inline hxa(A::KA, slot, d) = A.hx + 8XSZ + 4 * ((Int64(slot) * ND + d) * PMAX * 2)
# workgroup 0: the die's best over the workgroups, into the argmax exchange (row as a global vocabulary id)
function hred_ph!(A::KA, slot::Int32, seq::Int32, p::Int32)
    (wgid() == Int32(0) && tid() == Int32(0)) || return
    t0 = memtime(); v = -Inf32; ix = Int32(0)
    for q in Int32(0):A.nwg-Int32(1)
        o = Int64(A.nwg) * PMAX + 2 * (Int64(q) * PMAX + p); v2 = ld(A.red, o); i2 = reinterpret(Int32, ld(A.red, o + 1))
        (v2 > v || (v2 == v && i2 < ix)) && (v = v2; ix = i2)
    end
    hb = hxa(A, slot, A.d); st!(hb, v, 0); st!(hb, reinterpret(Float32, ix + Int32(VPD) * A.d), 1)
    done!(A, seq, t0, true)
end
# the token: the best of the four dies; tok[0] for the next forward, and die 0 writes it out
function tok_ph!(A::KA, slot::Int32, seq::Int32, step::Int32)
    (wgid() == Int32(0) && tid() == Int32(0)) || return
    fence_acq_sys(); t0 = memtime(); v = -Inf32; ix = Int32(0)
    for d in Int32(0):Int32(ND - 1)
        hb = hxa(A, slot, d); v2 = ld(hb, 0); i2 = reinterpret(Int32, ld(hb, 1))
        (v2 > v || (v2 == v && i2 < ix)) && (v = v2; ix = i2)
    end
    st!(A.tok, ix, 0); A.d == Int32(0) && st!(A.out, ix, step)
    done!(A, seq, t0, true)
end
function tokin_ph!(A::KA, pos0::Int32, P::Int32)           # the forward's input tokens (prefill, score)
    wgid() == Int32(0) && tid() < P && st!(A.tok, ld(A.inp, pos0 + tid()), tid())
end

# ---- one forward of P positions at pos0 (P is PMAX or 1) ----
function fwd!(A::KA, pos0::Int32, seq::Int32, xs::Int32, head::Bool, frominp::Bool, step::Int32, ::Val{P}) where {P}
    Pi = Int32(P)
    frominp && (tokin_ph!(A, pos0, Pi); gbar!(A))
    xwr_ph!(A, Int32(0), Int32(-1), fvec(A, 0, 0), true, xs, seq, Pi); seq += Int32(2); xs += Int32(1); gbar!(A)
    nlr = Int32(ldacq(A.bar + 12)); nl = nlr == Int32(0) ? Int32(NL) : nlr   # bar[3]: layers to run (0: all), for debugging
    for l in Int32(0):nl-Int32(1)
        if isrec(l)
            gemv_ph!(A, l, JQKV, seq, Val(P)); seq += Int32(1); gbar!(A)
            conv_ph!(A, l, seq, Pi); seq += Int32(1); gbar!(A)
            rec_ph!(A, l, seq, Val(P)); seq += Int32(1); gbar!(A)
        else
            gemv_ph!(A, l, JQKVA, seq, Val(P)); seq += Int32(1); gbar!(A)
            aprep_ph!(A, l, pos0, seq, Pi); seq += Int32(1); gbar!(A)
            attn_ph!(A, l, pos0, seq, Pi); seq += Int32(1); gbar!(A)
            acomb_ph!(A, pos0, seq, Pi); seq += Int32(1); gbar!(A)
        end
        gemv_ph!(A, l, (Int32(4),), seq, Val(P)); seq += Int32(1); gbar!(A)
        xwr_ph!(A, l, Int32(4), fvec(A, l, 1), false, xs, seq, Pi); seq += Int32(2); xs += Int32(1); gbar!(A)
        gemv_ph!(A, l, JFF, seq, Val(P)); seq += Int32(1); gbar!(A)
        act_ph!(A, l, seq, Pi); seq += Int32(1); gbar!(A)
        gemv_ph!(A, l, (Int32(7),), seq, Val(P)); seq += Int32(1); gbar!(A)
        wn = l < nl - Int32(1) ? fvec(A, l + Int32(1), 0) : fvec(A, 64, 0)
        xwr_ph!(A, l, Int32(7), wn, false, xs, seq, Pi); seq += Int32(2); xs += Int32(1); gbar!(A)
    end
    if head
        gemv_ph!(A, Int32(64), (Int32(1),), seq, Val(P)); seq += Int32(1); gbar!(A)
        gen = A.mode == Int32(0)
        hmax_ph!(A, pos0, seq, Pi, gen ? Pi - Int32(1) : Int32(0)); seq += Int32(1); gbar!(A)
        if gen
            hred_ph!(A, xs & Int32(1), seq, Pi - Int32(1)); seq += Int32(1); gbar!(A)
            xsig!(A, xs); gbar!(A)
            tok_ph!(A, xs & Int32(1), seq, step); seq += Int32(1); xs += Int32(1); gbar!(A)
        end
    end
    seq, xs
end

# the request: prefill (or score) the input PMAX positions a forward (the rest one at a time), then decode ngen
function k_req!(A::KA)
    tid() == Int32(0) && umin64(A.acct, memtime())
    seq = Int32(0); xs = Int32(0); pos = Int32(0); step = Int32(0)
    while pos < A.ninp
        P = A.ninp - pos >= Int32(PMAX) ? Int32(PMAX) : Int32(1)
        head = A.mode == Int32(1) || (pos + P == A.ninp && A.ngen > Int32(0))
        seq, xs = P == Int32(PMAX) ? fwd!(A, pos, seq, xs, head, true, step, Val(PMAX)) : fwd!(A, pos, seq, xs, head, true, step, Val(1))
        pos += P
    end
    tid() == Int32(0) && umax64(A.acct + 16, memtime())   # the prefill ends
    A.ngen > Int32(0) && (step += Int32(1))
    while step < A.ngen
        seq, xs = fwd!(A, pos, seq, xs, true, false, step, Val(1)); pos += Int32(1); step += Int32(1)
    end
    tid() == Int32(0) && umax64(A.acct + 8, memtime())
    return
end

# ---- weights and buffers of a die ----
struct DLayer
    rec::Bool
    an::ROCVector{Float32}; pn::ROCVector{Float32}
    gate::DMat; up::DMat; down::DMat
    m1::DMat; m2::DMat; m3::DMat; m4::Union{DMat,Nothing}  # qkv z alpha beta | q k v -
    out::DMat
    w1::ROCVector{Float32}; w2::ROCVector{Float32}         # conv, dt bias | q norm, k norm
    w3::ROCVector{Float32}; w4::ROCVector{Float32}         # ssm_a, ssm_norm | -
end
f32(g, name) = (t = g.t[name]; copy(reinterpret(Float32, view(g.map, t.off+1:t.off+4prod(t.dims)))))
const NOV = Float32[]
function load_layer(g::GGUF, l, d)
    p = "blk.$l."; all20 = [(0, 20)]
    gate = DMat(g, p * "ffn_gate.weight", [(4352d, 4352)], all20); up = DMat(g, p * "ffn_up.weight", [(4352d, 4352)], all20)
    down = DMat(g, p * "ffn_down.weight", [(0, NE)], [(17d, 17)])
    an = ROCArray(f32(g, p * "attn_norm.weight")); pn = ROCArray(f32(g, p * "post_attention_norm.weight"))
    if isrec(l)
        vh = [16j + 4d for j in 0:2]                       # the first value head of each 4-head run
        heads = [16j + 4d + i for j in 0:2 for i in 0:3]
        qkv = DMat(g, p * "attn_qkv.weight", [(512d, 512); (2048 + 512d, 512); [(4096 + 128h, 512) for h in vh]], all20)
        z = DMat(g, p * "attn_gate.weight", [(128h, 512) for h in vh], all20)
        al = DMat(g, p * "ssm_alpha.weight", [(h, 4) for h in vh], all20); be = DMat(g, p * "ssm_beta.weight", [(h, 4) for h in vh], all20)
        out = DMat(g, p * "ssm_out.weight", [(0, NE)], [(8j + 2d, 2) for j in 0:2])
        chans = [512d:512d+511; 2048+512d:2048+512d+511; [4096+128h:4096+128h+511 for h in vh]...]
        cw = f32(g, p * "ssm_conv1d.weight"); cwd = [cw[4c+k] for c in chans for k in 1:4]
        dtb = f32(g, p * "ssm_dt.bias")[heads .+ 1]; ssa = f32(g, p * "ssm_a")[heads .+ 1]
        return DLayer(true, an, pn, gate, up, down, qkv, z, al, be, out, ROCArray(cwd), ROCArray(dtb), ROCArray(ssa), ROCArray(f32(g, p * "ssm_norm.weight")))
    end
    q = DMat(g, p * "attn_q.weight", [(3072d, 3072)], all20)
    k = DMat(g, p * "attn_k.weight", [(256d, 256)], all20); v = DMat(g, p * "attn_v.weight", [(256d, 256)], all20)
    out = DMat(g, p * "attn_output.weight", [(0, NE)], [(6d, 6)])
    DLayer(false, an, pn, gate, up, down, q, k, v, nothing, out, ROCArray(f32(g, p * "attn_q_norm.weight")), ROCArray(f32(g, p * "attn_k_norm.weight")), ROCArray(NOV), ROCArray(NOV))
end
struct Die
    dev::HIPDevice; d::Int
    L::Vector{DLayer}; emb::DMat; head::DMat; onrm::ROCVector{Float32}
    gb::GBuf; tok::ROCVector{Int32}
    x::ROCVector{Float32}; h::ROCVector{Float32}; y::ROCVector{Float32}; tb::ROCVector{Float32}
    b1::ROCVector{Float32}; b2::ROCVector{Float32}; b3::ROCVector{Float32}; b4::ROCVector{Float32}
    u::ROCVector{Float32}; o::ROCVector{Float32}; qh::ROCVector{Float32}; gh::ROCVector{Float32}
    ap::ROCVector{Float32}; logits::ROCVector{Float32}
end
# a request's state on a die: DeltaNet states and conv rings, the attention cache (f16) up to maxctx
struct DState
    S::ROCVector{Float32}; ring::ROCVector{Float32}; kc::ROCVector{Float16}; vc::ROCVector{Float16}; maxctx::Int
end
DState(maxctx) = DState(AMDGPU.zeros(Float32, 48 * VHD * DK * DK), AMDGPU.zeros(Float32, 48 * QKVD * 3),
    AMDGPU.zeros(Float16, 16 * maxctx * HD), AMDGPU.zeros(Float16, 16 * maxctx * HD), maxctx)
function load_die(g::GGUF, dev, d, L; maxctx = 4096)
    AMDGPU.device!(dev)
    emb = DMat(g, "token_embd.weight", VPD * d, VPD); head = DMat(g, "output.weight", VPD * d, VPD)
    z(n) = AMDGPU.zeros(Float32, n)
    Die(dev, d, L, emb, head, ROCArray(f32(g, "output_norm.weight")), GBuf(VPD * 10 * PMAX, NE, PMAX), AMDGPU.zeros(Int32, PMAX),
        z(NE * PMAX), z(NE * PMAX), z(NE * PMAX), z(NE * PMAX), z(4352 * PMAX), z(4352 * PMAX), z(256 * PMAX), z(64 * PMAX),
        z(QKVD * PMAX), z(1536 * PMAX), z(1536 * PMAX), z(1536 * PMAX), z(6 * PMAX * cld(maxctx, CH) * (HD + 2)), z(VPD * PMAX))
end

# ---- the forward: P tokens at positions pos0 .. pos0 + P - 1 ----
struct Engine
    dies::Vector{Die}
end
# x on every die += the sum over dies of y (the first n floats); through the host for now
function allreduce_add!(E::Engine, n; set = false)
    s = zeros(Float32, n); hb = Vector{Float32}(undef, n)
    for D in E.dies
        AMDGPU.device!(D.dev); @x(copyto!(hb, 1, D.y, 1, n)); s .+= hb
    end
    for D in E.dies
        AMDGPU.device!(D.dev); @x copyto!(D.tb, 1, s, 1, n)
        set && @x(copyto!(D.x, 1, s, 1, n))
        set || @k @roc groupsize = 256 gridsize = cld(n, 256) k_add!(D.x, D.tb, Int32(n))
    end
end
function forward!(E::Engine, st::Vector{DState}, toks::Vector{Int}, pos0::Int; logits = true, chk = nothing)
    P = length(toks); n = NE * P
    for D in E.dies
        AMDGPU.device!(D.dev); @x copyto!(D.tok, 1, Int32.(toks), 1, P)
        @k @roc groupsize = 256 gridsize = (20, P) k_embed!(D.y, D.emb.W, D.tok, Int32(VPD * D.d), Int32(VPD))
    end
    allreduce_add!(E, n; set = true)
    chk === nothing || chk(-1, :embed)
    for l in 0:NL-1
        for (D, S) in zip(E.dies, st)
            AMDGPU.device!(D.dev); W = D.L[l+1]
            @k @roc groupsize = 256 gridsize = P k_rms!(D.h, D.x, W.an, Int32(NE))
            if W.rec
                ri = ridx(l)
                gemv!(D.b1, D.gb, W.m1, D.h, P); gemv!(D.b2, D.gb, W.m2, D.h, P); gemv!(D.b3, D.gb, W.m3, D.h, P); gemv!(D.b4, D.gb, W.m4, D.h, P)
                @k @roc groupsize = 256 gridsize = cld(QKVD, 256) k_dnconv!(D.u, D.b1, view(S.ring, 3QKVD*ri+1:3QKVD*(ri+1)), W.w1, Int32(P), Int32(QKVD))
                @k @roc groupsize = 256 gridsize = VHD k_dnrec!(D.o, view(S.S, VHD*DK*DK*ri+1:VHD*DK*DK*(ri+1)), D.u, D.b2, D.b3, D.b4, W.w2, W.w3, W.w4, Int32(P))
            else
                ai = aidx(l); koff = Int32(ai * S.maxctx * HD); nch = cld(pos0 + P, CH)
                gemv!(D.b1, D.gb, W.m1, D.h, P); gemv!(D.b2, D.gb, W.m2, D.h, P); gemv!(D.b3, D.gb, W.m3, D.h, P)
                @k @roc groupsize = 256 gridsize = (7, P) k_attn_prep!(D.qh, D.gh, S.kc, S.vc, D.b1, D.b2, D.b3, W.w1, W.w2, Int32(pos0), koff)
                @k @roc groupsize = 256 gridsize = (6P, nch) k_attn!(D.ap, D.qh, S.kc, S.vc, koff, Int32(pos0), Int32(nch))
                @k @roc groupsize = 256 gridsize = 6P k_attn_comb!(D.o, D.ap, D.gh, Int32(nch))
            end
            gemv!(D.y, D.gb, W.out, D.o, P)
        end
        allreduce_add!(E, n)
        chk === nothing || chk(l, :mixer)
        for D in E.dies
            AMDGPU.device!(D.dev); W = D.L[l+1]
            @k @roc groupsize = 256 gridsize = P k_rms!(D.h, D.x, W.pn, Int32(NE))
            gemv!(D.b1, D.gb, W.gate, D.h, P); gemv!(D.b2, D.gb, W.up, D.h, P)
            @k @roc groupsize = 256 gridsize = cld(4352P, 256) k_silu_mul!(D.b1, D.b2, Int32(4352P))
            gemv!(D.y, D.gb, W.down, D.b1, P)
        end
        allreduce_add!(E, n)
        chk === nothing || chk(l, :ffn)
    end
    logits || return nothing
    Lg = Matrix{Float32}(undef, NV, P); hb = Vector{Float32}(undef, VPD * P)
    for D in E.dies
        AMDGPU.device!(D.dev)
        @k @roc groupsize = 256 gridsize = P k_rms!(D.h, D.x, D.onrm, Int32(NE))
        gemv!(D.logits, D.gb, D.head, D.h, P)
        @x copyto!(hb, 1, D.logits, 1, VPD * P)
        for p in 1:P; Lg[VPD*D.d+1:VPD*(D.d+1), p] = view(hb, (p-1)*VPD+1:p*VPD); end
    end
    Lg
end
# the layers are read in file order, each one for all four dies at once (one pass over the file: the GGUF is on a
# spinning disk and sentry's RAM can't cache all of it)
function load_engine(path; maxctx = 4096)
    g = gguf(path); devs = AMDGPU.devices()
    length(devs) >= ND || error("need $ND dies, have $(length(devs))")
    t0 = time(); Ls = [Vector{DLayer}(undef, NL) for _ in 1:ND]
    onall(f) = @sync for d in 0:ND-1
        Threads.@spawn (AMDGPU.device!(devs[d+1]); f(d))
    end
    for l in 0:NL-1
        onall(d -> (Ls[d+1][l+1] = load_layer(g, l, d)))
    end
    dies = Vector{Die}(undef, ND)
    onall(d -> (dies[d+1] = load_die(g, devs[d+1], d, Ls[d+1]; maxctx)))
    @printf("weights resident on %d dies (%.0f s)\n", ND, time() - t0)
    Engine(dies), g
end
# the first stage (layer, mixer / ffn) whose buffers hold a non-finite value, die by die: forward with a check after
# every all-reduce
function probe(io, E::Engine, toks::Vector{Int})
    st = newstate(E, CH); P = length(toks); bad = Ref(false)
    function chk(l, stage)
        bad[] && return
        for D in E.dies
            AMDGPU.device!(D.dev)
            for f in (:x, :h, :y, :b1, :b2, :b3, :b4, :u, :o, :qh, :gh)
                v = Array(getfield(D, f))
                if !all(isfinite, v)
                    k = findfirst(!isfinite, v)
                    println(io, "layer $l $stage: die $(D.d) $f[$k] = $(v[k]) (max finite $(maximum(abs, filter(isfinite, v); init = 0f0)))")
                    bad[] = true
                end
            end
        end
        bad[] || l < 0 || l % 8 == 0 && println(io, "layer $l $stage ok, |x| max $(maximum(abs, Array(E.dies[1].x)[1:NE*P]))")
    end
    forward!(E, st, toks, 0; logits = false, chk)
    bad[] || println(io, "all finite")
end
newstate(E::Engine, maxctx) = [(AMDGPU.device!(D.dev); DState(maxctx)) for D in E.dies]

# ---- the persistent kernel's host side ----
addr(a) = reinterpret(Int64, UInt64(pointer(a)))
# a die's table: every matrix's address and shape, every vector's address (the weights stay where they are)
function ktable(D::Die)
    tab = zeros(Int64, TABF + 65 * NMAT)
    m!(l, j, m) = (e = (l * NMAT + j) * 5; tab[e+1:e+5] = [addr(m.W), m.rows, m.K ÷ 256, m.typ, m.typ == GQ6_K ? addr(m.D) : 0])
    v!(l, j, a) = (tab[TABF+l*NMAT+j+1] = addr(a))
    for l in 0:NL-1
        W = D.L[l+1]
        m!(l, 0, W.m1); m!(l, 1, W.m2); m!(l, 2, W.m3); W.rec && m!(l, 3, W.m4)
        m!(l, 4, W.out); m!(l, 5, W.gate); m!(l, 6, W.up); m!(l, 7, W.down)
        v!(l, 0, W.an); v!(l, 1, W.pn); v!(l, 2, W.w1); v!(l, 3, W.w2)
        W.rec && (v!(l, 4, W.w3); v!(l, 5, W.w4))
    end
    m!(64, 0, D.emb); m!(64, 1, D.head); v!(64, 0, D.onrm)
    ROCArray(tab)
end
# the kernel's own buffers on a die, its table, and its workgroup count (all resident: the occupancy times 56 CUs)
struct KDie
    tab::ROCVector{Int64}; x::ROCVector{Float32}; xt::ROCVector{Float32}; x16::ROCVector{Float32}; x32::ROCVector{Float32}
    part::ROCVector{Float32}; red::ROCVector{Float32}; u::ROCVector{Float32}; qh::ROCVector{Float32}; gh::ROCVector{Float32}
    ap::ROCVector{Float32}; tok::ROCVector{Int32}; bar::ROCVector{UInt32}
end
const NWG = 256                                             # buffers sized for up to this many workgroups
function KDie(D::Die, maxctx)
    AMDGPU.device!(D.dev); z(n) = AMDGPU.zeros(Float32, n)
    KDie(ktable(D), z(NE * PMAX), z(NE * PMAX), z(NE ÷ 16 * PMAX), z(NE ÷ 32 * PMAX), z(VPD * 10 * PMAX), z(3 * NWG * PMAX),
        z(QKVD * PMAX), z(1536 * PMAX), z(1536 * PMAX), z(6 * PMAX * cld(maxctx, CH) * (HD + 2)), AMDGPU.zeros(Int32, PMAX), AMDGPU.zeros(UInt32, 4))
end
# page-locked host memory every die maps: the exchanges, the counters, the generated tokens
struct HostX
    p::Ptr{UInt8}; nbytes::Int
end
const HXOUT = 8XSZ + 4 * (2 * ND * PMAX * 2)                 # byte offsets: counters, then tokens
const HXTOK = HXOUT + 64
function HostX(maxgen)
    n = HXTOK + 4 * maxgen; r = Ref{Ptr{Cvoid}}(C_NULL)
    AMDGPU.HIP.hipHostMalloc(r, n, AMDGPU.HIP.hipHostMallocPortable | AMDGPU.HIP.hipHostMallocMapped | AMDGPU.HIP.hipHostMallocCoherent)
    HostX(Ptr{UInt8}(r[]), n)
end
lp(T, p::Ptr) = reinterpret(LP{T}, p)
lp(T, a::ROCArray) = reinterpret(LP{T}, pointer(a))
struct PEngine
    E::Engine; K::Vector{KDie}; hx::HostX; maxctx::Int; nwg::Int
end
function PEngine(E::Engine; maxctx = 4096, maxgen = 4096)
    K = [KDie(D, maxctx) for D in E.dies]; hx = HostX(maxgen)
    AMDGPU.device!(E.dies[1].dev)
    k = @roc launch = false maxthreads = 256 waves_per_eu = KWPE[] k_req!(kargs(E, K, hx, 1, nothing, AMDGPU.zeros(UInt64, 4), AMDGPU.zeros(Int32, 1), 0, 0, 0, 1, nothing))
    nb = Ref{Cint}(0); AMDGPU.HIP.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(nb, k.fun, 256, 0)
    nwg = Int(nb[]) * 56
    0 < nwg <= NWG || error("persistent kernel: $(nb[]) workgroups a CU")
    PEngine(E, K, hx, maxctx, nwg)
end
function kargs(E::Engine, K, hx::HostX, d, st, acct, inp, ninp, ngen, mode, nwg, lg)
    D = E.dies[d]; k = K[d]
    S, ring, kc, vc, mc = st === nothing ? (k.x, k.x, reinterpret(Float16, k.x), reinterpret(Float16, k.x), 0) : (st.S, st.ring, st.kc, st.vc, st.maxctx)
    KA(lp(Int64, k.tab), lp(Float32, k.x), lp(Float32, k.xt), lp(Float32, k.x16), lp(Float32, k.x32), lp(Float32, k.part), lp(Float32, k.red),
       lp(Float32, k.u), lp(Float32, k.qh), lp(Float32, k.gh), lp(Float32, k.ap), lg === nothing ? lp(Float32, k.x) : lp(Float32, lg),
       lp(Float32, S), lp(Float32, ring), lp(Float16, kc), lp(Float16, vc), Int32(mc),
       lp(Float32, hx.p), lp(UInt32, hx.p + HXOUT), Int32(d - 1), lp(UInt32, k.bar), lp(UInt64, acct),
       lp(Int32, inp), Int32(ninp), Int32(ngen), lp(Int32, hx.p + HXTOK), Int32(mode), Int32(nwg), lp(Int32, k.tok))
end
# one request on the four dies: one dispatch a die. mode 0: greedy, ngen tokens after inp; mode 1: score inp (the
# logits of every position into lg, a die's vocabulary rows). Returns the tokens and each die's (instruction,
# idle, wall) seconds.
function request!(PE::PEngine, st::Vector{DState}, inp::Vector{Int}, ngen::Int, mode::Int; lgs = nothing)
    E = PE.E; nf = cld(length(inp), PMAX) + PMAX + ngen; nphase = nf * 1100
    ccall(:memset, Ptr{Cvoid}, (Ptr{UInt8}, Cint, Csize_t), PE.hx.p, 0, HXTOK)   # exchange words and counters
    accts = ROCVector{UInt64}[]
    for d in 1:ND
        AMDGPU.device!(E.dies[d].dev)
        a = AMDGPU.zeros(UInt64, 3 + 2nphase); ah = zeros(UInt64, 3 + 2nphase); ah[1] = typemax(UInt64); ah[4:2:end] .= typemax(UInt64); copyto!(a, ah); push!(accts, a)
        copyto!(PE.K[d].bar, UInt32[0, 0, 0, NLRUN[]])
        ib = ROCArray(Int32.(inp)); AMDGPU.synchronize()
        A = kargs(E, PE.K, PE.hx, d, st[d], a, ib, length(inp), ngen, mode, PE.nwg, lgs === nothing ? nothing : lgs[d])
        @roc groupsize = 256 gridsize = PE.nwg maxthreads = 256 waves_per_eu = KWPE[] k_req!(A)
    end
    res = NTuple{3,Float64}[]; tpre = 0.0; tdec = 0.0
    for d in 1:ND
        AMDGPU.device!(E.dies[d].dev); AMDGPU.synchronize()
        ah = Array(accts[d]); wall = (ah[2] - ah[1]) / 1e8; iv = Tuple{Float64,Float64}[]
        d == 1 && (tpre = (ah[3] - ah[1]) / 1e8; tdec = (ah[2] - ah[3]) / 1e8)
        for s in 0:nphase-1
            b = ah[4+2s]; e = ah[5+2s]; b == typemax(UInt64) && continue
            push!(iv, ((b - ah[1]) / 1e8, (e - ah[1]) / 1e8))
        end
        ins = union_len(iv)                                # the phases overlap (an exchange writes while it reads)
        push!(res, (ins, wall - ins, wall))
        AMDGPU.unsafe_free!(accts[d])
    end
    ab = [(AMDGPU.device!(D.dev); Array(K.bar)[3]) for (D, K) in zip(E.dies, PE.K)]
    any(!=(0), ab) && error("persistent kernel aborted (a wait passed 10 s): $ab")
    toks = [Int(unsafe_load(Ptr{Int32}(PE.hx.p + HXTOK), i)) for i in 1:ngen]
    toks, res, tpre, tdec
end
if !isdefined(Main, :KWPE)
    const KWPE = Ref(2)                                    # waves a SIMD the persistent kernel is compiled for (2: 38.8 vs 31.6 tok/s decode at 1)
end
if !isdefined(Main, :NLRUN)
    const NLRUN = Ref(0)                                   # layers the persistent kernel runs (0: all), for debugging
end
idlepct(res) = 100sum(r[2] for r in res) / sum(r[3] for r in res)

# ---- requests through the persistent kernel ----
function pgen(io, PE::PEngine, vocab, ids::Vector{Int}, n::Int)
    st = newstate(PE.E, cld(length(ids) + n, CH) * CH)
    toks, res, tp, td = request!(PE, st, ids, n, 0)
    foreach(s -> foreach(AMDGPU.unsafe_free!, (s.S, s.ring, s.kc, s.vc)), st)
    println(io, "ids: ", join(toks, " ")); println(io, "text: ", repr(detok(vocab, toks)))
    @printf(io, "prefill %.2f tok/s\ndecode %.2f tok/s\nidle time %.1f%%\n", length(ids) / tp, (n - 1) / td, idlepct(res))
end
function pkl(io, PE::PEngine, ids, refpath, n)
    E = PE.E; st = newstate(E, cld(n, CH) * CH)
    lgs = [(AMDGPU.device!(D.dev); AMDGPU.zeros(Float32, n * VPD)) for D in E.dies]
    _, res, tp, _ = request!(PE, st, ids[1:n], 0, 1; lgs)
    Lg = Matrix{Float32}(undef, NV, n)
    for (D, a) in zip(E.dies, lgs)
        AMDGPU.device!(D.dev); h = Array(a)
        for p in 1:n; Lg[VPD*D.d+1:VPD*(D.d+1), p] = view(h, (p-1)*VPD+1:p*VPD); end
        AMDGPU.unsafe_free!(a)
    end
    foreach(s -> foreach(AMDGPU.unsafe_free!, (s.S, s.ring, s.kc, s.vc)), st)
    ref = Mmap.mmap(refpath, Matrix{Float16}, (NV, n)); kls = Float64[]; top = 0
    for p in 1:n
        r = Float64.(view(ref, :, p)); o = Float64.(view(Lg, :, p)); all(isfinite, r) || continue
        lr = r .- (maximum(r) + log(sum(exp.(r .- maximum(r))))); lo = o .- (maximum(o) + log(sum(exp.(o .- maximum(o)))))
        push!(kls, sum(exp.(lr) .* (lr .- lo))); top += argmax(r) == argmax(o)
    end
    m = length(kls)
    @printf(io, "KL vs BF16 over %d positions: mean %.4f, median %.4f, max %.3f; top-1 %.3f\nprefill %.2f tok/s\nidle time %.1f%%\n",
        m, sum(kls) / m, sort(kls)[cld(m, 2)], maximum(kls), top / m, n / tp, idlepct(res))
end

# ---- byte-level BPE detokenizing (GPT-2 byte map) ----
const BDEC = let bs = [33:126; 161:172; 174:255], cs = copy(bs), n = 0
    for b in 0:255
        b in bs || (push!(bs, b); push!(cs, 256 + n); n += 1)
    end
    Dict(Char(c) => UInt8(b) for (b, c) in zip(bs, cs))
end
detok(vocab, ids) = String(UInt8[get(BDEC, ch, UInt8('?')) for i in ids for ch in vocab[i+1]])

# ---- greedy generation from token ids: prefill PMAX at a time, then one token a forward ----
function gen(io, E::Engine, vocab, ids::Vector{Int}, n::Int)
    gen1(devnull, E, vocab, ids, n)                        # warm: compile every kernel, fill the event pools
    account(() -> gen1(devnull, E, vocab, ids, n), E)
    _, res = account(() -> gen1(io, E, vocab, ids, n), E)
    print_account(io, res)
end
function gen1(io, E::Engine, vocab, ids::Vector{Int}, n::Int)
    st = newstate(E, cld(length(ids) + n, CH) * CH)
    t0 = time(); Lg = nothing
    for p0 in 0:PMAX:length(ids)-1
        Lg = forward!(E, st, ids[p0+1:min(p0 + PMAX, end)], p0)
    end
    tp = time() - t0; out = Int[argmax(view(Lg, :, size(Lg, 2))) - 1]; t1 = time()
    for i in 1:n-1
        Lg = forward!(E, st, [out[end]], length(ids) + i - 1)
        push!(out, argmax(view(Lg, :, 1)) - 1)
    end
    td = time() - t1
    println(io, "ids: ", join(out, " "))
    println(io, "text: ", repr(detok(vocab, out)))
    @printf(io, "prefill %.2f tok/s, decode %.2f tok/s\n", length(ids) / tp, (n - 1) / td)
end

# ---- the resident server: weights load once; each connection sends one command line ----
#   kl N REF.logits.f16        KL against BF16 reference logits over the first N tokens of REF's sequence
#   gen N ID ID ...            greedy: N tokens after the given token ids
#   echo 'gen 32 9707 11' | socat - UNIX-CONNECT:SOCK
# the server's engine, kept across reloads of this file
if !isdefined(Main, :SRV)
    const SRV = Ref{Any}(nothing)
end
if !isdefined(Main, :PK)
    const PK = Ref{Any}(nothing)                           # the persistent kernel engine (pinit)
end
function serve(path, sock)
    E, g = load_engine(path; maxctx = 4096)
    SRV[] = (E = E, g = g, vocab = g.kv["tokenizer.ggml.tokens"], file = abspath(@__FILE__))
    lk = ReentrantLock()
    ispath(sock) && rm(sock)
    srv = listen(sock); println("serving on $sock"); flush(stdout)
    while true
        c = accept(srv)
        @async try
            line = readline(c)
            lock(lk) do
                Base.invokelatest(handle, c, line)        # the newest definitions, after a reload
            end
        catch e
            println(c, "error: ", sprint(showerror, e, catch_backtrace()))
        finally
            close(c)
        end
    end
end
#   kl N REF.logits.f16        KL against BF16 reference logits over the first N tokens of REF's sequence
#   gen N ID ID ...            greedy: N tokens after the given token ids
#   reload                     include this file again (new code, the weights stay resident)
#   eval CODE                  run Julia code in the server, print its value
function handle(c, line)
    a = split(line); s = SRV[]
    if a[1] == "kl"
        n = parse(Int, a[2]); ids = Int.(reinterpret(Int32, read(replace(a[3], ".logits.f16" => ".ids"))))
        kl_test(c, s.E, ids, a[3], n)
    elseif a[1] == "gen"
        gen(c, s.E, s.vocab, parse.(Int, a[3:end]), parse(Int, a[2]))
    elseif a[1] == "pinit"
        t0 = time(); PK[] = PEngine(s.E); @printf(c, "persistent kernel: %d workgroups a die, compiled in %.0f s\n", PK[].nwg, time() - t0)
    elseif a[1] == "pgen"
        pgen(c, PK[], s.vocab, parse.(Int, a[3:end]), parse(Int, a[2]))
    elseif a[1] == "pkl"
        n = parse(Int, a[2]); ids = Int.(reinterpret(Int32, read(replace(a[3], ".logits.f16" => ".ids"))))
        pkl(c, PK[], ids, a[3], n)
    elseif a[1] == "reload"
        Base.include(Main, s.file); println(c, "reloaded ", s.file)
    elseif a[1] == "eval"
        println(c, repr(Base.include_string(Main, line[6:end])))
    else
        println(c, "unknown command: ", a[1])
    end
end

# ---- KL of the forward against BF16 reference logits ([position][vocab] f16) over the first n tokens ----
function kl_test(io, E::Engine, ids, refpath, n)
    st = newstate(E, cld(n, CH) * CH)
    ref = Mmap.mmap(refpath, Matrix{Float16}, (NV, n))
    kls = Float64[]; top = 0; skip = 0; t0 = time()
    for p0 in 0:PMAX:n-1
        P = min(PMAX, n - p0)
        Lg = forward!(E, st, ids[p0+1:p0+P], p0)
        for p in 1:P
            r = Float64.(view(ref, :, p0 + p)); o = Float64.(view(Lg, :, p))
            all(isfinite, r) || (skip += 1; continue)              # positions the reference run lost
            lr = r .- (maximum(r) + log(sum(exp.(r .- maximum(r))))); lo = o .- (maximum(o) + log(sum(exp.(o .- maximum(o)))))
            push!(kls, sum(exp.(lr) .* (lr .- lo))); top += argmax(r) == argmax(o)
        end
    end
    m = length(kls)
    @printf(io, "KL vs BF16 over %d positions (%d with a non-finite reference skipped): mean %.4f, median %.4f, max %.3f; top-1 agreement %.3f (%.1f s)\n",
        m, skip, sum(kls) / m, sort(kls)[cld(m, 2)], maximum(kls), top / m, time() - t0)
end

# ---------------------------------------------------------------------------------------------
# --bench-gemv: each format against the host dequant on a few rows, then GB/s of weight streamed at P = 1, 4, 8
# ---------------------------------------------------------------------------------------------
function bench_gemv(g::GGUF)
    for name in ("blk.0.ffn_up.weight", "blk.0.ffn_down.weight", "blk.3.attn_output.weight")
        t = g.t[name]; K = t.dims[1]; R = nrows(t)
        m = DMat(g, name)
        for P in (1, 2, 4, 8)
            xh = randn(Float32, K * P); x = ROCArray(xh); y = AMDGPU.zeros(Float32, R * P); gb = GBuf(m, P)
            for _ in 1:5; gemv!(y, gb, m, x, P); end; AMDGPU.synchronize()          # warm (clocks ramp)
            yh = Array(y); err = 0.0; ref = 0.0
            for r in (0, 1, 63, 64, R ÷ 2 + 5, R - 1)
                wr = dq_row(t, rowview(g, t, r, 1))
                for p in 1:P
                    e = sum(Float64.(wr) .* Float64.(view(xh, (p-1)*K+1:p*K)))
                    err = max(err, abs(yh[(p-1)*R+r+1] - e)); ref = max(ref, abs(e))
                end
            end
            n = 50; tm = @elapsed (for _ in 1:n; gemv!(y, gb, m, x, P); end; AMDGPU.synchronize())
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
    if (i = findfirst(==("--kl"), args)) !== nothing
        n = parse(Int, args[i+1]); ref = args[i+2]
        ids = Int.(reinterpret(Int32, read(replace(ref, ".logits.f16" => ".ids"))))
        E, _ = load_engine(args[k+1]; maxctx = n)
        return kl_test(stdout, E, ids, ref, n)
    end
    (i = findfirst(==("--serve"), args)) !== nothing && return serve(args[k+1], args[i+1])
    error("nothing to do: --bench-gemv")
end
if abspath(PROGRAM_FILE) == @__FILE__() && SRV[] === nothing
    main(ARGS)
end
