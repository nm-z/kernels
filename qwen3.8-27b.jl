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
# Each workgroup (4 wavefronts) takes 16 rows, 4 a wavefront; rows must be a multiple of 16. x goes through shared
# memory 1,024 elements at a time, one plane per position (xs[p KC + k]). A wavefront step covers two 256-element
# blocks: half the lanes each, a lane 8 elements of its block (one 32-bit word of Q5_K quants or two 16-bit words of
# Q6_K quants). The lane reads its 8 x values for all P positions into registers once and uses them for its 4 rows.
const KC = 1024
@inline ld32(W, off) = unsafe_load(reinterpret(Core.LLVMPtr{UInt32,1}, pointer(W) + off))
@inline ld16(W, off) = unsafe_load(reinterpret(Core.LLVMPtr{UInt16,1}, pointer(W) + off))
@inline f16(u::UInt16) = Float32(reinterpret(Float16, u))
@inline function stage_x!(xs, x, t, kb, ldx, ::Val{P}) where {P}
    i = t
    @inbounds while i < Int32(KC * P)
        p = i ÷ Int32(KC)
        xs[i+Int32(1)] = x[p*ldx+kb+(i-p*Int32(KC))+Int32(1)]
        i += Int32(256)
    end
end
@inline xv(xs, k, ::Val{P}) where {P} = ntuple(p -> @inbounds(xs[(p-1)*KC+k+1]), Val(P))
@inline fmav(a::NTuple{P,Float32}, w, x::NTuple{P,Float32}) where {P} = ntuple(p -> fma(w, x[p], a[p]), Val(P))
@inline function wsum!(y, acc::NTuple{P,Float32}, lane, row, ldy) where {P}
    for p in 1:P
        v = AMDGPU.Device.wfred(+, acc[p])
        lane == Int32(0) && @inbounds(y[(p-1)*ldy+row+1] = v)
    end
end
function k_gemv_q5k!(y, W, x, nb::Int32, ldx::Int32, ldy::Int32, ::Val{P}) where {P}
    xs = @ROCStaticLocalArray(Float32, 8192, false)
    t = Int32(workitemIdx().x) - Int32(1); lane = t & Int32(63); wave = t >> Int32(6)
    r0 = (Int32(workgroupIdx().x) - Int32(1)) * Int32(16) + wave * Int32(4)
    bs = lane >> Int32(5); jj = (lane >> Int32(3)) & Int32(3); e0 = (lane & Int32(7)) * Int32(4); s0 = Int32(2) * jj
    rbytes = nb * Int32(176)
    Base.Cartesian.@nexprs 4 r -> (acc_r = ntuple(_ -> 0.0f0, Val(P)))
    for kb in Int32(0):Int32(KC):nb*Int32(256)-Int32(1)
        sync_workgroup(); stage_x!(xs, x, t, kb, ldx, Val(P)); sync_workgroup()
        for st in Int32(0):Int32(1)
            bl = Int32(2) * st + bs; b = (kb >> Int32(8)) + bl
            kx = Int32(256) * bl + Int32(64) * jj + e0
            Base.Cartesian.@nexprs 4 i -> (xl_i = xv(xs, kx + Int32(i - 1), Val(P)); xh_i = xv(xs, kx + Int32(31 + i), Val(P)))
            Base.Cartesian.@nexprs 4 r -> begin
                o = (r0 + Int32(r - 1)) * rbytes + b * Int32(176)
                dd = ld32(W, o); d = f16(UInt16(dd & 0xffff)); dmin = f16(UInt16(dd >> 16))
                sc0, mn0 = smk4(W, o + Int32(4), s0); sc1, mn1 = smk4(W, o + Int32(4), s0 + Int32(1))
                a0 = d * sc0; m0 = dmin * mn0; a1 = d * sc1; m1 = dmin * mn1
                qs = ld32(W, o + Int32(48) + Int32(32) * jj + e0); qh = ld32(W, o + Int32(16) + e0) >> s0
                Base.Cartesian.@nexprs 4 i -> begin
                    bt = qs >> (8 * (i - 1)); hb = qh >> (8 * (i - 1))
                    acc_r = fmav(acc_r, a0 * Float32((bt & 0x0f) | (hb & 0x1) << 4) - m0, xl_i)
                    acc_r = fmav(acc_r, a1 * Float32(((bt >> 4) & 0x0f) | ((hb >> 1) & 0x1) << 4) - m1, xh_i)
                end
            end
        end
    end
    Base.Cartesian.@nexprs 4 r -> wsum!(y, acc_r, lane, r0 + Int32(r - 1), ldy)
    return
end
function k_gemv_q6k!(y, W, x, nb::Int32, ldx::Int32, ldy::Int32, ::Val{P}) where {P}
    xs = @ROCStaticLocalArray(Float32, 8192, false)
    t = Int32(workitemIdx().x) - Int32(1); lane = t & Int32(63); wave = t >> Int32(6)
    r0 = (Int32(workgroupIdx().x) - Int32(1)) * Int32(16) + wave * Int32(4)
    # a lane: elements l2, l2 + 1 of half h in each of the half's four 32-element groups
    bs = lane >> Int32(5); h = (lane >> Int32(4)) & Int32(1); l2 = (lane & Int32(15)) * Int32(2); is = l2 >> Int32(4)
    rbytes = nb * Int32(210)
    Base.Cartesian.@nexprs 4 r -> (acc_r = ntuple(_ -> 0.0f0, Val(P)))
    for kb in Int32(0):Int32(KC):nb*Int32(256)-Int32(1)
        sync_workgroup(); stage_x!(xs, x, t, kb, ldx, Val(P)); sync_workgroup()
        for st in Int32(0):Int32(1)
            bl = Int32(2) * st + bs; b = (kb >> Int32(8)) + bl
            kx = Int32(256) * bl + Int32(128) * h + l2
            Base.Cartesian.@nexprs 2 e -> Base.Cartesian.@nexprs 4 g -> (x_g_e = xv(xs, kx + Int32(32 * (g - 1) + e - 1), Val(P)))
            Base.Cartesian.@nexprs 4 r -> begin
                o = (r0 + Int32(r - 1)) * rbytes + b * Int32(210)
                d = f16(ld16(W, o + Int32(208)))
                q0 = ld16(W, o + Int32(64) * h + l2); q1 = ld16(W, o + Int32(64) * h + l2 + Int32(32)); qh = ld16(W, o + Int32(128) + Int32(32) * h + l2)
                so = o + Int32(192) + Int32(8) * h + is
                Base.Cartesian.@nexprs 4 g -> (c_g = d * Float32(reinterpret(Int8, ldb(W, so + Int32(2 * (g - 1))))))
                Base.Cartesian.@nexprs 2 e -> begin
                    b0 = q0 >> (8 * (e - 1)); b1 = q1 >> (8 * (e - 1)); bh = qh >> (8 * (e - 1))
                    acc_r = fmav(acc_r, c_1 * Float32(Int32((b0 & 0x0f) | (bh & 0x03) << 4) - Int32(32)), x_1_e)
                    acc_r = fmav(acc_r, c_2 * Float32(Int32((b1 & 0x0f) | ((bh >> 2) & 0x03) << 4) - Int32(32)), x_2_e)
                    acc_r = fmav(acc_r, c_3 * Float32(Int32(((b0 >> 4) & 0x0f) | ((bh >> 4) & 0x03) << 4) - Int32(32)), x_3_e)
                    acc_r = fmav(acc_r, c_4 * Float32(Int32(((b1 >> 4) & 0x0f) | ((bh >> 6) & 0x03) << 4) - Int32(32)), x_4_e)
                end
            end
        end
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
        @roc groupsize = 256 gridsize = g k_gemv_q5k!(y, m.W, x, nb, Int32(ldx), Int32(ldy), Val(P))
    elseif m.typ == GQ6_K
        @roc groupsize = 256 gridsize = g k_gemv_q6k!(y, m.W, x, nb, Int32(ldx), Int32(ldy), Val(P))
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
