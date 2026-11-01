# qwen3.8-27b.jl: Qwen3.8-27B (qwen35: 48 Gated DeltaNet + 16 gated-attention layers, hidden 5120) on sentry's four
# Radeon Pro V340 dies (gfx900, wave64, 56 CUs, 8 GB HBM2 each), AMDGPU.jl. One cooperative persistent kernel a die
# runs a whole request: the prefill (GEMMs over as many positions as fit) then the decode loop (GEMVs).
# Model: mradermacher/Qwen3.8-27B-GGUF Q5_K_M (static K-quants: Q5_K, Q6_K, F32; its MTP layer is blk.64).
# Reports only: prefill tok/s, decode tok/s, idle time %.
#   julia -t 24 qwen3.8-27b.jl --model FILE --serve SOCK
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
# Device GEMV for decode and verify widths: y[p][r] = sum_k W[r][k] x[p][k] for P positions (1..8), the weights read
# once for all P. One lane per row: a wavefront takes 64 rows, a workgroup 256, and K is cut into slices of KS
# elements, one workgroup per (256 rows, slice); a wavefront needs nothing from the others (no shared memory, no
# barrier). The weights are repacked once at load so a wavefront's 16-byte loads of its 64 rows are one contiguous
# 1 KB run: row r = 64 G + lane, block b, 16-byte chunk c of the block at ((G nb + b) nchunk + c) 1024 + 16 lane
# (Q6_K's 2-byte d goes to its own array, so its other 208 bytes are 13 chunks). x is the same for every lane, so it
# is read with scalar loads into SGPRs, which the multiply-adds take as an operand; it is laid out [k][p]
# (xt[k P + p]) with the sums of each scale group beside it (xsm[g P + p]), both made by the exchange (xwr_ph!). A lane sums q x
# over each scale group and applies the group's sum of x once: Q5_K group value = d sc Σqx - dmin mn Σx, Q6_K
# = d sc (Σqx - 32 Σx), so a weight element costs a byte extract and a convert before its P multiply-adds. Each
# slice writes partial sums part[slice][p][row]; the consumer adds the slices (psum).
# ---------------------------------------------------------------------------------------------
function union_len(iv)                                     # total length of a set of intervals
    isempty(iv) && return 0.0
    sort!(iv); tot = 0.0; (a, b) = iv[1]
    for (c, d) in iv[2:end]
        c > b ? (tot += b - a; (a, b) = (c, d)) : (b = max(b, d))
    end
    tot + b - a
end
const KS = 512
const NCH = Dict(GQ5_K => 11, GQ6_K => 13)                 # 16-byte chunks a block
const GSZ = Dict(GQ5_K => 32, GQ6_K => 16)                 # elements a scale group
const V4U = NTuple{4,VecElement{UInt32}}
# memory either as a device array or a raw pointer
@inline wptr(a) = pointer(a)
@inline wptr(a::Core.LLVMPtr) = a
@inline pst!(a, v, i) = unsafe_store!(wptr(a), v, i)
@inline dld(a, i) = unsafe_load(wptr(a), i)
# 16 weight bytes a lane, non-temporal: each weight is read once a token, and a decode forward streams about 76 MB a
# layer through the 4 MB L2, which would otherwise evict the code, x, the partial sums and the state every phase
@inline function ld128(W, off)
    p = reinterpret(Core.LLVMPtr{V4U,1}, wptr(W) + (off % UInt32))
    v = Base.llvmcall(("""
        define <4 x i32> @ldnt(ptr addrspace(1) %p) #0 {
            %v = load <4 x i32>, ptr addrspace(1) %p, align 16, !nontemporal !0
            ret <4 x i32> %v
        }
        attributes #0 = { alwaysinline }
        !0 = !{i32 1}
        """, "ldnt"), V4U, Tuple{Core.LLVMPtr{V4U,1}}, p)
    (v[1].value, v[2].value, v[3].value, v[4].value)
end
# q, made to depend on dep (an empty asm the compiler can't see through): a word's quants are converted only after
# the word LA(P) back has been multiplied in, so the scheduler can't hold a whole block of converted floats
@inline vgate(q::UInt32, dep::Float32) = Base.llvmcall("""
    %r = call i32 asm "; vgate \$0 \$2", "=v,0,v"(i32 %0, float %1)
    ret i32 %r""", UInt32, Tuple{UInt32,Float32}, q, dep)
@inline LA(P) = clamp(8 ÷ P, 1, 8)
@inline f16(u) = Float32(reinterpret(Float16, UInt16(u)))
@inline fmav(a::NTuple{P,Float32}, w, x, o) where {P} = ntuple(p -> fma(w, x[o+p], a[p]), Val(P))
# Q5_K (scale, min) of sub-block s from the three scale words
@inline function scmn(u0, u1, u2, ::Val{s}) where {s}
    if s < 4
        return Float32((u0 >> (8s)) & 0x3f), Float32((u1 >> (8s)) & 0x3f)
    end
    h = 8 * (s - 4)
    Float32(((u2 >> h) & 0x0f) | ((u0 >> (h + 6)) & 0x03) << 4), Float32(((u2 >> (h + 4)) & 0x0f) | ((u1 >> (h + 6)) & 0x03) << 4)
end
# one wavefront: rows 64 G .. 64 G + 63, K slice sl, into part
@inline function gemv5!(part, W, xt, xsm, nb::Int32, rows::Int32, G::Int32, sl::Int32, lane::Int32, ::Val{P}) where {P}
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
# one wavefront: rows 64 G .. 64 G + 63, K slice sl, into part
@inline function gemv6!(part, W, D, xt, xsm, nb::Int32, rows::Int32, G::Int32, sl::Int32, lane::Int32, ::Val{P}) where {P}
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
# a weight matrix on the device, repacked: rows r0 .. r0 + n of a GGUF tensor
# host -> device copies that finish before they return, the source kept alive until then (an asynchronous copy
# that outlives its source reads freed host pages: a GPU fault)
function dev(a::Array)
    d = ROCArray{eltype(a)}(undef, size(a))
    GC.@preserve a begin; copyto!(d, a); AMDGPU.synchronize(); end
    d
end
function up!(d::ROCArray, a::Array, n = length(a))
    GC.@preserve a begin; copyto!(d, 1, a, 1, n); AMDGPU.synchronize(); end
    d
end
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
    DMat(t.typ, 256nbk, np, dev(W), dev(D))
end

# ---------------------------------------------------------------------------------------------
# The model on four dies. Die d holds DeltaNet key heads 4d .. 4d + 3 and the 12 value heads that read them
# (h = 16j + 4d + i, j = 0..2, i = 0..3; value head h reads key head h mod 16), attention query heads 6d .. 6d + 5 and
# KV head d, FFN rows (and down-projection columns) 4352d .., and a quarter of the vocabulary rows of the embedding
# and the head. The residual stream x is on every die; each mixer and FFN ends in a partial sum on each die, added
# across the dies (the exchange, xwr_ph!).
# ---------------------------------------------------------------------------------------------
const NE, NFF, NV, NL = 5120, 17408, 248320, 64
const DK, NKH, NVH, HD, NH, NKV, NROT = 128, 16, 48, 256, 24, 4, 64
const EPS, THETA = 1.0f-6, 1.0e7
const ND, VPD = 4, 62080                                   # dies, vocabulary rows a die
const QKVD, VHD = 2560, 12                                 # DeltaNet qkv channels and value heads a die
const CH = 256                                             # attention keys a workgroup
const MTPL = 65                                            # the MTP layer's slot in the table
isrec(l) = l != MTPL && (l + 1) % 4 != 0
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
@inline function wgsum2(a, b, red)                         # two sums at once (red: 8 shared floats)
    sa = AMDGPU.Device.wfred(+, a); sb = AMDGPU.Device.wfred(+, b); t = Int32(workitemIdx().x) - Int32(1)
    sync_workgroup()
    t & Int32(63) == Int32(0) && @inbounds(begin red[(t>>6)+1] = sa; red[(t>>6)+5] = sb end)
    sync_workgroup()
    @inbounds (red[1] + red[2] + red[3] + red[4], red[5] + red[6] + red[7] + red[8])
end
@inline function wgmax(v, red)
    s = AMDGPU.Device.wfred(max, v); t = Int32(workitemIdx().x) - Int32(1)
    sync_workgroup()
    t & Int32(63) == Int32(0) && @inbounds(red[(t>>6)+1] = s)
    sync_workgroup()
    @inbounds max(max(red[1], red[2]), max(red[3], red[4]))
end

# ---------------------------------------------------------------------------------------------
# The cooperative persistent kernel: one dispatch a die runs a whole request (every forward of the prefill, then
# every decode step, the token choice included). All its workgroups are resident at once and step through the
# same sequence of phases with a grid barrier (gbar!) between phases. Between dies, each mixer and FFN output, the
# embedding and the argmax go through an exchange: each die writes its share into page-locked host memory that
# every die maps, raises its counter, waits for the other three, and reads theirs.
# Time accounting: for each phase, the first start and the last end over the workgroups that worked in it
# (memrealtime: 27 MHz here, its rate measured at startup) go into acct; their union is instruction time, the rest of the die's request time
# is idle (the exchanges compute while they move data, so their time is instruction time).
# ---------------------------------------------------------------------------------------------
const LP{T} = Core.LLVMPtr{T,1}
@inline memtime() = ccall("llvm.amdgcn.s.memrealtime", llvmcall, UInt64, ())
@inline dcinv() = ccall("llvm.amdgcn.s.dcache.inv", llvmcall, Cvoid, ())
const SLP = 2                                              # s_sleep units (64 clocks) between a waiter's polls
@inline ssleep() = ccall("llvm.amdgcn.s.sleep", llvmcall, Cvoid, (Int32,), Int32(SLP))
@inline atadd(p::LP{UInt32}, v::UInt32) = Base.llvmcall("%r = atomicrmw add ptr addrspace(1) %0, i32 %1 syncscope(\"agent\") acq_rel, align 4\nret i32 %r", UInt32, Tuple{LP{UInt32},UInt32}, p, v)
@inline ldacq(p::LP{UInt32}) = Base.llvmcall("%r = load atomic i32, ptr addrspace(1) %0 syncscope(\"agent\") acquire, align 4\nret i32 %r", UInt32, Tuple{LP{UInt32}}, p)
@inline ldmono(p::LP{UInt32}) = Base.llvmcall("%r = load atomic i32, ptr addrspace(1) %0 syncscope(\"agent\") monotonic, align 4\nret i32 %r", UInt32, Tuple{LP{UInt32}}, p)
@inline stmono(p::LP{UInt32}, v::UInt32) = Base.llvmcall("store atomic i32 %1, ptr addrspace(1) %0 syncscope(\"agent\") monotonic, align 4\nret void", Cvoid, Tuple{LP{UInt32},UInt32}, p, v)
@inline ldacq_sys(p::LP{UInt32}) = Base.llvmcall("%r = load atomic volatile i32, ptr addrspace(1) %0 acquire, align 4\nret i32 %r", UInt32, Tuple{LP{UInt32}}, p)
@inline strel_sys(p::LP{UInt32}, v::UInt32) = Base.llvmcall("store atomic volatile i32 %1, ptr addrspace(1) %0 release, align 4\nret void", Cvoid, Tuple{LP{UInt32},UInt32}, p, v)
@inline umin64(p::LP{UInt64}, v::UInt64) = Base.llvmcall("%r = atomicrmw umin ptr addrspace(1) %0, i64 %1 syncscope(\"agent\") monotonic, align 8\nret i64 %r", UInt64, Tuple{LP{UInt64},UInt64}, p, v)
@inline umax64(p::LP{UInt64}, v::UInt64) = Base.llvmcall("%r = atomicrmw umax ptr addrspace(1) %0, i64 %1 syncscope(\"agent\") monotonic, align 8\nret i64 %r", UInt64, Tuple{LP{UInt64},UInt64}, p, v)
@inline fence_rel() = Base.llvmcall("fence syncscope(\"agent\") release\nret void", Cvoid, Tuple{})
@inline fence_acq() = Base.llvmcall("fence syncscope(\"agent\") acquire\nret void", Cvoid, Tuple{})
@inline fence_rel_sys() = Base.llvmcall("fence release\nret void", Cvoid, Tuple{})
@inline fence_acq_sys() = Base.llvmcall("fence acquire\nret void", Cvoid, Tuple{})
@inline opv(v::Int32) = Base.llvmcall("%r = call i32 asm sideeffect \"; opv \$0\", \"=v,0\"(i32 %0)\nret i32 %r", Int32, Tuple{Int32}, v)
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
    sl::LP{Int64}; maxctx::Int32                            # a request slot's state: (S, ring, kc, vc) addresses
    hx::LP{Float32}; hf::LP{UInt32}; d::Int32
    bar::LP{UInt32}; acct::LP{UInt64}
    inp::LP{Int32}; rq::LP{Int32}; nreq::Int32; out::LP{Int32}; mode::Int32; nwg::Int32; tok::LP{Int32}   # rq: (inp offset, ninp, ngen, out offset) a request
    posr::LP{Int32}; posa::LP{Int32}; tks::LP{Int32}         # a forward position's request and position; a request's last token
    npos::Int32                                            # positions a prefill forward (the most that fit)
    ssq::LP{Int64}                                         # sum of squares of x a position (fixed point, 2^-16)
    rid::Int32                                             # request number
    pk::LP{Int32}                                          # a position whose DeltaNet state and conv ring also go to the backup slots
    nem::LP{Int32}; drf::LP{Int32}; msel::LP{Int32}        # a request's tokens out, its draft, the draft's source (0, 1)
    spec::Int32                                            # decode with MTP drafts
end
const NR = 2                                               # requests at once
const DW = 4                                               # the widest decode forward (with MTP: 2 positions a request)
@inline slp(A::KA, r, j) = ld(A.sl, 6Int64(r) + j)      # (S, ring, kc, vc, S backup, ring backup)
const NMAT = 8                                             # matrix (and vector) slots a layer in the table
const TABF = 66 * NMAT * 5                                 # where the f32-vector offsets start in the table
const TOUT = Int64(270_000_000)                            # 10 s of the 27 MHz clock: a wait longer than this aborts
# table entry i (0-based) as an invariant load: the table is written before the kernel, so with a uniform i these are
# scalar loads the compiler can issue together (plain loads were vector loads, each waited for before the next)
@inline ldt(p::LP{Int64}, i::Int64) = Base.llvmcall(("""
    define i64 @ldt(ptr addrspace(1) %p, i64 %i) #0 {
        %q = getelementptr inbounds i64, ptr addrspace(1) %p, i64 %i
        %v = load i64, ptr addrspace(1) %q, align 8, !invariant.load !0
        ret i64 %v
    }
    attributes #0 = { alwaysinline }
    !0 = !{}
    """, "ldt"), Int64, Tuple{LP{Int64},Int64}, p, i)
@inline mat(A::KA, l, j) = (e = (Int64(l) * NMAT + j) * 5; (ldt(A.tab, e), Int32(ldt(A.tab, e + 1)), Int32(ldt(A.tab, e + 2)), Int32(ldt(A.tab, e + 3)), ldt(A.tab, e + 4)))
@inline invf(A::KA, i) = reinterpret(Float64, ldt(A.tab, Int64(TABF + 66NMAT + i)))   # rope inverse frequency i
@inline fvec(A::KA, l, j) = reinterpret(LP{Float32}, ldt(A.tab, Int64(TABF + Int64(l) * NMAT + j)))
@inline ld(p, i) = unsafe_load(p, i + 1)                   # 0-based
@inline st!(p, v, i) = unsafe_store!(p, v, i + 1)
@inline ld(a::AbstractArray, i) = @inbounds a[i+1]           # shared-memory arrays
@inline st!(a::AbstractArray, v, i) = (@inbounds a[i+1] = v; nothing)
@inline aborted(A::KA) = ldacq(A.bar + 8) != UInt32(0)
# grid barrier (count, generation, abort); a wait past TOUT sets abort, after which every barrier passes at once
@inline function gbar!(A::KA)
    fence_rel(); sync_workgroup()
    if workitemIdx().x == 1
        # the generation (same cache line as the count, so it is read before the arrive lands) with a plain load,
        # and the abort flag only while waiting: two acquire loads in a row cost a round trip each
        # (one counter: two levels, groups of 8 on their own cache lines, measured 2.55 us a barrier against 2.0)
        g = ldmono(A.bar + 4)
        if atadd(A.bar, UInt32(1)) == UInt32(A.nwg - 1)
            stmono(A.bar, UInt32(0)); atadd(A.bar + 4, UInt32(1))
        else
            t0 = memtime()
            while ldacq(A.bar + 4) == g
                (memtime() - t0 > TOUT || aborted(A)) && (atadd(A.bar + 8, UInt32(1)); break)
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
@inline nsl(nb) = (nb + Int32(1)) >> Int32(1)              # K slices (KS = 512, two blocks) of a GEMV
@inline isgemm(P) = P > Int32(DW) || P == Int32(3)
@inline nsx(nb, P) = isgemm(P) ? Int32(1) : nsl(nb)       # slices of a matrix phase's output (a GEMM: one)
@inline gtid() = Int32(256) * wgid() + tid()
@inline nthr(A::KA) = Int32(256) * A.nwg
# partial sums of matrix slot js[m]: floats after the ones of js[1 .. m - 1]
@inline function pof(A::KA, l, j0, m, P)                  # where matrix j0 + m - 1's sums start in part (the group j0 ..)
    o = Int64(0)
    for i in Int32(0):Int32(m - 2)
        _, rows, nb, _, _ = mat(A, l, Int32(j0) + i); o += Int64(nsx(nb, P)) * P * rows
    end
    o
end
# Σ over the K slices of output row r at position p: the loads in groups of 8 independent ones (a plain loop waits
# for each load before the next add)
@inline function psum(A::KA, o::Int64, ns, rows, P, p, r)
    s = 0.0f0; b = A.part + 4o; st = Int64(P) * rows; e = Int64(p) * rows + r
    ns == Int32(1) && return ld(b, e)                      # a GEMM's (prefill) one complete slot
    for q in Int32(0):Int32(16):ns-Int32(1)
        v = let q = q; ntuple(j -> ld(b, Int64(min(q + Int32(j - 1), ns - Int32(1))) * st + e), Val(16)) end
        for j in 1:16; s += q + Int32(j - 1) < ns ? v[j] : 0.0f0; end
    end
    s
end
# the items (64 rows, one K slice) of the matrices js of layer l, over every wavefront of the die
function gemv_ph!(A::KA, l::Int32, j0::Int32, nm::Int32, seq::Int32, ::Val{P}, zs::Bool = false) where {P}
    zs && zero_ssq!(A, Int32(P))
    # the wavefront index, uniform (readfirstlane): so is everything an item derives from it, and x is read with
    # scalar loads
    t = tid(); lane = t & Int32(63); gw = AMDGPU.Device.readfirstlane(Int32(4) * wgid() + (t >> Int32(6))); NW = Int32(4) * A.nwg
    t0 = memtime(); did = false
    xr = sbuf(A.xt, NE * 4 * A.npos); x16 = sbuf(A.x16, NE ÷ 4 * A.npos); x32 = sbuf(A.x32, NE ÷ 8 * A.npos)
    base = Int32(0); o = Int64(0)
    for m in Int32(0):nm-Int32(1)
        woff, rows, nb, typ, doff = mat(A, l, j0 + m); ns = nsl(nb); n = (rows >> Int32(6)) * ns
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
# The exchange: an all-reduce over the four dies of this die's share (the K-slice sums of matrix (l, j), or with
# j < 0 its embedding rows of tok[p]), through page-locked host memory. A die reads host memory at about 4.5 GB/s
# with 2 us a round trip (writes are posted). At decode width (P <= DW) the round trips are the cost: each die writes
# its share once (hxd) and reads the other three. A prefill's bytes are the cost: a reduce-scatter then an
# all-gather, 3/4 fewer bytes read (decode 31.5 tok/s with the one-round form, 29.2 with the two-round; a 512-token
# prefill 1.5x faster with the two-round): die d owns quarter d of the elements (QTR of them);
#   1. every die writes its share of each other quarter to that quarter's owner (hxr: owner, source);
#   2. the owner reads the three, sums the four in die order, and writes the total to each other die (hxg: reader,
#      owner); each die reads the other three quarters' totals.
# A word is 8 bytes: the float and a tag (request number, exchange number), so a reader polls the data itself (one
# round trip, nothing to clear). Every die adds the same totals, so x stays bit-identical on the four dies. Then
# x (+)= the total, and the next norm's input is prepared: x w into xt with its 16- and 32-sums, and the sum of squares
# a position (the consumers apply 1 / rms). Phase seq is the computing and writing; seq + 1 is the rest, from when
# the first share arrived (the wait before it is idle).
const QTR = NE ÷ ND
@inline hxr(A::KA, slot, q, s) = reinterpret(LP{UInt64}, A.hx + 8 * (((Int64(slot) * ND + q) * ND + s) * A.npos * QTR))
@inline hxg(A::KA, slot, r, s) = reinterpret(LP{UInt64}, A.hx + 8 * ((((Int64(2) + slot) * ND + r) * ND + s) * A.npos * QTR))
@inline hxd(A::KA, slot, s) = reinterpret(LP{UInt64}, A.hx + 8 * ((Int64(slot) * ND + s) * DW * NE))   # (inside hxr's space)
@inline ld64a_sys(p::LP{UInt64}) = Base.llvmcall("%r = load atomic i64, ptr addrspace(1) %0 monotonic, align 8\nret i64 %r", UInt64, Tuple{LP{UInt64}}, p)
@inline st64_sys(p::LP{UInt64}, v::UInt64) = Base.llvmcall("store atomic i64 %1, ptr addrspace(1) %0 monotonic, align 8\nret void", Cvoid, Tuple{LP{UInt64},UInt64}, p, v)
@inline xword(v::Float32, tag::UInt64) = UInt64(reinterpret(UInt32, v)) | tag
@inline function xwait(A::KA, q::LP{UInt64}, u::UInt64, tag::UInt64)   # the word at q with this tag (u: its first read)
    if u & 0xffffffff00000000 != tag
        tw = memtime()
        while u & 0xffffffff00000000 != tag
            (memtime() - tw > TOUT || aborted(A)) && (atadd(A.bar + 8, UInt32(1)); break)
            u = ld64a_sys(q)
        end
    end
    reinterpret(Float32, UInt32(u & 0xffffffff))
end
function xwr_ph!(A::KA, l::Int32, j::Int32, w::LP{Float32}, set::Bool, xs::Int32, seq::Int32, P::Int32)
    t0 = memtime(); slot = xs & Int32(1); d = A.d
    tag = UInt64(A.rid & Int32(255)) << 56 | UInt64((xs + Int32(1)) & Int32(0xffffff)) << 32
    rows, ns, W = if j >= Int32(0)                       # (one assignment each: the closure below captures them)
        _, r_, nb, _, _ = mat(A, l, j); (r_, nsx(nb, P), reinterpret(LP{UInt8}, Int64(0)))
    else
        woff, _, _, _, _ = mat(A, Int32(64), Int32(0)); (Int32(0), Int32(0), reinterpret(LP{UInt8}, woff))
    end
    share(p, k) = if j >= Int32(0)
        psum(A, Int64(0), ns, rows, P, p, k)
    else
        r = ld(A.tok, p) - Int32(VPD) * d
        r >= Int32(0) && r < Int32(VPD) ? embval(W, r, k) : 0.0f0
    end
    ta = UInt64(0)
    if P <= Int32(DW)
        # decode width: one round trip. Every die writes its whole share (hxd), reads the other three and sums the
        # four in die order (its own recomputed, not read back)
        i = gtid(); did = i < Int32(NE) * P
        while i < Int32(NE) * P
            p = i ÷ Int32(NE); k = i - p * Int32(NE)
            st64_sys(hxd(A, slot, d) + 8Int64(i), xword(share(p, k), tag)); i += nthr(A)
        end
        done!(A, seq, t0, did)
        i = gtid()
        while i < Int32(NE) * P
            p = i ÷ Int32(NE); k = i - p * Int32(NE); o = 8Int64(i)      # (not i itself in the closures: it is reassigned)
            u3 = ntuple(e -> ld64a_sys(hxd(A, slot, (d + Int32(e)) & Int32(ND - 1)) + o), Val(ND - 1))
            mine = share(p, k)
            sh = ntuple(e -> xwait(A, hxd(A, slot, (d + Int32(e)) & Int32(ND - 1)) + o, u3[e], tag), Val(ND - 1))
            ta == UInt64(0) && (ta = memtime())
            xput!(A, w, set, p, k, dsum(mine, sh, d), P)
            i += nthr(A)
        end
    else
        # prefill: bytes matter more than round trips. 1. this die's share of the other quarters, to their owners
        i = gtid(); did = i < Int32(QTR) * P
        while i < Int32(QTR) * P
            p = i ÷ Int32(QTR); kk = i - p * Int32(QTR); o = Int64(p) * QTR + kk
            for e in Int32(1):Int32(ND - 1)
                q = (d + e) & Int32(ND - 1)
                st64_sys(hxr(A, slot, q, d) + 8o, xword(share(p, q * Int32(QTR) + kk), tag))
            end
            i += nthr(A)
        end
        done!(A, seq, t0, did)
        # 2. this die's quarter: the four shares summed and sent out; then the other quarters' totals
        i = gtid()
        while i < Int32(QTR) * P
            p = i ÷ Int32(QTR); kk = i - p * Int32(QTR); o = Int64(p) * QTR + kk
            u3 = ntuple(e -> ld64a_sys(hxr(A, slot, d, (d + Int32(e)) & Int32(ND - 1)) + 8o), Val(ND - 1))
            mine = share(p, d * Int32(QTR) + kk)
            sh = ntuple(e -> xwait(A, hxr(A, slot, d, (d + Int32(e)) & Int32(ND - 1)) + 8o, u3[e], tag), Val(ND - 1))
            ta == UInt64(0) && (ta = memtime())
            tot = dsum(mine, sh, d)
            for e in Int32(1):Int32(ND - 1)
                st64_sys(hxg(A, slot, (d + e) & Int32(ND - 1), d) + 8o, xword(tot, tag))
            end
            g3 = ntuple(e -> ld64a_sys(hxg(A, slot, d, (d + Int32(e)) & Int32(ND - 1)) + 8o), Val(ND - 1))
            tq = ntuple(e -> xwait(A, hxg(A, slot, d, (d + Int32(e)) & Int32(ND - 1)) + 8o, g3[e], tag), Val(ND - 1))
            for e in Int32(0):Int32(ND - 1)                 # quarter (d + e) mod 4
                xput!(A, w, set, p, ((d + e) & Int32(ND - 1)) * Int32(QTR) + kk, e == Int32(0) ? tot : tq[e], P)
            end
            i += nthr(A)
        end
    end
    did && tid() == Int32(0) && (pa = A.acct + 8 * (3 + 2Int64(seq + Int32(1))); umin64(pa, ta); umax64(pa + 8, memtime()))
end
# the four shares in die order: die s's is sh[(s - d) mod 4], this die's own mine
@inline function dsum(mine::Float32, sh::NTuple{3,Float32}, d::Int32)
    v(s) = s == d ? mine : sh[(s - d) & Int32(ND - 1)]
    ((v(Int32(0)) + v(Int32(1))) + v(Int32(2))) + v(Int32(3))
end
# x[p][k] (+)= u, and the next norm's input: x w into xt, its 16- and 32-sums, the position's sum of squares (a
# wavefront's lanes hold 64 consecutive k of one position)
@inline function xput!(A::KA, w::LP{Float32}, set::Bool, p, k, u::Float32, P)
    ox = Int64(p) * NE + k
    v = (set ? 0.0f0 : ld(A.x, ox)) + u
    st!(A.x, v, ox); xw = v * ld(w, k); st!(A.xt, xw, Int64(k) * P + p)
    sq = AMDGPU.Device.wfred(+, v * v)
    tid() & Int32(63) == Int32(0) && atadd64(A.ssq + 8p, round(Int64, sq * 65536.0f0))
    a = xw
    for sh in (Cint(1), Cint(2), Cint(4), Cint(8)); a += AMDGPU.Device.shfl_xor(a, sh); end
    c = a + AMDGPU.Device.shfl_xor(a, Cint(16))
    k & Int32(15) == Int32(0) && st!(A.x16, a, Int64(k >> Int32(4)) * P + p)
    k & Int32(31) == Int32(0) && st!(A.x32, c, Int64(k >> Int32(5)) * P + p)
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
# 1 / rms of position p of x (from the fixed-point sum of squares the exchange made)
@inline rinv(A::KA, p) = 1.0f0 / sqrt(Float32(ld(A.ssq, p)) * (1.0f0 / 65536.0f0) / Float32(NE) + EPS)
@inline atadd64(p::LP{Int64}, v::Int64) = Base.llvmcall("%r = atomicrmw add ptr addrspace(1) %0, i64 %1 syncscope(\"agent\") monotonic, align 8\nret i64 %r", Int64, Tuple{LP{Int64},Int64}, p, v)
# zero the sums of squares (in the phase before an exchange: their readers are done, the exchange adds)
@inline function zero_ssq!(A::KA, P)
    i = gtid()
    while i < P; st!(A.ssq, Int64(0), i); i += nthr(A); end
end
# table slots: 0..3 qkv z alpha beta (DeltaNet) | 0..2 q k v (attention); 4 out; 5, 6 gate up; 7 down
# DeltaNet conv + silu, one thread a channel
function conv_ph!(A::KA, l::Int32, seq::Int32, P::Int32)
    t0 = memtime(); 
    c = gtid(); did = c < Int32(QKVD)
    if did
        _, rows, nb, _, _ = mat(A, l, Int32(0)); ns = nsx(nb, P)
        ro = 4 * (Int64(ridx(l)) * 3 * QKVD); cw = fvec(A, l, 2); rc = Int32(-1); ring = reinterpret(LP{Float32}, Int64(0))
        r1 = 0.0f0; r2 = 0.0f0; r3 = 0.0f0
        w1 = ld(cw, 4c); w2 = ld(cw, 4c + 1); w3 = ld(cw, 4c + 2); w4 = ld(cw, 4c + 3)
        for p in Int32(0):P-Int32(1)
            r = ld(A.posr, p)
            if r != rc                                     # the request's ring
                rc >= Int32(0) && (st!(ring, r1, 3c); st!(ring, r2, 3c + 1); st!(ring, r3, 3c + 2))
                ring = reinterpret(LP{Float32}, slp(A, r, 1)) + ro; rc = r
                r1 = ld(ring, 3c); r2 = ld(ring, 3c + 1); r3 = ld(ring, 3c + 2)
            end
            x = psum(A, Int64(0), ns, rows, P, p, c) * rinv(A, p)
            st!(A.u, silu(w1 * r1 + w2 * r2 + w3 * r3 + w4 * x), Int64(p) * QKVD + c)
            r1, r2, r3 = r2, r3, x
            if ld(A.pk, p) == Int32(1)
                rb = reinterpret(LP{Float32}, slp(A, r, 5)) + ro; st!(rb, r1, 3c); st!(rb, r2, 3c + 1); st!(rb, r3, 3c + 2)
            end
        end
        st!(ring, r1, 3c); st!(ring, r2, 3c + 1); st!(ring, r3, 3c + 2)
    end
    done!(A, seq, t0, did)
end
# the gated delta rule: a value head's 128 state rows in RSPL slices of 16, one workgroup a slice (workgroup
# RSPL lv + q); a row's 128 columns over 16 lanes, 8 each, so a wavefront's state loads are one contiguous 2 KB run.
# Every load a position needs is issued before its first reduction. A slice writes its rows of
# S q / sqrt(128) ssm_norm silu(z) (all but the head's rmsnorm) into xt, their 16-sum into x16 and their raw sum of
# squares into x16's unused groups (96 + 8 lv + q); the head's last slice to finish (a counter a head, bar words
# 4 ..) scales the head's 128 outputs and sums by 1 / rms.
const RSPL = 8
@inline function xor16(v)                                  # sum over a row's 16 lanes
    v += AMDGPU.Device.shfl_xor(v, Cint(1)); v += AMDGPU.Device.shfl_xor(v, Cint(2))
    v += AMDGPU.Device.shfl_xor(v, Cint(4)); v + AMDGPU.Device.shfl_xor(v, Cint(8))
end
function rec_ph!(A::KA, l::Int32, seq::Int32, P::Int32)
    qs = @ROCStaticLocalArray(Float32, DK, false); ks = @ROCStaticLocalArray(Float32, DK, false)
    red = @ROCStaticLocalArray(Float32, 2RED, false); fl = @ROCStaticLocalArray(Int32, 1, false)
    w = wgid(); w < Int32(VHD * RSPL) || return
    t0 = memtime()
    lv = w ÷ Int32(RSPL); q = w - lv * Int32(RSPL)
    kh = lv & Int32(3); t = tid(); i = Int32(16) * q + (t >> Int32(4)); c0 = (t & Int32(15)) * Int32(8); k = DK * lv + i
    ri = ridx(l); so = (Int64(lv) * DK + i) * DK + c0; rc = Int32(-1); Sp = reinterpret(LP{Float32}, Int64(0))
    _, _, nbz, _, _ = mat(A, l, Int32(1)); nsz = nsx(nbz, P); oz = pof(A, l, 0, 2, P)
    _, _, nba, _, _ = mat(A, l, Int32(2)); nsa = nsx(nba, P); oa = pof(A, l, 0, 3, P); ob4 = pof(A, l, 0, 4, P)
    dtb = ld(fvec(A, l, 3), lv); ssa = ld(fvec(A, l, 4), lv); sni = ld(fvec(A, l, 5), i)
    s = ntuple(_ -> 0.0f0, Val(8))
    isc = 1.0f0 / sqrt(Float32(DK))
    for p in Int32(0):P-Int32(1)
        r = ld(A.posr, p)
        if r != rc                                         # the request's state
            rc >= Int32(0) && (for j in 1:8; st!(Sp, s[j], so + j - 1); end)
            Sp = reinterpret(LP{Float32}, slp(A, r, 0)) + 4 * (Int64(ri) * VHD * DK * DK); rc = r
            s = let Sp = Sp; ntuple(j -> ld(Sp, so + j - 1), Val(8)) end
        end
        ub = Int64(p) * QKVD; ri_p = rinv(A, p)
        qv = t < Int32(DK) ? ld(A.u, ub + DK * kh + t) : 0.0f0
        kv = t < Int32(DK) ? ld(A.u, ub + 512 + DK * kh + t) : 0.0f0
        vv = ld(A.u, ub + 1024 + k)
        xa = psum(A, oa, nsa, Int32(64), P, p, lv); xb = psum(A, ob4, nsa, Int32(64), P, p, lv)
        z = psum(A, oz, nsz, Int32(DK * VHD), P, p, k)
        nq2, nk2 = wgsum2(qv * qv, kv * kv, red)
        if t < Int32(DK)
            st!(qs, qv / max(sqrt(nq2), EPS), t); st!(ks, kv / max(sqrt(nk2), EPS), t)
        end
        sync_workgroup()
        x = xa * ri_p + dtb; sp = x > 20.0f0 ? x : log1p(exp(x))
        a = exp(ssa * sp); bt = 1.0f0 / (1.0f0 + exp(-xb * ri_p))
        # S = a S + e kᵀ with e = bt (v - a S k): the decay folded into the one update
        kk = ntuple(j -> ld(ks, c0 + Int32(j - 1)), Val(8))
        e = 0.0f0
        for j in 1:8; e = fma(s[j], kk[j], e); end
        e = bt * (vv - a * xor16(e))
        s = let s = s, e = e, a = a, kk = kk; ntuple(j -> fma(e, kk[j], a * s[j]), Val(8)) end
        o = 0.0f0
        for j in 1:8; o = fma(s[j], ld(qs, c0 + Int32(j - 1)), o); end
        o = xor16(o) * isc; g = o * sni * silu(z * ri_p)
        (t & Int32(15)) == Int32(0) && st!(A.xt, g, Int64(k) * P + p)
        if ld(A.pk, p) == Int32(1)
            Sb = reinterpret(LP{Float32}, slp(A, r, 4)) + 4 * (Int64(ri) * VHD * DK * DK)
            for j in 1:8; st!(Sb, s[j], so + j - 1); end
        end
        # the slice's 16 rows: their sum (x16) and raw sum of squares; lanes 0, 16, 32, 48 hold a row each
        so2 = o * o; sg = g
        so2 += AMDGPU.Device.shfl_xor(so2, Cint(16)); so2 += AMDGPU.Device.shfl_xor(so2, Cint(32))
        sg += AMDGPU.Device.shfl_xor(sg, Cint(16)); sg += AMDGPU.Device.shfl_xor(sg, Cint(32))
        sync_workgroup()
        (t & Int32(63)) == Int32(0) && (st!(red, so2, t >> Int32(6)); st!(red, sg, Int32(4) + (t >> Int32(6))))
        sync_workgroup()
        if t == Int32(0)
            st!(A.x16, ld(red, 4) + ld(red, 5) + ld(red, 6) + ld(red, 7), Int64(8lv + q) * P + p)
            st!(A.x16, ld(red, 0) + ld(red, 1) + ld(red, 2) + ld(red, 3), Int64(96 + 8lv + q) * P + p)
        end
    end
    for j in 1:8; st!(Sp, s[j], so + j - 1); end
    # the head's last slice: 1 / rms over its 128 rows
    fence_rel(); sync_workgroup()
    if t == Int32(0)
        c = A.bar + 16 + 4Int64(lv)
        last = atadd(c, UInt32(1)) == UInt32(RSPL - 1)
        last && stmono(c, UInt32(0))
        st!(fl, last ? Int32(1) : Int32(0), 0)
    end
    sync_workgroup()
    if ld(fl, 0) == Int32(1)
        fence_acq()
        rr(p) = (s2 = 0.0f0; for j in Int32(0):Int32(RSPL - 1); s2 += ld(A.x16, Int64(96 + 8lv + j) * P + p); end; 1.0f0 / sqrt(s2 / Float32(DK) + EPS))
        j = t
        while j < P * Int32(DK)
            p = j >> Int32(7); kj = Int64(DK * lv + (j & Int32(127))) * P + p
            st!(A.xt, ld(A.xt, kj) * rr(p), kj); j += Int32(256)
        end
        j = t
        while j < P * Int32(4)
            p = j >> Int32(2); m = j & Int32(3); f = rr(p)
            g0 = Int64(8lv + 2m) * P + p; g1 = Int64(8lv + 2m + 1) * P + p
            a0 = ld(A.x16, g0) * f; a1 = ld(A.x16, g1) * f
            st!(A.x16, a0, g0); st!(A.x16, a1, g1); st!(A.x32, a0 + a1, Int64(4lv + m) * P + p); j += Int32(256)
        end
    end
    done!(A, seq, t0, true)
end
# ---- TurboQuant KV, 3.5 bits a coordinate: a key or value vector is rotated (random signs, then the orthonormal
# Walsh-Hadamard transform), its norm kept (f32), and each coordinate of the rotated unit vector (about N(0, 1/256)),
# times 16, quantized to the Lloyd-Max levels of N(0, 1): coordinates 0..127 at 4 bits, 128..255 at 3 bits. A row
# is 64 bytes of 4-bit codes (coordinate 2i in the low nibble of byte i), 48 bytes of 3-bit codes (coordinate
# 128 + j at bit 3j), the norm: KVROW bytes. Scores use the rotated query (the rotation keeps dot products); the
# value sum stays rotated until the combine rotates it back.
const KVROW = 116
@inline rsgn(t) = (UInt32(t) * 0x9e3779b1) >> 31 == 0 ? 1.0f0 : -1.0f0
@inline function lm4(i)                                    # 4-bit Lloyd-Max level i (0..15) of N(0, 1)
    j = i < Int32(8) ? Int32(7) - i : i - Int32(8)
    m = j == Int32(0) ? 0.1284f0 : j == Int32(1) ? 0.3881f0 : j == Int32(2) ? 0.6568f0 : j == Int32(3) ? 0.9424f0 :
        j == Int32(4) ? 1.2562f0 : j == Int32(5) ? 1.6181f0 : j == Int32(6) ? 2.0690f0 : 2.7326f0
    i < Int32(8) ? -m : m
end
@inline function lm3(i)                                    # 3-bit level i (0..7)
    j = i < Int32(4) ? Int32(3) - i : i - Int32(4)
    m = j == Int32(0) ? 0.2451f0 : j == Int32(1) ? 0.7560f0 : j == Int32(2) ? 1.3439f0 : 2.1520f0
    i < Int32(4) ? -m : m
end
@inline function qcode(u, ::Val{B}) where {B}              # the nearest level: the count of midpoints below u
    n = Int32(1) << B; c = Int32(0)
    for i in Int32(0):n-Int32(2)
        a = B == 4 ? lm4(i) : lm3(i); b = B == 4 ? lm4(i + Int32(1)) : lm3(i + Int32(1))
        u > 0.5f0 * (a + b) && (c += Int32(1))
    end
    c
end
@inline function fwht!(L, t)                               # the Walsh-Hadamard transform of 256 shared floats
    for h in (Int32(1), Int32(2), Int32(4), Int32(8), Int32(16), Int32(32), Int32(64), Int32(128))
        a = ld(L, t); b = ld(L, t ⊻ h)
        sync_workgroup()
        st!(L, t & h == Int32(0) ? a + b : b - a, t)
        sync_workgroup()
    end
end
@inline ld32b(p::LP{UInt8}, o) = unsafe_load(reinterpret(LP{UInt32}, p + o))
@inline function wgred6(op, v::NTuple{6,Float32}, red)      # 6 reductions over the workgroup in one round
    t = tid(); s = ntuple(h -> AMDGPU.Device.wfred(op, v[h]), Val(6))
    sync_workgroup()
    if t & Int32(63) == Int32(0)
        for h in 1:6; st!(red, s[h], (t >> Int32(6)) * Int32(6) + Int32(h - 1)); end
    end
    sync_workgroup()
    ntuple(h -> op(op(ld(red, h - 1), ld(red, h + 5)), op(ld(red, h + 11), ld(red, h + 17))), Val(6))
end
# attention input: rmsnorm and neox rope of the 6 query heads and of the key, then the rotation; the rotated queries
# into qh (with their gates into gh), the key and the value quantized into the cache
function aprep_ph!(A::KA, l::Int32, seq::Int32, P::Int32)
    L = @ROCStaticLocalArray(Float32, HD, false); red = @ROCStaticLocalArray(Float32, RED, false)
    cd = @ROCStaticLocalArray(UInt8, HD, false)
    t0 = memtime(); did = false; t = tid()
    _, rq, nbq, _, _ = mat(A, l, Int32(0)); nsq = nsx(nbq, P)
    _, rk, nbk, _, _ = mat(A, l, Int32(1)); nsk = nsx(nbk, P); ok = pof(A, l, 0, 2, P)
    _, rv, nbv, _, _ = mat(A, l, Int32(2)); nsv = nsx(nbv, P); ov = pof(A, l, 0, 3, P)
    qn = fvec(A, l, 2); kn = fvec(A, l, 3); koff = Int64(aidx(l)) * A.maxctx * KVROW
    it = wgid()
    while it < Int32(8) * P                                 # 6 query heads, the key, the value
        hx = it & Int32(7); p = it >> Int32(3)
        v = if hx < Int32(6)
            psum(A, Int64(0), nsq, rq, P, p, Int32(512) * hx + t) * rinv(A, p)
        elseif hx == Int32(6)
            psum(A, ok, nsk, rk, P, p, t) * rinv(A, p)
        else
            psum(A, ov, nsv, rv, P, p, t) * rinv(A, p)
        end
        if hx < Int32(7)                                    # per-head rmsnorm and rope
            w = hx < Int32(6) ? ld(qn, t) : ld(kn, t)
            v = v * (1.0f0 / sqrt(wgsum(v * v, red) / Float32(HD) + EPS)) * w
            st!(L, v, t)
            sync_workgroup()
            if t < Int32(NROT)
                ii = t & Int32(31); th = Float64(ld(A.posa, p)) * invf(A, ii)
                r = Float32(th - 2π * round(th * (1 / 2π)))       # reduced in Float64, the sine and cosine in Float32
                c = cos(r); sn = sin(r)
                a = ld(L, ii); b = ld(L, ii + Int32(32))
                v = t < Int32(32) ? a * c - b * sn : b * c + a * sn
            end
            sync_workgroup()
        end
        st!(L, v * rsgn(t), t)
        sync_workgroup()
        fwht!(L, t); y = ld(L, t) * (1.0f0 / 16.0f0)
        if hx < Int32(6)
            st!(A.qh, y, Int64(p) * 1536 + HD * hx + t)
            st!(A.gh, psum(A, Int64(0), nsq, rq, P, p, Int32(512) * hx + Int32(HD) + t) * rinv(A, p), Int64(p) * 1536 + HD * hx + t)
        else
            nrm = sqrt(wgsum(y * y, red)); u = nrm > 0.0f0 ? y * 16.0f0 / nrm : 0.0f0
            st!(cd, UInt8(t < Int32(128) ? qcode(u, Val(4)) : qcode(u, Val(3))), t)
            sync_workgroup()
            row = reinterpret(LP{UInt8}, slp(A, ld(A.posr, p), hx == Int32(6) ? 2 : 3)) + koff + Int64(ld(A.posa, p)) * KVROW
            if t < Int32(64)
                st!(row, ld(cd, 2t) | ld(cd, 2t + Int32(1)) << 4, t)
            elseif t < Int32(80)
                g = t - Int32(64); bits = UInt32(0)
                for i in Int32(0):Int32(7); bits |= UInt32(ld(cd, Int32(128) + Int32(8) * g + i)) << (Int32(3) * i); end
                st!(row, UInt8(bits & 0xff), Int32(64) + Int32(3) * g); st!(row, UInt8((bits >> 8) & 0xff), Int32(65) + Int32(3) * g)
                st!(row, UInt8((bits >> 16) & 0xff), Int32(66) + Int32(3) * g)
            elseif t == Int32(80)
                unsafe_store!(reinterpret(LP{Float32}, row + 112), nrm)
            end
        end
        sync_workgroup()
        did = true; it += A.nwg
    end
    done!(A, seq, t0, did)
end
# the splits of a position's keys: enough items to fill the die (decode), one a position when the positions do
@inline asplits(A::KA, amax, P) = max(Int32(1), min(cld(A.nwg, P), cld(amax, Int32(CH))))
# attention, one implementation for prefill and decode. Item (position p, split s): the key chunks
# s nch ÷ S .. (s + 1) nch ÷ S of position p, for the 6 query heads at once (they share this die's KV head), with an
# online softmax; the scores with a key a thread, the value sum with a coordinate a thread. (max, sum, rotated
# value sum) a head into ap
function attn_ph!(A::KA, l::Int32, amax::Int32, seq::Int32, P::Int32, T)
    qs = T[1]; ps = T[2]                                    # the 6 rotated queries; the weights a head, the value norms
    red = @ROCStaticLocalArray(Float32, 24, false); tb = @ROCStaticLocalArray(Float32, 24, false)
    t0 = memtime(); did = false; t = tid()
    t < Int32(16) && st!(tb, lm4(t), t); (t >= Int32(16) && t < Int32(24)) && st!(tb, lm3(t - Int32(16)), t)
    koff = Int64(aidx(l)) * A.maxctx * KVROW; S = asplits(A, amax, P); isc = 1.0f0 / (16.0f0 * sqrt(Float32(HD)))
    it = wgid()
    while it < P * S
        p = it ÷ S; s = it - p * S; nk = ld(A.posa, p) + Int32(1); nchp = cld(nk, Int32(CH))
        r = ld(A.posr, p); kcp = reinterpret(LP{UInt8}, slp(A, r, 2)); vcp = reinterpret(LP{UInt8}, slp(A, r, 3))
        c0 = s * nchp ÷ S; c1 = (s + Int32(1)) * nchp ÷ S
        for h in Int32(0):Int32(5); st!(qs, ld(A.qh, Int64(p) * 1536 + HD * h + t), HD * h + t); end
        sync_workgroup()
        M = ntuple(_ -> -Inf32, Val(6)); Ls = ntuple(_ -> 0.0f0, Val(6)); acc = ntuple(_ -> 0.0f0, Val(6))
        for c in c0:c1-Int32(1)
            kk = c * Int32(CH) + t; sc = ntuple(_ -> -Inf32, Val(6)); vn = 0.0f0
            if kk < nk
                row = kcp + koff + Int64(kk) * KVROW; dots = ntuple(_ -> 0.0f0, Val(6))
                for i in Int32(0):Int32(15)
                    w = ld32b(row, Int32(4) * i)
                    Base.Cartesian.@nexprs 8 e -> begin
                        cv = ld(tb, Int32((w >> (4 * (e - 1))) & 0xf)); d = Int32(8) * i + Int32(e - 1)
                        dots = let dt = dots, cv = cv, d = d; ntuple(h -> fma(ld(qs, HD * (h - 1) + d), cv, dt[h]), Val(6)) end
                    end
                end
                for g in Int32(0):Int32(3)
                    w0 = ld32b(row, Int32(64) + Int32(12) * g); w1 = ld32b(row, Int32(68) + Int32(12) * g); w2 = ld32b(row, Int32(72) + Int32(12) * g)
                    Base.Cartesian.@nexprs 32 j -> begin
                        b3 = 3 * (j - 1); wa = b3 < 32 ? w0 : b3 < 64 ? w1 : w2; wb = b3 < 32 ? w1 : w2; sh = b3 & 31
                        code = (sh <= 29 ? (wa >> sh) : ((wa >> sh) | (wb << (32 - sh)))) & 0x7
                        cv = ld(tb, Int32(16) + Int32(code)); d = Int32(128) + Int32(32) * g + Int32(j - 1)
                        dots = let dt = dots, cv = cv, d = d; ntuple(h -> fma(ld(qs, HD * (h - 1) + d), cv, dt[h]), Val(6)) end
                    end
                end
                nr = reinterpret(Float32, ld32b(row, Int32(112))) * isc
                sc = let dots = dots, nr = nr; ntuple(h -> dots[h] * nr, Val(6)) end
                vn = reinterpret(Float32, ld32b(vcp + koff + Int64(kk) * KVROW, Int32(112))) * (1.0f0 / 16.0f0)
            end
            cm = wgred6(max, sc, red)
            Mn = let M = M, cm = cm; ntuple(h -> max(M[h], cm[h]), Val(6)) end
            e = let sc = sc, Mn = Mn, ok = kk < nk; ntuple(h -> ok ? exp(sc[h] - Mn[h]) : 0.0f0, Val(6)) end
            es = wgred6(+, e, red)
            Ls, acc = let M = M, Mn = Mn, Ls = Ls, acc = acc, es = es
                f = ntuple(h -> exp(M[h] - Mn[h]), Val(6))
                ntuple(h -> Ls[h] * f[h] + es[h], Val(6)), ntuple(h -> acc[h] * f[h], Val(6))
            end
            M = Mn
            for h in 1:6; st!(ps, e[h], CH * (h - 1) + t); end
            st!(ps, vn, Int32(6CH) + t)
            sync_workgroup()
            jn = min(Int32(CH), nk - c * Int32(CH))
            for j in Int32(0):jn-Int32(1)
                vr = vcp + koff + Int64(c * Int32(CH) + j) * KVROW
                cv = if t < Int32(128)
                    b = ld(vr, t >> Int32(1)); ld(tb, Int32(t & Int32(1) == Int32(0) ? b & 0x0f : b >> 4))
                else
                    bit = Int32(3) * (t - Int32(128)); by = Int32(64) + (bit >> Int32(3))
                    wv = UInt32(ld(vr, by)) | UInt32(ld(vr, by + Int32(1))) << 8
                    ld(tb, Int32(16) + Int32((wv >> (bit & Int32(7))) & 0x7))
                end
                fv = cv * ld(ps, Int32(6CH) + j)
                acc = let a = acc, fv = fv, j = j; ntuple(h -> fma(ld(ps, CH * (h - 1) + j), fv, a[h]), Val(6)) end
            end
            sync_workgroup()
        end
        for h in 1:6
            o = ((Int64(p) * S + s) * 6 + (h - 1)) * (HD + 2)
            t == Int32(0) && (st!(A.ap, M[h], o); st!(A.ap, Ls[h], o + 1))
            st!(A.ap, acc[h], o + 2 + t)
        end
        did = true; it += A.nwg
    end
    done!(A, seq, t0, did)
end
# combine the splits, rotate back, times sigmoid(gate): into xt (the out-projection's x) with its 16- and 32-sums
function acomb_ph!(A::KA, amax::Int32, seq::Int32, P::Int32)
    L = @ROCStaticLocalArray(Float32, HD, false)
    t0 = memtime(); did = false; t = tid(); S = asplits(A, amax, P); it = wgid()
    while it < Int32(6) * P
        hx = it % Int32(6); p = it ÷ Int32(6)
        Mx = -Inf32
        for s in Int32(0):S-Int32(1); Mx = max(Mx, ld(A.ap, ((Int64(p) * S + s) * 6 + hx) * (HD + 2))); end
        Lt = 0.0f0; acc = 0.0f0
        for s in Int32(0):S-Int32(1)
            o = ((Int64(p) * S + s) * 6 + hx) * (HD + 2); m = ld(A.ap, o)
            if m > -Inf32
                f = exp(m - Mx); Lt += ld(A.ap, o + 1) * f; acc += ld(A.ap, o + 2 + t) * f
            end
        end
        st!(L, acc / Lt, t)
        sync_workgroup()
        fwht!(L, t); y = ld(L, t) * rsgn(t) * (1.0f0 / 16.0f0)
        g = ld(A.gh, Int64(p) * 1536 + HD * hx + t); v = y / (1.0f0 + exp(-g)); k = HD * hx + t
        st!(A.xt, v, Int64(k) * P + p); st!(L, v, t)
        sync_workgroup()
        if t < Int32(8)
            a2 = 0.0f0; b2 = 0.0f0
            for e2 in Int32(0):Int32(15); a2 += ld(L, Int32(32) * t + e2); b2 += ld(L, Int32(32) * t + Int32(16) + e2); end
            st!(A.x32, a2 + b2, Int64(8hx + t) * P + p); st!(A.x16, a2, Int64(16hx + 2t) * P + p); st!(A.x16, b2, Int64(16hx + 2t + 1) * P + p)
        end
        sync_workgroup()
        did = true; it += A.nwg
    end
    done!(A, seq, t0, did)
end
# FFN: silu(gate) up into xt (the down-projection's x), an element a thread, with its 16- and 32-sums (Q6_K or Q5_K)
function act_ph!(A::KA, l::Int32, seq::Int32, P::Int32)
    t0 = memtime()
    did = false; i = gtid()
    _, rg, nbg, _, _ = mat(A, l, Int32(5)); ns = nsx(nbg, P); ou = pof(A, l, 5, 2, P)
    while i < Int32(4352) * P                              # a wavefront: 64 elements of one position (4352 = 68 x 64)
        p = i ÷ Int32(4352); k = i - p * Int32(4352)
        v = silu(psum(A, Int64(0), ns, rg, P, p, k) * rinv(A, p)) * (psum(A, ou, ns, rg, P, p, k) * rinv(A, p))
        st!(A.xt, v, Int64(k) * P + p)
        a = v
        for sh in (Cint(1), Cint(2), Cint(4), Cint(8)); a += AMDGPU.Device.shfl_xor(a, sh); end
        b = a + AMDGPU.Device.shfl_xor(a, Cint(16))
        k & Int32(15) == Int32(0) && st!(A.x16, a, Int64(k >> Int32(4)) * P + p)
        k & Int32(31) == Int32(0) && st!(A.x32, b, Int64(k >> Int32(5)) * P + p)
        did = true; i += nthr(A)
    end
    done!(A, seq, t0, did)
end
# the head's argmax a position (q = p - pfirst), the die's best into the argmax exchange, the token: the best of
# the four dies. The token goes to the request's slot (the next decode forward's input) and, from die 0, out.
function hmax_ph!(A::KA, seq::Int32, P::Int32, pfirst::Int32)
    bv = @ROCStaticLocalArray(Float32, RED, false); bi = @ROCStaticLocalArray(Int32, RED, false)
    t0 = memtime()
    did = false; t = tid(); gen = A.mode == Int32(0)
    _, rows, nb, _, _ = mat(A, Int32(64), Int32(1)); ns = nsx(nb, P)
    for p in pfirst:P-Int32(1)
        v = -Inf32; ix = Int32(0); r = gtid()
        while r < Int32(VPD)
            np, pp = isgemm(P) ? (P - pfirst, p - pfirst) : (P, p)   # a GEMM stored positions pfirst .. from 0
            lg = psum(A, Int64(0), ns, rows, np, pp, r) * rinv(A, p)
            gen || st!(A.lg, lg, Int64(ld(A.posa, p)) * VPD + r)
            lg > v && (v = lg; ix = r); did = true; r += nthr(A)
        end
        gen || continue
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
            o = 2 * (Int64(wgid()) * DW + (p - pfirst))
            st!(A.red, v, o); st!(A.red, reinterpret(Float32, ix), o + 1)
        end
    end
    done!(A, seq, t0, did)
end
@inline hxa(A::KA, slot, d, q) = A.hx + 8 * (4 * ND * ND * Int64(A.npos) * QTR) + 4 * (((Int64(slot) * ND + d) * DW + q) * 2)
function hred_ph!(A::KA, slot::Int32, seq::Int32, nq::Int32)
    (wgid() == Int32(0) && tid() < nq) || return
    t0 = memtime(); v = -Inf32; ix = Int32(0); q = tid()
    for w in Int32(0):A.nwg-Int32(1)
        o = 2 * (Int64(w) * DW + q); v2 = ld(A.red, o); i2 = reinterpret(Int32, ld(A.red, o + 1))
        (v2 > v || (v2 == v && i2 < ix)) && (v = v2; ix = i2)
    end
    hb = hxa(A, slot, A.d, q); st!(hb, v, 0); st!(hb, reinterpret(Float32, ix + Int32(VPD) * A.d), 1)
    done!(A, seq, t0, true)
end
function tok_ph!(A::KA, slot::Int32, seq::Int32, nq::Int32, pfirst::Int32)
    (wgid() == Int32(0) && tid() < nq) || return
    fence_acq_sys(); t0 = memtime(); v = -Inf32; ix = Int32(0); q = tid()
    for d in Int32(0):Int32(ND - 1)
        hb = hxa(A, slot, d, q); v2 = ld(hb, 0); i2 = reinterpret(Int32, ld(hb, 1))
        (v2 > v || (v2 == v && i2 < ix)) && (v = v2; ix = i2)
    end
    r = ld(A.posr, pfirst + q); st!(A.tks, ix, r); n = ld(A.nem, r)
    A.d == Int32(0) && st!(A.out, ix, ld(A.rq, 4r + Int32(3)) + n)
    st!(A.nem, n + Int32(1), r)
    done!(A, seq, t0, true)
end
# the forward's positions: prefill, positions a0 .. a0 + P - 1 of request r; decode (r < 0), the next position of
# every request still generating, and with MTP the position after it (its draft)
function setpos_ph!(A::KA, r::Int32, a0::Int32, P::Int32)
    i = gtid()
    if r >= Int32(0)
        while i < P
            st!(A.posr, r, i); st!(A.posa, a0 + i, i); st!(A.tok, ld(A.inp, ld(A.rq, 4r) + a0 + i), i); st!(A.pk, Int32(0), i); i += nthr(A)
        end
    elseif i == Int32(0)
        n = Int32(0)
        for q in Int32(0):A.nreq-Int32(1)
            m = ld(A.nem, q)
            if m < ld(A.rq, 4q + Int32(2))
                c = ld(A.rq, 4q + Int32(1)) + m - Int32(1)
                st!(A.posr, q, n); st!(A.posa, c, n); st!(A.tok, ld(A.tks, q), n); st!(A.pk, A.spec, n); n += Int32(1)
                if A.spec != Int32(0)
                    st!(A.posr, q, n); st!(A.posa, c + Int32(1), n); st!(A.tok, ld(A.drf, q), n); st!(A.pk, Int32(0), n); n += Int32(1)
                end
            end
        end
    end
end

# ---- MTP (the GGUF's nextn layer, blk.64; layer slot MTPL in the table): drafting one token a step ----
# Its input at position i is rmsnorm(emb(token i + 1)) enorm ++ rmsnorm(h_i) hnorm (h: the trunk's output after the
# output norm), projected by eh_proj (10240 -> 5120), then one gated-attention layer with its own KV cache (attention
# slot 16) and FFN, shared_head_norm and the shared head: the argmax is the draft for position i + 2.
# h_i, normalized and weighted (hnorm), into part (5120 floats a position); in a prefill, the MTP's input tokens
# (the next token of each position) into tok. One workgroup a position.
function mpre1_ph!(A::KA, seq::Int32, P::Int32, pre::Bool)
    red = @ROCStaticLocalArray(Float32, RED, false)
    t0 = memtime(); did = false; t = tid(); hw = fvec(A, MTPL, 5)
    p = wgid()
    while p < P
        ri = rinv(A, p); s = 0.0f0; k = t
        while k < Int32(NE); h = ld(A.xt, Int64(k) * P + p) * ri; s = fma(h, h, s); k += Int32(256); end
        rh = 1.0f0 / sqrt(wgsum(s, red) / Float32(NE) + EPS); k = t
        while k < Int32(NE); st!(A.part, ld(A.xt, Int64(k) * P + p) * ri * rh * ld(hw, k), Int64(p) * NE + k); k += Int32(256); end
        if pre && t == Int32(0)                            # a prefill: the next token (the last position's: tks)
            r = ld(A.posr, p); a1 = ld(A.posa, p) + Int32(1)
            st!(A.tok, a1 < ld(A.rq, 4r + Int32(1)) ? ld(A.inp, ld(A.rq, 4r) + a1) : ld(A.tks, r), p)
        end
        did = true; p += A.nwg
    end
    done!(A, seq, t0, did)
end
zero_ph!(A::KA, P::Int32) = zero_ssq!(A, P)
# eh_proj's input on this die: elements 2560 d .. of the concatenation (x is emb(token) after the exchange), with
# the 16- and 32-sums (a wavefront holds 64 consecutive elements of one position)
function mpre2_ph!(A::KA, seq::Int32, P::Int32)
    t0 = memtime(); ew = fvec(A, MTPL, 4); KL = Int32(2 * NE ÷ ND)
    i = gtid(); did = i < KL * P
    while i < KL * P
        p = i ÷ KL; k = i - p * KL; g = Int32(KL) * A.d + k
        v = g < Int32(NE) ? ld(A.x, Int64(p) * NE + g) * ld(ew, g) * rinv(A, p) : ld(A.part, Int64(p) * NE + g - Int32(NE))
        st!(A.xt, v, Int64(k) * P + p)
        a = v
        for sh in (Cint(1), Cint(2), Cint(4), Cint(8)); a += AMDGPU.Device.shfl_xor(a, sh); end
        c = a + AMDGPU.Device.shfl_xor(a, Cint(16))
        k & Int32(15) == Int32(0) && st!(A.x16, a, Int64(k >> Int32(4)) * P + p)
        k & Int32(31) == Int32(0) && st!(A.x32, c, Int64(k >> Int32(5)) * P + p)
        i += nthr(A)
    end
    done!(A, seq, t0, did)
end
# a verify forward's tokens (spec decode: positions 2j, 2j + 1 of each request: its last token, then the draft): the
# argmax at the last token's position is the next token; the draft is accepted if it equals it, and then the argmax
# at the draft's position is one more. Rejected: the DeltaNet state and conv ring after the first position (kept in
# the backup slots) become the request's state. The MTP's input tokens for both positions go to tok.
function stok_ph!(A::KA, slot::Int32, seq::Int32, nq::Int32)
    (wgid() == Int32(0) && tid() < (nq >> Int32(1))) || return
    fence_acq_sys(); t0 = memtime(); j = tid()
    g = ntuple(Val(2)) do e
        v = -Inf32; ix = Int32(0)
        for d in Int32(0):Int32(ND - 1)
            hb = hxa(A, slot, d, Int32(2) * j + Int32(e - 1)); v2 = ld(hb, 0); i2 = reinterpret(Int32, ld(hb, 1))
            (v2 > v || (v2 == v && i2 < ix)) && (v = v2; ix = i2)
        end
        ix
    end
    p = Int32(2) * j; r = ld(A.posr, p); acc = ld(A.tok, p + Int32(1)) == g[1]
    n = ld(A.nem, r); ng = ld(A.rq, 4r + Int32(2)); two = acc && n + Int32(1) < ng
    if A.d == Int32(0)
        st!(A.out, g[1], ld(A.rq, 4r + Int32(3)) + n)
        two && st!(A.out, g[2], ld(A.rq, 4r + Int32(3)) + n + Int32(1))
    end
    st!(A.nem, n + (two ? Int32(2) : Int32(1)), r); st!(A.tks, acc ? g[2] : g[1], r); st!(A.msel, acc ? Int32(1) : Int32(0), r)
    if !acc
        for f in (0, 1)
            a = ld(A.sl, 6Int64(r) + f); st!(A.sl, ld(A.sl, 6Int64(r) + f + 4), 6Int64(r) + f); st!(A.sl, a, 6Int64(r) + f + 4)
        end
    end
    acc && atadd(A.bar + 64, UInt32(1)); atadd(A.bar + 68, UInt32(1))   # (accepted, drafted: the acceptance rate)
    st!(A.tok, g[1], p); st!(A.tok, g[2], p + Int32(1))
    done!(A, seq, t0, true)
end
# the MTP head's argmax: a request's draft (in a verify forward, from the position after its last accepted token)
function dtok_ph!(A::KA, slot::Int32, seq::Int32, nq::Int32, pfirst::Int32, pre::Bool)
    (wgid() == Int32(0) && tid() < nq) || return
    fence_acq_sys(); t0 = memtime(); v = -Inf32; ix = Int32(0); q = tid()
    for d in Int32(0):Int32(ND - 1)
        hb = hxa(A, slot, d, q); v2 = ld(hb, 0); i2 = reinterpret(Int32, ld(hb, 1))
        (v2 > v || (v2 == v && i2 < ix)) && (v = v2; ix = i2)
    end
    p = pfirst + q; r = ld(A.posr, p)
    (pre || (p & Int32(1)) == ld(A.msel, r)) && st!(A.drf, ix, r)
    done!(A, seq, t0, true)
end
# ---- the prefill GEMM: all the forward's positions against each weight tile ----
@inline ld32p(W::LP{UInt8}, off) = unsafe_load(reinterpret(LP{UInt32}, W + off))
@inline function lds4(xs, i)                              # shared floats i .. i + 3 (i a multiple of 4)
    v = unsafe_load(reinterpret(Core.LLVMPtr{NTuple{4,VecElement{Float32}},3}, pointer(xs) + Int32(4) * i), 1, Val(16))
    (v[1].value, v[2].value, v[3].value, v[4].value)
end
@inline function scmn_rt(u0, u1, u2, s)                    # Q5_K (scale, min) of sub-block s (run time)
    if s < Int32(4)
        sh = Int32(8) * s
        return Float32((u0 >> sh) & 0x3f), Float32((u1 >> sh) & 0x3f)
    end
    h = Int32(8) * (s - Int32(4))
    Float32(((u2 >> h) & 0x0f) | ((u0 >> (h + Int32(6))) & 0x03) << 4), Float32(((u2 >> (h + Int32(4))) & 0x0f) | ((u1 >> (h + Int32(6))) & 0x03) << 4)
end
# 8 weights of row (G, lane lr) of a repacked matrix: block b, 32-element run s (Q5_K: sub-block s; Q6_K: half
# s >> 2, group s & 3), elements 8q .. 8q + 7 of the run
@inline function dq8(W::LP{UInt8}, D::LP{UInt16}, typ, nb, G, b, s, lr, q)
    blk = Int64(G) * nb + b
    if typ == Int32(GQ5_K)
        o = blk * (11 * 1024) + 16 * lr
        h0 = ld32p(W, o); d = f16(h0 & 0xffff); dmin = f16(h0 >> 16)
        sc, mn = scmn_rt(ld32p(W, o + 4), ld32p(W, o + 8), ld32p(W, o + 12), s)
        qso = o + (3 + 2 * (s >> Int32(1)) + (q >> Int32(1))) * 1024 + 8 * (q & Int32(1)); qho = o + (1 + (q >> Int32(1))) * 1024 + 8 * (q & Int32(1))
        qs0 = ld32p(W, qso); qs1 = ld32p(W, qso + 4); qh0 = ld32p(W, qho); qh1 = ld32p(W, qho + 4)
        sh = Int32(4) * (s & Int32(1)); a = d * sc; m = dmin * mn
        q5(w, wh, e) = fma(a, Float32(((w >> (8e + sh)) & 0x0f) | ((wh >> (8e + s)) & 0x01) << 4), -m)
        return (q5(qs0, qh0, 0), q5(qs0, qh0, 1), q5(qs0, qh0, 2), q5(qs0, qh0, 3), q5(qs1, qh1, 0), q5(qs1, qh1, 1), q5(qs1, qh1, 2), q5(qs1, qh1, 3))
    end
    o = blk * (13 * 1024) + 16 * lr; d = f16(ld(D, blk * 64 + lr))
    h = s >> Int32(2); g = s & Int32(3)
    qlo = o + (4h + 2 * (g & Int32(1)) + (q >> Int32(1))) * 1024 + 8 * (q & Int32(1)); qho = o + (8 + 2h + (q >> Int32(1))) * 1024 + 8 * (q & Int32(1))
    ds = d * Float32(reinterpret(Int8, ld(W, o + 12 * 1024 + 8h + 2g + (q >> Int32(1)))))
    ql0 = ld32p(W, qlo); ql1 = ld32p(W, qlo + 4); qh0 = ld32p(W, qho); qh1 = ld32p(W, qho + 4)
    shl = g >= Int32(2) ? Int32(4) : Int32(0); shh = Int32(2) * g
    q6(w, wh, e) = fma(ds, Float32(((w >> (8e + shl)) & 0x0f) | ((wh >> (8e + shh)) & 0x03) << 4), -32.0f0 * ds)
    (q6(ql0, qh0, 0), q6(ql0, qh0, 1), q6(ql0, qh0, 2), q6(ql0, qh0, 3), q6(ql1, qh1, 0), q6(ql1, qh1, 1), q6(ql1, qh1, 2), q6(ql1, qh1, 3))
end
# one workgroup: rows 64 G .. 64 G + 63 times positions p0 .. p0 + 63 (< P) over all of K, into part[p rows + r].
# 32 columns at a time: the rows dequantized into Ws[k][row], x into Xs[k][p]; thread t computes rows
# 4 (t >> 4) .. + 3 by positions 4 (t & 15) .. + 3
@noinline function gemm_tile!(part::LP{Float32}, W::LP{UInt8}, D::LP{UInt16}, typ::Int32, nb::Int32, rows::Int32, G::Int32, p0::Int32, P::Int32, pf::Int32, xt::LP{Float32}, Ws, Xs)
    t = tid(); ar = t >> Int32(4); bq = t & Int32(15); lr = t >> Int32(2); q = t & Int32(3); kk = t >> Int32(3); pb = (t & Int32(7)) * Int32(8)
    acc = ntuple(_ -> 0.0f0, Val(16))
    for b in Int32(0):nb-Int32(1), s in Int32(0):Int32(7)
        v = dq8(W, D, typ, nb, G, b, s, lr, q)
        for e in 1:8; st!(Ws, v[e], (Int32(8) * q + Int32(e - 1)) * Int32(64) + lr); end
        k = Int32(256) * b + Int32(32) * s + kk
        for e in Int32(0):Int32(7)
            p = p0 + pb + e; st!(Xs, p < P ? ld(xt, Int64(k) * P + p) : 0.0f0, kk * Int32(64) + pb + e)
        end
        sync_workgroup()
        for k2 in Int32(0):Int32(31)
            w4 = lds4(Ws, k2 * Int32(64) + Int32(4) * ar); x4 = lds4(Xs, k2 * Int32(64) + Int32(4) * bq)
            acc = let a = acc, w4 = w4, x4 = x4; ntuple(i -> fma(w4[(i-1)>>2+1], x4[(i-1)&3+1], a[i]), Val(16)) end
        end
        sync_workgroup()
    end
    for r in 0:3, c in 0:3
        p = p0 + Int32(4) * bq + Int32(c)
        p < P && st!(part, acc[4r+c+1], Int64(p - pf) * rows + Int32(64) * G + Int32(4) * ar + Int32(r))
    end
end
# the (64 rows, 64 positions) tiles of the matrices js of layer l, positions pf .. P - 1, over the workgroups
function gemm_ph!(A::KA, l::Int32, j0::Int32, nm::Int32, seq::Int32, P::Int32, Ws, Xs, zs::Bool = false, pf::Int32 = Int32(0))
    zs && zero_ssq!(A, P)
    t0 = memtime(); did = false; npt = cld(P - pf, Int32(64)); base = Int32(0); o = Int64(0); w = wgid()
    for m in Int32(0):nm-Int32(1)
        woff, rows, nb, typ, doff = mat(A, l, j0 + m); n = (rows >> Int32(6)) * npt
        g = w + max(Int32(0), cld(base - w, A.nwg)) * A.nwg
        while g < base + n
            i = g - base; G = i ÷ npt; pt = i - G * npt
            gemm_tile!(A.part + 4o, reinterpret(LP{UInt8}, woff), reinterpret(LP{UInt16}, doff), typ, nb, rows, G, pf + Int32(64) * pt, P, pf, A.xt, Ws, Xs)
            did = true; g += A.nwg
        end
        base += n; o += Int64(P - pf) * rows                # positions pf .. P - 1 are stored from 0
    end
    done!(A, seq, t0, did)
end
# matrices j0 .. j0 + nm - 1 of layer l (consecutive table slots)
@inline mat_ph!(A::KA, l, j0, nm, seq, P, T, zs::Bool = false, pf::Int32 = Int32(0)) =
    isgemm(P) ? gemm_ph!(A, Int32(l), Int32(j0), Int32(nm), seq, P, T[1], T[2], zs, pf) :   # prefill; decode: GEMV over all P positions
    P == Int32(1) ? gemv_ph!(A, Int32(l), Int32(j0), Int32(nm), seq, Val(1), zs) :
    P == Int32(2) ? gemv_ph!(A, Int32(l), Int32(j0), Int32(nm), seq, Val(2), zs) : gemv_ph!(A, Int32(l), Int32(j0), Int32(nm), seq, Val(4), zs)

# ---- one forward of P positions (setpos_ph! set them): GEMMs for a prefill (isgemm), GEMVs at decode (P <= DW);
# amax: one past the largest position; head: the head takes positions pfirst .. P - 1 (into tokens of step k).
# A layer is a short program of steps run by one loop, so each phase is in the kernel once (compile time).
const S_MAT, S_CONV, S_REC, S_APREP, S_ATTN, S_ACOMB, S_XWR, S_ACT, S_HMAX, S_HRED, S_XSIG, S_TOK, S_MPRE1, S_ZERO, S_MPRE2, S_DTOK = Int32.(1:16)
# step i of layer l (64: the head): (kind, first matrix slot, matrices, zero the sums of squares first)
@inline function lstep(l::Int32, i::Int32)
    if l == Int32(64)
        i == Int32(0) && return S_MAT, Int32(1), Int32(1), false
        return (i == Int32(1) ? S_HMAX : i == Int32(2) ? S_HRED : i == Int32(3) ? S_XSIG : S_TOK), Int32(0), Int32(0), false
    end
    if l == Int32(MTPL)                                    # the MTP layer (its head: steps 16 ..)
        i == Int32(0) && return S_MPRE1, Int32(0), Int32(0), false
        i == Int32(1) && return S_ZERO, Int32(0), Int32(0), false
        i == Int32(2) && return S_XWR, Int32(-1), Int32(0), false # emb(next token)
        i == Int32(3) && return S_MPRE2, Int32(0), Int32(0), false
        i == Int32(4) && return S_MAT, Int32(3), Int32(1), true    # eh_proj
        i == Int32(5) && return S_XWR, Int32(3), Int32(0), false
        i == Int32(16) && return S_MAT, Int32(1), Int32(1), false  # the shared head (layer 64's slot 1)
        i >= Int32(17) && return (i == Int32(17) ? S_HMAX : i == Int32(18) ? S_HRED : i == Int32(19) ? S_XSIG : S_DTOK), Int32(0), Int32(0), false
        i -= Int32(6)
    elseif isrec(l)
        i == Int32(0) && return S_MAT, Int32(0), Int32(4), false   # qkv z alpha beta
        i == Int32(1) && return S_CONV, Int32(0), Int32(0), false
        i == Int32(2) && return S_REC, Int32(0), Int32(0), false
        i -= Int32(3)
        @goto tail
    end
    i == Int32(0) && return S_MAT, Int32(0), Int32(3), false       # q k v
    i == Int32(1) && return S_APREP, Int32(0), Int32(0), false
    i == Int32(2) && return S_ATTN, Int32(0), Int32(0), false
    i == Int32(3) && return S_ACOMB, Int32(0), Int32(0), false
    i -= Int32(4)
    @label tail
    i == Int32(0) && return S_MAT, Int32(4), Int32(1), true         # out
    i == Int32(1) && return S_XWR, Int32(4), Int32(0), false
    i == Int32(2) && return S_MAT, Int32(5), Int32(2), false        # gate up
    i == Int32(3) && return S_ACT, Int32(0), Int32(0), false
    i == Int32(4) && return S_MAT, Int32(7), Int32(1), true         # down
    return S_XWR, Int32(7), Int32(0), false
end
@inline nsteps(l::Int32, gen::Bool, mhead::Bool) = l == Int32(64) ? (gen ? Int32(5) : Int32(2)) :
    l == Int32(MTPL) ? (mhead ? Int32(21) : Int32(16)) : isrec(l) ? Int32(9) : Int32(10)
# head: the trunk's head over positions pfirst .. P - 1; mtp: the MTP layer after it, its head (mhead) over
# positions mpf ..; pre: a prefill forward; spec: a verify forward (pairs of positions)
function fwd!(A::KA, P::Int32, amax::Int32, seq::Int32, xs::Int32, head::Bool, pfirst::Int32, mtp::Bool, mhead::Bool, mpf::Int32, pre::Bool, T)
    xwr_ph!(A, Int32(0), Int32(-1), fvec(A, 0, 0), true, xs, seq, P); seq += Int32(2); xs += Int32(1); gbar!(A)
    nlr = Int32(ldacq(A.bar + 12)); nl = nlr == Int32(0) ? Int32(NL) : nlr   # bar[3]: layers to run (0: all), for debugging
    gen = A.mode == Int32(0); spec = !pre && A.spec != Int32(0)
    l = Int32(0); i = Int32(0)
    while l <= Int32(MTPL)
        kind, j0, nm, zs = lstep(l, i)
        hd = l == Int32(64) || (l == Int32(MTPL) && j0 == Int32(1))   # a head step
        pf = l == Int32(MTPL) ? mpf : pfirst; nq = P - pf
        if kind == S_MAT
            mat_ph!(A, hd ? Int32(64) : l, j0, nm, seq, P, T, zs, hd ? pf : Int32(0)); seq += Int32(1)
        elseif kind == S_CONV
            conv_ph!(A, l, seq, P); seq += Int32(1)
        elseif kind == S_REC
            rec_ph!(A, l, seq, P); seq += Int32(1)
        elseif kind == S_APREP
            aprep_ph!(A, l, seq, P); seq += Int32(1)
        elseif kind == S_ATTN
            attn_ph!(A, l, amax, seq, P, T); seq += Int32(1)
        elseif kind == S_ACOMB
            acomb_ph!(A, amax, seq, P); seq += Int32(1)
        elseif kind == S_XWR
            wn = if l == Int32(MTPL)
                j0 == Int32(-1) ? fvec(A, l, 4) : j0 == Int32(3) ? fvec(A, l, 0) : j0 == Int32(4) ? fvec(A, l, 1) : fvec(A, l, 6)
            else
                j0 == Int32(4) ? fvec(A, l, 1) : l < nl - Int32(1) ? fvec(A, l + Int32(1), 0) : fvec(A, 64, 0)
            end
            xwr_ph!(A, l, j0, wn, j0 == Int32(-1) || j0 == Int32(3), xs, seq, P); seq += Int32(2); xs += Int32(1)
        elseif kind == S_ACT
            act_ph!(A, l, seq, P); seq += Int32(1)
        elseif kind == S_HMAX
            hmax_ph!(A, seq, P, pf); seq += Int32(1)
        elseif kind == S_HRED
            hred_ph!(A, xs & Int32(1), seq, nq); seq += Int32(1)
        elseif kind == S_XSIG
            xsig!(A, xs)
        elseif kind == S_TOK
            spec ? stok_ph!(A, xs & Int32(1), seq, nq) : tok_ph!(A, xs & Int32(1), seq, nq, pf); seq += Int32(1); xs += Int32(1)
        elseif kind == S_MPRE1
            mpre1_ph!(A, seq, P, pre); seq += Int32(1)
        elseif kind == S_ZERO
            zero_ssq!(A, P)
        elseif kind == S_MPRE2
            mpre2_ph!(A, seq, P); seq += Int32(1)
        else
            dtok_ph!(A, xs & Int32(1), seq, nq, pf, pre); seq += Int32(1); xs += Int32(1)
        end
        gbar!(A)
        i += Int32(1)
        if i == nsteps(l, gen, mhead)
            i = Int32(0)
            l = l < nl - Int32(1) ? l + Int32(1) : l < Int32(64) ? (head ? Int32(64) : mtp ? Int32(MTPL) : Int32(MTPL + 1)) :
                l == Int32(64) ? (mtp ? Int32(MTPL) : Int32(MTPL + 1)) : Int32(MTPL + 1)
        end
    end
    seq, xs
end
# the requests (up to NR), one dispatch: each request's prefill (or scoring) in forwards of as many positions as fit
# (npos), then the decode steps of all of them together: a position a request a forward, or with MTP (spec) two
# (the last token and the draft) and the MTP layer after. One call of fwd! (one copy of the forward in the kernel):
# each loop turn picks the next forward.
function k_req!(A::KA)
    T = (@ROCStaticLocalArray(Float32, 32 * 64, false), @ROCStaticLocalArray(Float32, 32 * 64, false))   # the GEMM tiles
    tid() == Int32(0) && umin64(A.acct, memtime())
    seq = Int32(0); xs = Int32(0); spec = A.spec != Int32(0) && A.mode == Int32(0)
    r = Int32(0); a = Int32(0); prefill = true
    while true
        if prefill && r >= A.nreq                          # every prefill done
            tid() == Int32(0) && umax64(A.acct + 16, memtime())
            prefill = false
        end
        if prefill
            ninp = ld(A.rq, 4r + Int32(1)); ngen = ld(A.rq, 4r + Int32(2))
            P = min(A.npos, ninp - a); amax = a + P
            head = A.mode == Int32(1) || (a + P == ninp && ngen > Int32(0)); pfirst = A.mode == Int32(1) ? Int32(0) : P - Int32(1)
            mtp = spec && ngen > Int32(0)
            zero_ssq!(A, P); setpos_ph!(A, r, a, P); gbar!(A)
            seq, xs = fwd!(A, P, amax, seq, xs, head, pfirst, mtp, head, P - Int32(1), true, T)
            a += P
            a >= ninp && (r += Int32(1); a = Int32(0))
        else
            P = Int32(0); amax = Int32(0); w = spec ? Int32(2) : Int32(1)
            for q in Int32(0):A.nreq-Int32(1)
                n = ld(A.nem, q)
                if n < ld(A.rq, 4q + Int32(2))
                    P += w; amax = max(amax, ld(A.rq, 4q + Int32(1)) + n - Int32(1) + w)
                end
            end
            P == Int32(0) && break
            zero_ssq!(A, P); setpos_ph!(A, Int32(-1), Int32(0), P); gbar!(A)
            seq, xs = fwd!(A, P, amax, seq, xs, true, Int32(0), spec, spec, Int32(0), false, T)
        end
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
function load_layer(g::GGUF, l, d; att = !isrec(l))
    p = "blk.$l."; all20 = [(0, 20)]
    gate = DMat(g, p * "ffn_gate.weight", [(4352d, 4352)], all20); up = DMat(g, p * "ffn_up.weight", [(4352d, 4352)], all20)
    down = DMat(g, p * "ffn_down.weight", [(0, NE)], [(17d, 17)])
    an = dev(f32(g, p * "attn_norm.weight")); pn = dev(f32(g, p * "post_attention_norm.weight"))
    if !att
        vh = [16j + 4d for j in 0:2]                       # the first value head of each 4-head run
        heads = [16j + 4d + i for j in 0:2 for i in 0:3]
        qkv = DMat(g, p * "attn_qkv.weight", [(512d, 512); (2048 + 512d, 512); [(4096 + 128h, 512) for h in vh]], all20)
        z = DMat(g, p * "attn_gate.weight", [(128h, 512) for h in vh], all20)
        al = DMat(g, p * "ssm_alpha.weight", [(h, 4) for h in vh], all20); be = DMat(g, p * "ssm_beta.weight", [(h, 4) for h in vh], all20)
        out = DMat(g, p * "ssm_out.weight", [(0, NE)], [(8j + 2d, 2) for j in 0:2])
        chans = [512d:512d+511; 2048+512d:2048+512d+511; [4096+128h:4096+128h+511 for h in vh]...]
        cw = f32(g, p * "ssm_conv1d.weight"); cwd = [cw[4c+k] for c in chans for k in 1:4]
        dtb = f32(g, p * "ssm_dt.bias")[heads .+ 1]; ssa = f32(g, p * "ssm_a")[heads .+ 1]
        return DLayer(true, an, pn, gate, up, down, qkv, z, al, be, out, dev(cwd), dev(dtb), dev(ssa), dev(f32(g, p * "ssm_norm.weight")))
    end
    q = DMat(g, p * "attn_q.weight", [(3072d, 3072)], all20)
    k = DMat(g, p * "attn_k.weight", [(256d, 256)], all20); v = DMat(g, p * "attn_v.weight", [(256d, 256)], all20)
    out = DMat(g, p * "attn_output.weight", [(0, NE)], [(6d, 6)])
    DLayer(false, an, pn, gate, up, down, q, k, v, nothing, out, dev(f32(g, p * "attn_q_norm.weight")), dev(f32(g, p * "attn_k_norm.weight")), dev(NOV), dev(NOV))
end
struct Die
    dev::HIPDevice; d::Int
    L::Vector{DLayer}; emb::DMat; head::DMat; onrm::ROCVector{Float32}
    mtp::DLayer; eh::DMat; mv::ROCVector{Float32}           # the MTP layer, its eh_proj; enorm, hnorm, shared_head_norm
end
# a request's state on a die: DeltaNet states and conv rings, the attention cache (TurboQuant rows) up to maxctx
struct DState
    S::ROCVector{Float32}; ring::ROCVector{Float32}; kc::ROCVector{UInt8}; vc::ROCVector{UInt8}; maxctx::Int
    S2::ROCVector{Float32}; ring2::ROCVector{Float32}       # the backups (a verify forward's state after its first position)
end
DState(maxctx) = DState(AMDGPU.zeros(Float32, 48 * VHD * DK * DK), AMDGPU.zeros(Float32, 48 * QKVD * 3),
    AMDGPU.zeros(UInt8, 17 * maxctx * KVROW), AMDGPU.zeros(UInt8, 17 * maxctx * KVROW), maxctx,
    AMDGPU.zeros(Float32, 48 * VHD * DK * DK), AMDGPU.zeros(Float32, 48 * QKVD * 3))
function load_die(g::GGUF, hd, d, L)
    AMDGPU.device!(hd)
    m = load_layer(g, 64, d; att = true); p = "blk.64.nextn."
    eh = DMat(g, p * "eh_proj.weight", [(0, NE)], [(10d, 10)])             # K: elements 2560 d .. of emb ++ h
    mv = dev([f32(g, p * "enorm.weight"); f32(g, p * "hnorm.weight"); f32(g, p * "shared_head_norm.weight")])
    Die(hd, d, L, DMat(g, "token_embd.weight", VPD * d, VPD), DMat(g, "output.weight", VPD * d, VPD), dev(f32(g, "output_norm.weight")), m, eh, mv)
end
# the layers are read in file order, each one for all four dies at once (one pass over the file: the GGUF is on a
# spinning disk and sentry's RAM can't cache all of it)
function load_engine(path)
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
    onall(d -> (dies[d+1] = load_die(g, devs[d+1], d, Ls[d+1])))
    @printf("weights resident on %d dies (%.0f s)\n", ND, time() - t0)
    dies, g
end
newstate(dies, maxctx) = [(AMDGPU.device!(D.dev); DState(maxctx)) for D in dies]
freestate!(st) = foreach(s -> foreach(AMDGPU.unsafe_free!, (s.S, s.ring, s.kc, s.vc, s.S2, s.ring2)), st)

# ---- the persistent kernel's host side ----
addr(a) = reinterpret(Int64, UInt64(pointer(a)))
# a die's table: every matrix's address and shape, every vector's address (the weights stay where they are)
function ktable(D::Die)
    tab = zeros(Int64, TABF + 66 * NMAT + NROT ÷ 2)
    tab[TABF+66NMAT+1:end] = reinterpret(Int64, [THETA^(-2i / NROT) for i in 0:NROT÷2-1])   # the rope inverse frequencies
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
    M = D.mtp; m!(MTPL, 0, M.m1); m!(MTPL, 1, M.m2); m!(MTPL, 2, M.m3); m!(MTPL, 3, D.eh)
    m!(MTPL, 4, M.out); m!(MTPL, 5, M.gate); m!(MTPL, 6, M.up); m!(MTPL, 7, M.down)
    v!(MTPL, 0, M.an); v!(MTPL, 1, M.pn); v!(MTPL, 2, M.w1); v!(MTPL, 3, M.w2)
    for (j, o) in ((4, 0), (5, NE), (6, 2NE)); tab[TABF+MTPL*NMAT+j+1] = addr(D.mv) + 4o; end
    dev(tab)
end
const NWG = 256                                             # most workgroups a die
const MAXCTX = 262144                                      # positions a request slot holds
const NSCORE = 128                                          # positions a scoring forward (its logits are all kept)
# the kernel's buffers on a die, for forwards of up to npos positions
struct KDie
    tab::ROCVector{Int64}; x::ROCVector{Float32}; xt::ROCVector{Float32}; x16::ROCVector{Float32}; x32::ROCVector{Float32}
    part::ROCVector{Float32}; red::ROCVector{Float32}; u::ROCVector{Float32}; qh::ROCVector{Float32}; gh::ROCVector{Float32}
    ap::ROCVector{Float32}; tok::ROCVector{Int32}; bar::ROCVector{UInt32}; ssq::ROCVector{Int64}
    posr::ROCVector{Int32}; posa::ROCVector{Int32}; tks::ROCVector{Int32}; sl::ROCVector{Int64}; rq::ROCVector{Int32}
    pk::ROCVector{Int32}; nem::ROCVector{Int32}; drf::ROCVector{Int32}; msel::ROCVector{Int32}
end
function KDie(D::Die, npos)
    AMDGPU.device!(D.dev); z(n) = AMDGPU.zeros(Float32, n); zi(n) = AMDGPU.zeros(Int32, n)
    KDie(ktable(D), z(NE * npos), z(NE * npos), z(NE ÷ 16 * npos), z(NE ÷ 32 * npos), z(max(VPD * 10, 2 * 4352 * npos, VPD * NSCORE)),
        z(2 * NWG * DW), z(QKVD * npos), z(1536 * npos), z(1536 * npos), z(6 * (HD + 2) * max(npos, NWG)),
        zi(npos), AMDGPU.zeros(UInt32, 64), AMDGPU.zeros(Int64, npos), zi(npos), zi(npos), zi(NR), AMDGPU.zeros(Int64, 6NR), zi(4NR), zi(npos), zi(NR), zi(NR), zi(NR))
end
# page-locked host memory every die maps. p: what only the dies touch, the exchange words (2 phases x 2 slots x dies x
# dies x npos x QTR) and the argmax exchange (2 slots x dies x DW); c: what the host reads or writes too, the counters
# (at out) and the generated tokens (at tok). Both coherent (four dies read it at 14 GB/s together); in non-coherent
# memory a die's L2 keeps the lines, so a poll does not see the other dies' writes (1 of 2560 words within 1 s).
struct HostX
    p::Ptr{UInt8}; c::Ptr{UInt8}; nbytes::Int; out::Int; tok::Int
end
function HostX(npos, maxgen)
    n = 8 * 4 * ND * ND * npos * QTR + 4 * 2 * ND * DW * 2; out = 0; tok = 64; r = Ref{Ptr{Cvoid}}(C_NULL); rc = Ref{Ptr{Cvoid}}(C_NULL)
    H = AMDGPU.HIP
    H.hipHostMalloc(r, n, H.hipHostMallocPortable | H.hipHostMallocMapped | H.hipHostMallocCoherent)
    H.hipHostMalloc(rc, tok + 4maxgen, H.hipHostMallocPortable | H.hipHostMallocMapped | H.hipHostMallocCoherent)
    HostX(Ptr{UInt8}(r[]), Ptr{UInt8}(rc[]), n, out, tok)
end
lp(T, p::Ptr) = reinterpret(LP{T}, p)
lp(T, a::ROCArray) = reinterpret(LP{T}, pointer(a))
struct PEngine
    dies::Vector{Die}; K::Vector{KDie}; hx::HostX; nwg::Int; npos::Int; maxgen::Int; rid::Base.RefValue{Int}; hz::Float64   # hz: memtime ticks a second
end
# memtime's rate (s_memrealtime: 27 MHz on these dies), from two spins of different lengths against the host clock
function k_spin!(o, n::UInt64)
    t0 = memtime()
    while memtime() - t0 < n; end
    unsafe_store!(pointer(o), memtime() - t0, 1)
    return
end
function clockhz(D::Die)
    AMDGPU.device!(D.dev); o = AMDGPU.zeros(UInt64, 1)
    @roc k_spin!(o, UInt64(1000)); AMDGPU.synchronize()
    function t(n)
        t0 = time_ns(); @roc k_spin!(o, UInt64(n)); AMDGPU.synchronize()
        (time_ns() - t0) / 1e9, Array(o)[1]
    end
    # the median of 5 (one pair read 59 MHz on a loaded host)
    function rate()
        t1, n1 = t(2_000_000); t2, n2 = t(20_000_000)
        (n2 - n1) / (t2 - t1)
    end
    hz = sort([rate() for _ in 1:5])[3]
    20e6 < hz < 35e6 || @warn "memtime rate $(hz / 1e6) MHz, outside 20..35"
    hz
end
# npos: the positions a prefill forward takes, the most that fit (host exchange memory is the tighter bound here)
function PEngine(dies; maxgen = 8192, npos = 1024)
    K = [KDie(D, npos) for D in dies]; hx = HostX(npos, maxgen)
    AMDGPU.device!(dies[1].dev)
    A = kargs(K, hx, 1, 0, AMDGPU.zeros(UInt64, 4), AMDGPU.zeros(Int32, 1), 0, 0, 1, nothing, npos, 0)
    k = @roc launch = false maxthreads = 256 waves_per_eu = KWPE[] k_req!(A)
    nb = Ref{Cint}(0); AMDGPU.HIP.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(nb, k.fun, 256, 0)
    nwg = Int(nb[]) * 56
    VHD * RSPL <= nwg <= NWG || error("persistent kernel: $(nb[]) workgroups a CU")
    PEngine(dies, K, hx, nwg, npos, maxgen, Ref(0), clockhz(dies[1]))
end
function kargs(K, hx::HostX, d, maxctx, acct, inp, nreq, mode, nwg, lg, npos, rid)
    k = K[d]
    KA(lp(Int64, k.tab), lp(Float32, k.x), lp(Float32, k.xt), lp(Float32, k.x16), lp(Float32, k.x32), lp(Float32, k.part), lp(Float32, k.red),
       lp(Float32, k.u), lp(Float32, k.qh), lp(Float32, k.gh), lp(Float32, k.ap), lg === nothing ? lp(Float32, k.x) : lp(Float32, lg),
       lp(Int64, k.sl), Int32(maxctx), lp(Float32, hx.p), lp(UInt32, hx.c + hx.out), Int32(d - 1), lp(UInt32, k.bar), lp(UInt64, acct),
       lp(Int32, inp), lp(Int32, k.rq), Int32(nreq), lp(Int32, hx.c + hx.tok), Int32(mode), Int32(nwg), lp(Int32, k.tok),
       lp(Int32, k.posr), lp(Int32, k.posa), lp(Int32, k.tks), Int32(npos), lp(Int64, k.ssq), Int32(rid),
       lp(Int32, k.pk), lp(Int32, k.nem), lp(Int32, k.drf), lp(Int32, k.msel), Int32(mode == 0 && SPEC[] ? 1 : 0))
end
# up to NR requests at once on the four dies, one dispatch a die. reqs: (input ids, tokens to generate) each, with
# its state sts[r] (one DState a die). mode 0: greedy; mode 1: score one request\x27s input (the logits of every
# position into lg, a die\x27s vocabulary rows). Returns each request\x27s tokens, each die\x27s (instruction, idle, wall)
# seconds, and the prefill and decode seconds (die 0\x27s clock).
function request!(PE::PEngine, reqs, sts, mode::Int; lgs = nothing)
    dies = PE.dies; npos = mode == 1 ? min(PE.npos, NSCORE) : PE.npos; nr = length(reqs)
    1 <= nr <= NR || error("$nr requests: at most $NR at once")
    mc = minimum(s[1].maxctx for s in sts)
    for (inp, ngen) in reqs
        isempty(inp) && error("a request with no input tokens")
        length(inp) + ngen <= mc || error("request of $(length(inp) + ngen) positions: the state holds $mc")
    end
    sum(last, reqs) <= PE.maxgen || error("$(sum(last, reqs)) tokens: the output holds $(PE.maxgen)")
    inp = Int32.(vcat(first.(reqs)...)); ioff = cumsum([0; length.(first.(reqs))]); ooff = cumsum([0; last.(reqs)])
    rq = Int32.(vcat([[ioff[r], length(reqs[r][1]), reqs[r][2], ooff[r]] for r in 1:nr]...))
    nphase = (sum(cld(length(i), npos) for (i, _) in reqs) + maximum(last, reqs) + 2) * 1100; rid = (PE.rid[] += 1)
    unsafe_store!(Ptr{UInt64}(PE.hx.c + PE.hx.out), 0, 1); unsafe_store!(Ptr{UInt64}(PE.hx.c + PE.hx.out), 0, 2)   # the counters
    accts = ROCVector{UInt64}[]; ibs = ROCVector{Int32}[]
    for d in 1:ND
        AMDGPU.device!(dies[d].dev); K = PE.K[d]
        a = AMDGPU.zeros(UInt64, 3 + 2nphase); ah = zeros(UInt64, 3 + 2nphase); ah[1] = typemax(UInt64); ah[4:2:end] .= typemax(UInt64); up!(a, ah); push!(accts, a)
        up!(K.bar, [UInt32[0, 0, 0, NLRUN[]]; zeros(UInt32, 60)]); up!(K.rq, rq)
        for r in 1:nr; fill!(sts[r][d].S, 0.0f0); fill!(sts[r][d].ring, 0.0f0); end   # a request starts from zero state
        up!(K.sl, Int64[addr(getfield(sts[r][d], f)) for r in 1:nr for f in (:S, :ring, :kc, :vc, :S2, :ring2)]); up!(K.nem, zeros(Int32, NR))
        ib = dev(inp); push!(ibs, ib)
        A = kargs(PE.K, PE.hx, d, mc, a, ib, nr, mode, PE.nwg, lgs === nothing ? nothing : lgs[d], npos, rid)
        @roc groupsize = 256 gridsize = PE.nwg maxthreads = 256 waves_per_eu = KWPE[] k_req!(A)
    end
    res = NTuple{3,Float64}[]; tpre = 0.0; tdec = 0.0
    for d in 1:ND
        AMDGPU.device!(dies[d].dev); AMDGPU.synchronize()
        hz = PE.hz; ah = Array(accts[d]); wall = (ah[2] - ah[1]) / hz; iv = Tuple{Float64,Float64}[]
        d == 1 && (tpre = (ah[3] - ah[1]) / hz; tdec = (ah[2] - ah[3]) / hz; LASTACCT[] = ah)
        for s in 0:nphase-1
            b = ah[4+2s]; e = ah[5+2s]; b == typemax(UInt64) && continue
            push!(iv, ((b - ah[1]) / hz, (e - ah[1]) / hz))
        end
        ins = union_len(iv)                                # the phases overlap (an exchange writes while it reads)
        push!(res, (ins, wall - ins, wall))
        AMDGPU.unsafe_free!(accts[d]); AMDGPU.unsafe_free!(ibs[d])
    end
    ab = [(AMDGPU.device!(D.dev); Array(K.bar)[3]) for (D, K) in zip(dies, PE.K)]
    AMDGPU.device!(dies[1].dev); bh = Array(PE.K[1].bar); ACC[] = (Int(bh[17]), Int(bh[18]))
    any(!=(0), ab) && (error("persistent kernel aborted (a wait passed 10 s): $ab"))
    toks = [[Int(unsafe_load(Ptr{Int32}(PE.hx.c + PE.hx.tok), ooff[r] + i)) for i in 1:reqs[r][2]] for r in 1:nr]
    toks, res, tpre, tdec
end
if !isdefined(Main, :KWPE)
    const KWPE = Ref(2)                                    # waves a SIMD the persistent kernel is compiled for (2: 38.8 vs 31.6 tok/s decode at 1)
end
if !isdefined(Main, :LASTACCT)
    const LASTACCT = Ref{Vector{UInt64}}(UInt64[])        # die 0's phase times of the last request, for profiling
end
if !isdefined(Main, :SPEC)
    const SPEC = Ref(true)                                 # decode with MTP drafts (spec 0 | 1)
end
if !isdefined(Main, :ACC)
    const ACC = Ref((0, 0))                                # the last request's accepted and drafted counts
end
if !isdefined(Main, :NLRUN)
    const NLRUN = Ref(0)                                   # layers the persistent kernel runs (0: all), for debugging
end
idlepct(res) = 100sum(r[2] for r in res) / sum(r[3] for r in res)

# ---- requests ----
# greedy generation for one or two requests at once (each: its prompt ids and n new tokens), in the resident slots
function gen(io, PE::PEngine, vocab, reqs)
    toks, res, tp, td = request!(PE, reqs, SLOTS[][1:length(reqs)], 0)
    for t in toks; println(io, "text: ", repr(detok(vocab, t))); end
    @printf(io, "prefill %.2f tok/s\ndecode %.2f tok/s\nidle time %.1f%%\n", sum(length(i) for (i, _) in reqs) / tp, sum(n - 1 for (_, n) in reqs) / td, idlepct(res))
end
# KL against BF16 reference logits ([position][vocab] f16) over the first n tokens of the reference's sequence
function kl(io, PE::PEngine, ids, refpath, n)
    dies = PE.dies
    lgs = [(AMDGPU.device!(D.dev); AMDGPU.zeros(Float32, n * VPD)) for D in dies]
    _, res, tp, _ = request!(PE, [(ids[1:n], 0)], SLOTS[][1:1], 1; lgs)
    Lg = Matrix{Float32}(undef, NV, n)
    for (D, a) in zip(dies, lgs)
        AMDGPU.device!(D.dev); h = Array(a)
        for p in 1:n; Lg[VPD*D.d+1:VPD*(D.d+1), p] = view(h, (p-1)*VPD+1:p*VPD); end
        AMDGPU.unsafe_free!(a)
    end
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

# ---- the resident server: the weights load once, the persistent kernel compiles once; one command a connection ----
#   gen N ID ID ...            greedy: N tokens after the given token ids
#   gen2 N1 N2 F.ids L1 L2     two requests at once (each in a resident slot of MAXCTX positions): L1 then L2 prompt ids from F
#   kl N REF.logits.f16        KL against BF16 reference logits over the first N tokens of REF's sequence
#   reload                     include this file again and recompile the kernel (the weights stay resident)
#   eval CODE                  run Julia code in the server, print its value
#   echo 'gen 32 9707 11' | socat - UNIX-CONNECT:SOCK
if !isdefined(Main, :SRV)
    const SRV = Ref{Any}(nothing)
end
if !isdefined(Main, :SLOTS)
    const SLOTS = Ref{Any}(nothing)                        # NR request states of MAXCTX positions, resident
end
if !isdefined(Main, :PK)
    const PK = Ref{Any}(nothing)                           # the persistent kernel engine
end
function pinit()
    t0 = time(); PK[] === nothing || (foreach(K -> foreach(a -> a isa ROCArray && AMDGPU.unsafe_free!(a), (getfield(K, f) for f in fieldnames(typeof(K)))), PK[].K); AMDGPU.HIP.hipHostFree(PK[].hx.p); hasproperty(PK[].hx, :c) && AMDGPU.HIP.hipHostFree(PK[].hx.c))
    PK[] = PEngine(SRV[].dies)
    @printf("persistent kernel: %d workgroups a die, %d positions a prefill forward, clock %.2f MHz, compiled in %.0f s\n", PK[].nwg, PK[].npos, PK[].hz / 1e6, time() - t0); flush(stdout)
end
function serve(path, sock)
    dies, g = load_engine(path)
    SRV[] = (dies = dies, g = g, vocab = g.kv["tokenizer.ggml.tokens"], file = abspath(@__FILE__))
    SLOTS[] = [newstate(dies, MAXCTX) for _ in 1:NR]
    pinit()
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
function handle(c, line)
    a = split(line); s = SRV[]
    if a[1] == "gen"                                      # gen N ID ID ...
        gen(c, PK[], s.vocab, [(parse.(Int, a[3:end]), parse(Int, a[2]))])
    elseif a[1] == "gen2"                                 # gen2 N1 N2 FILE.ids L1 L2: two requests at once, prompts from FILE
        ids = Int.(reinterpret(Int32, read(a[4]))); l1 = parse(Int, a[5]); l2 = parse(Int, a[6])
        gen(c, PK[], s.vocab, [(ids[1:l1], parse(Int, a[2])), (ids[l1+1:l1+l2], parse(Int, a[3]))])
    elseif a[1] == "kl"
        n = parse(Int, a[2]); ids = Int.(reinterpret(Int32, read(replace(a[3], ".logits.f16" => ".ids"))))
        kl(c, PK[], ids, a[3], n)
    elseif a[1] == "spec"                                # spec 0 | 1: MTP drafting off or on
        SPEC[] = a[2] == "1"; println(c, "spec ", SPEC[])
    elseif a[1] == "reload"
        Base.include(Main, s.file); Base.invokelatest(pinit); println(c, "reloaded ", s.file)
    elseif a[1] == "eval"
        println(c, repr(Base.include_string(Main, line[6:end])))
    else
        println(c, "unknown command: ", a[1])
    end
end

function main(args)
    k = findfirst(==("--model"), args); i = findfirst(==("--serve"), args)
    (k === nothing || i === nothing) && error("usage: julia -t 24 qwen3.8-27b.jl --model FILE --serve SOCK")
    serve(args[k+1], args[i+1])
end
if abspath(PROGRAM_FILE) == @__FILE__() && SRV[] === nothing
    main(ARGS)
end
