# lfm3-1.7b.jl: LFM3-1.7B-Base (lfm3: 13 gated short-conv, 6 full-attention and 5 sliding-attention layers, hidden 2048)
# in BF16 on engi's Radeon RX 7700 XT (gfx1101, wave32, 12 GB GDDR6), KernelAbstractions over AMDGPU.jl. One cooperative
# persistent kernel runs a whole request: the prefill (positions in blocks of 16) then the decode loop.
# Model: LiquidAI/LFM3-1.7B-Base (model.safetensors, every tensor BF16).
# Reports only: prefill tok/s, decode tok/s, idle time %.
#   julia --project=env/amd -t 12 lfm3-1.7b.jl --model DIR --serve SOCK
using AMDGPU, KernelAbstractions, Mmap, Printf, Sockets
const KI = KernelAbstractions.KernelInterface

# ---------------------------------------------------------------------------------------------
# safetensors: the tensor table, the file mapped read-only; JSON for config and tokenizer
# ---------------------------------------------------------------------------------------------
struct ST
    t::Dict{String,Tuple{Vector{Int},Int,Int}}             # shape, first byte, bytes
    map::Vector{UInt8}
end
function sft(path)
    m = Mmap.mmap(path); n = Int(reinterpret(UInt64, view(m, 1:8))[1]); h = String(m[9:8+n]); t = Dict{String,Tuple{Vector{Int},Int,Int}}()
    for c in eachmatch(r"\"([^\"]+)\":\{\"dtype\":\"([A-Za-z0-9]+)\",\"shape\":\[([0-9,]*)\],\"data_offsets\":\[([0-9]+),([0-9]+)\]\}", h)
        c[2] == "BF16" || error("$(c[1]): $(c[2]), the kernel takes BF16")
        a = parse(Int, c[4]); b = parse(Int, c[5])
        t[c[1]] = (parse.(Int, split(c[3], ',')), 8 + n + a, b - a)
    end
    ST(t, m)
end
function jparse(s::String)
    i = Ref(1); ws() = while s[i[]] in (' ', '\n', '\t', '\r'); i[] = nextind(s, i[]); end
    function str()
        i[] += 1; io = IOBuffer()
        while true
            c = s[i[]]; i[] = nextind(s, i[])
            c == '"' && return String(take!(io))
            if c == '\\'
                e = s[i[]]; i[] += 1
                if e == 'u'
                    u = parse(UInt32, s[i[]:i[]+3], base = 16); i[] += 4
                    if 0xd800 <= u < 0xdc00                # a surrogate pair
                        v = parse(UInt32, s[i[]+2:i[]+5], base = 16); i[] += 6; u = 0x10000 + ((u - 0xd800) << 10) + (v - 0xdc00)
                    end
                    print(io, Char(u))
                else
                    print(io, get(Dict('n' => '\n', 't' => '\t', 'r' => '\r', 'b' => '\b', 'f' => '\f'), e, e))
                end
            else
                print(io, c)
            end
        end
    end
    function val()
        ws(); c = s[i[]]
        if c == '{'
            d = Dict{String,Any}(); i[] += 1; ws()
            s[i[]] == '}' && (i[] += 1; return d)
            while true
                ws(); k = str(); ws(); i[] += 1; d[k] = val(); ws(); e = s[i[]]; i[] += 1
                e == '}' && return d
            end
        elseif c == '['
            a = Any[]; i[] += 1; ws()
            s[i[]] == ']' && (i[] += 1; return a)
            while true
                push!(a, val()); ws(); e = s[i[]]; i[] += 1
                e == ']' && return a
            end
        elseif c == '"'
            return str()
        elseif startswith(SubString(s, i[]), "true")
            i[] += 4; return true
        elseif startswith(SubString(s, i[]), "false")
            i[] += 5; return false
        elseif startswith(SubString(s, i[]), "null")
            i[] += 4; return nothing
        end
        j = i[]
        while s[j] in "+-.eE0123456789"; j += 1; end
        v = s[i[]:j-1]; i[] = j
        occursin(r"[.eE]", v) ? parse(Float64, v) : parse(Int, v)
    end
    val()
end

# ---------------------------------------------------------------------------------------------
# The model. x stays in BF16: each tensor the vendor's forward rounds to BF16 is rounded here (bf), the sums between run
# in Float32 (a BF16 weight is its Float32's upper half, so unpacking one is a shift or a mask, no conversion). Live
# vectors are kept in Float32 registers and arrays holding BF16 values; the attention cache and conv ring are BF16.
# ---------------------------------------------------------------------------------------------
const NE, NFF, NV, NL = 2048, 7680, 128000, 24
const HD, NH, NKV = 128, 16, 4
const NEPS, SCALE, LOG2E = 1.0f-5, Float32(128.0^-0.5), Float32(log2(MathConstants.e))
const WIN, RING = 256, 288                                 # the sliding window; its ring (a window + a block, 9 row groups)
const CRING = 16                                           # a conv layer's ring of inputs: 7 taps + a block
const NPOS = 16                                             # positions a prefill forward
const MAXCTX = 262144                                      # positions in the cache, shared by the requests of a dispatch
const NR = 4                                               # request slots: requests at once, each with its own cache
const MAXGEN = 8192
isattn(l) = l >= 2 && iseven(l)
isfull(l) = isattn(l) && (l - 2) % 4 == 0
aidx(l) = (l - 2) >> 1                                     # 0 .. 10
cidx(l) = l < 2 ? l : (l + 1) >> 1                         # 0 .. 12
const HEADL = Int32(NL)

# a BF16 tensor's Float32 values
bf32(u::UInt16) = reinterpret(Float32, UInt32(u) << 16)
tensor(g::ST, name) = (t = g.t[name]; reinterpret(UInt16, view(g.map, t[2]+1:t[2]+t[3])))

# ---------------------------------------------------------------------------------------------
# GEMV: y[p][r] = sum_k W[r][k] x[p][k] over P positions, the weights read once for all P. One lane a row: a wavefront
# takes 32 rows, and K is cut into slices of KS elements, one wavefront a (32 rows, slice). A lane's 16-byte load is 8
# BF16 weights, and the weights are repacked at load so that a wavefront's 32 loads are one contiguous 512 bytes:
# row r = 32 G + lane, chunk c of 8 elements at (G nc + c) 512 + 16 lane (nc = K / 8). x is the same on every lane, so it
# is read with scalar loads into SGPRs, which the fused multiply-adds take as an operand; its layout is [k][p]. Each
# slice writes partial sums part[slice][p][r]; the consumer adds the slices (psum). The attention cache has the same
# format, so a score and a value sum are this GEMV too.
# ---------------------------------------------------------------------------------------------
const LANES, WGS = Int32(32), Int32(256)
const WVS = WGS ÷ LANES                                    # wavefronts a workgroup
const CHUNK = 512                                          # bytes of a wavefront's 32 lanes of 16
const KS = 512                                             # elements a K slice
const CKS = KS ÷ 8                                         # chunks a K slice
const LP{T} = Core.LLVMPtr{T,1}
const V4U = NTuple{4,VecElement{UInt32}}
const V4F = NTuple{4,VecElement{Float32}}
# 16 weight bytes a lane, non-temporal: each weight is read once a token, and a decode forward streams more than 100 MB a
# layer through the L2, which would otherwise evict the code, x, the partial sums and the state every phase
@inline function ld128(W::LP{UInt8}, lo::UInt32)
    p = reinterpret(LP{V4U}, W + lo)
    v = Base.llvmcall(("""
        define <4 x i32> @ldnt(ptr addrspace(1) %p) #0 {
            %v = load <4 x i32>, ptr addrspace(1) %p, align 16, !nontemporal !0
            ret <4 x i32> %v
        }
        attributes #0 = { alwaysinline }
        !0 = !{i32 1}
        """, "ldnt"), V4U, Tuple{LP{V4U}}, p)
    (v[1].value, v[2].value, v[3].value, v[4].value)
end
# P floats of x as one scalar load: invariant, so with a uniform address it is a load into SGPRs. It can't move above the
# barrier that orders it after x was written, because the base comes through fresh, an opaque copy each phase.
for P in (1, 2, 3, 4, 8, 16)
    ir = """
        define <$P x i32> @xv(ptr addrspace(1) %p) #0 {
            %v = load <$P x i32>, ptr addrspace(1) %p, align 4, !invariant.load !0
            ret <$P x i32> %v
        }
        attributes #0 = { alwaysinline }
        !0 = !{}
        """
    @eval @inline function xv(p::LP{Float32}, ::Val{$P})
        v = Base.llvmcall(($ir, "xv"), NTuple{$P,VecElement{UInt32}}, Tuple{LP{Float32}}, p)
        ntuple(i -> v[i].value, Val($P))
    end
end
@inline fresh(p::LP{T}) where {T} = reinterpret(LP{T}, Base.llvmcall("""
    %r = call i64 asm sideeffect "; fresh \$0", "=s,0"(i64 %0)
    ret i64 %r""", Int64, Tuple{Int64}, reinterpret(Int64, p)))
# v_dot2_f32_bf16: a pair of BF16 weights on a pair of BF16 x, added to a Float32 sum, one instruction
@inline dot2(a::UInt32, b::UInt32, c::Float32) = Base.llvmcall(("""
    declare float @llvm.amdgcn.fdot2.f32.bf16(<2 x bfloat>, <2 x bfloat>, float, i1 immarg)
    define float @d2(i32 %a, i32 %b, float %c) #0 {
        %x = bitcast i32 %a to <2 x bfloat>
        %y = bitcast i32 %b to <2 x bfloat>
        %r = call float @llvm.amdgcn.fdot2.f32.bf16(<2 x bfloat> %x, <2 x bfloat> %y, float %c, i1 false)
        ret float %r
    }
    attributes #0 = { alwaysinline }
    """, "d2"), Float32, Tuple{UInt32,UInt32,Float32}, a, b, c)
@inline dot2v(a::NTuple{P,Float32}, w::UInt32, x::NTuple{P,UInt32}) where {P} = ntuple(p -> dot2(w, x[p], a[p]), Val(P))
# one wavefront: nch chunks of W (a chunk every cs bytes) against x (at a chunk's first element), the P sums of its lanes
@inline function gchunk(acc::NTuple{P,Float32}, w::NTuple{4,UInt32}, x::LP{Float32}) where {P}
    Base.Cartesian.@nexprs 4 i -> begin
        acc = dot2v(acc, w[i], xv(x, Val(P))); x += 4P
    end
    acc
end
@inline ldg(q::LP{UInt8}, cs::Int64, lo::UInt32, ::Val{L}) where {L} = ntuple(i -> ld128(q + (i - 1) * cs, lo), Val(L))
# The value mix, where a row a position can't see meets a score of 0 in the same pair as a row it can. v_dot2_f32_bf16 is not
# exactly rounded, so what such a row holds (a later row of a prefill block, an earlier request's row) reached the sum, by up
# to a BF16 step of a logit. Exact multiply-adds here: a hidden row can't reach it, and a position's output doesn't depend on
# the block it ran in or the request before it.
@inline fma2v(a::NTuple{P,Float32}, w::UInt32, x::NTuple{P,UInt32}) where {P} = ntuple(p -> fma(reinterpret(Float32, w & 0xffff0000), reinterpret(Float32, x[p] & 0xffff0000), fma(reinterpret(Float32, w << 16), reinterpret(Float32, x[p] << 16), a[p])), Val(P))
@inline function gemvx(W::LP{UInt8}, lo::UInt32, cs::Int64, nch::Int32, x::LP{Float32}, ::Val{P}) where {P}
    acc = ntuple(_ -> 0.0f0, Val(P)); q = W; xq = x
    for _ in Int32(1):nch
        w = ld128(q, lo); q += cs
        Base.Cartesian.@nexprs 4 i -> begin
            acc = fma2v(acc, w[i], xv(xq, Val(P))); xq += 4P
        end
    end
    acc
end
const LOADS = 4                                            # chunks a wavefront's load group: loads in flight together
@inline function gemv(W::LP{UInt8}, lo::UInt32, cs::Int64, nch::Int32, x::LP{Float32}, ::Val{P}) where {P}
    acc = ntuple(_ -> 0.0f0, Val(P)); q = W; xq = x; n = nch
    while n >= Int32(LOADS)
        w = ldg(q, cs, lo, Val(LOADS)); q += LOADS * cs; n -= Int32(LOADS)
        Base.Cartesian.@nexprs 4 g -> begin
            acc = gchunk(acc, w[g], xq); xq += 16P
        end
    end
    while n > Int32(0)
        acc = gchunk(acc, ld128(q, lo), xq); xq += 16P; q += cs; n -= Int32(1)
    end
    acc
end
# round to BF16 (to nearest, ties to even), kept in a Float32
@inline function bf(x::Float32)
    u = reinterpret(UInt32, x)
    reinterpret(Float32, (u + 0x00007fff + ((u >> 16) & 0x00000001)) & 0xffff0000)
end
@inline rsq(x::Float32) = ccall("llvm.amdgcn.rsq.f32", llvmcall, Float32, (Float32,), x)
@inline exp2f(x::Float32) = ccall("llvm.exp2.f32", llvmcall, Float32, (Float32,), x)
@inline sigm(x::Float32) = 1.0f0 / (1.0f0 + exp2f(-LOG2E * x))
# a matrix on the device, repacked (rows padded to a multiple of 32 with zeros)
function dev(a::Array)
    d = ROCVector{eltype(a)}(undef, length(a))
    GC.@preserve a begin copyto!(d, a); AMDGPU.synchronize() end
    d
end
function up!(d::ROCArray, a::Array)
    GC.@preserve a begin copyto!(d, 1, a, 1, length(a)); AMDGPU.synchronize() end
    d
end
struct DMat
    W::ROCVector{UInt8}; rows::Int; K::Int; ns::Int
end
function repack(src, rows, K)
    ng = cld(rows, 32); nc = K ÷ 8; o = zeros(UInt16, ng * nc * CHUNK ÷ 2)
    Threads.@threads for G in 0:ng-1
        for lane in 0:31
            r = 32G + lane; r < rows || continue
            for c in 0:nc-1, e in 0:7
                o[(G*nc+c)*CHUNK÷2+8lane+e+1] = src[r*K+8c+e+1]
            end
        end
    end
    o
end
function DMat(g::ST, name; ns = cld(g.t[name][1][2], KS))
    rows, K = g.t[name][1]
    K % 8 == 0 || error("$name: K $K")
    DMat(dev(reinterpret(UInt8, repack(tensor(g, name), rows, K)) |> collect), cld(rows, 32) * 32, K, ns)
end

# ---------------------------------------------------------------------------------------------
# The cooperative persistent kernel: one dispatch runs a whole request (every forward of the prefill, then every decode
# step, the token choice included). All its workgroups are resident at once and step through the same sequence of phases
# with a grid barrier (gbar!) between phases; the kernel's every wait has a time limit, so a stuck wave can't hold the GPU.
# Time accounting: for each phase, the first start and the last end over the workgroups that worked in it go into acct;
# their union is instruction time, the rest of the request's time is idle.
# ---------------------------------------------------------------------------------------------
# the 100 MHz reference clock (gfx11 has no s_memrealtime, and readcyclecounter is a 20-bit shader-cycle count)
@inline memtime() = Base.llvmcall(("""
    declare i64 @llvm.amdgcn.s.sendmsg.rtn.i64(i32 immarg)
    define i64 @mt() #0 {
        %v = call i64 @llvm.amdgcn.s.sendmsg.rtn.i64(i32 131)
        ret i64 %v
    }
    attributes #0 = { alwaysinline }
    """, "mt"), UInt64, Tuple{})
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
@inline add64(p::LP{Int64}, v::Int64) = Base.llvmcall("%r = atomicrmw add ptr addrspace(1) %0, i64 %1 syncscope(\"agent\") monotonic, align 8\nret i64 %r", Int64, Tuple{LP{Int64},Int64}, p, v)
@inline fence_rel() = Base.llvmcall("fence syncscope(\"agent\") release\nret void", Cvoid, Tuple{})
@inline fence_acq() = Base.llvmcall("fence syncscope(\"agent\") acquire\nret void", Cvoid, Tuple{})
@inline rdf(v::Int32) = AMDGPU.Device.readfirstlane(v)     # a value the same on every lane, uniform to the compiler

# everything the kernel reads and writes
struct KR
    tab::LP{Int64}                                         # (address, rows, chunks, slices) a matrix; vector addresses
    x::LP{Float32}; o::LP{Float32}; xt::LP{Float32}; part::LP{Float32}
    qh::LP{Float32}; rc::LP{Float32}; rf::LP{Float32}; gh::LP{Float32}; et::LP{Float32}; pv::LP{Float32}
    eb::LP{Float32}; es::LP{Float32}; lg::LP{Float32}
    bar::LP{UInt32}; acct::LP{UInt64}; ssq::LP{Int64}; best::LP{UInt64}
    inp::LP{Int32}; rq::LP{Int32}; tok::LP{Int32}; posr::LP{Int32}; posa::LP{Int32}; tks::LP{Int32}; nem::LP{Int32}
    out::LP{Int32}; hf::LP{UInt32}
    nreq::Int32; mode::Int32; nwg::Int32; npos::Int32
    temp::Float32                                          # above 0: a slot samples (the slots' settings are in sp)
    dyn::Int32                                             # wavefronts claim matrix items (0: a fixed share each)
    sl::LP{Float32}; hs::LP{UInt32}; th::LP{Float32}; cn::LP{UInt32}; cd::LP{Int32}; rng::LP{UInt64}
    pen::Float32; seen::LP{UInt32}                         # any slot penalizes (1: none); a bit a vocabulary row, a slot
    sp::LP{Float32}; si::LP{Int32}                         # a slot's (temperature, top_k, top_p, penalty); its stop tokens
    apos::LP{Int32}; lim::LP{Int32}                        # a slot's next prompt position; where its prompt stops for now (a snapshot)
end
const SEENW = NV ÷ 32                                      # words of a request's seen-token bits
const TABV = (NL + 1) * 8 * 4                              # where the vector addresses start in the table
const ERW = MAXCTX + NR * RING + 32                        # rows of a score buffer (the arena, then the rings)
const EGW = ERW ÷ 32                                       # row groups of a score buffer
const PVS = MAXCTX ÷ KS + 2                                # K slices of a value sum
const TOUT = Int64(1_000_000_000)                          # 10 s of the 100 MHz clock: a wait longer than this aborts
@inline ld(p, i) = unsafe_load(p, i + 1)                   # 0-based
@inline st!(p, v, i) = unsafe_store!(p, v, i + 1)
@inline ldt(p::LP{Int64}, i::Int64) = Base.llvmcall(("""
    define i64 @ldt(ptr addrspace(1) %p, i64 %i) #0 {
        %q = getelementptr inbounds i64, ptr addrspace(1) %p, i64 %i
        %v = load i64, ptr addrspace(1) %q, align 8, !invariant.load !0
        ret i64 %v
    }
    attributes #0 = { alwaysinline }
    !0 = !{}
    """, "ldt"), Int64, Tuple{LP{Int64},Int64}, p, i)
@inline mat(A::KR, l, j) = (e = (Int64(l) * 8 + j) * 4; (reinterpret(LP{UInt8}, ldt(A.tab, e)), Int32(ldt(A.tab, e + 1)), Int32(ldt(A.tab, e + 2)), Int32(ldt(A.tab, e + 3))))
@inline vaddr(A::KR, l, j) = ldt(A.tab, Int64(TABV + Int64(l) * 8 + j))
@inline fvec(A::KR, l, j) = reinterpret(LP{Float32}, vaddr(A, l, j))
@inline aborted(A::KR) = ldacq(A.bar + 8) != UInt32(0)
@inline wgid() = KI.get_group_id(Int32).x - Int32(1)
@inline tid() = KI.get_local_id(Int32).x - Int32(1)
@inline lanei() = tid() & (LANES - Int32(1))
@inline waveid() = rdf(WVS * wgid() + (tid() >> Int32(5))) # a wavefront's number, uniform; so is what an item derives from it
@inline nwv(A::KR) = WVS * A.nwg
# grid barrier (count, generation, abort); a wait past TOUT sets abort, after which every barrier passes at once
@inline function gbar!(A::KR)
    fence_rel(); KI.barrier()
    if tid() == Int32(0)
        g = ldmono(A.bar + 4)
        if atadd(A.bar, UInt32(1)) == UInt32(A.nwg - 1)
            stmono(A.bar, UInt32(0)); stmono(A.bar + 128, UInt32(0)); atadd(A.bar + 4, UInt32(1))
        else
            t0 = memtime()
            while ldacq(A.bar + 4) == g
                (memtime() - t0 > TOUT || aborted(A)) && (atadd(A.bar + 8, UInt32(1)); break)
                ssleep()
            end
        end
    end
    KI.barrier(); fence_acq(); dcinv()
end
@inline function done!(A::KR, seq, t0, did)                # a workgroup's part of phase seq into the accounting
    (did && tid() == Int32(0)) && (p = A.acct + 8 * (3 + 2Int64(seq)); umin64(p, t0); umax64(p + 8, memtime()))
    return
end
# the first item at or past base of a wavefront taking every NW-th
@inline itemat(gw::Int32, base::Int32, NW::Int32) = base <= gw ? gw : gw + cld(base - gw, NW) * NW

# ---- sums of squares: two banks per position, fixed point (2^-32), so the sum doesn't depend on who adds first ----
const BSSQ = 16                                            # positions a bank
@inline sq0!(A::KR, b, p) = st!(A.ssq, Int64(0), Int64(b) * BSSQ + p)
@inline function sqadd!(A::KR, b, p, lane, s::Float32)
    t = KI.sub_group_reduce(+, s)
    lane == Int32(0) && add64(A.ssq + 8 * (Int64(b) * BSSQ + p), unsafe_trunc(Int64, t * 4294967296.0f0))
    return
end
@inline rinv(A::KR, b, p) = rsq(Float32(ld(A.ssq, Int64(b) * BSSQ + p)) * Float32(2.0^-32 / NE) + NEPS)

# ---- phases (every workgroup calls each one; the work is spread over wavefronts) ----
# a BF16 value at a byte offset; x's element k at position p of P: x for a GEMV is pairs, [k / 2][p], so a pair is one
# dword for the dot product
@inline stb!(b::LP{Float32}, v::Float32, o) = unsafe_store!(reinterpret(LP{UInt16}, b + o), (reinterpret(UInt32, v) >> 16) % UInt16)
@inline stx!(A::KR, v::Float32, k::Int32, P, p) = stb!(A.xt, v, (Int64(k >> Int32(1)) * P + p) * 4 + (k & Int32(1)) * 2)
# partial sums of matrix slot j0 + m - 1 of the group j0 ..: floats before it in part
@inline function pof(A::KR, l, j0, m, P)
    o = Int64(0)
    for i in Int32(0):Int32(m - 2)
        _, rows, _, ns = mat(A, l, Int32(j0) + i); o += Int64(ns) * P * rows
    end
    o
end
# Σ over the K slices of output row r at position p: the loads in groups of 16 independent ones (a plain loop waits for
# each load before the next add)
@inline function psum(A::KR, o::Int64, ns::Int32, rows::Int32, P::Int32, p::Int32, r::Int32)
    b = A.part + 4o; st = Int64(P) * rows; e = Int64(p) * rows + r; s = 0.0f0
    for q in Int32(0):Int32(16):ns-Int32(1)
        v = let q = q; ntuple(j -> ld(b, Int64(min(q + Int32(j - 1), ns - Int32(1))) * st + e), Val(16)) end
        Base.Cartesian.@nexprs 16 j -> (s += q + Int32(j - 1) < ns ? v[j] : 0.0f0)
    end
    s
end
# the items (32 rows, one K slice) of the matrices j0 .. of layer l, over every wavefront
@inline function mat_item!(A::KR, W::LP{UInt8}, o::Int64, nc, ns, rows, G, sl, lane, xp::LP{Float32}, ::Val{P}) where {P}
    c0 = sl * Int32(CKS); nch = min(Int32(CKS), nc - c0)
    acc = gemv(W + (Int64(G) * nc + c0) * CHUNK, (lane % UInt32) << 4, Int64(CHUNK), nch, xp + 16 * P * Int64(c0), Val(P))
    r = G * LANES + lane
    for p in 1:P
        st!(A.part, acc[p], o + (Int64(sl) * P + (p - 1)) * rows + r)
    end
end
# Dynamic dispatch of a phase's items: a wavefront claims its next item from a counter the barrier zeroes, so one held up (a
# slow load, a desktop shader sharing its SIMD) takes fewer and the others more, or (dyn 0) each takes every NW-th.
@inline function claim(A::KR, lane::Int32)
    v = Int32(0)
    lane == Int32(0) && (v = reinterpret(Int32, atadd(A.bar + 128, UInt32(1))))
    rdf(v)
end
# (the matrix of the group j0 .. holding item g: found, its table entry, the item within it, the floats before its sums)
@inline function matof(A::KR, l::Int32, j0::Int32, nm::Int32, g::Int32, P::Int32)
    base = Int32(0); o = Int64(0); m = Int32(0)
    while true
        W, rows, nc, ns = mat(A, l, j0 + m); n = (rows >> Int32(5)) * ns
        g < base + n && return true, W, rows, nc, ns, g - base, o
        base += n; o += Int64(ns) * P * rows; m += Int32(1)
        m == nm && return false, W, rows, nc, ns, Int32(0), o
    end
end
@inline function mat_ph!(A::KR, l::Int32, j0::Int32, nm::Int32, seq::Int32, P::Int32)
    t0 = memtime(); did = false; lane = lanei(); gw = waveid(); NW = nwv(A); xp = fresh(A.xt); dyn = A.dyn != Int32(0)
    g = dyn ? claim(A, lane) : gw
    while true
        ok, W, rows, nc, ns, it, o = matof(A, l, j0, nm, g, P)
        ok || break
        G = it ÷ ns; sl = it - G * ns
        if P == Int32(1)
            mat_item!(A, W, o, nc, ns, rows, G, sl, lane, xp, Val(1))
        elseif P == Int32(2)
            mat_item!(A, W, o, nc, ns, rows, G, sl, lane, xp, Val(2))
        elseif P == Int32(3)
            mat_item!(A, W, o, nc, ns, rows, G, sl, lane, xp, Val(3))
        elseif P == Int32(4)
            mat_item!(A, W, o, nc, ns, rows, G, sl, lane, xp, Val(4))
        elseif P == Int32(8)
            mat_item!(A, W, o, nc, ns, rows, G, sl, lane, xp, Val(8))
        else
            mat_item!(A, W, o, nc, ns, rows, G, sl, lane, xp, Val(16))
        end
        did = true; g = dyn ? claim(A, lane) : g + NW
    end
    done!(A, seq, t0, did)
end
# the forward's positions: prefill, positions a0 .. a0 + P - 1 of request r; decode (r < 0), the next position of every
# request still generating
@inline function setpos_ph!(A::KR, r::Int32, a0::Int32, P::Int32)
    i = WGS * wgid() + tid()
    if r >= Int32(0)
        while i < P
            st!(A.posr, r, i); st!(A.posa, a0 + i, i); st!(A.tok, ld(A.inp, ld(A.rq, 5r) + a0 + i), i); sq0!(A, 1, i); i += WGS * A.nwg
        end
    elseif i == Int32(0)
        n = Int32(0)
        for q in Int32(0):A.nreq-Int32(1)
            m = ld(A.nem, q); ni = ld(A.rq, 5q + Int32(1))
            if ld(A.apos, q) >= ni && m >= Int32(1) && m < ld(A.rq, 5q + Int32(2))
                st!(A.posr, q, n); st!(A.posa, ni + m - Int32(1), n); st!(A.tok, ld(A.tks, q), n); sq0!(A, 1, n); n += Int32(1)
            end
        end
    end
end
# the embeddings of the positions (the head's rows) into x, with their sums of squares (bank 1); the positions' rope
# tables: rc[(p, full or sliding)][angle] (cos, sin), BF16 as the vendor's forward has them
@inline function emb_ph!(A::KR, seq::Int32, P::Int32)
    t0 = memtime(); lane = lanei(); gw = waveid(); NW = nwv(A)
    W, _, nc, _ = mat(A, HEADL, Int32(0)); n = Int32(64) * P
    g = gw
    while g < n
        p = g >> Int32(6); G = g & Int32(63); k = G * LANES + lane
        tok = rdf(ld(A.tok, p))
        o = (Int64(tok >> Int32(5)) * nc + (k >> Int32(3))) * CHUNK + Int64(tok & Int32(31)) * 16 + Int64(k & Int32(7)) * 2
        v = bf32(ld(reinterpret(LP{UInt16}, W + o), 0))
        st!(A.x, v, Int64(k) * P + p)
        sqadd!(A, 1, p, lane, v * v)
        g += NW
    end
    # (the second pass of items: a wavefront a (position, table, 32 angles))
    g = itemat(gw, n, NW); n2 = n + Int32(4) * P
    while g < n2
        it = g - n; p = it >> Int32(2); tt = (it >> Int32(1)) & Int32(1); i = (it & Int32(1)) * LANES + lane
        a = Float32(rdf(ld(A.posa, p))) * ld(A.rf, tt * Int32(64) + i)
        st!(A.rc, bf(cos(a)), ((Int64(p) * 2 + tt) * 64 + i) * 2); st!(A.rc, bf(sin(a)), ((Int64(p) * 2 + tt) * 64 + i) * 2 + 1)
        g += NW
    end
    done!(A, seq, t0, true)
end
# x normalized and weighted into xt, positions pf ..; reads bank 1, zeroes bank 0 and the head's argmax
@inline function pre_ph!(A::KR, w::LP{Float32}, seq::Int32, P::Int32, pf::Int32)
    t0 = memtime(); lane = lanei(); gw = waveid(); NW = nwv(A); Np = P - pf; n = Int32(64) * Np
    g = gw
    while g < n
        q = g >> Int32(6); G = g & Int32(63); k = G * LANES + lane; p = pf + q
        rs = rinv(A, 1, p)
        if G == Int32(0) && lane == Int32(0)
            sq0!(A, 0, p); st!(A.best, UInt64(0), q)
        end
        stx!(A, bf(bf(ld(A.x, Int64(k) * P + p) * rs) * ld(w, k)), k, Np, q)
        g += NW
    end
    done!(A, seq, t0, true)
end
# a mixer's or FFN's output, its slices added, into o, with its sums of squares (bank 0); zeroes bank 1
@inline function ssq_ph!(A::KR, l::Int32, j::Int32, seq::Int32, P::Int32)
    t0 = memtime(); lane = lanei(); gw = waveid(); NW = nwv(A); n = Int32(64) * P
    _, rows, _, ns = mat(A, l, j)
    g = gw
    while g < n
        p = g >> Int32(6); G = g & Int32(63); k = G * LANES + lane
        G == Int32(0) && lane == Int32(0) && sq0!(A, 1, p)
        v = bf(psum(A, Int64(0), ns, rows, P, p, k))
        st!(A.o, v, Int64(k) * P + p)
        sqadd!(A, 0, p, lane, v * v)
        g += NW
    end
    done!(A, seq, t0, true)
end
# o normalized and weighted, added to x, x's sums of squares (bank 1)
@inline function norm_ph!(A::KR, w::LP{Float32}, seq::Int32, P::Int32)
    t0 = memtime(); lane = lanei(); gw = waveid(); NW = nwv(A); n = Int32(64) * P
    g = gw
    while g < n
        p = g >> Int32(6); G = g & Int32(63); k = G * LANES + lane; e = Int64(k) * P + p
        rs = rinv(A, 0, p)
        v = bf(ld(A.x, e) + bf(bf(ld(A.o, e) * rs) * ld(w, k)))
        st!(A.x, v, e)
        sqadd!(A, 1, p, lane, v * v)
        g += NW
    end
    done!(A, seq, t0, true)
end
# the gated short conv after its in_proj: h = B x, the 7-tap causal depthwise conv over a ring of h, C conv into xt
@inline function conv_ph!(A::KR, l::Int32, seq::Int32, P::Int32, pre::Bool)
    t0 = memtime(); lane = lanei(); gw = waveid(); NW = nwv(A); n = Int32(64)
    _, rows, _, ns = mat(A, l, Int32(0)); cw = fvec(A, l, 4); rg = reinterpret(LP{UInt16}, vaddr(A, l, 6))
    g = gw
    while g < n
        k = g * LANES + lane
        w = ntuple(i -> ld(cw, Int64(k) * 7 + (i - 1)), Val(7))
        r0 = Int32(-1); pos = Int32(0); win = ntuple(_ -> 0.0f0, Val(6))
        for p in Int32(0):P-Int32(1)
            r = rdf(ld(A.posr, p)); pos = rdf(ld(A.posa, p))
            if r != r0
                r0 = r; po = pos
                win = ntuple(i -> bf32(ld(rg, (Int64(r) * NE + k) * CRING + ((po - Int32(7) + Int32(i)) & Int32(CRING - 1)))), Val(6))
            end
            B = bf(psum(A, Int64(0), ns, rows, P, p, k)); C = bf(psum(A, Int64(0), ns, rows, P, p, Int32(NE) + k)); X = bf(psum(A, Int64(0), ns, rows, P, p, Int32(2NE) + k))
            h = bf(B * X)
            s = 0.0f0
            for i in 1:6; s += w[i] * win[i]; end
            s += w[7] * h
            stx!(A, bf(C * bf(s)), k, P, p)
            st!(rg, (reinterpret(UInt32, h) >> 16) % UInt16, (Int64(r) * NE + k) * CRING + (pos & Int32(CRING - 1)))
            win = (win[2], win[3], win[4], win[5], win[6], h)
        end
        g += NW
    end
    done!(A, seq, t0, true)
end
# the FFN's activation: silu(w1 x) (w3 x) into xt
@inline function act_ph!(A::KR, l::Int32, seq::Int32, P::Int32)
    t0 = memtime(); lane = lanei(); gw = waveid(); NW = nwv(A); n = Int32(NFF ÷ 32) * P
    _, rows, _, ns = mat(A, l, Int32(5)); ou = Int64(ns) * P * rows
    g = gw
    while g < n
        p = g ÷ Int32(NFF ÷ 32); G = g - p * Int32(NFF ÷ 32); k = G * LANES + lane
        a = bf(psum(A, Int64(0), ns, rows, P, p, k)); b = bf(psum(A, ou, ns, rows, P, p, k))
        stx!(A, bf(bf(a * sigm(a)) * b), k, P, p)
        g += NW
    end
    done!(A, seq, t0, true)
end
# ---- attention. A key head's rows are positions: the cache's keys of layer a are 16 chunks a head, row r = 32 G + lane
# at (G 64 + j 16 + c) 512 + 16 lane, so a score is a GEMV of the keys on the head's 4 queries. Its values have the
# positions as K: chunk tk of 8 positions at (tk 16 + j 4 + Gc) 512 + 16 lane for column 32 Gc + lane, so the mix is a GEMV
# of the values on the scores. Full attention's rows are an arena (a request's base + position), sliding attention's a
# ring (request r's rows r RING + position mod RING).
@inline ring_t(amax::Int32, slot::Int32) = amax - (amax - slot) % Int32(RING)  # the position in a ring slot
@inline function srow(A::KR, full::Bool, r::Int32, pos::Int32)
    full ? rdf(ld(A.rq, 5r + Int32(4))) + pos : r * Int32(RING) + pos % Int32(RING)
end
# (the rows' first group, how many groups a position's scores take; the first value chunk and how many)
@inline function grange(A::KR, full::Bool, r::Int32, pos::Int32, amax::Int32)
    if full
        rb = rdf(ld(A.rq, 5r + Int32(4)))
        G0 = rb >> Int32(5); tk0 = rb >> Int32(3)
        return G0, ((rb + pos) >> Int32(5)) - G0 + Int32(1), tk0, ((rb + pos) >> Int32(3)) - tk0 + Int32(1)
    end
    n = min(amax + Int32(1), Int32(RING))
    r * Int32(RING ÷ 32), (n + Int32(31)) >> Int32(5), r * Int32(RING ÷ 8), (n + Int32(7)) >> Int32(3)
end
# q and k normalized and rotated, v and the gate out of their sums: a workgroup's waves 0 .. 3 take a head of q or k
# (128 lanes, the head's sum of squares and the rotation's other half through shared memory), its waves 4 .. 7 the rows
# of v and the gate
@inline function aprep_ph!(A::KR, lds, l::Int32, seq::Int32, P::Int32)
    t0 = memtime(); t = tid(); w = wgid(); full = isfull(l); tt = full ? Int32(0) : Int32(1)
    kc = reinterpret(LP{UInt8}, vaddr(A, l, 6)); vc = reinterpret(LP{UInt8}, vaddr(A, l, 7))
    qn = fvec(A, l, 4); kn = fvec(A, l, 5); did = false
    _, rq, _, nsq = mat(A, l, Int32(0)); _, rk, _, _ = mat(A, l, Int32(1)); _, rv, _, _ = mat(A, l, Int32(2)); _, rg, _, _ = mat(A, l, Int32(3))
    ok = pof(A, l, 0, 2, P); ov = pof(A, l, 0, 3, P); og = pof(A, l, 0, 4, P)
    if t >= Int32(128)
        lane = t & (LANES - Int32(1)); gw = rdf(Int32(4) * w + ((t - Int32(128)) >> Int32(5))); NW = Int32(4) * A.nwg
        n = Int32(17) * P; g = gw
        while g < n
            p = g ÷ Int32(17); G = g - p * Int32(17)
            r = rdf(ld(A.posr, p)); pos = rdf(ld(A.posa, p))
            if G < Int32(16)                               # v: column 32 (G mod 4) + lane of key head G >> 2
                c = G * LANES + lane; v = bf(psum(A, ov, nsq, rv, P, p, c))
                row = srow(A, full, r, pos)
                o = (Int64(row >> Int32(3)) * 16 + G) * CHUNK + Int64(lane) * 16 + Int64(row & Int32(7)) * 2
                unsafe_store!(reinterpret(LP{UInt16}, vc + o), (reinterpret(UInt32, v) >> 16) % UInt16)
            elseif lane < Int32(16)                        # the gate of head lane
                st!(A.gh, bf(sigm(bf(psum(A, og, nsq, rg, P, p, lane)))), Int64(p) * NH + lane)
            end
            did = true; g += NW
        end
    end
    i = w; n = Int32(20) * P
    while i < n
        p = i ÷ Int32(20); h = i - p * Int32(20); isk = h >= Int32(16)
        c = t; v = 0.0f0; s = 0.0f0
        if t < Int32(128)
            v = isk ? bf(psum(A, ok, nsq, rk, P, p, (h - Int32(16)) * Int32(HD) + c)) : bf(psum(A, Int64(0), nsq, rq, P, p, h * Int32(HD) + c))
            s = v * v
        end
        s = KI.sub_group_reduce(+, s)
        (t < Int32(128) && t & (LANES - Int32(1)) == Int32(0)) && (@inbounds lds[Int(HD) + 1 + (t >> Int32(5))] = s)
        KI.barrier()
        x = 0.0f0
        if t < Int32(128)
            s = @inbounds lds[Int(HD)+1] + lds[Int(HD)+2] + lds[Int(HD)+3] + lds[Int(HD)+4]
            rs = rsq(s / Float32(HD) + NEPS)
            x = bf(bf(v * rs) * ld(isk ? kn : qn, c)); @inbounds lds[c+1] = x
        end
        KI.barrier()
        if t < Int32(128)
            oth = @inbounds lds[(c ⊻ Int32(64))+1]; a = c & Int32(63)
            e = ((Int64(p) * 2 + tt) * 64 + a) * 2; cs = ld(A.rc, e); sn = ld(A.rc, e + 1)
            y = bf(c < Int32(64) ? x * cs - oth * sn : x * cs + oth * sn)
            if isk
                j = h - Int32(16); r = rdf(ld(A.posr, p)); pos = rdf(ld(A.posa, p)); row = srow(A, full, r, pos)
                o = (Int64(row >> Int32(5)) * 64 + j * Int32(16) + (c >> Int32(3))) * CHUNK + Int64(row & Int32(31)) * 16 + Int64(c & Int32(7)) * 2
                unsafe_store!(reinterpret(LP{UInt16}, kc + o), (reinterpret(UInt32, y) >> 16) % UInt16)
            else
                stb!(A.qh, y, ((Int64(p) * 4 + (h >> Int32(2))) * 64 + (c >> Int32(1))) * 16 + (h & Int32(3)) * 4 + (c & Int32(1)) * 2)
            end
        end
        did = true; i += A.nwg
    end
    done!(A, seq, t0, did)
end
# the scores of every visible position: exp(q k), into eb[p][j][row][4 heads] with each wavefront's sums into es
@inline function st4!(p::LP{Float32}, v::NTuple{4,Float32})
    unsafe_store!(reinterpret(LP{V4F}, p), (VecElement(v[1]), VecElement(v[2]), VecElement(v[3]), VecElement(v[4])))
end
@inline function ld4(p::LP{Float32})
    v = unsafe_load(reinterpret(LP{V4F}, p))
    (v[1].value, v[2].value, v[3].value, v[4].value)
end
@inline function score_ph!(A::KR, l::Int32, seq::Int32, P::Int32, pre::Bool)
    t0 = memtime(); did = false; lane = lanei(); gw = waveid(); NW = nwv(A); full = isfull(l)
    kc = reinterpret(LP{UInt8}, vaddr(A, l, 6)); qb = fresh(A.qh); lo = (lane % UInt32) << 4; base = Int32(0)
    amaxb = rdf(ld(A.posa, P - Int32(1)))
    for p in Int32(0):P-Int32(1)
        r = rdf(ld(A.posr, p)); pos = rdf(ld(A.posa, p)); amax = pre ? amaxb : pos
        G0, ng, _, _ = grange(A, full, r, pos, amax); rb = full ? rdf(ld(A.rq, 5r + Int32(4))) : r * Int32(RING); n = Int32(4) * ng
        g = itemat(gw, base, NW)
        while g < base + n
            it = g - base; j = it & Int32(3); gi = it >> Int32(2); G = G0 + gi; row = G * LANES + lane
            acc = gemv(kc + (Int64(G) * 64 + j * Int32(16)) * CHUNK, lo, Int64(CHUNK), Int32(16), qb + ((Int64(p) * 4 + j) * 64) * 16, Val(4))
            ts = full ? row - rb : ring_t(amax, row - rb)
            vis = ts >= Int32(0) && ts <= pos && (full || pos - ts < Int32(WIN))
            e = ntuple(i -> vis ? bf(exp2f(bf(bf(acc[i]) * SCALE) * LOG2E)) : 0.0f0, Val(4))
            eo = ((Int64(p) * 4 + j) * (ERW ÷ 2) + G * Int32(16) + (lane >> Int32(1))) * 16 + (lane & Int32(1)) * 2
            Base.Cartesian.@nexprs 4 i -> stb!(A.eb, e[i], eo + (i - 1) * 4)
            se = ntuple(i -> KI.sub_group_reduce(+, e[i]), Val(4))
            lane == Int32(0) && st4!(A.es + ((Int64(p) * 4 + j) * EGW + gi) * 16, se)
            did = true; g += NW
        end
        base += n
    end
    done!(A, seq, t0, did)
end
# each head's total of the scores into et (items 0 .. 4 P - 1), then the mix: Σ score v, slices into pv
@inline function vmix_ph!(A::KR, l::Int32, seq::Int32, P::Int32, pre::Bool)
    t0 = memtime(); did = false; lane = lanei(); gw = waveid(); NW = nwv(A); full = isfull(l)
    vc = reinterpret(LP{UInt8}, vaddr(A, l, 7)); ebf = fresh(A.eb); lo = (lane % UInt32) << 4
    amaxb = rdf(ld(A.posa, P - Int32(1))); n0 = Int32(4) * P; g = gw
    while g < n0
        p = g >> Int32(2); j = g & Int32(3)
        r = rdf(ld(A.posr, p)); pos = rdf(ld(A.posa, p)); amax = pre ? amaxb : pos
        _, ng, _, _ = grange(A, full, r, pos, amax); tot = (0.0f0, 0.0f0, 0.0f0, 0.0f0)
        for gi in Int32(0):LANES:ng-Int32(1)
            e = gi + lane < ng ? ld4(A.es + ((Int64(p) * 4 + j) * EGW) * 16 + ((gi + lane) % UInt32) << 4) : (0.0f0, 0.0f0, 0.0f0, 0.0f0)
            s = ntuple(i -> KI.sub_group_reduce(+, e[i]), Val(4))
            tot = (tot[1] + s[1], tot[2] + s[2], tot[3] + s[3], tot[4] + s[4])
        end
        if lane == Int32(0)
            st4!(A.et + (Int64(p) * 16 + j * Int32(4)) * 4, tot)
        end
        did = true; g += NW
    end
    base = n0
    for p in Int32(0):P-Int32(1)
        r = rdf(ld(A.posr, p)); pos = rdf(ld(A.posa, p)); amax = pre ? amaxb : pos
        _, _, tk0, ntk = grange(A, full, r, pos, amax); nsv = (ntk + Int32(CKS - 1)) >> Int32(6); n = Int32(16) * nsv
        g = itemat(gw, base, NW)
        while g < base + n
            it = g - base; sl = it >> Int32(4); jg = it & Int32(15); j = jg >> Int32(2); Gc = jg & Int32(3)
            c0 = sl * Int32(CKS); nch = min(Int32(CKS), ntk - c0); tk = tk0 + c0
            acc = gemvx(vc + (Int64(tk) * 16 + jg) * CHUNK, lo, Int64(16 * CHUNK), nch, ebf + ((Int64(p) * 4 + j) * (ERW ÷ 2) + Int64(tk) * 4) * 16, Val(4))
            st4!(A.pv + (((Int64(p) * 4 + j) * PVS + sl) * 128 + Gc * LANES) * 16 + lo, acc)
            did = true; g += NW
        end
        base += n
    end
    done!(A, seq, t0, did)
end
# the mix divided by the head's total, gated, into xt for o_proj
@inline function acomb_ph!(A::KR, l::Int32, seq::Int32, P::Int32, pre::Bool)
    t0 = memtime(); did = false; lane = lanei(); gw = waveid(); NW = nwv(A); full = isfull(l); n = Int32(64) * P; g = gw
    amaxb = rdf(ld(A.posa, P - Int32(1)))
    while g < n
        p = g >> Int32(6); h = (g & Int32(63)) >> Int32(2); Gc = g & Int32(3); j = h >> Int32(2); i = h & Int32(3); c = Gc * LANES + lane
        r = rdf(ld(A.posr, p)); pos = rdf(ld(A.posa, p)); amax = pre ? amaxb : pos
        _, _, _, ntk = grange(A, full, r, pos, amax); nsv = (ntk + Int32(CKS - 1)) >> Int32(6)
        s = 0.0f0; b = A.pv + (Int64(p) * 4 + j) * PVS * 128 * 16 + (Int64(c) * 16 + i * Int32(4))
        for sl in Int32(0):nsv-Int32(1)
            s += ld(b, Int64(sl) * 128 * 4)
        end
        stx!(A, bf(bf(s / ld(A.et, Int64(p) * 16 + h)) * ld(A.gh, Int64(p) * NH + h)), h * Int32(HD) + c, P, p)
        did = true; g += NW
    end
    done!(A, seq, t0, did)
end
# the head (the embedding's rows on x), the whole K a wavefront. Generating: each position's argmax through an atomic
# maximum of (the value, in an order-preserving code; the lowest row first on a tie); scoring: the logits into lg.
@inline function ordk(v::Float32)
    u = reinterpret(UInt32, v)
    u >> 31 != UInt32(0) ? ~u : u | 0x80000000
end
@inline function head_item!(A::KR, W::LP{UInt8}, nc::Int32, G::Int32, lane::Int32, xp::LP{Float32}, pf::Int32, gen::Bool, ::Val{N}) where {N}
    acc = gemv(W + Int64(G) * nc * CHUNK, (lane % UInt32) << 4, Int64(CHUNK), nc, xp, Val(N)); r = G * LANES + lane
    for q in 0:N-1
        v = bf(acc[q+1])
        # the repetition penalty on a row the request has seen, after the rounding: the vendor applies it to the logits in Float32
        pr = (gen && A.pen != 1.0f0) ? rdf(ld(A.posr, pf + Int32(q))) : Int32(0); pn = (gen && A.pen != 1.0f0) ? ld(A.sp, 4pr + Int32(3)) : 1.0f0
        if pn != 1.0f0 && (ld(A.seen, Int64(pr) * SEENW + G) >> lane) & UInt32(1) != UInt32(0)
            v = v < 0.0f0 ? v * pn : v / pn
        end
        (gen && A.temp > 0.0f0) && st!(A.sl, v, Int64(q) * NV + r)
        if gen
            ks = ordk(v)
            if ks >= (ld(A.best, q) >> 32) % UInt32
                umax64(A.best + 8q, UInt64(ks) << 32 | UInt64(~reinterpret(UInt32, r)))
            end
        else
            st!(A.lg, v, Int64(rdf(ld(A.posa, pf + Int32(q)))) * NV + r)
        end
    end
end
@inline function head_ph!(A::KR, seq::Int32, P::Int32, pf::Int32)
    t0 = memtime(); did = false; lane = lanei(); gw = waveid(); NW = nwv(A); gen = A.mode == Int32(0); xp = fresh(A.xt); Np = P - pf
    W, rows, nc, _ = mat(A, HEADL, Int32(0)); n = rows >> Int32(5); dyn = A.dyn != Int32(0); g = dyn ? claim(A, lane) : gw
    if gen && A.temp > 0.0f0                               # the sampler's histogram and list start empty
        i = WGS * wgid() + tid(); i < Int32(NR) * HBINS && st!(A.hs, UInt32(0), i)
    end
    if gen && A.temp > 0.0f0 && wgid() == Int32(0)
        t = tid(); t < Int32(NR) && st!(A.cn, UInt32(0), t)
    end
    while g < n
        if Np == Int32(1)
            head_item!(A, W, nc, g, lane, xp, pf, gen, Val(1))
        elseif Np == Int32(2)
            head_item!(A, W, nc, g, lane, xp, pf, gen, Val(2))
        elseif Np == Int32(3)
            head_item!(A, W, nc, g, lane, xp, pf, gen, Val(3))
        elseif Np == Int32(4)
            head_item!(A, W, nc, g, lane, xp, pf, gen, Val(4))
        elseif Np == Int32(8)
            head_item!(A, W, nc, g, lane, xp, pf, gen, Val(8))
        else
            head_item!(A, W, nc, g, lane, xp, pf, gen, Val(16))
        end
        did = true; g = dyn ? claim(A, lane) : g + NW
    end
    done!(A, seq, t0, did)
end
@inline isstop(A::KR, r, t) = t == ld(A.si, 2r) || t == ld(A.si, 2r + Int32(1))
@inline function tok_out!(A::KR, ix::Int32, r::Int32)       # request r's next token (its last at a stop token)
    st!(A.tks, ix, r); n = ld(A.nem, r)
    if A.pen != 1.0f0                                      # (one lane a slot: no other writer of its bits)
        e = Int64(r) * SEENW + (ix >> Int32(5)); st!(A.seen, ld(A.seen, e) | (UInt32(1) << (ix & Int32(31))), e)
    end
    st!(A.out, ix, ld(A.rq, 5r + Int32(3)) + n)
    st!(A.nem, isstop(A, r, ix) ? ld(A.rq, 5r + Int32(2)) : n + Int32(1), r)
end
# a request's next token (its last at a stop token), a lane a position
@inline function tok_ph!(A::KR, seq::Int32, P::Int32, pf::Int32)
    (wgid() == Int32(0) && tid() < P - pf) || return
    t0 = memtime(); q = tid()
    tok_out!(A, reinterpret(Int32, ~(ld(A.best, q) % UInt32)), ld(A.posr, pf + q))
    done!(A, seq, t0, true)
end

# ---- sampling: temperature, top-k, top-p, as flashd's sample line asks. The head's phase also keeps the logits (sl); a
# histogram below each position's maximum finds the k-th largest without sorting 128000 rows; the rows at or above it go
# into a short list; one lane a position sorts that list in shared memory and draws. A row more than 22 temperatures below
# the maximum has a probability under 3e-10, so the histogram stops there. The draw depends on the seed alone: the list's
# order (who appended first) does not reach the sort.
const SMARGIN, HBINS, SCAND, SKMAX = 22.0f0, Int32(256), Int32(128), Int32(64)
@inline function bestv(A::KR, q::Int32)                    # a position's maximum, from the argmax's order-preserving code
    u = reinterpret(UInt32, rdf(reinterpret(Int32, (ld(A.best, q) >> 32) % UInt32)))
    reinterpret(Float32, u >> 31 != UInt32(0) ? u & 0x7fffffff : ~u)
end
@inline function hist_ph!(A::KR, seq::Int32, P::Int32, pf::Int32)
    t0 = memtime(); lane = lanei(); gw = waveid(); NW = nwv(A); Np = P - pf; nq = Int32(NV ÷ 32)
    g = gw; did = false
    while g < nq * Np
        q = g ÷ nq; r = (g - q * nq) * LANES + lane; mx = bestv(A, q); w = ld(A.sp, 4rdf(ld(A.posr, pf + q))) * SMARGIN / Float32(HBINS)
        d = (mx - ld(A.sl, Int64(q) * NV + r)) / w
        if d < Float32(HBINS)
            atadd(A.hs + 4 * (Int64(q) * HBINS + unsafe_trunc(Int32, d)), UInt32(1))
        end
        did = true; g += NW
    end
    done!(A, seq, t0, did)
end
@inline function thr_ph!(A::KR, seq::Int32, P::Int32, pf::Int32)
    (wgid() == Int32(0) && tid() < P - pf) || return
    t0 = memtime(); q = tid(); rs = ld(A.posr, pf + q); T = ld(A.sp, 4rs); w = T * SMARGIN / Float32(HBINS); k = min(unsafe_trunc(Int32, ld(A.sp, 4rs + Int32(1))), SKMAX); c = Int32(0); b = Int32(0)
    while b < HBINS - Int32(1)
        c += reinterpret(Int32, ld(A.hs, Int64(q) * HBINS + b))
        if c >= k
            # (more than the list holds: the bin before, so what arrives first never decides what is listed)
            (c > SCAND && b > Int32(0)) && (b -= Int32(1))
            break
        end
        b += Int32(1)
    end
    st!(A.th, T > 0.0f0 ? unordf(ld(A.best, q)) - Float32(b + Int32(1)) * w : Inf32, q)   # (a greedy slot lists nothing)
    done!(A, seq, t0, true)
end
@inline function unordf(k::UInt64)                         # (a lane's own maximum)
    u = (k >> 32) % UInt32
    reinterpret(Float32, u >> 31 != UInt32(0) ? u & 0x7fffffff : ~u)
end
@inline function cand_ph!(A::KR, seq::Int32, P::Int32, pf::Int32)
    t0 = memtime(); lane = lanei(); gw = waveid(); NW = nwv(A); Np = P - pf; nq = Int32(NV ÷ 32)
    g = gw; did = false
    while g < nq * Np
        q = g ÷ nq; r = (g - q * nq) * LANES + lane
        if ld(A.sl, Int64(q) * NV + r) > ld(A.th, q)
            i = reinterpret(Int32, atadd(A.cn + 4q, UInt32(1)))
            i < SCAND && st!(A.cd, r, Int64(q) * SCAND + i)
        end
        did = true; g += NW
    end
    done!(A, seq, t0, did)
end
@inline function stok_ph!(A::KR, lds2, seq::Int32, P::Int32, pf::Int32)
    (wgid() == Int32(0) && tid() < P - pf) || return
    t0 = memtime(); q = tid(); rq = ld(A.posr, pf + q); T = ld(A.sp, 4rq); n = min(reinterpret(Int32, ld(A.cn, q)), SCAND); o = Int(q) * 2 * Int(SCAND)
    if !(T > 0.0f0)                                        # a greedy slot in a forward with sampling ones
        tok_out!(A, reinterpret(Int32, ~(ld(A.best, q) % UInt32)), rq); done!(A, seq, t0, true); return
    end
    i = Int32(0)
    while i < n                                            # insertion sort: the value down, the row up on a tie
        r = ld(A.cd, Int64(q) * SCAND + i); v = ld(A.sl, Int64(q) * NV + r); j = i
        while j > Int32(0)
            vp = @inbounds lds2[o+2Int(j)-1]; rp = reinterpret(Int32, @inbounds lds2[o+2Int(j)])
            (vp > v || (vp == v && rp < r)) && break
            @inbounds lds2[o+2Int(j)+1] = vp; @inbounds lds2[o+2Int(j)+2] = reinterpret(Float32, rp); j -= Int32(1)
        end
        @inbounds lds2[o+2Int(j)+1] = v; @inbounds lds2[o+2Int(j)+2] = reinterpret(Float32, r); i += Int32(1)
    end
    k = min(n, min(unsafe_trunc(Int32, ld(A.sp, 4rq + Int32(1))), SKMAX)); tp = ld(A.sp, 4rq + Int32(2)); mx = @inbounds lds2[o+1]; s = 0.0f0; i = Int32(0); m = Int32(0)
    while i < k                                            # the top k, then as many as reach top_p of them
        s += exp2f((@inbounds(lds2[o+2Int(i)+1]) - mx) / T * LOG2E); i += Int32(1)
    end
    c = 0.0f0; i = Int32(0)
    while i < k
        c += exp2f((@inbounds(lds2[o+2Int(i)+1]) - mx) / T * LOG2E); i += Int32(1); m = i
        c >= tp * s && break
    end
    x = ld(A.rng, rq)                                      # xorshift64*
    x ⊻= x >> 12; x ⊻= x << 25; x ⊻= x >> 27; st!(A.rng, x, rq)
    u = Float32(((x * 0x106689d45497fdb5) >> 40) % UInt32) * Float32(2.0^-24) * c
    c2 = 0.0f0; i = Int32(0); ix = reinterpret(Int32, @inbounds lds2[o+2])
    while i < m
        c2 += exp2f((@inbounds(lds2[o+2Int(i)+1]) - mx) / T * LOG2E)
        if u < c2
            ix = reinterpret(Int32, @inbounds lds2[o+2Int(i)+2]); break
        end
        i += Int32(1)
    end
    tok_out!(A, ix, rq)
    done!(A, seq, t0, true)
end
# ---- the forward as a step program: each phase once in the kernel ----
const S_PRE, S_MAT, S_CONV, S_SSQ, S_NORM, S_ACT, S_APREP, S_SCORE, S_VMIX, S_ACOMB, S_HEAD, S_TOK, S_HIST, S_THR, S_CAND, S_STOK = Int32.(1:16)
# step i of layer l: (kind, first matrix slot, matrices, the vector a norm reads)
@inline function lstep(l::Int32, i::Int32)
    if l == HEADL
        i == Int32(0) && return S_PRE, Int32(0), Int32(0), Int32(0)
        return (i == Int32(1) ? S_HEAD : S_TOK), Int32(0), Int32(0), Int32(0)
    end
    i == Int32(0) && return S_PRE, Int32(0), Int32(0), Int32(0)    # operator_norm
    if isattn(l)
        i == Int32(1) && return S_MAT, Int32(0), Int32(4), Int32(0) # q k v gate
        i == Int32(2) && return S_APREP, Int32(0), Int32(0), Int32(0)
        i == Int32(3) && return S_SCORE, Int32(0), Int32(0), Int32(0)
        i == Int32(4) && return S_VMIX, Int32(0), Int32(0), Int32(0)
        i == Int32(5) && return S_ACOMB, Int32(0), Int32(0), Int32(0)
        i == Int32(6) && return S_MAT, Int32(4), Int32(1), Int32(0) # o
        i == Int32(7) && return S_SSQ, Int32(4), Int32(0), Int32(0)
        i -= Int32(5)
    else
        i == Int32(1) && return S_MAT, Int32(0), Int32(1), Int32(0) # in
        i == Int32(2) && return S_CONV, Int32(0), Int32(0), Int32(0)
        i == Int32(3) && return S_MAT, Int32(1), Int32(1), Int32(0) # out
        i == Int32(4) && return S_SSQ, Int32(1), Int32(0), Int32(0)
        i -= Int32(2)
    end
    i == Int32(3) && return S_NORM, Int32(0), Int32(0), Int32(1)    # post_operator_norm
    i == Int32(4) && return S_PRE, Int32(0), Int32(0), Int32(2)     # ffn_norm
    i == Int32(5) && return S_MAT, Int32(5), Int32(2), Int32(0)     # w1 w3
    i == Int32(6) && return S_ACT, Int32(0), Int32(0), Int32(0)
    i == Int32(7) && return S_MAT, Int32(7), Int32(1), Int32(0)     # w2
    i == Int32(8) && return S_SSQ, Int32(7), Int32(0), Int32(0)
    return S_NORM, Int32(0), Int32(0), Int32(3)                     # post_ffn_norm
end
@inline nsteps(l::Int32, gen::Bool, smp::Bool) = l == HEADL ? (gen ? (smp ? Int32(6) : Int32(3)) : Int32(2)) : isattn(l) ? Int32(15) : Int32(12)
# A cancel (host word 13 set by the host): workgroup 0 posts the decision for this check (bar word 19); after the next
# barrier every workgroup reads it (stopped).
@inline function cancelq!(A::KR)
    if wgid() == Int32(0) && tid() == Int32(0)
        stmono(A.bar + 76, ldacq_sys(A.hf + 52) != UInt32(0) ? UInt32(1) : UInt32(0))
    end
end
@inline stopped(A::KR) = rdf(reinterpret(Int32, ldmono(A.bar + 76))) != Int32(0)
# head: the head over positions pfirst .. P - 1; pre: a prefill forward (a block of one request; a cancel check a layer)
@inline function fwd!(A::KR, lds, lds2, P::Int32, seq::Int32, head::Bool, pfirst::Int32, pre::Bool)
    gen = A.mode == Int32(0); smp = gen && A.temp > 0.0f0
    emb_ph!(A, seq, P); seq += Int32(1)
    nlr = rdf(reinterpret(Int32, ldacq(A.bar + 12))); nl = nlr == Int32(0) ? Int32(NL) : nlr   # bar[3]: layers to run (0: all), for debugging
    l = Int32(0); i = Int32(0)
    while l <= HEADL
        kind, j0, nm, vn = lstep(l, i)
        (smp && l == HEADL && i >= Int32(2)) && (kind = S_HIST + i - Int32(2))
        gbar!(A)
        if kind == S_MAT
            mat_ph!(A, l, j0, nm, seq, P)
        elseif kind == S_PRE
            if l == HEADL
                pre_ph!(A, fvec(A, HEADL, 0), seq, P, pfirst)
            else
                pre_ph!(A, fvec(A, l, vn), seq, P, Int32(0))
            end
        elseif kind == S_SSQ
            ssq_ph!(A, l, j0, seq, P)
        elseif kind == S_NORM
            norm_ph!(A, fvec(A, l, vn), seq, P)
        elseif kind == S_CONV
            conv_ph!(A, l, seq, P, pre)
        elseif kind == S_ACT
            act_ph!(A, l, seq, P)
        elseif kind == S_APREP
            aprep_ph!(A, lds, l, seq, P)
        elseif kind == S_SCORE
            score_ph!(A, l, seq, P, pre)
        elseif kind == S_VMIX
            vmix_ph!(A, l, seq, P, pre)
        elseif kind == S_ACOMB
            acomb_ph!(A, l, seq, P, pre)
        elseif kind == S_HIST
            hist_ph!(A, seq, P, pfirst)
        elseif kind == S_THR
            thr_ph!(A, seq, P, pfirst)
        elseif kind == S_CAND
            cand_ph!(A, seq, P, pfirst)
        elseif kind == S_STOK
            stok_ph!(A, lds2, seq, P, pfirst)
        elseif kind == S_HEAD
            head_ph!(A, seq, P, pfirst)
        else
            tok_ph!(A, seq, P, pfirst)
        end
        seq += Int32(1); i += Int32(1)
        if i == nsteps(l, gen, smp)
            i = Int32(0)
            l = l < nl - Int32(1) ? l + Int32(1) : l < HEADL ? (head ? HEADL : HEADL + Int32(1)) : HEADL + Int32(1)
        end
    end
    gbar!(A)
    seq, false
end
@inline bigblock(n::Int32, npos::Int32) = n >= npos ? npos : n >= Int32(8) ? Int32(8) : n >= Int32(4) ? Int32(4) : n >= Int32(2) ? Int32(2) : Int32(1)
# The slots: each turn of the loop picks the next forward from what the slots need, a block of a prompt or a decode step of
# every slot generating, taking turns so a long prompt doesn't hold up the others' tokens. One call of fwd! (one copy of the
# forward in the kernel). The dispatch ends when nothing is left, when the host asks (word 13: a request arriving or leaving
# joins at the next one, between forwards, so nothing has to be redone) or when a prompt reaches its snapshot.
@kernel unsafe_indices = true function k_req!(A)
    lds = @localmem Float32 (132,); lds2 = @localmem Float32 (512,)
    tid() == Int32(0) && umin64(A.acct, memtime())
    seq = Int32(0); lastpre = false; prefill = true; stop = false; rr = Int32(0)
    while true
        cancelq!(A); gbar!(A)                              # the host's check, each forward
        stopped(A) && break
        rp = Int32(-1); Pd = Int32(0)
        for j in Int32(0):A.nreq-Int32(1)
            q = (rr + j) % A.nreq; ap = rdf(ld(A.apos, q)); ni = rdf(ld(A.rq, 5q + Int32(1))); m = rdf(ld(A.nem, q))
            (rp < Int32(0) && ap < rdf(ld(A.lim, q))) && (rp = q)
            (ap >= ni && m >= Int32(1) && m < rdf(ld(A.rq, 5q + Int32(2)))) && (Pd += Int32(1))
        end
        if prefill && rp < Int32(0)                        # every prompt done
            tid() == Int32(0) && umax64(A.acct + 16, memtime())
            prefill = false
        end
        if rp >= Int32(0) && (!lastpre || Pd == Int32(0))
            a = rdf(ld(A.apos, rp)); l = rdf(ld(A.lim, rp)); ninp = rdf(ld(A.rq, 5rp + Int32(1))); ngen = rdf(ld(A.rq, 5rp + Int32(2)))
            P = bigblock(l - a, A.npos)
            head = A.mode == Int32(1) || (a + P == ninp && ngen > Int32(0)); pfirst = A.mode == Int32(1) ? Int32(0) : P - Int32(1)
            setpos_ph!(A, rp, a, P); gbar!(A)
            seq, stop = fwd!(A, lds, lds2, P, seq, head, pfirst, true)
            # the slot's position, and for the host's progress lines its host word
            if wgid() == Int32(0) && tid() == Int32(0)
                st!(A.apos, a + P, rp); strel_sys(A.hf + 4 * (Int64(16) + rp), UInt32(a + P))
            end
            lastpre = true; rr = rp + Int32(1)
            (a + P == l && l < ninp) && break              # a snapshot: the host copies the slot's rings and carries on
        elseif Pd > Int32(0)
            setpos_ph!(A, Int32(-1), Int32(0), Pd); gbar!(A)
            seq, stop = fwd!(A, lds, lds2, Pd, seq, true, Int32(0), false)
            lastpre = false
        else
            break
        end
    end
    fence_rel(); KI.barrier()
    tid() == Int32(0) && umax64(A.acct + 8, memtime())
end

# ---- weights and buffers ----
struct DLayer
    M::Vector{DMat}; V::Vector{ROCVector{Float32}}         # matrix slots 0 .. 7 (conv: 0, 1, 5, 6, 7), vector slots 0 .. 5
end
f32(g, name) = bf32.(collect(tensor(g, name)))
function load_layer(g::ST, l)
    p = "model.layers.$l."; ffn = [DMat(g, p * "feed_forward.w$i.weight") for i in (1, 3, 2)]
    nm = [dev(f32(g, p * n * ".weight")) for n in ("operator_norm", "post_operator_norm", "ffn_norm", "post_ffn_norm")]
    if isattn(l)
        a = p * "self_attn."
        DLayer([DMat(g, a * "q_proj.weight"); DMat(g, a * "k_proj.weight"); DMat(g, a * "v_proj.weight"); DMat(g, a * "gate_proj.weight"); DMat(g, a * "o_proj.weight"); ffn],
               ROCVector{Float32}[nm..., dev(f32(g, a * "q_norm.weight")), dev(f32(g, a * "k_norm.weight"))])
    else
        c = p * "conv."
        DLayer([DMat(g, c * "in_proj.weight"); DMat(g, c * "out_proj.weight"); ffn[1]; ffn[1]; ffn[1]; ffn],
               ROCVector{Float32}[nm..., dev(f32(g, c * "conv.weight")), dev(f32(g, c * "conv.weight"))])
    end
end
struct Model
    L::Vector{DLayer}; head::DMat; onrm::ROCVector{Float32}
    kc::Vector{ROCVector{UInt16}}; vc::Vector{ROCVector{UInt16}}; ring::Vector{ROCVector{UInt16}}
end
function load_engine(dir)
    cfg = jparse(read(joinpath(dir, "config.json"), String))
    (cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"], cfg["num_hidden_layers"], cfg["head_dim"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["conv_L_cache"], cfg["sliding_window"]) ==
        (NE, NFF, NV, NL, HD, NH, NKV, 7, WIN) || error("config.json: not the model this kernel is written for")
    for (l, t) in enumerate(cfg["layer_types"])
        t == (isfull(l - 1) ? "full_attention" : isattn(l - 1) ? "sliding_attention" : "conv") || error("layer $(l - 1) is $t")
    end
    t0 = time(); g = sft(joinpath(dir, "model.safetensors"))
    L = DLayer[load_layer(g, l) for l in 0:NL-1]
    head = DMat(g, "model.embed_tokens.weight"; ns = 1)
    onrm = dev(f32(g, "model.embedding_norm.weight"))
    # (the caches are zeroed once: a score of 0 times a stale value must not meet a NaN pattern)
    na = count(isattn, 0:NL-1); kc = ROCVector{UInt16}[]; vc = ROCVector{UInt16}[]; ring = ROCVector{UInt16}[]
    for l in 0:NL-1
        if isattn(l)
            n = isfull(l) ? MAXCTX + 32 : NR * RING
            push!(kc, AMDGPU.zeros(UInt16, n * NKV * HD)); push!(vc, AMDGPU.zeros(UInt16, n * NKV * HD))
        else
            push!(ring, AMDGPU.zeros(UInt16, NR * NE * CRING))
        end
    end
    @printf("weights resident (%.0f s)\n", time() - t0)
    Model(L, head, onrm, kc, vc, ring), g
end
addr(a) = reinterpret(Int64, UInt64(pointer(a)))
# the table: every matrix's address and shape, every vector's address (the weights stay where they are)
function ktable(M::Model)
    tab = zeros(Int64, TABV + (NL + 1) * 8)
    m!(l, j, d) = (e = (l * 8 + j) * 4; tab[e+1:e+4] = [addr(d.W), d.rows, d.K ÷ 8, d.ns])
    v!(l, j, a) = (tab[TABV+l*8+j+1] = addr(a))
    for l in 0:NL-1
        W = M.L[l+1]
        for j in 0:7; m!(l, j, W.M[j+1]); end
        for j in 0:5; v!(l, j, W.V[j+1]); end
        if isattn(l)
            v!(l, 6, M.kc[aidx(l)+1]); v!(l, 7, M.vc[aidx(l)+1])
        else
            v!(l, 6, M.ring[cidx(l)+1])
        end
    end
    m!(NL, 0, M.head); v!(NL, 0, M.onrm)
    dev(tab)
end
rftab() = Float32[i < (t == 0 ? 32 : 64) ? 1.0f0 / (Float32(t == 0 ? 1.0e6 : 1.0e4)^(Float32(2i) / Float32(HD))) : 0.0f0 for t in 0:1 for i in 0:63]
# page-locked host memory the kernel writes: the counters (at out) and the generated tokens (at tok)
struct HostX
    c::Ptr{UInt8}; out::Int; tok::Int
end
function HostX(maxgen)
    out = 0; tok = 128; rc = Ref{Ptr{Cvoid}}(C_NULL); H = AMDGPU.HIP   # (host words: 13 end the dispatch, 16 .. a slot's prompt position)
    H.hipHostMalloc(rc, tok + 4NR * maxgen, H.hipHostMallocPortable | H.hipHostMallocMapped | H.hipHostMallocCoherent)
    HostX(Ptr{UInt8}(rc[]), out, tok)
end
lp(T, p::Ptr) = reinterpret(LP{T}, p)
lp(T, a::ROCArray) = reinterpret(LP{T}, pointer(a))
if !isdefined(Main, :KWPE)
    const KWPE = Ref(0)                                    # waves a SIMD the kernel is compiled for (0: the compiler's choice)
end
# (KernelAbstractions' hook for a kernel's compile options: the static workgroup's maxthreads, and the wave count)
KernelAbstractions.compiler_options(obj::KernelAbstractions.Kernel{ROCBackend,<:Any,<:Any,typeof(gpu_k_req!)}) = (; maxthreads = Int(WGS), waves_per_eu = KWPE[])
struct KBuf
    tab::ROCVector{Int64}; x::ROCVector{Float32}; o::ROCVector{Float32}; xt::ROCVector{Float32}; part::ROCVector{Float32}
    qh::ROCVector{Float32}; rc::ROCVector{Float32}; rf::ROCVector{Float32}; gh::ROCVector{Float32}; et::ROCVector{Float32}; pv::ROCVector{Float32}
    eb::ROCVector{Float32}; es::ROCVector{Float32}
    bar::ROCVector{UInt32}; ssq::ROCVector{Int64}; best::ROCVector{UInt64}
    sl::ROCVector{Float32}; hs::ROCVector{UInt32}; th::ROCVector{Float32}; cn::ROCVector{UInt32}; cd::ROCVector{Int32}; rng::ROCVector{UInt64}; seen::ROCVector{UInt32}
    sp::ROCVector{Float32}; si::ROCVector{Int32}; apos::ROCVector{Int32}; lim::ROCVector{Int32}
    rq::ROCVector{Int32}; tok::ROCVector{Int32}; posr::ROCVector{Int32}; posa::ROCVector{Int32}; tks::ROCVector{Int32}; nem::ROCVector{Int32}
end
function KBuf(M::Model, npos)
    z(n) = AMDGPU.zeros(Float32, n); zi(n) = AMDGPU.zeros(Int32, n)
    KBuf(ktable(M), z(NE * npos), z(NE * npos), z(NFF * npos),
        z(max(6144 * 4, 3104 * 4, 2 * 7680 * 4, 2048 * 15) * npos),
        z(npos * NKV * HD * 4), z(npos * 2 * 64 * 2), dev(rftab()), z(npos * NH), z(npos * NH),
        z(npos * NKV * PVS * HD * 4), z(npos * NKV * (ERW ÷ 2) * 4), z(npos * NKV * EGW * 4),
        AMDGPU.zeros(UInt32, 64), AMDGPU.zeros(Int64, 2 * BSSQ), AMDGPU.zeros(UInt64, npos),
        z(NR * NV), AMDGPU.zeros(UInt32, NR * HBINS), z(NR), AMDGPU.zeros(UInt32, NR), zi(NR * SCAND), AMDGPU.zeros(UInt64, NR), AMDGPU.zeros(UInt32, NR * SEENW),
        z(4NR), zi(2NR), zi(NR), zi(NR),
        zi(5NR), zi(npos), zi(npos), zi(npos), zi(NR), zi(NR))
end
struct PEngine
    M::Model; K::KBuf; hx::HostX; nwg::Int; npos::Int; maxgen::Int; hz::Float64; rings::Vector{ROCVector{UInt16}}
end
# the clock's rate, from two spins of different lengths against the host clock
@kernel unsafe_indices = true function k_spin!(o, n)
    t0 = memtime()
    while memtime() - t0 < n; end
    o[1] = memtime() - t0
end
function clockhz()
    o = AMDGPU.zeros(UInt64, 1); k = k_spin!(ROCBackend(), 1)
    k(o, UInt64(1000); ndrange = 1); AMDGPU.synchronize()
    function t(n)
        t0 = time_ns(); k(o, UInt64(n); ndrange = 1); AMDGPU.synchronize()
        (time_ns() - t0) / 1e9, Array(o)[1]
    end
    function rate()
        t1, n1 = t(2_000_000); t2, n2 = t(20_000_000)
        (n2 - n1) / (t2 - t1)
    end
    sort([rate() for _ in 1:5])[3]
end
# how many workgroups of the kernel are resident at once, from the compiled kernel's register use
function occupancy(k, A)
    b = ROCBackend(); nd, wg, it, _ = KernelAbstractions.launch_config(k, Int(WGS), nothing)
    launch = KernelAbstractions.select_launch(KernelAbstractions.sizes_only(k), wg, it)
    ctx = KernelAbstractions.mkcontext(k, nd, it, launch; subgroups = KernelAbstractions.kernel_subgroups(k))
    _, tt = KernelAbstractions.prepare_arguments(b, k.f, ctx, (A,))
    kf = KI.kernel_function(b, k.f, tt; KernelAbstractions.compiler_options(k)...)
    nb = Ref{Cint}(0); AMDGPU.HIP.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(nb, kf.kern.kernel.fun, WGS, 0)
    Int(nb[]) * KI.multiprocessor_count(b)
end
function kargs(K::KBuf, hx::HostX, acct, inp, lg, nreq, mode, nwg, npos; smp = false, pen = false)
    KR(lp(Int64, K.tab), lp(Float32, K.x), lp(Float32, K.o), lp(Float32, K.xt), lp(Float32, K.part),
       lp(Float32, K.qh), lp(Float32, K.rc), lp(Float32, K.rf), lp(Float32, K.gh), lp(Float32, K.et), lp(Float32, K.pv),
       lp(Float32, K.eb), lp(Float32, K.es), lg === nothing ? lp(Float32, K.x) : lp(Float32, lg),
       lp(UInt32, K.bar), lp(UInt64, acct), lp(Int64, K.ssq), lp(UInt64, K.best),
       lp(Int32, inp), lp(Int32, K.rq), lp(Int32, K.tok), lp(Int32, K.posr), lp(Int32, K.posa), lp(Int32, K.tks), lp(Int32, K.nem),
       lp(Int32, hx.c + hx.tok), lp(UInt32, hx.c + hx.out),
       Int32(nreq), Int32(mode), Int32(nwg), Int32(npos),
       smp ? 1.0f0 : 0.0f0, Int32(DYNCLAIM[] ? 1 : 0),
       lp(Float32, K.sl), lp(UInt32, K.hs), lp(Float32, K.th), lp(UInt32, K.cn), lp(Int32, K.cd), lp(UInt64, K.rng),
       pen ? 1.1f0 : 1.0f0, lp(UInt32, K.seen), lp(Float32, K.sp), lp(Int32, K.si), lp(Int32, K.apos), lp(Int32, K.lim))
end
# The card also runs the desktop: a kernel holding every wave slot starves the graphics ring until its timeout resets it
# (gfx_0.0.0 timeout, GPU Hang, at 108 of 108 workgroups). The cap leaves the graphics shaders their room.
if !isdefined(Main, :NWGMAX)
    const NWGMAX = Ref(27)                                 # workgroups the kernel may use (nwg N)
end
function PEngine(M::Model; maxgen = MAXGEN, npos = NPOS)
    K = KBuf(M, npos); hx = HostX(maxgen)
    A = kargs(K, hx, AMDGPU.zeros(UInt64, 4), AMDGPU.zeros(Int32, 1), nothing, 0, 0, 1, npos)
    k = k_req!(ROCBackend(), Int(WGS))
    nwg = occupancy(k, A); NWGMAX[] > 0 && (nwg = min(nwg, NWGMAX[]))
    PEngine(M, K, hx, nwg, npos, maxgen, clockhz(), M.ring)
end
# ---- the slots ----
# Each slot holds a request, or between requests its conversation: the arena rows of a region (in units of a quarter of the
# arena) and, at a snapshot position flashd names, a copy of what the rings hold, so that the conversation's next turn runs
# only its new positions. A request takes the slot whose snapshot starts its prompt; failing that, an idle slot, the one
# longest unused; a long one takes several units, dropping the conversations there that aren't running.
const UNIT = MAXCTX ÷ NR
mutable struct Rq
    ids::Vector{Int}; nnew::Int; stops::NTuple{2,Int}; sample::NTuple{3,Float64}; seed::UInt64; penalty::Float64; snap::Int
    slot::Int; resumed::Int; snapped::Int; cancelled::Bool; done::Bool; err::String
end
Rq(ids, nnew; stops = (-1, -1), sample = (0.0, 0.0, 1.0), seed = UInt64(1), penalty = 1.0, snap = 0) =
    Rq(ids, nnew, stops, Float64.(sample), seed, penalty, snap, -1, 0, 0, false, false, "")
mutable struct Slot
    unit::Int; units::Int; cached::Vector{Int}; used::Float64; rq::Union{Nothing,Rq}
end
if !isdefined(Main, :SLOTS)
    const SLOTS = Ref{Vector{Slot}}(Slot[])
    const PENDING = Rq[]; const SLK = ReentrantLock()      # requests waiting for a slot
    const GLK = ReentrantLock()                            # the GPU: a dispatch, or a command that needs it alone
    const SCHEDSTOP = Ref(false); const SCHEDTASK = Ref{Any}(nothing)
    const SNAPB = Ref{Any}(nothing)                        # the rings' copies, allocated once
    const LASTRES = Ref((0.0, 0.0, 1.0))                   # the last dispatch's (instruction, idle, wall) seconds
end
hword(PE::PEngine, i) = Ptr{UInt32}(PE.hx.c + PE.hx.out + 4i)
yield!(PE::PEngine) = unsafe_store!(hword(PE, 13), UInt32(1))
# what a slot's rings hold, in each ring array: (the live array, its copy, first element, elements)
function ringspans(PE::PEngine, r)
    M = PE.M; sl = [aidx(l) + 1 for l in 0:NL-1 if isattn(l) && !isfull(l)]
    live = [M.ring; [M.kc[i] for i in sl]; [M.vc[i] for i in sl]]
    SNAPB[] === nothing && (SNAPB[] = [similar(a) for a in live])
    nc = length(M.ring); ns = length(sl); s = Tuple{Any,Any,Int,Int}[]
    for (i, a) in enumerate(live)
        b = SNAPB[][i]
        if i <= nc
            push!(s, (a, b, r * NE * CRING, NE * CRING))           # [slot][channel][ring position]
        elseif i <= nc + ns
            push!(s, (a, b, r * (RING ÷ 32) * 64 * CHUNK ÷ 2, (RING ÷ 32) * 64 * CHUNK ÷ 2))   # keys: [row group][head][chunk][lane]
        else
            push!(s, (a, b, r * (RING ÷ 8) * 16 * CHUNK ÷ 2, (RING ÷ 8) * 16 * CHUNK ÷ 2))     # values: [position chunk][head, group][lane]
        end
    end
    s
end
function ringcopy!(PE::PEngine, r, save::Bool)
    for (a, b, o, n) in ringspans(PE, r)
        save ? copyto!(b, o + 1, a, o + 1, n) : copyto!(a, o + 1, b, o + 1, n)
    end
    AMDGPU.synchronize()
end
function ringzero!(PE::PEngine, r)
    z = zeros(UInt16, NE * CRING)
    GC.@preserve z begin
        for rg in PE.M.ring; copyto!(rg, r * NE * CRING + 1, z, 1, length(z)); end
        AMDGPU.synchronize()
    end
end
# k free units in a row for slot i: the units of idle conversations are given up, oldest first, only when it takes that
function carve!(i, k)
    S = SLOTS[]; held(u) = any(s -> s !== S[i] && s.units > 0 && s.unit <= u < s.unit + s.units, S)
    busy(u) = any(s -> s !== S[i] && s.rq !== nothing && s.unit <= u < s.unit + s.units, S)
    starts = [u for u in 0:NR-k if !any(busy, u:u+k-1)]
    isempty(starts) && return false
    cost(u) = sum(sum(s.used for s in S if s !== S[i] && s.units > 0 && s.unit <= v < s.unit + s.units; init = 0.0) for v in u:u+k-1)
    u = starts[argmin([cost(u) for u in starts])]
    for s in S
        (s !== S[i] && s.units > 0 && s.unit < u + k && u < s.unit + s.units) && (s.units = 0; s.cached = Int[])
    end
    S[i].unit = u; S[i].units = k; true
end
function place!(PE::PEngine, q::Rq, H)
    S = SLOTS[]; n = length(q.ids); need = cld(n + q.nnew, 32) * 32; i = 0; start = 0
    for (j, s) in enumerate(S)
        c = length(s.cached)
        if s.rq === nothing && c > start && c < n && s.units * UNIT >= need && view(q.ids, 1:c) == s.cached
            i = j; start = c
        end
    end
    r = i - 1
    if i > 0
        ringcopy!(PE, r, false)
    else
        idle = [j for j in eachindex(S) if S[j].rq === nothing]
        isempty(idle) && return false
        sort!(idle; by = j -> (isempty(S[j].cached) ? 0 : 1, S[j].used)); i = idle[1]; r = i - 1; k = cld(need, UNIT)
        k > NR && (q.err = "the prompt and its tokens take $need positions: the cache holds $MAXCTX"; q.done = true; return true)
        if S[i].units < k || S[i].units > k
            carve!(i, k) || return false
        end
        S[i].cached = Int[]; ringzero!(PE, r)
    end
    s = S[i]; s.rq = q; s.used = time(); q.resumed = start
    H.apos[i] = start; H.lim[i] = q.snap > start && q.snap < n ? q.snap : n; H.nem[i] = 0
    H.sp[4r+1:4r+4] = Float32[q.sample[1], q.sample[2], q.sample[3], q.penalty]; H.si[2r+1:2r+2] = Int32[q.stops[1], q.stops[2]]
    x = (q.seed + 0x9e3779b97f4a7c15) * 0xbf58476d1ce4e5b9; H.rng[i] = (x ⊻ (x >> 31)) | 1
    if q.penalty != 1.0                                    # the bits of what the slot has seen so far: its prompt
        sb = zeros(UInt32, SEENW)
        for t in q.ids; sb[t>>5+1] |= UInt32(1) << (t & 31); end
        GC.@preserve sb begin copyto!(PE.K.seen, r * SEENW + 1, sb, 1, SEENW); AMDGPU.synchronize() end
    end
    outp = Ptr{Int32}(PE.hx.c + PE.hx.tok) + 4r * PE.maxgen
    for j in 1:q.nnew; unsafe_store!(outp, Int32(-1), j); end
    unsafe_store!(hword(PE, 16 + r), UInt32(start))
    q.slot = r                                             # (last: the chat's task starts reading the slot's tokens)
    true
end
# One dispatch over the slots as they stand. Returns the prefill and decode seconds.
function dispatch!(PE::PEngine, H, mode; lg = nothing)
    K = PE.K; S = SLOTS[]; rq = zeros(Int32, 5NR); inp = Int32[]; npre = 0; rem = 0; smp = false; pen = false
    for (i, s) in enumerate(S)
        r = i - 1; q = s.rq
        if q === nothing
            H.apos[i] = 0; H.lim[i] = 0; H.nem[i] = 0; continue
        end
        rq[5r+1:5r+5] = Int32[length(inp), length(q.ids), q.nnew, r * PE.maxgen, s.unit * UNIT]
        append!(inp, Int32.(q.ids))
        npre += (H.lim[i] - H.apos[i]) ÷ PE.npos + 5; rem = max(rem, q.nnew - H.nem[i])
        smp |= q.sample[1] > 0; pen |= q.penalty != 1.0
    end
    nphase = (npre + rem + 3) * 340
    DPMLOCK[] && dpm!("high")                              # the clocks up for the dispatch
    a = AMDGPU.zeros(UInt64, 3 + 2nphase); ah = zeros(UInt64, 3 + 2nphase); ah[1] = typemax(UInt64); ah[4:2:end] .= typemax(UInt64); up!(a, ah)
    up!(K.bar, [UInt32[0, 0, 0, NLRUN[]]; zeros(UInt32, 60)]); up!(K.rq, rq); up!(K.nem, H.nem); up!(K.apos, H.apos); up!(K.lim, H.lim)
    up!(K.sp, H.sp); up!(K.si, H.si); up!(K.rng, H.rng)
    ib = dev(isempty(inp) ? Int32[0] : inp)
    A = kargs(K, PE.hx, a, ib, lg, NR, mode, PE.nwg, PE.npos; smp, pen)
    k_req!(ROCBackend(), Int(WGS))(A; ndrange = Int(WGS) * PE.nwg)
    AMDGPU.synchronize()
    hz = PE.hz; ah = Array(a); wall = (ah[2] - ah[1]) / hz; iv = Tuple{Float64,Float64}[]
    ah[3] == 0 && (ah[3] = ah[2])
    LASTACCT[] = ah
    for s in 0:nphase-1
        b = ah[4+2s]; e = ah[5+2s]; b == typemax(UInt64) && continue
        push!(iv, ((b - ah[1]) / hz, (e - ah[1]) / hz))
    end
    ins = union_len(iv); LASTRES[] = (ins, wall - ins, wall)
    AMDGPU.unsafe_free!(a); AMDGPU.unsafe_free!(ib)
    DPMLOCK[] && dpm!("auto")                              # and back down while the model waits
    copyto!(H.apos, Array(K.apos)); copyto!(H.nem, Array(K.nem)); copyto!(H.rng, Array(K.rng))
    ab = Array(K.bar)[3]
    ab != 0 && error("persistent kernel aborted (a wait passed 10 s): $ab")
    (ah[3] - ah[1]) / hz, (ah[2] - ah[3]) / hz
end
# after a dispatch: a prompt at its snapshot gets its rings copied and goes on; a finished or cancelled request leaves its slot
function settle!(PE::PEngine, H)
    S = SLOTS[]
    for (i, s) in enumerate(S)
        q = s.rq; q === nothing && continue
        r = i - 1
        if !q.cancelled && H.apos[i] == H.lim[i] && H.lim[i] < length(q.ids)
            ringcopy!(PE, r, true); s.cached = q.ids[1:H.lim[i]]; q.snapped = H.lim[i]; H.lim[i] = length(q.ids)
        end
        if q.cancelled || (H.apos[i] >= length(q.ids) && H.nem[i] >= q.nnew)
            q.done = true; s.rq = nothing; s.used = time()
        end
    end
end
struct HostState
    apos::Vector{Int32}; lim::Vector{Int32}; nem::Vector{Int32}; sp::Vector{Float32}; si::Vector{Int32}; rng::Vector{UInt64}
end
HostState() = HostState(zeros(Int32, NR), zeros(Int32, NR), zeros(Int32, NR), zeros(Float32, 4NR), fill(Int32(-1), 2NR), ones(UInt64, NR))
# The scheduler: what is waiting joins between dispatches, and a dispatch is told to end (word 13) when something is waiting.
function sched_loop(PE::PEngine, H)
    while !SCHEDSTOP[]
        worked = false
        lock(GLK) do
            lock(SLK) do
                keep = Rq[]
                for q in PENDING
                    q.cancelled && (q.done = true; continue)
                    place!(PE, q, H) || push!(keep, q)
                end
                empty!(PENDING); append!(PENDING, keep)
            end
            for (i, s) in enumerate(SLOTS[])                # (cancelled while no dispatch ran)
                s.rq !== nothing && s.rq.cancelled && settle!(PE, H)
            end
            if any(s -> s.rq !== nothing, SLOTS[])
                unsafe_store!(hword(PE, 13), UInt32(0))
                lock(SLK) do
                    # (a request still waiting for a slot can't join until a request ends: no need to stop for it)
                    any(s -> s.rq === nothing, SLOTS[]) && !isempty(PENDING) && yield!(PE)
                end
                try
                    dispatch!(PE, H, 0)
                    settle!(PE, H)
                catch e
                    msg = sprint(showerror, e); println("dispatch: ", msg); flush(stdout)
                    for s in SLOTS[]
                        if s.rq !== nothing
                            s.rq.err = msg; s.rq.done = true; s.rq = nothing; s.units = 0; s.cached = Int[]
                        end
                    end
                end
                worked = true
            end
        end
        worked || sleep(0.002)
    end
end
# Up to NR requests at once, alone on the GPU (the commands gen, gen2, kl and the tests' scoring). mode 0: greedy; mode 1: score
# one request's input (the logits of every position into lg). Returns each request's tokens, (instruction, idle, wall) seconds,
# and the prefill and decode seconds. Every slot's conversation is dropped.
function request!(PE::PEngine, reqs, mode::Int; lg = nothing, stops = (-1, -1), sample = (0.0, 0.0, 1.0), seed = UInt64(1), penalty = 1.0)
    nr = length(reqs); 1 <= nr <= NR || error("$nr requests: at most $NR at once")
    b = sum(cld(length(i) + g, 32) * 32 for (i, g) in reqs); b <= MAXCTX || error("requests of $b positions: the cache holds $MAXCTX")
    lock(GLK) do
        any(s -> s.rq !== nothing, SLOTS[]) && error("chats are running")
        H = HostState(); unsafe_store!(hword(PE, 13), UInt32(0)); o = 0; S = SLOTS[]
        for s in S; s.units = 0; s.cached = Int[]; end
        rbs = Int[]
        qs = map(enumerate(reqs)) do (j, (ids, g))
            isempty(ids) && error("a request with no input tokens")
            g <= PE.maxgen || error("$g tokens: the output holds $(PE.maxgen)")
            q = Rq(ids, g; stops, sample, seed, penalty); r = j - 1; s = S[j]
            ringzero!(PE, r); s.rq = q; q.slot = r
            H.lim[j] = length(ids); H.sp[4r+1:4r+4] = Float32[sample[1], sample[2], sample[3], penalty]; H.si[2r+1:2r+2] = Int32[stops[1], stops[2]]
            x = (seed + 0x9e3779b97f4a7c15) * 0xbf58476d1ce4e5b9; H.rng[j] = (x ⊻ (x >> 31)) | 1
            if penalty != 1.0
                sb = zeros(UInt32, SEENW); for t in ids; sb[t>>5+1] |= UInt32(1) << (t & 31); end
                GC.@preserve sb begin copyto!(PE.K.seen, r * SEENW + 1, sb, 1, SEENW); AMDGPU.synchronize() end
            end
            q
        end
        # (the regions here are by position, not whole units: a request may be shorter than a unit and these run alone)
        tp, td = dispatch_at!(PE, H, mode, reqs; lg)
        toks = [[Int(unsafe_load(Ptr{Int32}(PE.hx.c + PE.hx.tok) + 4(r - 1) * PE.maxgen, i)) for i in 1:reqs[r][2]] for r in 1:nr]
        for s in S; s.rq = nothing; end
        LASTREQ[] = (reqs, mode, 0, sample[1] > 0)
        toks, LASTRES[], tp, td
    end
end
function dispatch_at!(PE, H, mode, reqs; lg)
    S = SLOTS[]; b = 0; tp = 0.0; td = 0.0
    for (j, (ids, g)) in enumerate(reqs)
        S[j].unit = b ÷ UNIT; b += cld(length(ids) + g, UNIT) * UNIT
    end
    b <= MAXCTX || error("requests take more than the arena's $NR units")
    p, d = dispatch!(PE, H, mode; lg); (tp + p, td + d)
end
function union_len(iv)                                     # total length of a set of intervals
    isempty(iv) && return 0.0
    sort!(iv); t = 0.0; a, b = iv[1]
    for (x, y) in iv
        if x > b
            t += b - a; a = x; b = y
        else
            b = max(b, y)
        end
    end
    t + b - a
end
if !isdefined(Main, :LASTACCT)
    const LASTACCT = Ref{Vector{UInt64}}(UInt64[])        # the phase times of the last request, for profiling
    const LASTREQ = Ref{Any}(nothing)                     # the last request's (reqs, mode)
end
# the card's performance level (sysfs power_dpm_force_performance_level of its PCI device): forced high for a request's
# dispatch, auto between requests; where the file isn't writable the level stays as it is
dpmfile() = (b = AMDGPU.HIP.properties(AMDGPU.device()).pciBusID; "/sys/bus/pci/devices/0000:" * string(b, base = 16, pad = 2) * ":00.0/power_dpm_force_performance_level")
dpm!(lv) = try write(dpmfile(), lv) catch end
if !isdefined(Main, :DPMLOCK)
    const DPMLOCK = Ref(true)                              # dpm on | off
end
if !isdefined(Main, :DYNCLAIM)
    const DYNCLAIM = Ref(true)                             # dyn 0 | 1
end
if !isdefined(Main, :NLRUN)
    const NLRUN = Ref(0)                                   # layers the kernel runs (0: all), for debugging
end
idlepct(res) = 100res[2] / res[3]

# ---- requests ----
# greedy generation for one or two requests at once (each: its prompt ids and n new tokens)
function gen(io, PE::PEngine, vocab, reqs)
    toks, res, tp, td = request!(PE, reqs, 0)
    for t in toks; println(io, "text: ", repr(detok(vocab, t))); println(io, "ids: ", join(t, ' ')); end
    @printf(io, "prefill %.2f tok/s\ndecode %.2f tok/s\nidle time %.1f%%\n", sum(length(i) for (i, _) in reqs) / tp, sum(max(n - 1, 0) for (_, n) in reqs) / td, idlepct(res))
end
# KL against BF16 reference logits ([position][vocab]) over the first n tokens of the reference's sequence
function kl(io, PE::PEngine, ids, refpath, n)
    lg = AMDGPU.zeros(Float32, n * NV)
    _, res, tp, _ = request!(PE, [(ids[1:n], 0)], 1; lg)
    Lg = reshape(Array(lg), NV, n); AMDGPU.unsafe_free!(lg)
    ref = Mmap.mmap(refpath, Matrix{UInt16}, (NV, n)); kls = Float64[]; top = 0
    for p in 1:n
        r = Float64.(bf32.(view(ref, :, p))); o = Float64.(view(Lg, :, p)); (all(isfinite, r) && all(isfinite, o)) || (push!(kls, Inf); continue)
        lr = r .- (maximum(r) + log(sum(exp.(r .- maximum(r))))); lo = o .- (maximum(o) + log(sum(exp.(o .- maximum(o)))))
        push!(kls, sum(exp.(lr) .* (lr .- lo))); top += argmax(r) == argmax(o)
    end
    m = length(kls)
    @printf(io, "KL vs BF16 over %d positions: mean %.6f, median %.6f, max %.5f; top-1 %.4f\nprefill %.2f tok/s\nidle time %.1f%%\n",
        m, sum(kls) / m, sort(kls)[cld(m, 2)], maximum(kls), top / m, n / tp, idlepct(res))
end

# the last request's phases: the milliseconds each step of a forward took, added over the forwards, and the gaps
function prof(io, PE::PEngine)
    ah = LASTACCT[]; hz = PE.hz; tot = Dict{String,Float64}(); gap = 0.0; tl = ah[1]
    kn = ["pre", "mat", "conv", "ssq", "norm", "act", "aprep", "score", "vmix", "acomb", "head", "tok"]
    seqk = String[]
    function fw!(head, gen)                                # a forward's steps, as fwd! takes them
        push!(seqk, "emb")
        for l in 0:NL-1, i in 0:(isattn(l) ? 14 : 11)
            k = lstep(Int32(l), Int32(i)); push!(seqk, k[1] == S_MAT ? "mat$(k[2])" : kn[k[1]])
        end
        head && (push!(seqk, "pre"); push!(seqk, "head"); gen && (smp ? append!(seqk, ["hist", "thr", "cand", "stok"]) : push!(seqk, "tok")))
    end
    reqs, mode, start, smp = LASTREQ[]
    for (inp, ngen) in reqs
        a = start; n = length(inp)
        while a < n
            P = n - a >= PE.npos ? PE.npos : n - a >= 8 ? 8 : n - a >= 4 ? 4 : n - a >= 2 ? 2 : 1
            fw!(mode == 1 || (a + P == n && ngen > 0), mode == 0); a += P
        end
    end
    npre = length(seqk); left = [g - 1 for (_, g) in reqs]
    while any(>(0), left)
        fw!(true, true); left .-= 1
    end
    for s in 0:length(seqk)-1
        b = ah[4+2s]; e = ah[5+2s]; b == typemax(UInt64) && continue
        nm = (s < npre ? "p:" : "d:") * seqk[s+1]; tot[nm] = get(tot, nm, 0.0) + (e - b) / hz; b > tl && (gap += (b - tl) / hz); tl = max(tl, e)
    end
    for (k, v) in sort(collect(tot); by = x -> x[2], rev = true); @printf(io, "%-8s %9.2f ms\n", k, 1e3v); end
    @printf(io, "gaps     %9.2f ms\nwall     %9.2f ms\n", 1e3gap, 1e3(ah[2] - ah[1]) / hz)
end

# ---- byte-level BPE detokenizing (GPT-2 byte map) ----
const BDEC = let bs = [33:126; 161:172; 174:255], cs = copy(bs), n = 0
    for b in 0:255
        b in bs || (push!(bs, b); push!(cs, 256 + n); n += 1)
    end
    Dict(Char(c) => UInt8(b) for (b, c) in zip(bs, cs))
end
detok(vocab, ids) = String(UInt8[get(BDEC, ch, UInt8('?')) for i in ids for ch in vocab[i+1]])
function vocabulary(dir)
    t = jparse(read(joinpath(dir, "tokenizer.json"), String)); v = fill("", NV)
    for (s, i) in t["model"]["vocab"]; v[i+1] = s; end
    for a in t["added_tokens"]; v[a["id"]+1] = a["content"]; end
    v
end

# ---- the resident server: the weights load once, the persistent kernel compiles once; one command a connection ----
#   gen N ID ID ...            greedy: N tokens after the given token ids
#   gen2 N1 N2 F.ids L1 L2     two requests at once: L1 then L2 prompt ids from F
#   kl N REF.logits.bf16       KL against BF16 reference logits over the first N tokens of REF's sequence
#   reload                     include this file again and recompile the kernel (the weights stay resident)
#   eval CODE                  run Julia code in the server, print its value
#   echo 'gen 32 9707 11' | socat - UNIX-CONNECT:SOCK
if !isdefined(Main, :SRV)
    const SRV = Ref{Any}(nothing)
end
if !isdefined(Main, :PK)
    const PK = Ref{Any}(nothing)                           # the persistent kernel engine
end
function pinit()
    t0 = time()
    if SCHEDTASK[] !== nothing                             # the scheduler of the engine being replaced
        SCHEDSTOP[] = true; wait(SCHEDTASK[]); SCHEDSTOP[] = false
    end
    PK[] === nothing || (foreach(a -> a isa ROCArray && AMDGPU.unsafe_free!(a), (getfield(PK[].K, f) for f in fieldnames(typeof(PK[].K)))); AMDGPU.HIP.hipHostFree(PK[].hx.c))
    PK[] = PEngine(SRV[].M)
    SLOTS[] = [Slot(0, 0, Int[], 0.0, nothing) for _ in 1:NR]; empty!(PENDING)
    SCHEDTASK[] = Threads.@spawn Base.invokelatest(sched_loop, PK[], HostState())
    # (the first dispatch compiles nothing more: the occupancy query compiled the same kernel)
    @printf("persistent kernel: %d workgroups, %d positions a prefill forward, clock %.2f MHz, compiled in %.0f s\n", PK[].nwg, PK[].npos, PK[].hz / 1e6, time() - t0); flush(stdout)
end
function serve(dir, sock)
    M, g = load_engine(dir)
    dpm!("auto")                                          # (a previous engine killed mid-request left it forced high)
    SRV[] = (M = M, g = g, vocab = vocabulary(dir), file = abspath(@__FILE__))
    pinit()
    lk = ReentrantLock()
    ispath(sock) && rm(sock)
    srv = listen(sock); println("serving on $sock"); flush(stdout)
    while true
        c = accept(srv)
        @async try
            Base.invokelatest(session, c, lk)              # the newest definitions, after a reload
        catch e
            println(c, "error: ", sprint(showerror, e, catch_backtrace()))
        finally
            close(c)
        end
    end
end
# flashd's engine protocol (flash-sass kern/flashd.c): "timings 1" and "cancel_on_eof 1" are acknowledged, "stop A B"
# sets the stop tokens, "sample ..." (greedy here), "img" and "snap" are taken and ignored, "info" answers the positions,
# "chat N ID ..." runs the request and streams "tok ID" lines as the kernel writes them to host memory, then
# "done ngen ended nprompt run_ms first_ms 0 0". Any other first line is one of handle's commands.
function session(c, lk)
    stops = (-1, -1); snap = 0; sample = (0.0, 0.0, 1.0); seed = UInt64(1); penalty = 1.0
    while !eof(c)
        line = readline(c); a = split(line); isempty(a) && continue
        if a[1] in ("timings", "cancel_on_eof")
            println(c, "ok: ", line)
        elseif a[1] == "stop"
            v = parse.(Int, a[2:end]); stops = (length(v) >= 1 ? v[1] : -1, length(v) >= 2 && v[2] != 0xffffffff ? v[2] : -1)
        elseif a[1] == "snap"
            snap = parse(Int, a[2])
        elseif a[1] == "sample"                            # sample SEED TEMP TOPK TOPP
            seed = parse(UInt64, a[2]); sample = (parse(Float64, a[3]), parse(Float64, a[4]), parse(Float64, a[5]))
        elseif a[1] == "penalty"                           # penalty P: repetition penalty (flashd sends it only when it isn't 1)
            penalty = parse(Float64, a[2])
        elseif a[1] == "img"
        elseif a[1] == "info"
            println(c, "info ", MAXCTX, " 0")
        elseif a[1] == "chat"
            chat(c, parse(Int, a[2]), parse.(Int, a[3:end]), stops, snap, sample, seed, penalty); return
        else
            lock(() -> handle(c, line), lk); return
        end
    end
end
function chat(c, nnew, ids, stops, snap = 0, sample = (0.0, 0.0, 1.0), seed = UInt64(1), penalty = 1.0)
    PE = PK[]; nnew = clamp(nnew, 1, min(PE.maxgen, MAXCTX - length(ids)))
    q = Rq(ids, nnew; stops, sample, seed, penalty, snap)
    t0 = time(); lock(() -> push!(PENDING, q), SLK); yield!(PE)
    n = 0; t1 = 0.0; ended = 0; pr = 0; tp = 0.0; gone = false
    # a write to a client that has gone fails: the request is cancelled
    cancel() = (q.cancelled = true; yield!(PE))
    say(a...) = gone || try
        println(c, a...)
    catch
        gone = true; cancel()
    end
    # cancel_on_eof: flashd closes its side when the client goes; the slot leaves at the next forward
    @async try
        eof(c) && !q.done && cancel()
    catch
        cancel()
    end
    while q.slot < 0 && !q.done; sleep(0.002); end
    isempty(q.err) || (say("error: ", q.err); return)
    r = q.slot; outp = Ptr{Int32}(PE.hx.c + PE.hx.tok) + 4r * PE.maxgen
    function progress()                                    # "prog ran total cached", at most one every 500 ms
        n == 0 || return
        ran = Int(unsafe_load(hword(PE, 16 + r))); now = time()
        (ran > pr && now - tp >= 0.5) && (say("prog ", ran, " ", length(ids), " ", q.resumed); pr = ran; tp = now)
    end
    pump() = while n < nnew && ended == 0
        t = unsafe_load(outp, n + 1); t < 0 && break
        n += 1; n == 1 && (t1 = time()); say("tok ", t); (t == stops[1] || t == stops[2]) && (ended = 1)
    end
    while !q.done && n < nnew && ended == 0
        progress(); pump(); gone || try flush(c) catch; gone = true; cancel() end; sleep(0.003)
    end
    pump(); t2 = time()
    isempty(q.err) || (say("error: ", q.err); return)
    gone || try
        @printf(c, "done %d %d %d %.1f %.1f %d %d\n", n, ended, length(ids), 1e3(t2 - t0), 1e3(t1 - t0), q.resumed, q.snapped); flush(c)
    catch
    end
    idle = LASTRES[]
    @printf("chat: %d prompt, %d out, %d cached, slot %d, prefill %.2f tok/s, decode %.2f tok/s, idle time %.1f%%%s\n", length(ids), n, q.resumed, r,
            (length(ids) - q.resumed) / max(t1 - t0, 1e-9), max(n - 1, 0) / max(t2 - t1, 1e-9), idlepct(idle), q.cancelled ? " (cancelled)" : ""); flush(stdout)
end
function handle(c, line)
    a = split(line); s = SRV[]
    if a[1] == "gen"                                      # gen N ID ID ...
        gen(c, PK[], s.vocab, [(parse.(Int, a[3:end]), parse(Int, a[2]))])
    elseif a[1] == "gen2"                                 # gen2 N1 N2 FILE.ids L1 L2: two requests at once, prompts from FILE
        ids = Int.(reinterpret(Int32, read(a[4]))); l1 = parse(Int, a[5]); l2 = parse(Int, a[6])
        gen(c, PK[], s.vocab, [(ids[1:l1], parse(Int, a[2])), (ids[l1+1:l1+l2], parse(Int, a[3]))])
    elseif a[1] == "kl"
        n = parse(Int, a[2]); ids = Int.(reinterpret(Int32, read(replace(a[3], ".logits.bf16" => ".ids"))))
        kl(c, PK[], ids, a[3], n)
    elseif a[1] == "dpm"                                 # dpm 0 | 1: lock the clocks high for each request
        DPMLOCK[] = a[2] == "1"; println(c, "dpm ", DPMLOCK[])
    elseif a[1] == "dyn"                                 # dyn 0 | 1: wavefronts claim matrix items, or a fixed share each
        DYNCLAIM[] = a[2] == "1"; println(c, "dyn ", DYNCLAIM[])
    elseif a[1] == "nwg"                                 # nwg N: the workgroups the kernel uses
        NWGMAX[] = parse(Int, a[2]); pinit(); println(c, "nwg ", PK[].nwg)
    elseif a[1] == "prof"                                # the last request's time by phase kind
        prof(c, PK[])
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
    (k === nothing || i === nothing) && error("usage: julia --project=env/amd -t 12 lfm3-1.7b.jl --model DIR --serve SOCK")
    serve(args[k+1], args[i+1])
end
if abspath(PROGRAM_FILE) == @__FILE__() && SRV[] === nothing
    main(ARGS)
end
