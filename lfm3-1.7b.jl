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
# (the clock is a message to the SPI, shared by a whole shader engine's waves: only the lane that writes the accounting reads it)
@inline wtime() = (AMDGPU.Device.readfirstlane((KI.get_local_id(Int32).x - Int32(1)) >> Int32(5)) == Int32(0) ? memtime() : UInt64(0))   # (uniform: a wavefront's branch)
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
    flow::Int32; part2::LP{Float32}; flg::LP{UInt32}       # dataflow (below): 0, 1, 2; the sums of a matrix step that overlaps its producer; a slice's x items done
    xt2::LP{Float32}; dc::LP{UInt32}                       # flow 2: x from conv, activation, mix; a forward's counters
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
            stmono(A.bar, UInt32(0)); stmono(A.bar + 128, UInt32(0))
            for f in Int32(0):Int32(15); stmono(A.flg + 4f, UInt32(0)); end
            atadd(A.bar + 4, UInt32(1))
        else
            n = Int64(0)                                   # (polls, not the clock; about 10 s at the sleep and a read)
            while ldmono(A.bar + 4) == g                   # (the fence below does the one invalidate the barrier needs)
                ((n += Int64(1)) > Int64(30_000_000) || ldmono(A.bar + 8) != UInt32(0)) && (atadd(A.bar + 8, UInt32(1)); break)
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
@inline stx2!(A::KR, v::Float32, k::Int32, P, p) = stb!(A.xt2, v, (Int64(k >> Int32(1)) * P + p) * 4 + (k & Int32(1)) * 2)
# ---- flow 2: no grid barrier inside a forward. Every item starts when what it reads is done, as the dependencies say:
#   a matrix's K slice (64 chunks, 32 rows) when its slice of x is written;
#   an element group after a matrix (conv, activation, sums of squares) when its row group's K slices are all done;
#   what reads a whole result (a norm after a sum of squares over every element; a head's sum; attention's keys and values)
#   when all of it is done.
# A forward's counters (dc), zeroed before it: each step's items done and items claimed, its x slices, a matrix's row groups.
const D_FLG = 680; const D_RG = D_FLG + 340 * 16; const D_N = D_RG + 100 * 512
@inline dcdone(A::KR, si) = A.dc + 4 * (2Int64(si))
@inline dcflg(A::KR, si) = A.dc + 4 * (D_FLG + 16Int64(si))
@inline dcrg(A::KR, mi) = A.dc + 4 * (D_RG + 512Int64(mi))
@inline function dn!(A::KR, si, lane::Int32)               # an item of step si is done (its stores out first)
    A.flow == Int32(2) || return
    fence_rel()
    lane == Int32(0) && atadd(dcdone(A, si), UInt32(1))
    return
end
@inline function waitn!(A::KR, p::LP{UInt32}, need::Int32) # a plain read until a counter reaches need
    rdf(reinterpret(Int32, ldmono(p))) >= need && return
    n = Int64(0)
    while rdf(reinterpret(Int32, ldmono(p))) < need
        (n += Int64(1)) > Int64(30_000_000) && (atadd(A.bar + 8, UInt32(1)); break)
        rdf(reinterpret(Int32, ldmono(A.bar + 8))) != Int32(0) && break
        ssleep()
    end
end
@inline stepin!() = (fence_acq(); dcinv())                 # what the barrier did for this wavefront: old lines of what it reads next go
@inline function zero_dc!(A::KR)                           # (before a forward's barrier: a lane a word)
    i = WGS * wgid() + tid()
    while i < Int32(D_N)
        stmono(A.dc + 4i, UInt32(0)); i += WGS * A.nwg
    end
end
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
@inline function claim(cl::LP{UInt32}, lane::Int32)
    v = Int32(0)
    lane == Int32(0) && (v = reinterpret(Int32, atadd(cl, UInt32(1))))
    rdf(v)
end
# (the matrix of the group j0 .. holding item g: found, its table entry, the item within it, the floats before its sums, the row groups before it)
@inline function matof(A::KR, l::Int32, j0::Int32, nm::Int32, g::Int32, P::Int32)
    base = Int32(0); o = Int64(0); m = Int32(0); bg = Int32(0)
    while true
        W, rows, nc, ns = mat(A, l, j0 + m); n = (rows >> Int32(5)) * ns
        g < base + n && return true, W, rows, nc, ns, g - base, o, bg
        base += n; o += Int64(ns) * P * rows; bg += rows >> Int32(5); m += Int32(1)
        m == nm && return false, W, rows, nc, ns, Int32(0), o, bg
    end
end
@inline function matn(A::KR, l::Int32, j0::Int32, nm::Int32)   # the group's items
    n = Int32(0)
    for m in Int32(0):nm-Int32(1)
        _, rows, _, ns = mat(A, l, j0 + m); n += (rows >> Int32(5)) * ns
    end
    n
end
# Dataflow: a matrix step doesn't wait at a grid barrier for the step that makes its x. Each item of that step adds to its K
# slice's counter (the barrier before it zeroed them) and a wavefront waits only for the slices of its items, so a matrix
# starts on a slice as soon as that slice is ready. A step whose sums overlap a producer still reading sums (conv, activation)
# writes the second buffer. In flow 1 steps that need a whole result (a norm's sum of squares, what follows a matrix) keep the
# barrier; in flow 2 (above) they wait on that step's count, and only the tail of the head and a forward's end have one.
@inline function xdone!(A::KR, xb::LP{UInt32}, sl::Int32, lane::Int32)
    A.flow == Int32(0) && return
    fence_rel()
    lane == Int32(0) && atadd(xb + 4sl, UInt32(1))
    return
end
# (a plain read: an acquire load invalidates the L0 and L1 caches at every poll, under the other wavefronts. No invalidate after
# it either: x wasn't read since the barrier before its producer invalidated, so no cache holds an old line of it. A slice is
# a whole number of cache lines, so reading a ready one brings in none of a slice not ready.)
@inline xready!(A::KR, xb::LP{UInt32}, sl::Int32, need::Int32) = waitn!(A, xb + 4sl, need)   # (polls, not the clock: see wtime)
@generated function withpart2(A::KR)
    f = [:(A.$(n == :part ? :part2 : n == :xt ? :xt2 : n)) for n in fieldnames(KR)]
    quote
        $(Expr(:meta, :inline))
        KR($(f...))
    end
end
@inline function mat_ph!(A::KR, l::Int32, j0::Int32, nm::Int32, seq::Int32, P::Int32, need::Int32, xb::LP{UInt32}, cl::LP{UInt32}, si::Int32, mi::Int32)
    t0 = wtime(); did = false; lane = lanei(); gw = waveid(); NW = nwv(A); xp = fresh(A.xt); dyn = A.dyn != Int32(0); okm = UInt32(0); flow = A.flow != Int32(0); f2 = A.flow == Int32(2)
    g = dyn ? claim(cl, lane) : gw
    while true
        ok, W, rows, nc, ns, it, o, bg = matof(A, l, j0, nm, g, P)
        ok || break
        G = it ÷ ns; sl = it - G * ns
        if flow && (okm >> sl) & UInt32(1) == UInt32(0)    # (okm: the slices this wavefront has seen ready: no read for those again)
            xready!(A, xb, sl, need); okm |= UInt32(1) << sl; xp = fresh(A.xt)
            did || (t0 = wtime())
        end
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
        if f2                                              # its row group's slice, then the whole step's count
            fence_rel()
            lane == Int32(0) && (atadd(dcrg(A, mi) + 4(bg + G), UInt32(1)); atadd(dcdone(A, si), UInt32(1)))
        end
        did = true; g = dyn ? claim(cl, lane) : g + NW
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
    t0 = wtime(); lane = lanei(); gw = waveid(); NW = nwv(A)
    W, _, nc, _ = mat(A, HEADL, Int32(0)); n = Int32(64) * P
    g = gw
    while g < n
        p = g >> Int32(6); G = g & Int32(63); k = G * LANES + lane
        tok = rdf(ld(A.tok, p))
        o = (Int64(tok >> Int32(5)) * nc + (k >> Int32(3))) * CHUNK + Int64(tok & Int32(31)) * 16 + Int64(k & Int32(7)) * 2
        v = bf32(ld(reinterpret(LP{UInt16}, W + o), 0))
        st!(A.x, v, Int64(k) * P + p)
        sqadd!(A, 1, p, lane, v * v)
        dn!(A, Int32(0), lane)
        g += NW
    end
    # (the second pass of items: a wavefront a (position, table, 32 angles))
    g = itemat(gw, n, NW); n2 = n + Int32(4) * P
    while g < n2
        it = g - n; p = it >> Int32(2); tt = (it >> Int32(1)) & Int32(1); i = (it & Int32(1)) * LANES + lane
        a = Float32(rdf(ld(A.posa, p))) * ld(A.rf, tt * Int32(64) + i)
        st!(A.rc, bf(cos(a)), ((Int64(p) * 2 + tt) * 64 + i) * 2); st!(A.rc, bf(sin(a)), ((Int64(p) * 2 + tt) * 64 + i) * 2 + 1)
        dn!(A, Int32(0), lane)
        g += NW
    end
    done!(A, seq, t0, true)
end
# x normalized and weighted into xt, positions pf ..; reads bank 1, zeroes bank 0 and the head's argmax
@inline function pre_ph!(A::KR, w::LP{Float32}, seq::Int32, P::Int32, pf::Int32, xb::LP{UInt32}, si::Int32, pn::Int32)
    t0 = wtime(); lane = lanei(); gw = waveid(); NW = nwv(A); Np = P - pf; n = Int32(64) * Np
    A.flow == Int32(2) && (waitn!(A, dcdone(A, si - Int32(1)), pn); stepin!())   # a norm reads the sum over every element
    g = gw
    while g < n
        q = g >> Int32(6); G = g & Int32(63); k = G * LANES + lane; p = pf + q
        rs = rinv(A, 1, p)
        if G == Int32(0) && lane == Int32(0)
            sq0!(A, 0, p); st!(A.best, UInt64(0), q)
        end
        stx!(A, bf(bf(ld(A.x, Int64(k) * P + p) * rs) * ld(w, k)), k, Np, q)
        xdone!(A, xb, G >> Int32(4), lane)
        g += NW
    end
    done!(A, seq, t0, true)
end
# a mixer's or FFN's output, its slices added, into o, with its sums of squares (bank 0); zeroes bank 1
@inline function ssq_ph!(A::KR, l::Int32, j::Int32, seq::Int32, P::Int32, si::Int32, mi::Int32)
    t0 = wtime(); lane = lanei(); gw = waveid(); NW = nwv(A); n = Int32(64) * P; f2 = A.flow == Int32(2)
    _, rows, _, ns = mat(A, l, j)
    g = gw
    while g < n
        p = g >> Int32(6); G = g & Int32(63); k = G * LANES + lane
        f2 && waitn!(A, dcrg(A, mi) + 4G, ns)              # its row group's K slices
        G == Int32(0) && lane == Int32(0) && sq0!(A, 1, p)
        v = bf(psum(A, Int64(0), ns, rows, P, p, k))
        st!(A.o, v, Int64(k) * P + p)
        sqadd!(A, 0, p, lane, v * v)
        dn!(A, si, lane)
        g += NW
    end
    done!(A, seq, t0, true)
end
# o normalized and weighted, added to x, x's sums of squares (bank 1)
@inline function norm_ph!(A::KR, w::LP{Float32}, seq::Int32, P::Int32, si::Int32)
    t0 = wtime(); lane = lanei(); gw = waveid(); NW = nwv(A); n = Int32(64) * P
    A.flow == Int32(2) && (waitn!(A, dcdone(A, si - Int32(1)), n); stepin!())   # the sum of squares over every element
    g = gw
    while g < n
        p = g >> Int32(6); G = g & Int32(63); k = G * LANES + lane; e = Int64(k) * P + p
        rs = rinv(A, 0, p)
        v = bf(ld(A.x, e) + bf(bf(ld(A.o, e) * rs) * ld(w, k)))
        st!(A.x, v, e)
        sqadd!(A, 1, p, lane, v * v)
        dn!(A, si, lane)
        g += NW
    end
    done!(A, seq, t0, true)
end
# the gated short conv after its in_proj: h = B x, the 7-tap causal depthwise conv over a ring of h, C conv into xt
@inline function conv_ph!(A::KR, l::Int32, seq::Int32, P::Int32, pre::Bool, xb::LP{UInt32}, mi::Int32)
    t0 = wtime(); lane = lanei(); gw = waveid(); NW = nwv(A); n = Int32(64); f2 = A.flow == Int32(2)
    _, rows, _, ns = mat(A, l, Int32(0)); cw = fvec(A, l, 4); rg = reinterpret(LP{UInt16}, vaddr(A, l, 6))
    g = gw
    while g < n
        k = g * LANES + lane
        if f2                                              # the row groups of its B, C and x
            waitn!(A, dcrg(A, mi) + 4g, ns); waitn!(A, dcrg(A, mi) + 4(g + Int32(64)), ns); waitn!(A, dcrg(A, mi) + 4(g + Int32(128)), ns)
        end
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
            stx2!(A, bf(C * bf(s)), k, P, p)
            st!(rg, (reinterpret(UInt32, h) >> 16) % UInt16, (Int64(r) * NE + k) * CRING + (pos & Int32(CRING - 1)))
            win = (win[2], win[3], win[4], win[5], win[6], h)
        end
        xdone!(A, xb, g >> Int32(4), lane)
        g += NW
    end
    done!(A, seq, t0, true)
end
# the FFN's activation: silu(w1 x) (w3 x) into xt
@inline function act_ph!(A::KR, l::Int32, seq::Int32, P::Int32, xb::LP{UInt32}, mi::Int32)
    t0 = wtime(); lane = lanei(); gw = waveid(); NW = nwv(A); n = Int32(NFF ÷ 32) * P; f2 = A.flow == Int32(2)
    _, rows, _, ns = mat(A, l, Int32(5)); ou = Int64(ns) * P * rows
    g = gw
    while g < n
        p = g ÷ Int32(NFF ÷ 32); G = g - p * Int32(NFF ÷ 32); k = G * LANES + lane
        if f2                                              # the row groups of its w1 and w3
            waitn!(A, dcrg(A, mi) + 4G, ns); waitn!(A, dcrg(A, mi) + 4(G + Int32(NFF ÷ 32)), ns)
        end
        a = bf(psum(A, Int64(0), ns, rows, P, p, k)); b = bf(psum(A, ou, ns, rows, P, p, k))
        stx2!(A, bf(bf(a * sigm(a)) * b), k, P, p)
        xdone!(A, xb, G >> Int32(4), lane)
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
@inline function aprep_ph!(A::KR, lds, l::Int32, seq::Int32, P::Int32, si::Int32, pn::Int32)
    t0 = wtime(); t = tid(); w = wgid(); full = isfull(l); tt = full ? Int32(0) : Int32(1)
    A.flow == Int32(2) && (waitn!(A, dcdone(A, si - Int32(1)), pn); stepin!())   # q, k, v and the gate, whole
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
            dn!(A, si, lane)
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
        if A.flow == Int32(2)                              # (every lane's store out, then this head's one count)
            fence_rel(); KI.barrier()
            dn!(A, si, t)
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
# (a step's items, for the step that waits for all of them: as score_ph! and vmix_ph! lay them out)
@inline function score_n(A::KR, l::Int32, P::Int32, pre::Bool)
    full = isfull(l); amaxb = rdf(ld(A.posa, P - Int32(1))); n = Int32(0)
    for p in Int32(0):P-Int32(1)
        r = rdf(ld(A.posr, p)); pos = rdf(ld(A.posa, p)); _, ng, _, _ = grange(A, full, r, pos, pre ? amaxb : pos); n += Int32(4) * ng
    end
    n
end
@inline function vmix_n(A::KR, l::Int32, P::Int32, pre::Bool)
    full = isfull(l); amaxb = rdf(ld(A.posa, P - Int32(1))); n = Int32(4) * P
    for p in Int32(0):P-Int32(1)
        r = rdf(ld(A.posr, p)); pos = rdf(ld(A.posa, p)); _, _, _, ntk = grange(A, full, r, pos, pre ? amaxb : pos); n += Int32(16) * ((ntk + Int32(CKS - 1)) >> Int32(6))
    end
    n
end
@inline function score_ph!(A::KR, l::Int32, seq::Int32, P::Int32, pre::Bool, si::Int32)
    t0 = wtime(); did = false; lane = lanei(); gw = waveid(); NW = nwv(A); full = isfull(l)
    A.flow == Int32(2) && (waitn!(A, dcdone(A, si - Int32(1)), Int32(37) * P); stepin!())   # the queries, keys and values just written
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
            dn!(A, si, lane)
            did = true; g += NW
        end
        base += n
    end
    done!(A, seq, t0, did)
end
# each head's total of the scores into et (items 0 .. 4 P - 1), then the mix: Σ score v, slices into pv
@inline function vmix_ph!(A::KR, l::Int32, seq::Int32, P::Int32, pre::Bool, si::Int32)
    t0 = wtime(); did = false; lane = lanei(); gw = waveid(); NW = nwv(A); full = isfull(l)
    A.flow == Int32(2) && (waitn!(A, dcdone(A, si - Int32(1)), score_n(A, l, P, pre)); stepin!())
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
        dn!(A, si, lane)
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
            dn!(A, si, lane)
            did = true; g += NW
        end
        base += n
    end
    done!(A, seq, t0, did)
end
# the mix divided by the head's total, gated, into xt for o_proj
@inline function acomb_ph!(A::KR, l::Int32, seq::Int32, P::Int32, pre::Bool, xb::LP{UInt32}, si::Int32)
    t0 = wtime(); did = false; lane = lanei(); gw = waveid(); NW = nwv(A); full = isfull(l); n = Int32(64) * P; g = gw
    A.flow == Int32(2) && (waitn!(A, dcdone(A, si - Int32(1)), vmix_n(A, l, P, pre)); stepin!())
    amaxb = rdf(ld(A.posa, P - Int32(1)))
    while g < n
        p = g >> Int32(6); h = (g & Int32(63)) >> Int32(2); Gc = g & Int32(3); j = h >> Int32(2); i = h & Int32(3); c = Gc * LANES + lane
        r = rdf(ld(A.posr, p)); pos = rdf(ld(A.posa, p)); amax = pre ? amaxb : pos
        _, _, _, ntk = grange(A, full, r, pos, amax); nsv = (ntk + Int32(CKS - 1)) >> Int32(6)
        s = 0.0f0; b = A.pv + (Int64(p) * 4 + j) * PVS * 128 * 16 + (Int64(c) * 16 + i * Int32(4))
        for sl in Int32(0):nsv-Int32(1)
            s += ld(b, Int64(sl) * 128 * 4)
        end
        stx2!(A, bf(bf(s / ld(A.et, Int64(p) * 16 + h)) * ld(A.gh, Int64(p) * NH + h)), h * Int32(HD) + c, P, p)
        xdone!(A, xb, h >> Int32(2), lane)
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
@inline function head_ph!(A::KR, seq::Int32, P::Int32, pf::Int32, xb::LP{UInt32}, cl::LP{UInt32})
    t0 = wtime(); did = false; lane = lanei(); gw = waveid(); NW = nwv(A); gen = A.mode == Int32(0); xp = fresh(A.xt); Np = P - pf
    W, rows, nc, _ = mat(A, HEADL, Int32(0)); n = rows >> Int32(5); dyn = A.dyn != Int32(0); g = dyn ? claim(cl, lane) : gw
    if A.flow != Int32(0) && g < n                         # (the whole K a wavefront: every slice)
        for f in Int32(0):Int32(3); xready!(A, xb, f, Int32(16) * Np); end
        xp = fresh(A.xt); t0 = wtime()
    end
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
        did = true; g = dyn ? claim(cl, lane) : g + NW
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
    t0 = wtime(); q = tid()
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
    t0 = wtime(); lane = lanei(); gw = waveid(); NW = nwv(A); Np = P - pf; nq = Int32(NV ÷ 32)
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
    t0 = wtime(); q = tid(); rs = ld(A.posr, pf + q); T = ld(A.sp, 4rs); w = T * SMARGIN / Float32(HBINS); k = min(unsafe_trunc(Int32, ld(A.sp, 4rs + Int32(1))), SKMAX); c = Int32(0); b = Int32(0)
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
    t0 = wtime(); lane = lanei(); gw = waveid(); NW = nwv(A); Np = P - pf; nq = Int32(NV ÷ 32)
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
    t0 = wtime(); q = tid(); rq = ld(A.posr, pf + q); T = ld(A.sp, 4rq); n = min(reinterpret(Int32, ld(A.cn, q)), SCAND); o = Int(q) * 2 * Int(SCAND)
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
    # si: this step in the forward (emb is step 0); pn: the items of the step before; mi: the matrix step last run
    sf = seq; f1 = A.flow == Int32(1); f2 = A.flow == Int32(2); pn = Int32(68) * P; mi = Int32(-1); pm = Int32(0)
    emb_ph!(A, seq, P); seq += Int32(1)
    nlr = rdf(reinterpret(Int32, ldacq(A.bar + 12))); nl = nlr == Int32(0) ? Int32(NL) : nlr   # bar[3]: layers to run (0: all), for debugging
    l = Int32(0); i = Int32(0)
    while l <= HEADL
        kind, j0, nm, vn = lstep(l, i); si = seq - sf
        (smp && l == HEADL && i >= Int32(2)) && (kind = S_HIST + i - Int32(2))
        # the wait each step needs: flow 1, a matrix step slice by slice; flow 2, every step up to the head's
        sb = f2 ? kind <= S_HEAD : f1 && (kind == S_MAT || kind == S_HEAD)
        sb || gbar!(A)
        # what's read next through caches holding what was read there before (a matrix step's x comes after this too: no
        # line of it can have been read since the step that wrote it started, and its old readers have finished)
        f2 && sb && stepin!()
        xp = f2 ? dcflg(A, si) : A.flg                     # the x flags this step writes
        xc = f2 ? dcflg(A, si - Int32(1)) : A.flg          # the ones it reads
        cl = f2 ? dcdone(A, si) + 4 : A.bar + 128          # its claim counter
        if kind == S_MAT
            mi += Int32(1); pm = matn(A, l, j0, nm)
            (j0 == Int32(1) || j0 == Int32(4) || j0 == Int32(7)) ? mat_ph!(withpart2(A), l, j0, nm, seq, P, j0 == Int32(1) && !isattn(l) ? Int32(16) : Int32(16) * P, xc, cl, si, mi) : mat_ph!(A, l, j0, nm, seq, P, Int32(16) * P, xc, cl, si, mi)
            pn = pm
        elseif kind == S_PRE
            if l == HEADL
                pre_ph!(A, fvec(A, HEADL, 0), seq, P, pfirst, xp, si, pn)
            else
                pre_ph!(A, fvec(A, l, vn), seq, P, Int32(0), xp, si, pn)
            end
        elseif kind == S_SSQ
            ssq_ph!(withpart2(A), l, j0, seq, P, si, mi); pn = Int32(64) * P
        elseif kind == S_NORM
            norm_ph!(A, fvec(A, l, vn), seq, P, si); pn = Int32(64) * P
        elseif kind == S_CONV
            conv_ph!(A, l, seq, P, pre, xp, mi)
        elseif kind == S_ACT
            act_ph!(A, l, seq, P, xp, mi)
        elseif kind == S_APREP
            aprep_ph!(A, lds, l, seq, P, si, pn)
        elseif kind == S_SCORE
            score_ph!(A, l, seq, P, pre, si)
        elseif kind == S_VMIX
            vmix_ph!(A, l, seq, P, pre, si)
        elseif kind == S_ACOMB
            acomb_ph!(A, l, seq, P, pre, xp, si)
        elseif kind == S_HIST
            hist_ph!(A, seq, P, pfirst)
        elseif kind == S_THR
            thr_ph!(A, seq, P, pfirst)
        elseif kind == S_CAND
            cand_ph!(A, seq, P, pfirst)
        elseif kind == S_STOK
            stok_ph!(A, lds2, seq, P, pfirst)
        elseif kind == S_HEAD
            head_ph!(A, seq, P, pfirst, xc, cl)
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
            setpos_ph!(A, rp, a, P); A.flow == Int32(2) && zero_dc!(A); gbar!(A)
            seq, stop = fwd!(A, lds, lds2, P, seq, head, pfirst, true)
            # the slot's position, and for the host's progress lines its host word
            if wgid() == Int32(0) && tid() == Int32(0)
                st!(A.apos, a + P, rp); strel_sys(A.hf + 4 * (Int64(16) + rp), UInt32(a + P))
            end
            lastpre = true; rr = rp + Int32(1)
            (a + P == l && l < ninp) && break              # a snapshot: the host copies the slot's rings and carries on
        elseif Pd > Int32(0)
            setpos_ph!(A, Int32(-1), Int32(0), Pd); A.flow == Int32(2) && zero_dc!(A); gbar!(A)
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
function load_engine(dir; caches = true)
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
        caches || break                                    # (--train runs its own forward and has no cache)
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
    part2::ROCVector{Float32}; flg::ROCVector{UInt32}; xt2::ROCVector{Float32}; dc::ROCVector{UInt32}
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
        z(max(6144 * 4, 3104 * 4, 2 * 7680 * 4, 2048 * 15) * npos), AMDGPU.zeros(UInt32, 16), z(NFF * npos), AMDGPU.zeros(UInt32, D_N),
        zi(5NR), zi(npos), zi(npos), zi(npos), zi(NR), zi(NR))
end
struct PEngine
    M::Model; K::KBuf; hx::HostX; nwg::Int; npos::Int; maxgen::Int; hz::Float64; rings::Vector{ROCVector{UInt16}}
end
# the clock's rate, from two spins of different lengths against the host clock
@kernel unsafe_indices = true function k_spin!(o, n)
    t0 = wtime()
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
       pen ? 1.1f0 : 1.0f0, lp(UInt32, K.seen), lp(Float32, K.sp), lp(Int32, K.si), lp(Int32, K.apos), lp(Int32, K.lim),
       Int32(DFLOW[]), lp(Float32, K.part2), lp(UInt32, K.flg), lp(Float32, DFLOW[] == 2 ? K.xt2 : K.xt), lp(UInt32, K.dc))
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
    const SCHEDSTOP = Ref(false); const SCHEDTASK = Ref{Any}(nothing); const SCHEDW = Ref{Any}(nothing)
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
# ---- the display watchdog: the card also runs the desktop. While a dispatch runs this reads the rings' fences (debugfs, through
# sudo) and the logs. A ring with work pending standing still for WSTALL seconds (the driver resets at 10), a stuck page flip, a
# failed KWin commit or a reset ends the dispatch at the next forward and holds the slots for WPAUSE seconds. Chats wait; their
# tokens are unchanged, since a slot resumes at the forward it stopped at.
const WSTALL, WPAUSE = 3.0, 30.0
if !isdefined(Main, :WPAUSED)
    const WPAUSED = Ref(0.0)                               # no dispatch before this time
    const WTRIPS = Ref(0)
    const WDISPATCH = Ref(false)                           # a dispatch is running
end
function watch_trip!(PE::PEngine, why)
    WPAUSED[] = time() + WPAUSE; WTRIPS[] += 1; yield!(PE)
    println("display watchdog: ", why, ": the slots hold for $(Int(WPAUSE)) s"); flush(stdout)
end
function watch_loop(PE::PEngine)
    dri = try
        first(d for d in readlines(`sudo -n sh -c 'ls -d /sys/kernel/debug/dri/*/'`) if occursin("gfx_0.0.0", read(`sudo -n sh -c "cat $(d)amdgpu_fence_info 2>/dev/null; true"`, String)))
    catch e
        println("display watchdog: no access to the rings' fences (", sprint(showerror, e), "): logs only"); flush(stdout); ""
    end
    last = Dict{String,Tuple{UInt64,Float64}}(); since = ""; tl = 0.0
    bad = "ring .*timeout|GPU reset|GPU Hang|flip_done|commit wait"
    while !SCHEDSTOP[]
        if !WDISPATCH[] || time() < WPAUSED[]
            empty!(last); since = ""; sleep(0.25); continue
        end
        isempty(since) && (since = strip(read(`date '+%Y-%m-%d %H:%M:%S'`, String)))
        now = time()
        if !isempty(dri)
            for m in eachmatch(r"--- ring \d+ \(((?:gfx|comp)[^)]*)\) ---\nLast signaled fence\s+0x([0-9a-f]+)\nLast emitted\s+0x([0-9a-f]+)", read(`sudo -n cat $(dri)amdgpu_fence_info`, String))
                s = parse(UInt64, m[2], base = 16); e = parse(UInt64, m[3], base = 16); a = get(last, m[1], (s, now))
                if s != a[1] || s == e
                    last[m[1]] = (s, now)
                else
                    last[m[1]] = a
                    now - a[2] > WSTALL && (watch_trip!(PE, "$(m[1]) stood still with work pending"); break)
                end
            end
        end
        if now - tl > 1.0
            tl = now
            k = read(pipeline(ignorestatus(`journalctl -k --since $since`), ignorestatus(`rg -i $bad`)), String)
            u = read(pipeline(ignorestatus(`journalctl --user --since $since`), ignorestatus(`rg -c "Atomic commit failed"`)), String)
            (!isempty(k) || !isempty(strip(u))) && time() >= WPAUSED[] && (watch_trip!(PE, isempty(k) ? "KWin's commits failed" : first(split(k, '\n'))); since = "")
        end
        sleep(0.25)
    end
end
# The scheduler: what is waiting joins between dispatches, and a dispatch is told to end (word 13) when something is waiting.
function sched_loop(PE::PEngine, H)
    while !SCHEDSTOP[]
        worked = false
        if time() < WPAUSED[]                              # the display watchdog holds the slots
            sleep(0.05); continue
        end
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
                    WDISPATCH[] = true; dispatch!(PE, H, 0); WDISPATCH[] = false
                    settle!(PE, H)
                catch e
                    WDISPATCH[] = false
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
if !isdefined(Main, :DFLOW)
    const DFLOW = Ref(2)                                   # flow 0 | 1 | 2
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
        SCHEDSTOP[] = true; wait(SCHEDTASK[]); wait(SCHEDW[]); SCHEDSTOP[] = false
    end
    PK[] === nothing || (foreach(a -> a isa ROCArray && AMDGPU.unsafe_free!(a), (getfield(PK[].K, f) for f in fieldnames(typeof(PK[].K)))); AMDGPU.HIP.hipHostFree(PK[].hx.c))
    PK[] = PEngine(SRV[].M)
    SLOTS[] = [Slot(0, 0, Int[], 0.0, nothing) for _ in 1:NR]; empty!(PENDING)
    SCHEDTASK[] = Threads.@spawn Base.invokelatest(sched_loop, PK[], HostState())
    SCHEDW[] = Threads.@spawn Base.invokelatest(watch_loop, PK[])
    # (the first dispatch compiles nothing more: the occupancy query compiled the same kernel)
    @printf("persistent kernel: %d workgroups, %d positions a prefill forward, clock %.2f MHz, compiled in %.0f s\n", PK[].nwg, PK[].npos, PK[].hz / 1e6, time() - t0); flush(stdout)
end
function serve(dir, sock)
    M, g = load_engine(dir)
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
    q.slot < 0 && return                                   # cancelled (the client went) before a slot took it: nothing to read
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
    elseif a[1] == "flow"                                # flow 0 | 1: matrix steps wait for their x slice by slice, or at a grid barrier
        DFLOW[] = parse(Int, a[2]); println(c, "flow ", DFLOW[])
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
# ---------------------------------------------------------------------------------------------
# Training (--train). Model: what the user says next, from what lfm3-data wrote: every weight trains. One window of a transcript
# at a time, with the loss on the replies in it, one layer at a time over all its positions:
#   forward: the layer inputs are kept (H); backward: each layer's forward again with what its adjoint needs, then the adjoint.
# A step is short dispatches, no persistent kernel: k_bwd! for forward and adjoint, k_wupd! for the update. A wavefront takes 32
# positions (a lane a position) or runs the GEMV of the forward, whose x stays BF16 pairs. The adjoint of y = W x is the same
# GEMV on Wt, W's transpose, packed beside it: gx = Wt gy; dW = Σ gy x' is a matrix product (t_gemm!).
# BF16 weights and the transposes stay in VRAM. FP32 master weights and momentum are in GTT: a matrix's gradient goes straight into
# its momentum as the layer finishes. Weight matrices step by Muon: the momentum orthogonalized (Newton-Schulz, in VRAM), times lr.
# The vector weights (norms, conv taps) step by plain momentum. The embedding, tied to the head, stays on the host (its gradient
# is the input rows plus a few head rows, and it's bigger than VRAM has room for).
# ---------------------------------------------------------------------------------------------
using LinearAlgebra
const T_NORM, T_NORMB, T_NORMW, T_PACK, T_GEMV, T_AXPBY, T_CONV1, T_CONV2, T_CONVB1, T_CONVB2, T_CONVW, T_SILU, T_SILUB, T_ROT, T_ROTB, T_TR, T_SOFT, T_SOFTB, T_GATE, T_GATEB, T_GEMM, T_UPD, T_GEMVT = Int32.(1:23)
const NPMAX = 1568                                         # positions a window (a multiple of 32)
struct TR
	op::Int32; nwg::Int32; N::Int32
	i1::Int32; i2::Int32; i3::Int32; i4::Int32; i5::Int32; i6::Int32
	f1::Float32; f2::Float32
	a::LP{Float32}; b::LP{Float32}; c::LP{Float32}; d::LP{Float32}; e::LP{Float32}
	W::LP{UInt8}; Wt::LP{UInt8}
	s1::Int64; s2::Int64; s3::Int64; s4::Int64; s5::Int64; s6::Int64; s7::Int64
end
const LP0 = reinterpret(LP{Float32}, Ptr{Float32}(C_NULL))
TR(op, nwg, N; i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0, i6 = 0, f1 = 1.0f0, f2 = 0.0f0, a = LP0, b = LP0, c = LP0, d = LP0, e = LP0, W = reinterpret(LP{UInt8}, LP0), Wt = reinterpret(LP{UInt8}, LP0), s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0) =
	TR(op, Int32(nwg), Int32(N), Int32(i1), Int32(i2), Int32(i3), Int32(i4), Int32(i5), Int32(i6), Float32(f1), Float32(f2), a, b, c, d, e, W, Wt, s1, s2, s3, s4, s5, s6, s7)
@inline tnw(T::TR) = WVS * T.nwg
# (x[n][bb][d] is x[n S + bb D + d]; a wavefront takes 32 positions of a block bb)
@inline function t_norm!(T::TR)                            # c = b x rsqrt(mean x² + eps) per (n, bb), d[n][bb] its rsqrt
	lane = lanei(); NW = tnw(T); D = T.i1; S = T.i2; nb = T.i3; n = ((T.N + Int32(31)) >> Int32(5)) * nb; g = waveid()
	while g < n
	G = g ÷ nb; bb = g - G * nb; r = G * LANES + lane; o = Int64(r) * S + bb * D
	s = 0.0f0
	for q in Int32(0):D-Int32(1); v = ld(T.a, o + q); s = fma(v, v, s); end
rs = rsq(s / Float32(D) + NEPS)
for q in Int32(0):D-Int32(1); st!(T.c, ld(T.b, q) * rs * ld(T.a, o + q), o + q); end
st!(T.d, rs, Int64(r) * nb + bb)
g += NW
    end
end
@inline function t_normb!(T::TR)                           # its adjoint: e (+) from a = the output's gradient, b = x, c = the weights, d = rsqrt
	lane = lanei(); NW = tnw(T); D = T.i1; S = T.i2; nb = T.i3; add = T.i4 != Int32(0); n = ((T.N + Int32(31)) >> Int32(5)) * nb; g = waveid()
	while g < n
	G = g ÷ nb; bb = g - G * nb; r = G * LANES + lane; o = Int64(r) * S + bb * D; rs = ld(T.d, Int64(r) * nb + bb)
	s = 0.0f0
	for q in Int32(0):D-Int32(1); s = fma(ld(T.a, o + q) * ld(T.c, q), ld(T.b, o + q), s); end
k = s * rs * rs * rs / Float32(D)
for q in Int32(0):D-Int32(1)
	v = ld(T.a, o + q) * ld(T.c, q) * rs - k * ld(T.b, o + q)
	st!(T.e, add ? ld(T.e, o + q) + v : v, o + q)
end
g += NW
end
end
@inline function t_normw!(T::TR)                           # c[q] = the gradient of the weight q: a lane a weight
	lane = lanei(); NW = tnw(T); D = T.i1; S = T.i2; nb = T.i3; n = (D + Int32(31)) >> Int32(5); g = waveid()
	while g < n
	q = g * LANES + lane; acc = 0.0f0
	for r in Int32(0):T.N-Int32(1), bb in Int32(0):nb-Int32(1)
	o = Int64(r) * S + bb * D + q
	acc = fma(ld(T.a, o) * ld(T.b, o), ld(T.d, Int64(r) * nb + bb), acc)
end
st!(T.c, acc, q)
g += NW
end
end
@inline function t_pack!(T::TR)                            # c, BF16 pairs for the GEMV: [block of 16][k / 2][p], from a[n S + k]
	lane = lanei(); NW = tnw(T); K = T.i1; S = T.i2; n = (T.N + Int32(31)) >> Int32(5); g = waveid()
	while g < n
	r = g * LANES + lane; b = r >> Int32(4); p = r & Int32(15)
	for k in Int32(0):K-Int32(1)
	stb!(T.c, bf(ld(T.a, Int64(r) * S + k)), (Int64(b) * (K >> Int32(1)) * 16 + Int64(k >> Int32(1)) * 16 + p) * 4 + (k & Int32(1)) * 2)
end
g += NW
    end
end
@inline function t_gemv!(T::TR)                            # d[n S + r] = W x, c = x packed; W with i1 groups of 32 rows, K = i3
	lane = lanei(); NW = tnw(T); ng = T.i1; K = T.i3; nc = K >> Int32(3); S = T.i4; lo = (lane % UInt32) << 4; n = ((T.N + Int32(15)) >> Int32(4)) * ng; g = waveid(); xp = fresh(T.c)
	while g < n
	b = g ÷ ng; G = g - b * ng
	acc = gemv(T.W + (Int64(G) * nc) * CHUNK, lo, Int64(CHUNK), nc, xp + Int64(b) * (K >> Int32(1)) * 64, Val(16))
	r = G * LANES + lane
	for p in 1:16
	st!(T.d, acc[p], (Int64(b) * 16 + (p - 1)) * S + r)
end
g += NW
    end
    end
    @inline function t_axpby!(T::TR)                           # c = f1 a + f2 b over N rows of D (stride S, a's stride i3, in i4 rows of ...)
	    lane = lanei(); NW = tnw(T); D = T.i1; S = T.i2; n = (T.N + Int32(31)) >> Int32(5); g = waveid(); hasb = T.f2 != 0.0f0; f1 = T.f1; f2 = T.f2
	    if S == D                                          # contiguous: a lane an element, 4 at a time (coalesced: the momentum is read over PCIe)
	        m = Int64(T.N) * D; i = Int64(g) * LANES * 4 + lane * 4; step = Int64(NW) * LANES * 4
	        while i + 3 < m
	            Base.Cartesian.@nexprs 4 k -> begin
	                v = f1 == 0.0f0 ? 0.0f0 : f1 * ld(T.a, i + (k - 1))
	                st!(T.c, hasb ? v + f2 * ld(T.b, i + (k - 1)) : v, i + (k - 1))
	            end
	            i += step
	        end
	        while i < m                                    # (the tail)
	            v = f1 == 0.0f0 ? 0.0f0 : f1 * ld(T.a, i)
	            st!(T.c, hasb ? v + f2 * ld(T.b, i) : v, i); i += Int64(1)
	        end
	        return
	    end
	    while g < n
	    r = g * LANES + lane; o = Int64(r) * S
	    for q in Int32(0):(r < T.N ? D : Int32(0))-Int32(1)
	    v = f1 == 0.0f0 ? 0.0f0 : f1 * ld(T.a, o + q)
	    st!(T.c, hasb ? v + f2 * ld(T.b, o + q) : v, o + q)
    end
    g += NW
end
    end
    # the conv: a = B C x [n][3 NE], b = h = B x, d = the taps [q][7], c = y, e = the conv's output
    @inline function t_conv1!(T::TR)
	    lane = lanei(); NW = tnw(T); n = (T.N + Int32(31)) >> Int32(5); g = waveid()
	    while g < n
	    r = g * LANES + lane; o = Int64(r) * 3NE
	    for q in Int32(0):Int32(NE - 1); st!(T.b, ld(T.a, o + q) * ld(T.a, o + q + 2NE), Int64(r) * NE + q); end
    g += NW
end
end
@inline function t_conv2!(T::TR)
	lane = lanei(); NW = tnw(T); n = (T.N + Int32(31)) >> Int32(5); g = waveid()
	while g < n
	r = g * LANES + lane
	for q in Int32(0):Int32(NE - 1)
	s = 0.0f0
	for t in Int32(0):Int32(6)
	m = r + t - Int32(6)
	s = fma(ld(T.d, Int64(q) * 7 + t), m >= Int32(0) ? ld(T.b, Int64(m) * NE + q) : 0.0f0, s)
end
st!(T.e, s, Int64(r) * NE + q); st!(T.c, ld(T.a, Int64(r) * 3NE + q + NE) * s, Int64(r) * NE + q)
    end
    g += NW
end
    end
    @inline function t_convb1!(T::TR)                          # a = gy, b = B C x, c = the conv's output, d = gc, e = the gradient of B C x (its C)
	    lane = lanei(); NW = tnw(T); n = (T.N + Int32(31)) >> Int32(5); g = waveid()
	    while g < n
	    r = g * LANES + lane
	    for q in Int32(0):Int32(NE - 1)
	    gy = ld(T.a, Int64(r) * NE + q)
	    st!(T.d, gy * ld(T.b, Int64(r) * 3NE + q + NE), Int64(r) * NE + q)
	    st!(T.e, gy * ld(T.c, Int64(r) * NE + q), Int64(r) * 3NE + q + NE)
    end
    g += NW
    end
    end
    @inline function t_convb2!(T::TR)                          # a = gc, b = B C x, d = the taps, e = the gradient of B C x (its B and x)
	    lane = lanei(); NW = tnw(T); n = (T.N + Int32(31)) >> Int32(5); g = waveid(); N = T.N
	    while g < n
	    r = g * LANES + lane
	    for q in Int32(0):Int32(NE - 1)
	    s = 0.0f0
	    for t in Int32(0):Int32(6)
	    m = r + Int32(6) - t
	    s = fma(ld(T.d, Int64(q) * 7 + t), m < N ? ld(T.a, Int64(m) * NE + q) : 0.0f0, s)
    end
    o = Int64(r) * 3NE + q
    st!(T.e, s * ld(T.b, o + 2NE), o); st!(T.e, s * ld(T.b, o), o + 2NE)
    end
    g += NW
    end
end
@inline function t_convw!(T::TR)                           # c[q][t] = the tap's gradient: a = gc, b = h
	lane = lanei(); NW = tnw(T); n = Int32(NE) >> Int32(5); g = waveid()
	while g < n
	q = g * LANES + lane
	for t in Int32(0):Int32(6)
	s = 0.0f0
	for r in Int32(6)-t:T.N-Int32(1)
	s = fma(ld(T.a, Int64(r) * NE + q), ld(T.b, Int64(r + t - Int32(6)) * NE + q), s)
end
st!(T.c, s, Int64(q) * 7 + t)
    end
    g += NW
    end
    end
    @inline function t_silu!(T::TR)                            # c = silu(a) b, rows of NFF
	    lane = lanei(); NW = tnw(T); n = (T.N + Int32(31)) >> Int32(5); g = waveid()
	    while g < n
	    r = g * LANES + lane; o = Int64(r) * NFF
	    for q in Int32(0):Int32(NFF - 1); x = ld(T.a, o + q); st!(T.c, x * sigm(x) * ld(T.b, o + q), o + q); end
    g += NW
    end
end
@inline function t_silub!(T::TR)                           # a = gs, b, c = what the forward's a, b were; d, e their gradients
	lane = lanei(); NW = tnw(T); n = (T.N + Int32(31)) >> Int32(5); g = waveid()
	while g < n
	r = g * LANES + lane; o = Int64(r) * NFF
	for q in Int32(0):Int32(NFF - 1)
	gs = ld(T.a, o + q); x = ld(T.b, o + q); s = sigm(x)
	st!(T.d, gs * ld(T.c, o + q) * (s + x * s * (1.0f0 - s)), o + q); st!(T.e, gs * x * s, o + q)
end
g += NW
end
end
# rotary: b = (cos, sin) [n][table][64 angles]; a, c rows [n S + j HD + q], i2 = heads, i3 = the table
@inline function t_rot!(T::TR)
	lane = lanei(); NW = tnw(T); S = T.i1; nh = T.i2; tt = T.i3; n = (T.N + Int32(31)) >> Int32(5); g = waveid()
	while g < n
	r = g * LANES + lane
	for j in Int32(0):nh-Int32(1), q in Int32(0):Int32(HD - 1)
	o = Int64(r) * S + j * Int32(HD); e = ((Int64(r) * 2 + tt) * 64 + (q & Int32(63))) * 2; cs = ld(T.b, e); sn = ld(T.b, e + 1)
	st!(T.c, q < Int32(64) ? ld(T.a, o + q) * cs - ld(T.a, o + q + 64) * sn : ld(T.a, o + q) * cs + ld(T.a, o + q - 64) * sn, o + q)
end
g += NW
end
	  end
	  @inline function t_rotb!(T::TR)
		  lane = lanei(); NW = tnw(T); S = T.i1; nh = T.i2; tt = T.i3; n = (T.N + Int32(31)) >> Int32(5); g = waveid()
		  while g < n
		  r = g * LANES + lane
		  for j in Int32(0):nh-Int32(1), q in Int32(0):Int32(HD - 1)
		  o = Int64(r) * S + j * Int32(HD); e = ((Int64(r) * 2 + tt) * 64 + (q & Int32(63))) * 2; cs = ld(T.b, e); sn = ld(T.b, e + 1)
		  st!(T.c, q < Int32(64) ? ld(T.a, o + q) * cs + ld(T.a, o + q + 64) * sn : ld(T.a, o + q) * cs - ld(T.a, o + q - 64) * sn, o + q)
	  end
	  g += NW
	  end
	  end
	  @inline function t_tr!(T::TR)                              # c[e][n] = a[n S + e] (a lane an e), i1 = e's, N = n's, s1 = c's row stride
		  lane = lanei(); NW = tnw(T); E = T.i1; S = T.i2; n = (E + Int32(31)) >> Int32(5); g = waveid()
		  while g < n
		  e = g * LANES + lane
		  for r in Int32(0):T.N-Int32(1); st!(T.c, ld(T.a, Int64(r) * S + e), Int64(e) * T.s1 + r); end
	  g += NW
	  end
	  end
using Random
# scores [j][n][m], j of 16 heads: softmax over what n sees (i1 = the window, 0: every earlier position); in place
@inline function t_soft!(T::TR)
    lane = lanei(); NW = tnw(T); N = T.N; win = T.i1; n = Int32(NH) * N; g = waveid()
    while g < n
        j = g ÷ N; r = g - j * N; o = (Int64(j) * N + r) * N; lo = win == Int32(0) ? Int32(0) : max(Int32(0), r - win + Int32(1))
        mx = -Inf32
        for m in lo+lane:LANES:r; mx = max(mx, ld(T.a, o + m)); end
        mx = KI.sub_group_reduce(max, mx)
        s = 0.0f0
        for m in lo+lane:LANES:r; e = exp2f((ld(T.a, o + m) - mx) * LOG2E); st!(T.a, e, o + m); s += e; end
        s = KI.sub_group_reduce(+, s); inv = 1.0f0 / s
        for m in lane:LANES:N-Int32(1); st!(T.a, m >= lo && m <= r ? ld(T.a, o + m) * inv : 0.0f0, o + m); end
        g += NW
    end
end
@inline function t_softb!(T::TR)                           # a = the probabilities' gradient, b = the probabilities: a = b (a - Σ a b), the scores' gradient
    lane = lanei(); NW = tnw(T); N = T.N; n = Int32(NH) * N; g = waveid()
    while g < n
        j = g ÷ N; r = g - j * N; o = (Int64(j) * N + r) * N
        s = 0.0f0
        for m in lane:LANES:N-Int32(1); s = fma(ld(T.a, o + m), ld(T.b, o + m), s); end
        s = KI.sub_group_reduce(+, s)
        for m in lane:LANES:N-Int32(1); p = ld(T.b, o + m); st!(T.a, p * (ld(T.a, o + m) - s), o + m); end
        g += NW
    end
end
# the gate: c = a x sigmoid(b[n][head]) over rows of NH HD, i1 = b's row stride
@inline function t_gate!(T::TR)
    lane = lanei(); NW = tnw(T); S = T.i1; n = (T.N + Int32(31)) >> Int32(5); g = waveid()
    while g < n
        r = g * LANES + lane; o = Int64(r) * NE
        for j in Int32(0):Int32(NH - 1)
            s = sigm(ld(T.b, Int64(r) * S + j))
            for q in Int32(0):Int32(HD - 1); e = o + j * Int32(HD) + q; st!(T.c, ld(T.a, e) * s, e); end
        end
        g += NW
    end
end
@inline function t_gateb!(T::TR)                           # a = the gated output's gradient, b = the mix, c = the gate's logits: d = the mix's gradient, e = the logits'
    lane = lanei(); NW = tnw(T); S = T.i1; n = (T.N + Int32(31)) >> Int32(5); g = waveid()
    while g < n
        r = g * LANES + lane; o = Int64(r) * NE
        for j in Int32(0):Int32(NH - 1)
            s = sigm(ld(T.c, Int64(r) * S + j)); acc = 0.0f0
            for q in Int32(0):Int32(HD - 1)
                e = o + j * Int32(HD) + q; gy = ld(T.a, e)
                st!(T.d, gy * s, e); acc = fma(gy, ld(T.b, e), acc)
            end
            st!(T.e, acc * s * (1.0f0 - s), Int64(r) * S + j)
        end
        g += NW
    end
end
# the update of a matrix: a = the master ([rows][K], GTT), b = its step: a -= f1 b, then W = a in BF16 packed as the GEMV reads
# it (i4 = 0), or Wt = a's transpose packed (i4 = 1, after the first). An item: 32 rows x 32 columns; a lane a row (0) or a column (1)
@inline function t_upd!(T::TR)
    lane = lanei(); NW = tnw(T); rows = T.i1; K = T.i2; nkb = K >> Int32(5); ng = (rows + Int32(31)) >> Int32(5); n = ng * nkb; g = waveid()
    if T.i4 == Int32(0)
        nc = K >> Int32(3)
        while g < n
            G = g ÷ nkb; kb = g - G * nkb; r = G * LANES + lane; k0 = kb * LANES
            if r < rows
                o = Int64(r) * K + k0
                for c in Int32(0):Int32(3)
                    v = ntuple(i -> ld(T.a, o + c * Int32(8) + (i - 1)) - T.f1 * ld(T.b, o + c * Int32(8) + (i - 1)), Val(8))
                    for i in 1:8; st!(T.a, v[i], o + c * Int32(8) + (i - 1)); end
                    w = ntuple(i -> (reinterpret(UInt32, bf(v[2i])) & 0xffff0000) | (reinterpret(UInt32, bf(v[2i-1])) >> 16), Val(4))
                    unsafe_store!(reinterpret(LP{V4U}, T.W + ((Int64(G) * nc + (k0 >> Int32(3)) + c) * CHUNK + Int64(lane) * 16)), (VecElement(w[1]), VecElement(w[2]), VecElement(w[3]), VecElement(w[4])))
                end
            end
            g += NW
        end
    else
        ncT = ng << Int32(2)                               # chunks a row of the transpose: the padded rows / 8
        while g < n
            G = g ÷ nkb; kb = g - G * nkb; k = kb * LANES + lane
            for c in Int32(0):Int32(3)
                r0 = G * LANES + c * Int32(8)
                v = ntuple(i -> r0 + (i - 1) < rows ? ld(T.a, Int64(r0 + (i - 1)) * K + k) : 0.0f0, Val(8))
                w = ntuple(i -> (reinterpret(UInt32, bf(v[2i])) & 0xffff0000) | (reinterpret(UInt32, bf(v[2i-1])) >> 16), Val(4))
                unsafe_store!(reinterpret(LP{V4U}, T.Wt + ((Int64(kb) * ncT + (r0 >> Int32(3))) * CHUNK + Int64(lane) * 16)), (VecElement(w[1]), VecElement(w[2]), VecElement(w[3]), VecElement(w[4])))
            end
            g += NW
        end
    end
end
@inline function ld128c(W::LP{UInt8}, lo::UInt32)        # (a cached 16-byte load: what is read again by the next item)
    v = unsafe_load(reinterpret(LP{V4U}, W + lo)); (v[1].value, v[2].value, v[3].value, v[4].value)
end
# gx = W' gy with W as the forward's GEMV reads it (no transpose is kept): c = gy packed as the forward packs x (its K = W's padded
# rows), d[n S + k] the result; i1 = W's row groups, i3 = W's K (the result's width), i4 = S. An item: a chunk's 8 columns of 16
# positions over every row group; a lane a row of the group, the 32 rows' products added across the lanes at the end
@inline function t_gemvt!(T::TR)
    lane = lanei(); NW = tnw(T); ng = T.i1; K = T.i3; nc = K >> Int32(3); S = T.i4; R = ng * LANES
    nb = (T.N + Int32(15)) >> Int32(4); n = nb * nc; g = waveid(); lo = (lane % UInt32) << 4; sh = (lane & Int32(1)) << Int32(4)
    xp = reinterpret(LP{UInt8}, fresh(T.c))
    while g < n
        b = g ÷ nc; c = g - b * nc
        Base.Cartesian.@nexprs 128 i -> (a_i = 0.0f0)
        xb = xp + Int64(b) * (R >> Int32(1)) * 64
        for G in Int32(0):ng-Int32(1)
            w = ld128c(T.W + (Int64(G) * nc + c) * CHUNK, lo); r = G * LANES + lane
            Base.Cartesian.@nexprs 8 e -> (w_e = reinterpret(Float32, isodd(e) ? (w[(e + 1) >> 1] << 16) : (w[e >> 1] & 0xffff0000)))
            q = xb + Int64(r >> Int32(1)) * 64
            y1 = ld128c(q, UInt32(0)); y2 = ld128c(q, UInt32(16)); y3 = ld128c(q, UInt32(32)); y4 = ld128c(q, UInt32(48))
            Base.Cartesian.@nexprs 4 p -> (v_p = y1[p]; v_{p + 4} = y2[p]; v_{p + 8} = y3[p]; v_{p + 12} = y4[p])
            Base.Cartesian.@nexprs 16 p -> (y_p = reinterpret(Float32, ((v_p >> sh) & 0x0000ffff) << 16))
            Base.Cartesian.@nexprs 16 p -> Base.Cartesian.@nexprs 8 e -> (a_{(p - 1) * 8 + e} = fma(y_p, w_e, a_{(p - 1) * 8 + e}))
        end
        Base.Cartesian.@nexprs 128 i -> (a_i = KI.sub_group_reduce(+, a_i))
        if lane == Int32(0)
            Base.Cartesian.@nexprs 16 p -> Base.Cartesian.@nexprs 8 e -> st!(T.d, a_{(p - 1) * 8 + e}, (Int64(b) * 16 + (p - 1)) * S + c * Int32(8) + (e - 1))
        end
        g += NW
    end
end
@kernel unsafe_indices = true function k_tr!(T)
    op = T.op
    if op == T_NORM; t_norm!(T)
    elseif op == T_NORMB; t_normb!(T)
    elseif op == T_NORMW; t_normw!(T)
    elseif op == T_PACK; t_pack!(T)
    elseif op == T_GEMV; t_gemv!(T)
    elseif op == T_AXPBY; t_axpby!(T)
    elseif op == T_CONV1; t_conv1!(T)
    elseif op == T_CONV2; t_conv2!(T)
    elseif op == T_CONVB1; t_convb1!(T)
    elseif op == T_CONVB2; t_convb2!(T)
    elseif op == T_CONVW; t_convw!(T)
    elseif op == T_SILU; t_silu!(T)
    elseif op == T_SILUB; t_silub!(T)
    elseif op == T_ROT; t_rot!(T)
    elseif op == T_ROTB; t_rotb!(T)
    elseif op == T_TR; t_tr!(T)
    elseif op == T_SOFT; t_soft!(T)
    elseif op == T_SOFTB; t_softb!(T)
    elseif op == T_GATE; t_gate!(T)
    elseif op == T_GATEB; t_gateb!(T)
    elseif op == T_UPD; t_upd!(T)
    elseif op == T_GEMVT; t_gemvt!(T)
    end
end
KernelAbstractions.compiler_options(obj::KernelAbstractions.Kernel{ROCBackend,<:Any,<:Any,typeof(gpu_k_tr!)}) = (; maxthreads = Int(WGS))
if !isdefined(Main, :KTR)
    const KTR = Ref{Any}(nothing)
end
if !isdefined(Main, :TDBG)
    const TPROF = Dict{Any,Tuple{Int,Float64}}()         # --prof: ops' counts and seconds, a line a window
    const TDBG = Ref(false)                                # --debug: a line after every op, to place a fault
    const TPAUSE = Ref(0.0)                                # no op before this time: the display watchdog's hold
end
# the gfx ring's fences, mirrored to a readable file by a root shell loop (debugfs is root's): (signaled, emitted)
const FENCE = "/tmp/lfm3-gfx-fence"
function fence_start()
    dri = first(d for d in readlines(`sudo -n sh -c 'ls -d /sys/kernel/debug/dri/*/'`) if occursin("gfx_0.0.0", read(`sudo -n sh -c "cat $(d)amdgpu_fence_info 2>/dev/null; true"`, String)))
    run(ignorestatus(`sudo -n pkill -f lfm3-gfx-fence`))
    run(pipeline(`sudo -n sh -c "while :; do grep -A2 'gfx_0.0.0' $(dri)amdgpu_fence_info > $FENCE.tmp; mv $FENCE.tmp $FENCE; sleep 0.05; done"`); wait = false)
    atexit(() -> run(ignorestatus(`sudo -n pkill -f lfm3-gfx-fence`)))
    sleep(0.3)
end
function fence()
    m = match(r"Last signaled fence\s+0x([0-9a-f]+)\nLast emitted\s+0x([0-9a-f]+)", read(FENCE, String))
    m === nothing ? (UInt64(0), UInt64(0)) : (parse(UInt64, m[1], base = 16), parse(UInt64, m[2], base = 16))
end
# the display watchdog, in the op loop: the gfx ring standing still with work pending for WSTALL seconds (the driver resets it
# at 10, and KWin doesn't come back from that) holds the ops for 2 s, so its job gets the card
if !isdefined(Main, :WS0)
    const WS0 = Ref(UInt64(0)); const WT0 = Ref(0.0); const WLAST = Ref(0.0); const WHOLDS = Ref(0)
end
function tgap()
    now = time()
    while now < TPAUSE[]; sleep(0.05); now = time(); end
    now - WLAST[] < 0.1 && return
    WLAST[] = now; s, e = fence()
    if s != WS0[] || s == e
        WS0[] = s; WT0[] = now
    elseif now - WT0[] > WSTALL
        TPAUSE[] = now + 2.0; WT0[] = now; WHOLDS[] += 1
        WHOLDS[] % 10 == 1 && (println("display watchdog: gfx ring stood still $WSTALL s with work pending: ops hold for 2 s (hold $(WHOLDS[]))"); flush(stdout))
        sleep(2.0)
    end
end
function trun!(op, N; nwg = NWGMAX[], kw...)
    tgap(); t0 = time()
    KTR[](TR(op, nwg, N; kw...); ndrange = nwg * Int(WGS))
    AMDGPU.synchronize()                                  # (a gap after every op, for the desktop's gfx ring)
    c = get(TPROF, op, (0, 0.0)); TPROF[op] = (c[1] + 1, c[2] + time() - t0)
    TDBG[] && (println("op ", op, " N ", N, " ", kw); flush(stdout))
end
dp(a::ROCVector{Float32}, o = 0) = reinterpret(LP{Float32}, pointer(a) + 4o)
dp(p::Ptr{Float32}, o = 0) = reinterpret(LP{Float32}, p + 4o)
fp(a::ROCVector{Float32}, o = 0) = pointer(a) + 4o
wp(d::DMat) = reinterpret(LP{UInt8}, pointer(d.W))
# ---- matrix products: rocBLAS, column-major. A row-major [n][k] array is the column-major k x n matrix with leading dimension its row stride
const OPN = AMDGPU.rocBLAS.rocblas_operation_none; const OPT = AMDGPU.rocBLAS.rocblas_operation_transpose
function sgemm!(ta::Bool, tb::Bool, m, n, k, alpha, A::Ptr{Float32}, lda, B::Ptr{Float32}, ldb, beta, C::Ptr{Float32}, ldc)
    tgap(); t0 = time()
    AMDGPU.rocBLAS.rocblas_sgemm(AMDGPU.rocBLAS.handle(), ta ? OPT : OPN, tb ? OPT : OPN, Cint(m), Cint(n), Cint(k), Ref(Float32(alpha)), A, Cint(lda), B, Cint(ldb), Ref(Float32(beta)), C, Cint(ldc))
    AMDGPU.synchronize()
    c = get(TPROF, :gemm, (0, 0.0)); TPROF[:gemm] = (c[1] + 1, c[2] + time() - t0)
    TDBG[] && (println("gemm ", (ta, tb, m, n, k, lda, ldb, ldc)); flush(stdout))
end
function snrm2(n, x::Ptr{Float32})
    r = Ref{Float32}(0); AMDGPU.rocBLAS.rocblas_snrm2(AMDGPU.rocBLAS.handle(), Cint(n), x, Cint(1), r); r[]
end
sscal!(n, a, x::Ptr{Float32}) = AMDGPU.rocBLAS.rocblas_sscal(AMDGPU.rocBLAS.handle(), Cint(n), Ref(Float32(a)), x, Cint(1))
# dW[r][k] (+)= Σ_n gy[n][r] x[n][k]
dw!(D::Ptr{Float32}, R, K, gy::Ptr{Float32}, ldg, x::Ptr{Float32}, ldx, N; beta = 0) = sgemm!(false, true, K, R, N, 1, x, ldx, gy, ldg, beta, D, K)

# ---- the trained weights ----
struct TW                                                  # a matrix: W as the GEMVs read it, the FP32 master and momentum (GTT)
    W::DMat; rows::Int; K::Int; m::Ptr{Float32}; mo::Ptr{Float32}; mh::Ptr{Float32}; moh::Ptr{Float32}; name::String
end
struct TLayer
    M::Vector{TW}                                          # attention: q k v gate o w1 w3 w2; conv: in out w1 w3 w2
    V::Vector{ROCVector{Float32}}; VM::Vector{ROCVector{Float32}}; VN::Vector{String}   # the vectors, their momenta, their names
end
function pinned(nbytes)                                    # host memory the device reads and writes through the GTT: (host pointer, device pointer)
    rc = Ref{Ptr{Cvoid}}(C_NULL); H = AMDGPU.HIP
    H.hipHostMalloc(rc, nbytes, H.hipHostMallocMapped | H.hipHostMallocNonCoherent)
    d = Ref{Ptr{Cvoid}}(C_NULL); H.hipHostGetDevicePointer(d, rc[], Cuint(0))
    Ptr{Float32}(rc[]), Ptr{Float32}(d[])
end
mnames(l) = isattn(l) ? ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.gate_proj", "self_attn.o_proj", "feed_forward.w1", "feed_forward.w3", "feed_forward.w2"] :
                        ["conv.in_proj", "conv.out_proj", "feed_forward.w1", "feed_forward.w3", "feed_forward.w2"]
mslots(l) = isattn(l) ? [1, 2, 3, 4, 5, 6, 7, 8] : [1, 2, 6, 7, 8]
vnames(l) = ["operator_norm", "post_operator_norm", "ffn_norm", "post_ffn_norm", isattn(l) ? "self_attn.q_norm" : "conv.conv", isattn(l) ? "self_attn.k_norm" : ""]   # (a conv layer's 6th vector is a second copy of its taps: not trained)
struct TS                                                  # the training state
    M::Model; g::ST; L::Vector{TLayer}; om::ROCVector{Float32}   # om: the embedding norm's momentum
    E::Matrix{Float32}; ME::Matrix{Float32}; OE::Matrix{Float32}  # the embedding (NE x NV: a column a row of it), its momentum, the step
    mh::Ptr{Float32}; moh::Ptr{Float32}; nf::Int              # the masters' host pointer, the momenta's, their floats
end
function load_train(dir)
    M, g = load_engine(dir; caches = false)
    AMDGPU.unsafe_free!(M.head.W)                          # (the head runs on the host)
    nf = sum(l -> sum(j -> M.L[l+1].M[j].rows * M.L[l+1].M[j].K, mslots(l)), 0:NL-1)   # (the rows as loaded: padded)
    t0 = time(); mh, md = pinned(4nf); moh, mod = pinned(4nf)
    ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), moh, 0, 4nf)
    @printf("masters and momenta: %.2f GB each in GTT (%.0f s)\n", 4nf / 1e9, time() - t0)
    off = 0; L = TLayer[]
    for l in 0:NL-1
        D = M.L[l+1]; tw = TW[]
        for (j, nm) in zip(mslots(l), mnames(l))
            name = "model.layers.$l.$nm.weight"; rows, K = g.t[name][1]; src = tensor(g, name)
            a = unsafe_wrap(Array, mh + 4off, rows * K); Threads.@threads for i in 1:rows*K; a[i] = bf32(src[i]); end
            push!(tw, TW(D.M[j], rows, K, md + 4off, mod + 4off, mh + 4off, moh + 4off, name))
            off += D.M[j].rows * K
        end
        push!(L, TLayer(tw, D.V[1:6], [AMDGPU.zeros(Float32, length(v)) for v in D.V[1:6]], [n == "" ? "" : "model.layers.$l.$n.weight" for n in vnames(l)]))
    end
    @printf("masters loaded (%.0f s)\n", time() - t0)
    E = Matrix{Float32}(undef, NE, NV); src = tensor(g, "model.embed_tokens.weight")
    Threads.@threads for i in 1:NE*NV; E[i] = bf32(src[i]); end
    TS(M, g, L, AMDGPU.zeros(Float32, NE), E, zeros(Float32, NE, NV), Matrix{Float32}(undef, NE, NV), mh, moh, nf)
end
# ---- the buffers of a window: rows of positions, NPMAX of them ----
struct TB
    H::ROCVector{Float32}                                  # [layer][n][NE]: every layer's input, then the last layer's output
    xn::ROCVector{Float32}; o::ROCVector{Float32}; on::ROCVector{Float32}; x1::ROCVector{Float32}; fn::ROCVector{Float32}; f::ROCVector{Float32}; fo::ROCVector{Float32}
    BCx::ROCVector{Float32}; h::ROCVector{Float32}; cv::ROCVector{Float32}; y::ROCVector{Float32}
    q::ROCVector{Float32}; k::ROCVector{Float32}; v::ROCVector{Float32}; gt::ROCVector{Float32}; qn::ROCVector{Float32}; kn::ROCVector{Float32}; qr::ROCVector{Float32}; kr::ROCVector{Float32}
    P::ROCVector{Float32}; A::ROCVector{Float32}; ao::ROCVector{Float32}
    u1::ROCVector{Float32}; u3::ROCVector{Float32}; s::ROCVector{Float32}
    rs1::ROCVector{Float32}; rso::ROCVector{Float32}; rsf::ROCVector{Float32}; rsff::ROCVector{Float32}; rsq::ROCVector{Float32}; rsk::ROCVector{Float32}; rc::ROCVector{Float32}
    xp::ROCVector{Float32}                                 # the GEMV's x, BF16 pairs
    gx::ROCVector{Float32}; gy::ROCVector{Float32}; go::ROCVector{Float32}; gxn::ROCVector{Float32}; gt1::ROCVector{Float32}; gt2::ROCVector{Float32}
    gBCx::ROCVector{Float32}; gc::ROCVector{Float32}
    gq::ROCVector{Float32}; gk::ROCVector{Float32}; gv::ROCVector{Float32}; ggt::ROCVector{Float32}; gqr::ROCVector{Float32}; gkr::ROCVector{Float32}; gP::ROCVector{Float32}; gA::ROCVector{Float32}
    gu1::ROCVector{Float32}; gu3::ROCVector{Float32}; gs::ROCVector{Float32}
    dW::ROCVector{Float32}; X::ROCVector{Float32}; X2::ROCVector{Float32}; NA::ROCVector{Float32}; NB::ROCVector{Float32}; vg::ROCVector{Float32}
    Xg::ROCVector{Float32}; X2g::ROCVector{Float32}       # the embedding's chunks
end
const ECH = 8192                                           # embedding rows a chunk of its orthogonalization
function TB()
    z(n) = AMDGPU.zeros(Float32, n); NP = NPMAX; rct = zeros(Float32, NP * 2 * 64 * 2); rf = rftab()
    for p in 0:NP-1, tt in 0:1, i in 0:63
        a = Float32(p) * rf[tt*64+i+1]; rct[((p*2+tt)*64+i)*2+1] = cos(a); rct[((p*2+tt)*64+i)*2+2] = sin(a)
    end
    TB(z((NL + 1) * NP * NE),
       z(NP * NE), z(NP * NE), z(NP * NE), z(NP * NE), z(NP * NE), z(NP * NE), z(NP * NE),
       z(3NP * NE), z(NP * NE), z(NP * NE), z(NP * NE),
       z(NP * NE), z(NP * NKV * HD), z(NP * NKV * HD), z(NP * 32), z(NP * NE), z(NP * NKV * HD), z(NP * NE), z(NP * NKV * HD),
       z(NH * NP * NP), z(NP * NE), z(NP * NE),
       z(NP * NFF), z(NP * NFF), z(NP * NFF),
       z(NP), z(NP), z(NP), z(NP), z(NP * NH), z(NP * NKV), dev(rct),
       z(NP * NFF ÷ 2),
       z(NP * NE), z(NP * NE), z(NP * NE), z(NP * NE), z(NP * NE), z(NP * NE),
       z(3NP * NE), z(NP * NE),
       z(NP * NE), z(NP * NKV * HD), z(NP * NKV * HD), z(NP * 32), z(NP * NE), z(NP * NKV * HD), z(NH * NP * NP), z(NP * NE),
       z(NP * NFF), z(NP * NFF), z(NP * NFF),
       z(NFF * NE), z(NFF * NE), z(NFF * NE), z(NE * NE), z(NE * NE), z(7NE),
       z(ECH * NE), z(ECH * NE))
end
hrow(B::TB, l) = dp(B.H, l * NPMAX * NE)
# ---- the ops over a window of N positions ----
norm!(x, w, out, rs, N; D = NE, S = NE, nb = 1) = trun!(T_NORM, N; i1 = D, i2 = S, i3 = nb, a = x, b = dp(w), c = dp(out), d = dp(rs))
normb!(gy, x, w, rs, gx, N, add; D = NE, S = NE, nb = 1) = trun!(T_NORMB, N; i1 = D, i2 = S, i3 = nb, i4 = add, a = gy, b = x, c = dp(w), d = dp(rs), e = gx)
normw!(gy, x, rs, out, N; D = NE, S = NE, nb = 1) = trun!(T_NORMW, N; i1 = D, i2 = S, i3 = nb, a = gy, b = x, d = dp(rs), c = dp(out))
axpby!(a, b, c, N, D, f1, f2; S = D) = trun!(T_AXPBY, N; i1 = D, i2 = S, a = a, b = b, c = c, f1 = f1, f2 = f2)
# out[n][r] = W x[n]: x rows of K at stride S, the result's rows at stride So (its padded rows)
function matmul!(B::TB, x, K, S, W::DMat, out, N)
    trun!(T_PACK, N; i1 = K, i2 = S, a = x, c = dp(B.xp))
    trun!(T_GEMV, N; i1 = W.rows ÷ 32, i3 = W.K, i4 = W.rows, c = dp(B.xp), d = out, W = wp(W))
end
# the layer's forward from H[l] into H[l + 1], its intermediates left in the buffers
function lfwd!(S::TS, B::TB, l, N)
    T = S.L[l+1]; x = hrow(B, l); full = isfull(l)
    norm!(x, T.V[1], B.xn, B.rs1, N)
    if isattn(l)
        q, k, v, gt, o = T.M[1:5]
        trun!(T_PACK, N; i1 = NE, i2 = NE, a = dp(B.xn), c = dp(B.xp))
        for (W, out) in ((q, B.q), (k, B.k), (v, B.v), (gt, B.gt))
            trun!(T_GEMV, N; i1 = W.W.rows ÷ 32, i3 = NE, i4 = W.W.rows, c = dp(B.xp), d = dp(out), W = wp(W.W))
        end
        norm!(dp(B.q), T.V[5], B.qn, B.rsq, N; D = HD, S = NE, nb = NH); norm!(dp(B.k), T.V[6], B.kn, B.rsk, N; D = HD, S = NKV * HD, nb = NKV)
        tt = full ? 0 : 1
        trun!(T_ROT, N; i1 = NE, i2 = NH, i3 = tt, a = dp(B.qn), b = dp(B.rc), c = dp(B.qr)); trun!(T_ROT, N; i1 = NKV * HD, i2 = NKV, i3 = tt, a = dp(B.kn), b = dp(B.rc), c = dp(B.kr))
        for j in 0:NH-1                                    # S_j = scale Q_j K_j'; Sc(m, n) = Σ_c Kc(c, m) Qc(c, n)
            sgemm!(true, false, N, N, HD, SCALE, fp(B.kr, (j ÷ 4) * HD), NKV * HD, fp(B.qr, j * HD), NE, 0, fp(B.P, j * N * N), N)
        end
        trun!(T_SOFT, N; i1 = full ? 0 : WIN, a = dp(B.P))
        for j in 0:NH-1                                    # A_j = P_j V_j; Ac(c, n) = Σ_m Vc(c, m) Pc(m, n)
            sgemm!(false, false, HD, N, N, 1, fp(B.v, (j ÷ 4) * HD), NKV * HD, fp(B.P, j * N * N), N, 0, fp(B.A, j * HD), NE)
        end
        trun!(T_GATE, N; i1 = 32, a = dp(B.A), b = dp(B.gt), c = dp(B.ao))
        matmul!(B, dp(B.ao), NE, NE, o.W, dp(B.o), N)
    else
        matmul!(B, dp(B.xn), NE, NE, T.M[1].W, dp(B.BCx), N)
        trun!(T_CONV1, N; a = dp(B.BCx), b = dp(B.h))
        trun!(T_CONV2, N; a = dp(B.BCx), b = dp(B.h), d = dp(T.V[5]), c = dp(B.y), e = dp(B.cv))
        matmul!(B, dp(B.y), NE, NE, T.M[2].W, dp(B.o), N)
    end
    norm!(dp(B.o), T.V[2], B.on, B.rso, N)
    axpby!(x, dp(B.on), dp(B.x1), N, NE, 1, 1)
    norm!(dp(B.x1), T.V[3], B.fn, B.rsf, N)
    w1, w3, w2 = T.M[end-2:end]
    trun!(T_PACK, N; i1 = NE, i2 = NE, a = dp(B.fn), c = dp(B.xp))
    trun!(T_GEMV, N; i1 = NFF ÷ 32, i3 = NE, i4 = NFF, c = dp(B.xp), d = dp(B.u1), W = wp(w1.W))
    trun!(T_GEMV, N; i1 = NFF ÷ 32, i3 = NE, i4 = NFF, c = dp(B.xp), d = dp(B.u3), W = wp(w3.W))
    trun!(T_SILU, N; a = dp(B.u1), b = dp(B.u3), c = dp(B.s))
    matmul!(B, dp(B.s), NFF, NFF, w2.W, dp(B.f), N)
    norm!(dp(B.f), T.V[4], B.fo, B.rsff, N)
    axpby!(dp(B.x1), dp(B.fo), hrow(B, l + 1), N, NE, 1, 1)
end
# the gradient of a matrix into its momentum: mo = f2 mo + (1 / acc) dW; of a vector likewise
function macc!(B::TB, tw::TW, f2, acc)
    axpby!(dp(B.dW), dp(tw.mo), dp(tw.mo), tw.rows, tw.K, 1 / acc, f2)
end
vacc!(B::TB, m::ROCVector{Float32}, f2, acc) = axpby!(dp(B.vg), dp(m), dp(m), length(m), 1, 1 / acc, f2)
# gx = W' gy: gy rows of R at stride ldg (W's padded rows), the result rows of K at stride K
function matmulT!(B::TB, gy, tw::TW, gx, N)
    trun!(T_PACK, N; i1 = tw.W.rows, i2 = tw.W.rows, a = gy, c = dp(B.xp))
    trun!(T_GEMVT, N; i1 = tw.W.rows ÷ 32, i3 = tw.K, i4 = tw.K, c = dp(B.xp), d = gx, W = wp(tw.W))
end
# a matrix's gradient dW = gy' x into its momentum and gx (+)= W' gy
function mback!(B::TB, tw::TW, gy, ldg, x, ldx, gx, N, f2, acc; add = false)
    dw!(fp(B.dW), tw.rows, tw.K, gy, ldg, x, ldx, N); macc!(B, tw, f2, acc)
    if add
        matmulT!(B, dp(gy), tw, dp(B.gt2), N); axpby!(gx, dp(B.gt2), gx, N, tw.K, 1, 1)
    else
        matmulT!(B, dp(gy), tw, gx, N)
    end
end
# the layer's adjoint: gx, the gradient of H[l + 1], becomes the gradient of H[l]; every weight's gradient into its momentum
function lbwd!(S::TS, B::TB, l, N, f2, acc)
    lfwd!(S, B, l, N)
    T = S.L[l+1]; x = hrow(B, l); gx = dp(B.gx); full = isfull(l)
    w1, w3, w2 = T.M[end-2:end]
    # the FFN: H[l + 1] = x1 + post_ffn_norm(f)
    normb!(gx, dp(B.f), T.V[4], B.rsff, dp(B.gy), N, 0); normw!(gx, dp(B.f), B.rsff, B.vg, N); vacc!(B, T.VM[4], f2, acc)
    mback!(B, w2, fp(B.gy), NE, fp(B.s), NFF, dp(B.gs), N, f2, acc)
    trun!(T_SILUB, N; a = dp(B.gs), b = dp(B.u1), c = dp(B.u3), d = dp(B.gu1), e = dp(B.gu3))
    mback!(B, w1, fp(B.gu1), NFF, fp(B.fn), NE, dp(B.gxn), N, f2, acc)
    mback!(B, w3, fp(B.gu3), NFF, fp(B.fn), NE, dp(B.gxn), N, f2, acc; add = true)
    normb!(dp(B.gxn), dp(B.x1), T.V[3], B.rsf, gx, N, 1); normw!(dp(B.gxn), dp(B.x1), B.rsf, B.vg, N); vacc!(B, T.VM[3], f2, acc)
    # the operator: x1 = x + post_operator_norm(o)
    normb!(gx, dp(B.o), T.V[2], B.rso, dp(B.go), N, 0); normw!(gx, dp(B.o), B.rso, B.vg, N); vacc!(B, T.VM[2], f2, acc)
    if isattn(l)
        q, k, v, gt, o = T.M[1:5]
        mback!(B, o, fp(B.go), NE, fp(B.ao), NE, dp(B.gy), N, f2, acc)        # gy: the gated mix's gradient
        trun!(T_GATEB, N; i1 = 32, a = dp(B.gy), b = dp(B.A), c = dp(B.gt), d = dp(B.gA), e = dp(B.ggt))
        for j in 0:NH-1                                    # gP_j = gA_j V_j': gPc(m, n) = Σ_c Vc(c, m) gAc(c, n); gV_j += P_j' gA_j: gVc(c, m) = Σ_n gAc(c, n) Pc(m, n)
            sgemm!(true, false, N, N, HD, 1, fp(B.v, (j ÷ 4) * HD), NKV * HD, fp(B.gA, j * HD), NE, 0, fp(B.gP, j * N * N), N)
            sgemm!(false, true, HD, N, N, 1, fp(B.gA, j * HD), NE, fp(B.P, j * N * N), N, j % 4 == 0 ? 0 : 1, fp(B.gv, (j ÷ 4) * HD), NKV * HD)
        end
        trun!(T_SOFTB, N; a = dp(B.gP), b = dp(B.P))
        for j in 0:NH-1                                    # gQ_j = scale gS_j K_j: gQc(c, n) = Σ_m Kc(c, m) gSc(m, n); gK_j += scale gS_j' Q_j: gKc(c, m) = Σ_n Qc(c, n) gSc(m, n)
            sgemm!(false, false, HD, N, N, SCALE, fp(B.kr, (j ÷ 4) * HD), NKV * HD, fp(B.gP, j * N * N), N, 0, fp(B.gqr, j * HD), NE)
            sgemm!(false, true, HD, N, N, SCALE, fp(B.qr, j * HD), NE, fp(B.gP, j * N * N), N, j % 4 == 0 ? 0 : 1, fp(B.gkr, (j ÷ 4) * HD), NKV * HD)
        end
        tt = full ? 0 : 1
        trun!(T_ROTB, N; i1 = NE, i2 = NH, i3 = tt, a = dp(B.gqr), b = dp(B.rc), c = dp(B.gA)); trun!(T_ROTB, N; i1 = NKV * HD, i2 = NKV, i3 = tt, a = dp(B.gkr), b = dp(B.rc), c = dp(B.gt1))
        normb!(dp(B.gA), dp(B.q), T.V[5], B.rsq, dp(B.gq), N, 0; D = HD, S = NE, nb = NH); normw!(dp(B.gA), dp(B.q), B.rsq, B.vg, N; D = HD, S = NE, nb = NH); vacc!(B, T.VM[5], f2, acc)
        normb!(dp(B.gt1), dp(B.k), T.V[6], B.rsk, dp(B.gk), N, 0; D = HD, S = NKV * HD, nb = NKV); normw!(dp(B.gt1), dp(B.k), B.rsk, B.vg, N; D = HD, S = NKV * HD, nb = NKV); vacc!(B, T.VM[6], f2, acc)
        mback!(B, q, fp(B.gq), NE, fp(B.xn), NE, dp(B.gxn), N, f2, acc)
        mback!(B, k, fp(B.gk), NKV * HD, fp(B.xn), NE, dp(B.gxn), N, f2, acc; add = true)
        mback!(B, v, fp(B.gv), NKV * HD, fp(B.xn), NE, dp(B.gxn), N, f2, acc; add = true)
        mback!(B, gt, fp(B.ggt), 32, fp(B.xn), NE, dp(B.gxn), N, f2, acc; add = true)
    else
        cin, cout = T.M[1:2]
        mback!(B, cout, fp(B.go), NE, fp(B.y), NE, dp(B.gy), N, f2, acc)
        trun!(T_CONVB1, N; a = dp(B.gy), b = dp(B.BCx), c = dp(B.cv), d = dp(B.gc), e = dp(B.gBCx))
        trun!(T_CONVB2, N; a = dp(B.gc), b = dp(B.BCx), d = dp(T.V[5]), e = dp(B.gBCx))
        trun!(T_CONVW, N; a = dp(B.gc), b = dp(B.h), c = dp(B.vg)); vacc!(B, T.VM[5], f2, acc)
        mback!(B, cin, fp(B.gBCx), 3NE, fp(B.xn), NE, dp(B.gxn), N, f2, acc)
    end
    normb!(dp(B.gxn), x, T.V[1], B.rs1, gx, N, 1); normw!(dp(B.gxn), x, B.rs1, B.vg, N); vacc!(B, T.VM[1], f2, acc)
end
# the forward of a window whose tokens are in H[0]: the final hidden state, normalized, into xn
function wfwd!(S::TS, B::TB, N)
    for l in 0:NL-1; lfwd!(S, B, l, N); end
    norm!(hrow(B, NL), S.M.onrm, B.xn, B.rs1, N)
end
# the backward from gy, the gradient of the normalized final state: gx ends as the gradient of the embeddings
function wbwd!(S::TS, B::TB, N, f2, acc)
    normb!(dp(B.gy), hrow(B, NL), S.M.onrm, B.rs1, dp(B.gx), N, 0); normw!(dp(B.gy), hrow(B, NL), B.rs1, B.vg, N); vacc!(B, S.om, f2, acc)
    for l in NL-1:-1:0; lbwd!(S, B, l, N, f2, acc); end
end
# ---- the head, on the host: the loss over the positions pos (targets tgt) and the gradient of the final state ----
struct Head
    L::Matrix{Float32}; Xr::Matrix{Float32}; dX::Matrix{Float32}; xf::Matrix{Float32}; gxf::Matrix{Float32}
end
Head() = Head(Matrix{Float32}(undef, NV, NPMAX), Matrix{Float32}(undef, NE, NPMAX), Matrix{Float32}(undef, NE, NPMAX), Matrix{Float32}(undef, NE, NPMAX), zeros(Float32, NE, NPMAX))
# the loss (the mean over the positions) and its gradient into gxf; the embedding's momentum ME = f2 ME + (1 / acc) dE when grad
function head!(S::TS, Hd::Head, N, pos, tgt, f2, acc; grad = true)
    R = length(pos); L = view(Hd.L, :, 1:R); Xr = view(Hd.Xr, :, 1:R); dX = view(Hd.dX, :, 1:R)
    for (i, p) in enumerate(pos); Xr[:, i] = view(Hd.xf, :, p + 1); end
    mul!(L, transpose(S.E), Xr)
    loss = zeros(Float64, Threads.maxthreadid())
    Threads.@threads for i in 1:R
        c = view(L, :, i); mx = maximum(c); s = 0.0
        for v in 1:NV; s += exp(Float64(c[v] - mx)); end
        lse = mx + Float32(log(s)); loss[Threads.threadid()] += lse - c[tgt[i]+1]
        for v in 1:NV; c[v] = exp(c[v] - lse) / R; end
        c[tgt[i]+1] -= 1.0f0 / R
    end
    grad || return sum(loss) / R
    mul!(dX, S.E, L)
    fill!(view(Hd.gxf, :, 1:N), 0.0f0)
    for (i, p) in enumerate(pos); Hd.gxf[:, p+1] = view(dX, :, i); end
    mul!(S.ME, Xr, transpose(L), Float32(1 / acc), Float32(f2))
    sum(loss) / R
end
# ---- the optimizer: Muon for the matrices (the momentum orthogonalized by 5 Newton-Schulz steps), momentum for the vectors ----
const NSA, NSB, NSC = 3.4445f0, -4.7750f0, 2.0315f0
function muon!(B::TB, tw::TW, lr)
    R, K = tw.rows, tw.K; n = R * K; X = B.X; X2 = B.X2
    axpby!(dp(tw.mo), dp(tw.mo), dp(X), R, K, 1, 0)
    sscal!(n, 1 / (snrm2(n, fp(X)) + 1.0f-7), fp(X))
    m = min(R, K)
    for _ in 1:5
        if R <= K                                          # A = X X' (R x R)
            sgemm!(true, false, R, R, K, 1, fp(X), K, fp(X), K, 0, fp(B.NA), R)
        else                                               # A = X' X (K x K)
            sgemm!(false, true, K, K, R, 1, fp(X), K, fp(X), K, 0, fp(B.NA), K)
        end
        sgemm!(false, false, m, m, m, NSC, fp(B.NA), m, fp(B.NA), m, 0, fp(B.NB), m); axpby!(dp(B.NB), dp(B.NA), dp(B.NB), m, m, 1, NSB)
        axpby!(dp(X), dp(X), dp(X2), R, K, 1, 0)
        if R <= K                                          # X = a X + B X: Xc B
            sgemm!(false, false, K, R, R, 1, fp(X), K, fp(B.NB), R, NSA, fp(X2), K)
        else                                               # X = a X + X B: B Xc
            sgemm!(false, false, K, R, K, 1, fp(B.NB), K, fp(X), K, NSA, fp(X2), K)
        end
        X, X2 = X2, X
    end
    f = lr * sqrt(max(1.0, R / K))
    trun!(T_UPD, R; i1 = R, i2 = K, i4 = 0, f1 = f, a = dp(tw.m), b = dp(X), W = wp(tw.W))
end
# the embedding's: its momentum (NE x NV on the host) orthogonalized in chunks of rows through the card, then E -= lr O
function muon_emb!(S::TS, B::TB, lr)
    O = S.OE; copyto!(O, S.ME); BLAS.scal!(1.0f0 / (BLAS.nrm2(vec(O)) + 1.0f-7), vec(O))
    chunks = [(c0, min(ECH, NV - c0)) for c0 in 0:ECH:NV-1]
    for it in 1:5
        for (c0, cn) in chunks                             # A = Σ over the rows' chunks of X_c' X_c (NE x NE)
            copyto!(B.Xg, 1, O, c0 * NE + 1, cn * NE)
            sgemm!(false, true, NE, NE, cn, 1, fp(B.Xg), NE, fp(B.Xg), NE, c0 == 0 ? 0 : 1, fp(B.NA), NE)
        end
        sgemm!(false, false, NE, NE, NE, NSC, fp(B.NA), NE, fp(B.NA), NE, 0, fp(B.NB), NE); axpby!(dp(B.NB), dp(B.NA), dp(B.NB), NE, NE, 1, NSB)
        for (c0, cn) in chunks                             # X_c = a X_c + X_c B: B Xc
            copyto!(B.Xg, 1, O, c0 * NE + 1, cn * NE); axpby!(dp(B.Xg), dp(B.Xg), dp(B.X2g), cn, NE, 1, 0)
            sgemm!(false, false, NE, cn, NE, 1, fp(B.NB), NE, fp(B.Xg), NE, NSA, fp(B.X2g), NE)
            AMDGPU.synchronize(); copyto!(O, c0 * NE + 1, B.X2g, 1, cn * NE)
        end
    end
    BLAS.axpy!(Float32(-lr), vec(O), vec(S.E))
end
function step!(S::TS, B::TB, lr, lrv, lre)
    t0 = time()
    for T in S.L
        for tw in T.M; muon!(B, tw, lr); end
        for (v, m) in zip(T.V, T.VM)
            axpby!(dp(v), dp(m), dp(v), length(v), 1, 1, -lrv)
        end
    end
    axpby!(dp(S.M.onrm), dp(S.om), dp(S.M.onrm), NE, 1, 1, -lrv)
    AMDGPU.synchronize(); t1 = time()
    muon_emb!(S, B, lre)
    @printf("step: matrices %.1f s, embedding %.1f s\n", t1 - t0, time() - t1); flush(stdout)
end
# ---- the data: what lfm3-data wrote ----
struct Data
    tok::Vector{Int32}; tr::Vector{NTuple{5,UInt32}}; ex::Vector{NTuple{4,UInt32}}   # tr: (first token lo, hi, tokens, date, held out)
end
function Data(dir)
    tok = reinterpret(Int32, read(joinpath(dir, "tokens.bin"))) |> collect
    t = reinterpret(UInt32, read(joinpath(dir, "trans.bin"))); e = reinterpret(UInt32, read(joinpath(dir, "examples.bin")))
    Data(tok, [(t[6i+1], t[6i+2], t[6i+3], t[6i+4], t[6i+5]) for i in 0:length(t)÷6-1], [(e[4i+1], e[4i+2], e[4i+3], e[4i+4]) for i in 0:length(e)÷4-1])
end
tfirst(D::Data, t) = Int(D.tr[t+1][1]) | Int(D.tr[t+1][2]) << 32
# the windows: a reply's context starts it; the replies that end within NPMAX join it. (transcript, first token, N, the loss positions, their targets)
function windows(D::Data, held::Bool)
    W = Tuple{Int,Int,Int,Vector{Int},Vector{Int}}[]; i = 1; ne = length(D.ex)
    while i <= ne
        t, w0, _, _ = D.ex[i]; last = i
        while last + 1 <= ne && D.ex[last+1][1] == t && D.ex[last+1][4] <= w0 + NPMAX; last += 1; end
        if (D.tr[t+1][5] != 0) == held
            N = Int(D.ex[last][4] - w0); pos = Int[]; tgt = Int[]; f = tfirst(D, Int(t))
            for j in i:last
                _, _, rs, re = D.ex[j]
                for p in Int(rs)-1:Int(re)-2; push!(pos, p - Int(w0)); push!(tgt, D.tok[f+p+2]); end
            end
            push!(W, (Int(t), f + Int(w0), N, pos, tgt))
        end
        i = last + 1
    end
    W
end
# one window: the embeddings in, forward, the loss on the host; its backward when grad
function window!(S::TS, B::TB, Hd::Head, D::Data, w, f2, acc; grad = true)
    t, f0, N, pos, tgt = w
    for p in 0:N-1; Hd.xf[:, p+1] = view(S.E, :, D.tok[f0+p+1] + 1); end
    copyto!(B.H, 1, Hd.xf, 1, N * NE)
    wfwd!(S, B, N)
    AMDGPU.synchronize(); copyto!(Hd.xf, 1, B.xn, 1, N * NE)
    loss = head!(S, Hd, N, pos, tgt, f2, acc; grad)
    grad || return loss
    copyto!(B.gy, 1, Hd.gxf, 1, N * NE)
    wbwd!(S, B, N, f2, acc)
    AMDGPU.synchronize(); copyto!(Hd.gxf, 1, B.gx, 1, N * NE)
    for p in 0:N-1                                         # the embeddings' gradient: the rows of the window's tokens
        BLAS.axpy!(Float32(1 / acc), view(Hd.gxf, :, p + 1), view(S.ME, :, D.tok[f0+p+1] + 1))
    end
    loss
end
function heldout(S::TS, B::TB, Hd::Head, D::Data, W)
    t0 = time(); tot = 0.0; n = 0
    for w in W
        tot += window!(S, B, Hd, D, w, 1, 1; grad = false) * length(w[4]); n += length(w[4])
    end
    @printf("held out: %.4f over %d reply tokens in %d windows (%.0f s)\n", tot / n, n, length(W), time() - t0); flush(stdout)
    tot / n
end
# ---- BF16 safetensors of the trained weights, with the model's other files ----
function bf16(a::AbstractVector{Float32})
    o = Vector{UInt16}(undef, length(a))
    Threads.@threads for i in eachindex(a)
        u = reinterpret(UInt32, a[i]); o[i] = ((u + 0x00007fff + ((u >> 16) & 0x00000001)) >> 16) % UInt16
    end
    o
end
function export_bf16(S::TS, src, out)
    mkpath(out); AMDGPU.synchronize()
    for f in readdir(src)
        f == "model.safetensors" && continue
        endswith(f, ".bak-nobos") && continue
        startswith(f, "ref.") && continue
        cp(joinpath(src, f), joinpath(out, f); force = true)
    end
    byname = Dict{String,Any}()
    byname["model.embed_tokens.weight"] = vec(S.E); byname["model.embedding_norm.weight"] = Array(S.M.onrm)
    for (l, T) in enumerate(S.L)
        for tw in T.M; byname[tw.name] = unsafe_wrap(Array, tw.mh, tw.rows * tw.K); end
        for (i, nm) in enumerate(T.VN); nm == "" || (byname[nm] = Array(T.V[i])); end
    end
    names = sort(collect(keys(S.g.t))); hdr = IOBuffer(); off = 0; print(hdr, "{")
    for (i, nm) in enumerate(names)
        haskey(byname, nm) || error("$nm: not trained")
        shape = S.g.t[nm][1]; n = prod(shape); length(byname[nm]) == n || error("$nm: $(length(byname[nm])) values, shape $shape")
        print(hdr, i == 1 ? "" : ",", "\"", nm, "\":{\"dtype\":\"BF16\",\"shape\":[", join(shape, ","), "],\"data_offsets\":[", off, ",", off + 2n, "]}"); off += 2n
    end
    print(hdr, "}"); h = take!(hdr); pad = (8 - length(h) % 8) % 8; append!(h, fill(UInt8(' '), pad))
    open(joinpath(out, "model.safetensors"), "w") do io
        write(io, UInt64(length(h))); write(io, h)
        for nm in names; write(io, bf16(byname[nm])); end
    end
    println("wrote $out/model.safetensors"); flush(stdout)
end
# ---- a checkpoint: every FP32 master and momentum, the embedding and its momentum, the vectors and theirs, and where the epoch
# stood; written to a temporary name and renamed, so a crash leaves the last one whole. --resume reads it, repacks W from the masters
# and carries on after the window it names (the shuffles are the seed's, so the order is the same)
function ckpt_write(S::TS, path, ep, i, best)
    AMDGPU.synchronize(); t0 = time(); tmp = path * ".tmp"
    open(tmp, "w") do io
        write(io, Int64(ep), Int64(i), Float64(best))
        write(io, unsafe_wrap(Array, S.mh, S.nf)); write(io, unsafe_wrap(Array, S.moh, S.nf))
        write(io, S.E); write(io, S.ME); write(io, Array(S.M.onrm)); write(io, Array(S.om))
        for T in S.L; for v in T.V; write(io, Array(v)); end; for m in T.VM; write(io, Array(m)); end; end
    end
    mv(tmp, path; force = true)
    @printf("checkpoint: epoch %d window %d (%.0f s)\n", ep, i, time() - t0); flush(stdout)
end
function ckpt_read(S::TS, path)
    t0 = time(); ep = 0; i = 0; best = 0.0
    open(path) do io
        ep = read(io, Int64); i = read(io, Int64); best = read(io, Float64)
        read!(io, unsafe_wrap(Array, S.mh, S.nf)); read!(io, unsafe_wrap(Array, S.moh, S.nf))
        read!(io, S.E); read!(io, S.ME); up!(S.M.onrm, read!(io, Vector{Float32}(undef, NE))); up!(S.om, read!(io, Vector{Float32}(undef, NE)))
        for T in S.L; for v in T.V; up!(v, read!(io, Vector{Float32}(undef, length(v)))); end; for m in T.VM; up!(m, read!(io, Vector{Float32}(undef, length(m)))); end; end
    end
    for T in S.L, tw in T.M                                # W in BF16 from the master: the update with a step of 0
        trun!(T_UPD, tw.rows; i1 = tw.rows, i2 = tw.K, i4 = 0, f1 = 0, a = dp(tw.m), b = dp(tw.m), W = wp(tw.W))
    end
    AMDGPU.synchronize()
    @printf("resumed: epoch %d window %d, best held out %.4f (%.0f s)\n", ep, i, best, time() - t0); flush(stdout)
    Int(ep), Int(i), best
end
# ---- the gradient check: one window's gradients (momentum with f2 = 0, acc = 1) of named tensors, for test/lfm3-grad.jl ----
const CHECKN = ["model.embed_tokens.weight", "model.embedding_norm.weight", "model.layers.0.conv.in_proj.weight", "model.layers.0.conv.out_proj.weight", "model.layers.0.conv.conv.weight",
                "model.layers.0.operator_norm.weight", "model.layers.0.post_operator_norm.weight", "model.layers.0.ffn_norm.weight", "model.layers.0.post_ffn_norm.weight",
                "model.layers.0.feed_forward.w1.weight", "model.layers.0.feed_forward.w3.weight", "model.layers.0.feed_forward.w2.weight",
                "model.layers.2.self_attn.q_proj.weight", "model.layers.2.self_attn.k_proj.weight", "model.layers.2.self_attn.v_proj.weight", "model.layers.2.self_attn.gate_proj.weight",
                "model.layers.2.self_attn.o_proj.weight", "model.layers.2.self_attn.q_norm.weight", "model.layers.2.self_attn.k_norm.weight",
                "model.layers.4.self_attn.q_proj.weight", "model.layers.4.self_attn.k_norm.weight", "model.layers.23.feed_forward.w2.weight", "model.layers.23.conv.conv.weight"]
function gradcheck(S::TS, B::TB, Hd::Head, D::Data, prefix, N)
    W = windows(D, false); w = W[findfirst(w -> any(p -> p + 1 < N, w[4]), W)]; t, f0, _, pos, tgt = w
    keep = findall(p -> p + 1 < N, pos); pos = pos[keep]; tgt = tgt[keep]
    isempty(pos) && (pos = [N - 2]; tgt = [D.tok[f0+N]])
    w = (t, f0, N, pos, tgt)
    write(prefix * ".ids", Int32[D.tok[f0+p+1] for p in 0:N-1]); write(prefix * ".pos", Int32.(pos))
    loss = window!(S, B, Hd, D, w, 0, 1); AMDGPU.synchronize()
    @printf("check: N %d, %d loss positions, loss %.5f\n", N, length(w[4]), loss)
    write(prefix * ".loss", Float32(loss))
    byname = Dict{String,Any}("model.embed_tokens.weight" => vec(S.ME), "model.embedding_norm.weight" => Array(S.om))
    for T in S.L
        for tw in T.M; byname[tw.name] = unsafe_wrap(Array, tw.moh, tw.rows * tw.K); end
        for (i, nm) in enumerate(T.VN); nm == "" || (byname[nm] = Array(T.VM[i])); end
    end
    for nm in CHECKN; write(prefix * "." * nm * ".f32", Vector{Float32}(byname[nm])); end
    println("wrote $prefix.*.f32"); flush(stdout)
end
# ---- training ----
function train(args)
    arg(k, d) = (i = findfirst(==(k), args); i === nothing ? d : args[i+1])
    dir = arg("--model", ""); data = arg("--train", ""); out = arg("--out", ""); epochs = parse(Int, arg("--epochs", "2"))
    lr = parse(Float64, arg("--lr", "0.002")); lrv = parse(Float64, arg("--lrv", "0.0002")); lre = parse(Float64, arg("--lre", "0.002")); beta = parse(Float64, arg("--beta", "0.95"))
    acc = parse(Int, arg("--acc", "8")); nwin = parse(Int, arg("--windows", "0")); seed = parse(Int, arg("--seed", "1"))
    (dir == "" || data == "") && error("usage: julia --project=env/amd -t 12 lfm3-1.7b.jl --train DATADIR --model DIR --out OUTDIR [--epochs E --lr L --lrv L --lrv L --beta B --acc A --windows W --check PREFIX N --resume]")
    BLAS.set_num_threads(Sys.CPU_THREADS)
    AMDGPU.stream!(AMDGPU.HIPStream(:low))                 # (the card also runs the desktop: its gfx ring goes first)
    fence_start()
    S = load_train(dir); B = TB(); Hd = Head(); D = Data(data)
    t0 = time(); KTR[] = k_tr!(ROCBackend(), Int(WGS))
    trun!(T_AXPBY, 1; i1 = 1, i2 = 1, a = dp(B.vg), b = dp(B.vg), c = dp(B.vg), f1 = 0, f2 = 0); AMDGPU.synchronize()
    @printf("training kernel compiled (%.0f s)\n", time() - t0); flush(stdout)
    TDBG[] = "--debug" in args
    i = findfirst(==("--check"), args)
    i === nothing || return gradcheck(S, B, Hd, D, args[i+1], parse(Int, args[i+2]))
    Wt = windows(D, false); Wh = windows(D, true)
    nwin > 0 && (Wt = Wt[1:min(nwin, end)]; Wh = Wh[1:min(cld(nwin, 3), end)])
    "--repeat" in args && (Wt = fill(Wt[1], length(Wt)); Wh = Wh[1:1])   # (the same window every time: what a step does to its loss)
    @printf("%d training windows, %d held out; %d windows a step\n", length(Wt), length(Wh), acc); flush(stdout)
    ck = joinpath(out, "checkpoint.bin"); ep0 = 1; i0 = 0
    if "--resume" in args && isfile(ck)
        ep0, i0, best = ckpt_read(S, ck)
    else
        best = heldout(S, B, Hd, D, Wh)
    end
    rng = Random.MersenneTwister(seed)
    for _ in 1:ep0-1; Random.shuffle(rng, 1:length(Wt)); end
    for ep in ep0:epochs
        order = Random.shuffle(rng, 1:length(Wt)); t0 = time(); tot = 0.0; n = 0
        for (i, wi) in enumerate(order)
            ep == ep0 && i <= i0 && continue
            w = Wt[wi]; fst = (i - 1) % acc == 0
            t1 = time(); loss = window!(S, B, Hd, D, w, fst ? beta : 1.0, acc)
            tot += loss * length(w[4]); n += length(w[4])
            @printf("epoch %d window %d/%d: N %d, %d reply tokens, loss %.4f (%.1f s)\n", ep, i, length(order), w[3], length(w[4]), loss, time() - t1); flush(stdout)
            "--prof" in args && (println(join(("$k $(v[1]) $(round(v[2]; digits = 2)) s" for (k, v) in sort(collect(TPROF); by = x -> -x[2][2])), ", ")); empty!(TPROF); flush(stdout))
            (i % acc == 0 || i == length(order)) && step!(S, B, lr, lrv, lre)
            i % (acc * 32) == 0 && ckpt_write(S, ck, ep, i, best)
        end
        @printf("epoch %d: train loss %.4f over %d reply tokens (%.0f min)\n", ep, tot / n, n, (time() - t0) / 60)
        h = heldout(S, B, Hd, D, Wh)
        if h < best
            best = h; export_bf16(S, dir, out)
        else
            println("not the best ($best): no export")
        end
        ckpt_write(S, ck, ep, length(order), best); flush(stdout)
    end
end


# ---- the arena: https://nm-z.com/arena/ — 16 random conversation states from the transcripts, the model's reply in 4 panes (one
# sampling setting each, streamed from the engine), the reply nate actually gave, and Jev's similarity score for each pane
# (jev-1.13 through OpenCode Zen: 4 Score questions in one request); the best pane is highlighted. The page is four quadrants, a
# blank page folded twice; a sample's replies appear whole once scored, 2 s apart. The engine serves the model on SOCK.
#   withkey "OpenCode Zen" -- julia --project=env/amd -t 4,1 lfm3-1.7b.jl --arena --data ~/models/lfm3-data --model DIR --sock SOCK [--listen 127.0.0.1:18650]
using HTTP, JSON3
const PANES = [("A · temp 0.1 top-k 50", 0.1, 50, 1.0), ("B · temp 0.6 top-k 50", 0.6, 50, 1.0),
               ("C · temp 0.7 top-k 20 top-p 0.95", 0.7, 20, 0.95), ("D · temp 1.0 top-p 0.9", 1.0, 0, 0.9)]
const NSAMP = 16; const NGEN = 96; const IM_END = 124900
const LEVELS = ["unrelated to what nate said, or not a sentence", "nate's voice, but a different subject", "the same subject, a different point",
                "the same point in different words", "the same reply: meaning and nearly the words"]

mutable struct Pane; toks::Vector{Int}; done::Bool; err::String; end
mutable struct Sample
    i::Int; t::Int; ex::Int; held::Bool; ctx::Vector{Int}; actual::Vector{Int}; panes::Vector{Pane}
    scores::Vector{Float64}; conf::Vector{Float64}; best::Int; jerr::String; scored::Bool
end
struct Run; id::String; seed::UInt32; samples::Vector{Sample}; cond::Threads.Condition; end
const RUNS = Dict{String,Run}(); const RLK = ReentrantLock()
const CFG = Dict{String,String}()
D = nothing; VOCAB = nothing; NAMES = String[]

esc(s) = replace(s, '&' => "&amp;", '<' => "&lt;", '>' => "&gt;")
query(target) = (i = findfirst('?', target); i === nothing ? Dict{String,String}() : Dict(String(first(p)) => String(get(p, 2, "")) for p in (split(kv, '='; limit = 2) for kv in split(target[i+1:end], '&')) if !isempty(first(p))))
cut(t) = (i = findfirst(==(IM_END), t); i === nothing ? t : t[1:i-1])
text(ids) = detok(VOCAB, ids)
notify_all(R) = lock(R.cond) do; notify(R.cond); end

function pick(rng)                                             # any example whose context fits the engine comfortably
    ok = [i for (i, e) in enumerate(D.ex) if e[3] - e[2] <= 1200 && e[4] - e[3] >= 3]
    [ok[rand(rng, 1:length(ok))] for _ in 1:NSAMP]
end
function newrun(seed::UInt32)
    rng = MersenneTwister(seed); S = Sample[]
    for (k, i) in enumerate(pick(rng))
        t, cs, rs, re = Int.(D.ex[i]); f = tfirst(D, t)
        push!(S, Sample(k, t, i, D.tr[t+1][5] != 0, D.tok[f+cs+1:f+rs], D.tok[f+rs+1:f+re], [Pane(Int[], false, "") for _ in PANES],
                        fill(NaN, 4), fill(NaN, 4), 0, "", false))
    end
    R = Run(string(seed, base = 16), seed, S, Threads.Condition()); lock(RLK) do; RUNS[R.id] = R; end
    errormonitor(@async drive(R)); R
end

function generate!(R::Run, s::Sample, k::Int)                   # one pane: flashd's engine protocol on the socket
    P = s.panes[k]; (_, temp, topk, topp) = PANES[k]
    try
        c = connect(CFG["sock"])
        println(c, "stop ", IM_END); println(c, "sample ", (R.seed % 1000) * 7 + k, " ", temp, " ", topk, " ", topp)
        println(c, "chat ", NGEN, " ", join(s.ctx, ' ')); flush(c)
        while !eof(c)
            l = readline(c)
            if startswith(l, "tok ")
                t = parse(Int, l[5:end]); t == IM_END && break
                lock(R.cond) do; push!(P.toks, t); end; notify_all(R)
            elseif startswith(l, "error")
                P.err = l; break
            elseif startswith(l, "done")
                break
            end
        end
        close(c)
    catch e
        P.err = sprint(showerror, e)
    end
    lock(R.cond) do; P.done = true; end; notify_all(R)
end

function judge!(s::Sample)
    tail = text(s.ctx); tail = last(tail, 2000)
    cands = Dict("ABCD"[k] => text(cut(s.panes[k].toks)) for k in 1:4)
    state = Dict("conversation_so_far_tail" => tail, "what_nate_actually_replied" => text(cut(s.actual)), "candidate_replies" => cands)
    qs = Dict(string(c) => Dict("type" => "score", "criteria" => LEVELS,
                                "instructions" => "How close is candidate reply $c to what nate actually replied, as the next message in this conversation?") for c in "ABCD")
    body = JSON3.write(Dict("model" => "jev-1.13", "state" => state, "questions" => qs))
    try
        r = HTTP.post("https://opencode.ai/zen/v1/systemone", ["authorization" => "Bearer " * ENV["KEY_OPENCODE_ZEN"], "content-type" => "application/json"], body; request_timeout = 30, retry = false)
        a = JSON3.read(r.body)["answers"]
        for k in 1:4; x = a[string("ABCD"[k])]; s.scores[k] = x["score"]; s.conf[k] = x["confidence"]; end
        s.best = argmax([(s.scores[k], s.conf[k]) for k in 1:4])
    catch e
        s.jerr = sprint(showerror, e)
    end
    s.scored = true
end

function drive(R::Run)                                          # samples one after another; a sample's 4 panes at once
    for s in R.samples
        ts = [Threads.@spawn generate!(R, s, k) for k in 1:4]; foreach(wait, ts)
        judge!(s); notify_all(R); sleep(2)
    end
end

const PAGE = """
<!doctype html><html><head><meta charset=utf-8><title>nlfm arena</title><style>
html,body{height:100%;margin:0;background:#000;color:#fff;font:16px/1.4 ui-monospace,monospace}
body{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr}
.q{padding:16px;overflow:auto;white-space:pre-wrap;word-break:break-word;border:0 solid #fff}
#q1,#q2{border-bottom-width:1px}#q1,#q3{border-right-width:1px}
.n{color:#277DFF}.best{color:#00AE6B}
</style></head><body><div class=q id=q1></div><div class=q id=q2></div><div class=q id=q3></div><div class=q id=q4></div>
<script>
(async () => {
  const q = [1,2,3,4].map(i => document.getElementById('q' + i));
  const line = (d, cls, t) => { const p = document.createElement('div'); if (cls) p.className = cls; p.textContent = t; d.appendChild(p); d.scrollTop = d.scrollHeight; };
  const r = await fetch('events' + location.search); const rd = r.body.getReader(); const dec = new TextDecoder(); let buf = '';
  for (;;) {
    const {value, done} = await rd.read(); if (done) break;
    buf += dec.decode(value, {stream: true}); let i;
    while ((i = buf.indexOf('\\n')) >= 0) {
      const l = buf.slice(0, i); buf = buf.slice(i + 1); if (!l) continue;
      const e = JSON.parse(l);
      q.forEach((d, k) => {
        line(d, 'n', 'nate: ' + e.nate);
        line(d, e.best === k + 1 ? 'best' : '', (e.scores[k] == null ? '' : e.scores[k].toFixed(2) + ' ') + e.replies[k]);
        line(d, '', '');
      });
    }
  }
})();
</script></body></html>
"""

function page(http::HTTP.Stream)
    HTTP.setstatus(http, 200); HTTP.setheader(http, "content-type" => "text/html; charset=utf-8"); HTTP.setheader(http, "cache-control" => "no-store")
    HTTP.startwrite(http); write(http, PAGE)
end

function events(http::HTTP.Stream)                              # one JSON line a sample, once its 4 replies are scored; 2 s between samples
    q = query(http.message.target)
    seed = haskey(q, "seed") ? parse(UInt32, q["seed"]) : rand(UInt32)
    R = newrun(seed)
    HTTP.setstatus(http, 200); HTTP.setheader(http, "content-type" => "application/x-ndjson"); HTTP.setheader(http, "cache-control" => "no-store"); HTTP.setheader(http, "x-accel-buffering" => "no")
    HTTP.startwrite(http)
    for s in R.samples
        lock(R.cond) do
            while !s.scored; wait(R.cond); end
        end
        e = Dict("i" => s.i, "seed" => seed, "held" => s.held, "nate" => text(cut(s.actual)), "replies" => [text(cut(p.toks)) for p in s.panes],
                 "scores" => [isnan(x) ? nothing : x for x in s.scores], "conf" => [isnan(x) ? nothing : x for x in s.conf], "best" => s.best, "error" => s.jerr)
        write(http, JSON3.write(e) * "\n"); flush(http)
    end
end

function handler(http::HTTP.Stream)
    p = first(split(http.message.target, '?'))
    try
        if endswith(p, "/events"); events(http)
        elseif endswith(p, "/"); page(http)
        else
            HTTP.setstatus(http, 302); HTTP.setheader(http, "location" => p * "/"); HTTP.startwrite(http)
        end
    catch e
        e isa Base.IOError || e isa HTTP.HTTPError || rethrow()
    end
end

function arena(args)
    arg(k, d) = (i = findfirst(==(k), args); i === nothing ? d : args[i+1])
    CFG["sock"] = arg("--sock", "/run/user/1000/lfm3.sock"); CFG["model"] = arg("--model", expanduser("~/models/lfm3-1.7b-nate"))
    listen = arg("--listen", "127.0.0.1:18650")
    global D = Data(arg("--data", expanduser("~/models/lfm3-data"))); global VOCAB = vocabulary(CFG["model"])
    nf = joinpath(arg("--data", expanduser("~/models/lfm3-data")), "names.txt"); isfile(nf) && (global NAMES = readlines(nf))
    println("arena: $(length(D.ex)) examples, $(length(D.tr)) transcripts, listening on $listen"); flush(stdout)
    srv = HTTP.Server(; address = listen, handler = handler, stream = true); HTTP.listen!(srv); wait(srv)
end

function main(args)
    if "--arena" in args
        arena(args)
    elseif "--train" in args
        train(args)
    else
        k = findfirst(==("--model"), args); i = findfirst(==("--serve"), args)
        (k === nothing || i === nothing) && error("usage: julia --project=env/amd -t 12 lfm3-1.7b.jl --model DIR --serve SOCK | --train DATADIR --model DIR --out OUTDIR")
        serve(args[k+1], args[i+1])
    end
end
if abspath(PROGRAM_FILE) == @__FILE__() && SRV[] === nothing
    main(ARGS)
end
