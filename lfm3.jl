#!/usr/bin/env -S julia --startup-file=no -t 8
# lfm3.jl: LFM3-1.7B as one graph of methods, each written once. A request (positions, wanted, memory free) is the only input; the
# dispatch derives in the kernel the tiles, where every intermediate lives and how long, the order of the work and the waits.
#   lfm3.jl --kl N REF.logits.bf16 --model DIR          logits against the vendor's, N positions
#   lfm3.jl --train DATADIR --model DIR --out DIR ...
#   lfm3.jl --serve DIR SOCK                            flashd's protocol on a unix socket
using Mmap, Printf, Random, Sockets, LinearAlgebra, KernelAbstractions
const CUDAON = Base.find_package("CUDA") !== nothing
@static if CUDAON; using CUDA; const BE = CUDABackend(); else; using AMDGPU; const BE = ROCBackend(); end
const KI = KernelAbstractions.KernelInterface

# ---- safetensors (mapped) ----
struct ST; t::Dict{String,Tuple{Vector{Int},Int,Int}}; map::Vector{UInt8}; end
function sft(path)
    m = Mmap.mmap(path); n = Int(reinterpret(UInt64, view(m, 1:8))[1]); h = String(m[9:8+n]); t = Dict{String,Tuple{Vector{Int},Int,Int}}()
    for c in eachmatch(r"\"([^\"]+)\":\{\"dtype\":\"([A-Za-z0-9]+)\",\"shape\":\[([0-9,]*)\],\"data_offsets\":\[([0-9]+),([0-9]+)\]\}", h)
        c[2] == "BF16" || error("$(c[1]): $(c[2]), the kernel takes BF16")
        a = parse(Int, c[4]); b = parse(Int, c[5]); t[c[1]] = (parse.(Int, split(c[3], ',')), 8 + n + a, b - a)
    end
    ST(t, m)
end
tensor(g::ST, name) = (t = g.t[name]; reinterpret(UInt16, view(g.map, t[2]+1:t[2]+t[3])))
bf32(u::UInt16) = reinterpret(Float32, UInt32(u) << 16)

# ---- the model and the card ----
const NE, NFF, NV, NL, HD, NH, NKV, WIN = 2048, 7680, 128000, 24, 128, 16, 4, 256
const EPS, SCALE = 1.0f-5, Float32(128.0^-0.5)
const NSA, NSB, NSC = 3.4445f0, -4.7750f0, 2.0315f0                 # Newton-Schulz
isattn(l) = l >= 2 && iseven(l); isfull(l) = isattn(l) && (l - 2) % 4 == 0
const LP{T} = Core.LLVMPtr{T,1}
lp(::Type{T}, p) where {T} = Core.Intrinsics.bitcast(LP{T}, UInt64(p))
const LP0 = lp(UInt8, 0)
const DEVS = Ref(haskey(ENV, "LFM3_DEVS") ? parse.(Int, split(ENV["LFM3_DEVS"], ',')) : Int[])   # cards (1-based): LFM3_DEVS=3,4
ndev() = isempty(DEVS[]) ? (CUDAON ? length(CUDA.devices()) : length(AMDGPU.devices())) : length(DEVS[])
dev!(i) = (j = isempty(DEVS[]) ? i : DEVS[][i]; CUDAON ? CUDA.device!(j - 1) : AMDGPU.device!(AMDGPU.devices()[j]))
function vfree()
    CUDAON && return Int(CUDA.free_memory())
    f = Ref{Csize_t}(0); t = Ref{Csize_t}(0); AMDGPU.HIP.hipMemGetInfo(f, t); Int(f[])
end
dalloc(n) = KernelAbstractions.allocate(BE, UInt8, max(n, 256))
addr(a) = UInt64(pointer(a))
nbytes(v) = length(v) * sizeof(eltype(v)); bytes(v) = unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(pointer(v)), nbytes(v))
up!(a, off, v) = (GC.@preserve v copyto!(a, off + 1, bytes(v), 1, nbytes(v)); off + nbytes(v))
dn!(v, a, off) = (GC.@preserve v copyto!(bytes(v), 1, a, off + 1, nbytes(v)); v)
function pinned(n)                                         # host memory all cards read and write
    if CUDAON
        b = CUDA.alloc(CUDA.HostMemory, n, CUDA.MEMHOSTALLOC_DEVICEMAP | CUDA.MEMHOSTALLOC_PORTABLE)
        return Ptr{UInt8}(convert(Ptr{Cvoid}, b)), UInt64(convert(CuPtr{Cvoid}, b))
    end
    rc = Ref{Ptr{Cvoid}}(C_NULL); H = AMDGPU.HIP; H.hipHostMalloc(rc, n, H.hipHostMallocMapped | H.hipHostMallocNonCoherent)
    d = Ref{Ptr{Cvoid}}(C_NULL); H.hipHostGetDevicePointer(d, rc[], Cuint(0)); Ptr{UInt8}(rc[]), UInt64(d[])
end
sync() = KernelAbstractions.synchronize(BE)

# ---- device primitives; a BF16 is a Float32's upper half ----
@inline ld(p::LP{T}, i) where {T} = unsafe_load(p, i + 1, Val(sizeof(T)))   # (natural alignment)
@inline st!(p::LP{T}, v, i) where {T} = unsafe_store!(p, v, i + 1, Val(sizeof(T)))
@inline bfr(x::Float32) = (u = reinterpret(UInt32, x); reinterpret(Float32, (u + 0x00007fff + ((u >> 16) & 0x1)) & 0xffff0000))
@inline rdv(p::LP{UInt8}, dt::Int32, i::Int64) = dt == Int32(1) ? reinterpret(Float32, UInt32(ld(reinterpret(LP{UInt16}, p), i)) << 16) : ld(reinterpret(LP{Float32}, p), i)
@inline wrv!(p::LP{UInt8}, dt::Int32, v::Float32, i::Int64) = dt == Int32(1) ? st!(reinterpret(LP{UInt16}, p), (reinterpret(UInt32, bfr(v)) >> 16) % UInt16, i) : st!(reinterpret(LP{Float32}, p), v, i)
@inline sigm(x::Float32) = 1.0f0 / (1.0f0 + exp(-x))
const SCOPE = CUDAON ? "syncscope(\"device\") " : "syncscope(\"agent\") "
@eval @inline atomadd!(p::LP{UInt32}, v::UInt32) = Base.llvmcall($("%p = inttoptr i64 %0 to ptr addrspace(1)\n%r = atomicrmw add ptr addrspace(1) %p, i32 %1 $(SCOPE)seq_cst\nret i32 %r"), UInt32, Tuple{Int64,UInt32}, Core.Intrinsics.bitcast(Int64, p), v)
@eval @inline fadd!(p::LP{Float32}, v::Float32) = Base.llvmcall($("%p = inttoptr i64 %0 to ptr addrspace(1)\n%r = atomicrmw fadd ptr addrspace(1) %p, float %1 $(SCOPE)seq_cst\nret float %r"), Float32, Tuple{Int64,Float32}, Core.Intrinsics.bitcast(Int64, p), v)
@eval @inline atomld(p::LP{UInt32}) = Base.llvmcall($("%p = inttoptr i64 %0 to ptr addrspace(1)\n%r = load atomic i32, ptr addrspace(1) %p $(SCOPE)seq_cst, align 4\nret i32 %r"), UInt32, Tuple{Int64}, Core.Intrinsics.bitcast(Int64, p))
@inline atomldr(p::LP{UInt32}) = Base.llvmcall("%p = inttoptr i64 %0 to ptr addrspace(1)\n%r = load volatile i32, ptr addrspace(1) %p, align 4\nret i32 %r", UInt32, Tuple{Int64}, Core.Intrinsics.bitcast(Int64, p))
@eval @inline tick() = $(CUDAON ? :(Int64(CUDA.clock(UInt64))) : :(ccall("llvm.readcyclecounter", llvmcall, Int64, ())))   # (the SM clock: cycles)
@eval @inline atomst!(p::LP{UInt32}, v::UInt32) = Base.llvmcall($("%p = inttoptr i64 %0 to ptr addrspace(1)\nstore atomic i32 %1, ptr addrspace(1) %p $(SCOPE)seq_cst, align 4\nret void"), Cvoid, Tuple{Int64,UInt32}, Core.Intrinsics.bitcast(Int64, p), v)
@inline fence() = Base.llvmcall("fence seq_cst\nret void", Cvoid, Tuple{})
@inline bar() = KI.barrier()

# ---- the graph: nodes (a method each), edges (what flows between them), mats and vecs ----
const EMB, NORM, GEMM, CONV, ROPE, ATTN, RESID, LSE, DLOG, NORMB, SILUB, ATTBKV, ATTBQ, ROTNB, CONVB1, CONVB2, CONVW, NORMW, EMBB, NRM, STEP, STEPV, NRMS, STAGE, MOMU = Int32.(1:25)
const NG = 16                                              # the attention gate's rows
const NQKV = NE + 2NKV * HD + NG                           # q | k | v | gate
const ALL, LOSS, WT = Int32(1), Int32(2), Int32(3)        # rows: all positions, loss positions, a weight's shape
const SAME, SPREV, UPTO, SNEXT, FROM, DALL = Int32.(0:5)  # which tiles of its inputs a tile reads
const NF, EF, MF, VF = 26, 8, 8, 4
const N_KIND, N_L, N_ROWS, N_DEP, N_CARD, N_LVL, N_I1, N_I2, N_I3, N_I4, N_O1, N_O2, N_M, N_NM, N_FL, N_EPI, N_RS, N_WV, N_D, N_K, N_NO, N_WIN, N_TBL, N_WR, N_O3, N_X = Int32.(0:25)
const E_W, E_DT, E_ROWS, E_R, E_CACHE, E_REACH, E_PROD, E_CROSS = Int32.(0:7)   # cross: read on another card than its producer's
const A_ROW, A_TR, B_MAT, B_TR, B_ROWK = Int32(1), Int32(2), Int32(4), Int32(8), Int32(16)
const EP_ST, EP_SILU, EP_ADD, EP_MOM, EP_AXPY, EP_LSE = Int32.(0:5)
const BF, F32 = Int32(1), Int32(2)
const LOGITS, LOSSW, GRAD, WEIGHTS = Int32.(1:4)          # a node's level; wanted: 1 logits 2 loss 4 grad 8 weights
@inline lvlmask(l) = l == LOGITS ? Int32(7) : l == LOSSW ? Int32(6) : l == GRAD ? Int32(4) : Int32(8)
mutable struct Graph
    node::Vector{Int32}; nodef::Vector{Float32}; edge::Vector{Int32}; nn::Int; ne::Int
    names::Dict{Symbol,Int}                                # an edge by (layer, name)
    nmat::Int; nvec::Int; matname::Vector{String}; vecname::Vector{String}
end
Graph() = Graph(Int32[], Float32[], Int32[], 0, 0, Dict{Symbol,Int}(), 0, 0, String[], String[])
function edge!(G::Graph, name, w, dt; rows = ALL, r = 0, cache = 0, reach = 0)
    G.ne += 1; append!(G.edge, Int32[w, dt, rows, r, cache, reach, 0, 0]); G.names[name] = G.ne; G.ne
end
function node!(G::Graph, kind, l, lvl; rows = ALL, dep = SAME, in = (0, 0, 0, 0), out = (0, 0, 0), m = 0, nm = 0, fl = 0, epi = EP_ST, rs = 0, wv = 0, D = 0, K = 0, no = 0, win = 0, tbl = 0, wr = 0, f = (0.0, 0.0, 0.0, 0.0))
    G.nn += 1; n = G.nn
    append!(G.node, Int32[kind, l, rows, dep, 0, lvl, in[1], in[2], in[3], in[4], out[1], out[2], m, nm, fl, epi, rs, wv, D, K, no, win, tbl, wr, length(out) > 2 ? out[3] : 0, 0]); append!(G.nodef, Float32.(f))
    for e in out; e == 0 || (G.edge[EF*(e-1)+E_PROD+1] = n); end
    n
end
mat!(G::Graph, name) = (G.nmat += 1; push!(G.matname, name); G.nmat)
vec!(G::Graph, name) = (G.nvec += 1; push!(G.vecname, name); G.nvec)
E(G, l, s) = G.names[Symbol(l, s)]
# Muon: X = mo / |mo|; 5 x (A = X X' or X' X; B = b A + c A A; X = a X + B X or X B); lr f X; W = bf16(master)
function muon!(G::Graph, m, R, K, lr, gd)
    X = edge!(G, Symbol(:X, m), K, F32; rows = WT, r = R); q = min(R, K); s = edge!(G, Symbol(:s, m), 1, F32; rows = WT, r = 1)
    sp = edge!(G, Symbol(:sp, m), 1, F32; rows = WT, r = cld(R, 8))
    node!(G, NRM, 0, WEIGHTS; rows = WT, dep = DALL, in = (gd, 0, 0, 0), m = m, out = (sp, 0), wr = R, K = K); node!(G, NRMS, 0, WEIGHTS; rows = WT, dep = DALL, in = (sp, 0, 0, 0), m = m, out = (s, 0), wr = 1)
    node!(G, STAGE, 0, WEIGHTS; rows = WT, dep = DALL, in = (s, 0, 0, 0), out = (X, 0), m = m, rs = s, wr = R, K = K)
    a = X                                                  # (every edge written once)
    for it in 1:5
        A = edge!(G, Symbol(:A, m, it), q, F32; rows = WT, r = q); Bm = edge!(G, Symbol(:B, m, it), q, F32; rows = WT, r = q); xo = edge!(G, Symbol(:X, m, it), K, F32; rows = WT, r = R)
        if R <= K; node!(G, GEMM, 0, WEIGHTS; rows = WT, dep = DALL, in = (a, a, 0, 0), out = (A, 0), m = m, rs = s, wr = R, K = K, no = R)   # A = X X'
        else; node!(G, GEMM, 0, WEIGHTS; rows = WT, dep = DALL, in = (a, a, 0, 0), out = (A, 0), m = m, fl = A_TR | B_TR, rs = s, wr = K, K = R, no = K); end   # A = X' X
        node!(G, GEMM, 0, WEIGHTS; rows = WT, dep = DALL, in = (A, A, A, 0), out = (Bm, 0), rs = s, epi = EP_AXPY, wr = q, K = q, no = q, f = (NSC, NSB, 0, 0))
        if R <= K; node!(G, GEMM, 0, WEIGHTS; rows = WT, dep = DALL, in = (Bm, a, a, 0), out = (xo, 0), m = m, fl = B_TR, rs = s, epi = EP_AXPY, wr = R, K = R, no = K, f = (1, NSA, 0, 0))   # X = a X + B X
        else; node!(G, GEMM, 0, WEIGHTS; rows = WT, dep = DALL, in = (a, Bm, a, 0), out = (xo, 0), m = m, rs = s, epi = EP_AXPY, wr = R, K = K, no = K, f = (1, NSA, 0, 0)); end   # X = a X + X B
        a = xo
    end
    node!(G, STEP, 0, WEIGHTS; rows = WT, dep = DALL, in = (a, 0, 0, 0), m = m, rs = s, wr = R, K = K, f = (lr * sqrt(max(1.0, R / K)), 0, 0, 0))
end
done!(G::Graph, name) = edge!(G, name, 1, F32; rows = WT, r = 1)   
# the recipe
function recipe(; lr = 0.002, lrv = 0.0002, lre = 0.002)
    G = Graph(); me = mat!(G, "model.embed_tokens.weight"); onv = vec!(G, "model.embedding_norm.weight"); gdm = Dict{Int,Int}(); gdv = Dict{Int,Int}()
    dmat(m) = (gdm[m] = done!(G, Symbol(:gd, m))); dvec(v) = (gdv[v] = done!(G, Symbol(:gv, v)))
    function gmat!(l, m, wr, no; kw...)                     # dW = A' B, then mo = f2 mo + dW / acc
        dW = edge!(G, Symbol(:dW, m), no, F32; rows = WT, r = wr); node!(G, GEMM, l, GRAD; rows = WT, dep = DALL, out = (dW, 0), m = m, K = -1, no = no, wr = wr, kw...)
        node!(G, MOMU, l, GRAD; rows = WT, dep = DALL, in = (dW, 0, 0, 0), out = (dmat(m), 0), m = m, wr = wr, K = no)
    end
    x = edge!(G, :x0, NE, F32); node!(G, EMB, 0, LOGITS; m = me, out = (x, 0))
    gx = 0; bwd = Tuple{Int,Vector{Any}}[]
    for l in 0:NL-1
        L(s) = Symbol(l, s); at = isattn(l); mats = Dict{String,Int}(); vecs = Dict{String,Int}()
        for nm in (at ? ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.gate_proj", "self_attn.o_proj"] : ["conv.in_proj", "conv.out_proj"]) ∪ ["feed_forward.w1", "feed_forward.w3", "feed_forward.w2"]
            mats[nm] = mat!(G, "model.layers.$l.$nm.weight")
        end
        for nm in ["operator_norm", "post_operator_norm", "ffn_norm", "post_ffn_norm"] ∪ (at ? ["self_attn.q_norm", "self_attn.k_norm"] : ["conv.conv"])
            vecs[nm] = vec!(G, "model.layers.$l.$nm.weight")
        end
        rs1 = edge!(G, L(:rs1), 1, F32); xa = edge!(G, L(:xa), NE, F32); n1 = node!(G, NORM, l, LOGITS; in = (x, 0, 0, 0), out = (rs1, xa), wv = vecs["operator_norm"], D = NE, K = NE)
        o = edge!(G, L(:o), NE, BF)
        if at
            qkv = edge!(G, L(:qkv), NQKV, BF)   # q | k | v | gate: one matrix product
            nq = node!(G, GEMM, l, LOGITS; in = (xa, 0, 0, 0), out = (qkv, 0), m = mats["self_attn.q_proj"], nm = 4, fl = A_ROW | B_MAT, K = NE, no = NQKV)
            rsq = edge!(G, L(:rsq), NH, F32); rsk = edge!(G, L(:rsk), NKV, F32)
            node!(G, NORM, l, LOGITS; in = (qkv, 0, 0, 0), out = (rsq, 0), wv = vecs["self_attn.q_norm"], D = HD, K = NE)
            node!(G, NORM, l, LOGITS; in = (qkv, 0, 0, 0), out = (rsk, 0), wv = vecs["self_attn.k_norm"], D = HD, K = NKV * HD, win = NE)   # win: the offset into qkv
            qr = edge!(G, L(:qr), NE, BF); kr = edge!(G, L(:kr), NKV * HD, BF; cache = 1, reach = isfull(l) ? 0 : WIN); v = edge!(G, L(:v), NKV * HD, BF; cache = 1, reach = isfull(l) ? 0 : WIN)
            node!(G, ROPE, l, LOGITS; in = (qkv, rsq, rsk, 0), out = (qr, kr, v), wv = vecs["self_attn.q_norm"], D = vecs["self_attn.k_norm"], tbl = isfull(l) ? 0 : 1)
            ao = edge!(G, L(:ao), NE, BF); lse = edge!(G, L(:lse), NH, F32)
            node!(G, ATTN, l, LOGITS; dep = UPTO, in = (qr, kr, v, qkv), out = (ao, lse), win = isfull(l) ? 0 : WIN)
            node!(G, GEMM, l, LOGITS; in = (ao, 0, 0, 0), out = (o, 0), m = mats["self_attn.o_proj"], nm = 1, fl = A_ROW | B_MAT, K = NE, no = NE)
        else
            bcx = edge!(G, L(:bcx), 3NE, BF); h = edge!(G, L(:h), NE, BF; cache = 1, reach = 7); y = edge!(G, L(:y), NE, BF); cv = edge!(G, L(:cv), NE, BF)
            node!(G, GEMM, l, LOGITS; in = (xa, 0, 0, 0), out = (bcx, 0), m = mats["conv.in_proj"], nm = 1, fl = A_ROW | B_MAT, K = NE, no = 3NE)
            node!(G, CONV, l, LOGITS; in = (bcx, 0, 0, 0), out = (h, 0))               # h = B x into the ring
            node!(G, CONV, l, LOGITS; dep = SPREV, in = (bcx, h, 0, 0), out = (y, cv), wv = vecs["conv.conv"], win = 1)   # y = C conv(h)
            node!(G, GEMM, l, LOGITS; in = (y, 0, 0, 0), out = (o, 0), m = mats["conv.out_proj"], nm = 1, fl = A_ROW | B_MAT, K = NE, no = NE)
        end
        rso = edge!(G, L(:rso), 1, F32); node!(G, NORM, l, LOGITS; in = (o, 0, 0, 0), out = (rso, 0), wv = vecs["post_operator_norm"], D = NE, K = NE)
        x1 = edge!(G, L(:x1), NE, F32); node!(G, RESID, l, LOGITS; in = (x, o, rso, 0), out = (x1, 0), wv = vecs["post_operator_norm"])
        rsf = edge!(G, L(:rsf), 1, F32); xb = edge!(G, L(:xb), NE, F32); node!(G, NORM, l, LOGITS; in = (x1, 0, 0, 0), out = (rsf, xb), wv = vecs["ffn_norm"], D = NE, K = NE)
        u = edge!(G, L(:u), 2NFF, BF); s = edge!(G, L(:s), NFF, BF)
        node!(G, GEMM, l, LOGITS; in = (xb, 0, 0, 0), out = (u, s), m = mats["feed_forward.w1"], nm = 2, fl = A_ROW | B_MAT, K = NE, no = NFF, epi = EP_SILU)
        f = edge!(G, L(:f), NE, BF); node!(G, GEMM, l, LOGITS; in = (s, 0, 0, 0), out = (f, 0), m = mats["feed_forward.w2"], nm = 1, fl = A_ROW | B_MAT, K = NFF, no = NE)
        rsff = edge!(G, L(:rsff), 1, F32); node!(G, NORM, l, LOGITS; in = (f, 0, 0, 0), out = (rsff, 0), wv = vecs["post_ffn_norm"], D = NE, K = NE)
        xn = edge!(G, Symbol(l + 1, :x0), NE, F32); node!(G, RESID, l, LOGITS; in = (x1, f, rsff, 0), out = (xn, 0), wv = vecs["post_ffn_norm"])
        push!(bwd, (l, Any[x, x1, xn, mats, vecs, rs1, rso, rsf, rsff, o, f, u, s, xa, xb, at ? (qkv, rsq, rsk, qr, kr, v, ao, lse) : (bcx, h, y, cv)]))
        x = xn
    end
    # the head over the loss positions
    rsh = edge!(G, :rsh, 1, F32; rows = LOSS); xh = edge!(G, :xh, NE, F32; rows = LOSS); node!(G, NORM, NL, LOGITS; rows = LOSS, in = (x, 0, 0, 0), out = (rsh, xh), wv = onv, D = NE, K = NE)
    lg = edge!(G, :lg, NV, F32; rows = LOSS); lpart = edge!(G, :lpart, 2 * cld(NV, 16), F32; rows = LOSS)
    node!(G, GEMM, NL, LOGITS; rows = LOSS, in = (xh, 0, 0, 0), out = (lg, lpart), m = me, nm = 1, fl = A_ROW | B_MAT, K = NE, no = NV, epi = EP_LSE)
    lsev = edge!(G, :lse, 2, F32; rows = LOSS); node!(G, LSE, NL, LOSSW; rows = LOSS, in = (lpart, lg, 0, 0), out = (lsev, 0))
    # the adjoint: dlg, dxn = dlg E, dE = dlg' xn
    dlg = edge!(G, :dlg, NV, F32; rows = LOSS); node!(G, DLOG, NL, GRAD; rows = LOSS, in = (lg, lsev, 0, 0), out = (dlg, 0))
    dxh = edge!(G, :dxh, NE, F32; rows = LOSS)
    node!(G, GEMM, NL, GRAD; rows = LOSS, dep = DALL, in = (dlg, 0, 0, 0), out = (dxh, 0), m = me, nm = 1, fl = A_ROW | B_MAT | B_TR, K = NV, no = NE)
    gmat!(NL, me, NV, NE; in = (dlg, xh, 0, 0), fl = A_ROW | A_TR | B_ROWK)
    gx = edge!(G, :gx, NE, F32); node!(G, NORMB, NL, GRAD; in = (dxh, x, rsh, 0), out = (gx, 0), wv = onv, D = NE, K = NE, win = 1)   # (win: from loss rows into every row)
    node!(G, NORMW, NL, GRAD; rows = WT, dep = DALL, in = (dxh, x, rsh, 0), out = (dvec(onv), 0), wv = onv, D = NE, K = NE, wr = NE, win = 1)
    for (l, (x, x1, xn, mats, vecs, rs1, rso, rsf, rsff, o, f, u, s, xa, xb, rest)) in reverse(bwd)
        L(s) = Symbol(l, s); at = isattn(l); M(nm) = mats[nm]; V(nm) = vecs[nm]
        gf = edge!(G, L(:gf), NE, BF); node!(G, NORMB, l, GRAD; in = (gx, f, rsff, 0), out = (gf, 0), wv = V("post_ffn_norm"), D = NE, K = NE)
        node!(G, NORMW, l, GRAD; rows = WT, dep = DALL, in = (gx, f, rsff, 0), out = (dvec(V("post_ffn_norm")), 0), wv = V("post_ffn_norm"), D = NE, K = NE, wr = NE)
        gs = edge!(G, L(:gs), NFF, BF); node!(G, GEMM, l, GRAD; in = (gf, 0, 0, 0), out = (gs, 0), m = M("feed_forward.w2"), nm = 1, fl = A_ROW | B_MAT | B_TR, K = NE, no = NFF)
        gmat!(l, M("feed_forward.w2"), NE, NFF; in = (gf, s, 0, 0), fl = A_ROW | A_TR | B_ROWK)
        gu = edge!(G, L(:gu), 2NFF, BF); node!(G, SILUB, l, GRAD; in = (gs, u, 0, 0), out = (gu, 0))
        gxn = edge!(G, L(:gxn), NE, BF)
        node!(G, GEMM, l, GRAD; in = (gu, 0, 0, 0), out = (gxn, 0), m = M("feed_forward.w1"), nm = 2, fl = A_ROW | B_MAT | B_TR, K = 2NFF, no = NE)   # gxn = W1' gu1 + W3' gu3
        gmat!(l, M("feed_forward.w1"), NFF, NE; in = (gu, xb, 0, 0), fl = A_ROW | A_TR | B_ROWK)
        gmat!(l, M("feed_forward.w3"), NFF, NE; in = (gu, xb, 0, 0), fl = A_ROW | A_TR | B_ROWK, win = NFF)
        gx1 = edge!(G, L(:gx1), NE, F32); node!(G, NORMB, l, GRAD; in = (gxn, x1, rsf, gx), out = (gx1, 0), wv = V("ffn_norm"), D = NE, K = NE)   # gx1 = gx + ffn_norm'(gxn)
        node!(G, NORMW, l, GRAD; rows = WT, dep = DALL, in = (gxn, x1, rsf, 0), out = (dvec(V("ffn_norm")), 0), wv = V("ffn_norm"), D = NE, K = NE, wr = NE)
        go = edge!(G, L(:go), NE, BF); node!(G, NORMB, l, GRAD; in = (gx1, o, rso, 0), out = (go, 0), wv = V("post_operator_norm"), D = NE, K = NE)
        node!(G, NORMW, l, GRAD; rows = WT, dep = DALL, in = (gx1, o, rso, 0), out = (dvec(V("post_operator_norm")), 0), wv = V("post_operator_norm"), D = NE, K = NE, wr = NE)
        gxn2 = edge!(G, L(:gxn2), NE, BF)
        if at
            qkv, rsq, rsk, qr, kr, v, ao, lse = rest
            gao = edge!(G, L(:gao), NE, BF); node!(G, GEMM, l, GRAD; in = (go, 0, 0, 0), out = (gao, 0), m = M("self_attn.o_proj"), nm = 1, fl = A_ROW | B_MAT | B_TR, K = NE, no = NE)
            gmat!(l, M("self_attn.o_proj"), NE, NE; in = (go, ao, 0, 0), fl = A_ROW | A_TR | B_ROWK)
            gkv = edge!(G, L(:gkv), 2NKV * HD, F32); gqg = edge!(G, L(:gqg), NE + NG, F32)
            node!(G, ATTBKV, l, GRAD; dep = FROM, in = (qr, kr, v, qkv), out = (gkv, 0), m = ao, nm = gao, D = lse, win = isfull(l) ? 0 : WIN)
            node!(G, ATTBQ, l, GRAD; dep = UPTO, in = (qr, kr, v, qkv), out = (gqg, 0), m = ao, nm = gao, D = lse, win = isfull(l) ? 0 : WIN)
            gqkv = edge!(G, L(:gqkv), NQKV, BF); gqn = edge!(G, L(:gqn), NE + NKV * HD, BF)
            node!(G, ROTNB, l, GRAD; in = (gqg, gkv, qkv, rsq), out = (gqkv, gqn), m = rsk, wv = V("self_attn.q_norm"), D = V("self_attn.k_norm"), tbl = isfull(l) ? 0 : 1)
            node!(G, NORMW, l, GRAD; rows = WT, dep = DALL, in = (gqn, qkv, rsq, 0), out = (dvec(V("self_attn.q_norm")), 0), wv = V("self_attn.q_norm"), D = HD, K = NE, wr = HD)
            node!(G, NORMW, l, GRAD; rows = WT, dep = DALL, in = (gqn, qkv, rsk, 0), out = (dvec(V("self_attn.k_norm")), 0), wv = V("self_attn.k_norm"), D = HD, K = NKV * HD, wr = HD, tbl = NE)
            node!(G, GEMM, l, GRAD; in = (gqkv, 0, 0, 0), out = (gxn2, 0), m = M("self_attn.q_proj"), nm = 4, fl = A_ROW | B_MAT | B_TR, K = NQKV, no = NE)
            for (i, nm) in enumerate(["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.gate_proj"])
                off = [0, NE, NE + NKV * HD, NE + 2NKV * HD][i]; rows = [NE, NKV * HD, NKV * HD, NG][i]
                gmat!(l, M(nm), rows, NE; in = (gqkv, xa, 0, 0), fl = A_ROW | A_TR | B_ROWK, win = off)
            end
        else
            bcx, h, y, cv = rest
            gy = edge!(G, L(:gy), NE, BF); node!(G, GEMM, l, GRAD; in = (go, 0, 0, 0), out = (gy, 0), m = M("conv.out_proj"), nm = 1, fl = A_ROW | B_MAT | B_TR, K = NE, no = NE)
            gmat!(l, M("conv.out_proj"), NE, NE; in = (go, y, 0, 0), fl = A_ROW | A_TR | B_ROWK)
            gc = edge!(G, L(:gc), NE, BF); gbcx = edge!(G, L(:gbcx), 3NE, BF)
            node!(G, CONVB1, l, GRAD; in = (gy, bcx, cv, 0), out = (gc, gbcx))
            node!(G, CONVB2, l, GRAD; dep = SNEXT, in = (gc, bcx, 0, 0), out = (gbcx, 0), wv = V("conv.conv"))
            node!(G, CONVW, l, GRAD; rows = WT, dep = DALL, in = (gc, h, 0, 0), out = (dvec(V("conv.conv")), 0), wv = V("conv.conv"), wr = NE)
            node!(G, GEMM, l, GRAD; in = (gbcx, 0, 0, 0), out = (gxn2, 0), m = M("conv.in_proj"), nm = 1, fl = A_ROW | B_MAT | B_TR, K = 3NE, no = NE)
            gmat!(l, M("conv.in_proj"), 3NE, NE; in = (gbcx, xa, 0, 0), fl = A_ROW | A_TR | B_ROWK)
        end
        gx0 = edge!(G, L(:gx0), NE, F32); node!(G, NORMB, l, GRAD; in = (gxn2, x, rs1, gx1), out = (gx0, 0), wv = V("operator_norm"), D = NE, K = NE)
        node!(G, NORMW, l, GRAD; rows = WT, dep = DALL, in = (gxn2, x, rs1, 0), out = (dvec(V("operator_norm")), 0), wv = V("operator_norm"), D = NE, K = NE, wr = NE)
        gx = gx0
    end
    node!(G, EMBB, 0, GRAD; rows = WT, dep = DALL, in = (gx, gdm[me], 0, 0), out = (dmat(me), 0), m = me, wr = 0)
    # the update: Muon, momentum for the rest
    for m in 1:G.nmat
        R, K = matdims(G.matname[m])
        m == me ? node!(G, STEP, 0, WEIGHTS; rows = WT, dep = DALL, in = (0, gdm[m], 0, 0), m = m, wr = R, K = K, f = (lre, 0, 0, 0)) : muon!(G, m, R, K, lr, gdm[m])
    end
    for v in 1:G.nvec; node!(G, STEPV, 0, WEIGHTS; rows = WT, dep = DALL, in = (gdv[v], 0, 0, 0), wv = v, wr = 1, f = (lrv, 0, 0, 0)); end
    G
end
function matdims(name)
    occursin("embed", name) && return (NV, NE)
    occursin("q_proj", name) && return (NE, NE); occursin("k_proj", name) && return (NKV * HD, NE); occursin("v_proj", name) && return (NKV * HD, NE)
    occursin("gate_proj", name) && return (NG, NE); occursin("o_proj", name) && return (NE, NE)
    occursin("in_proj", name) && return (3NE, NE); occursin("out_proj", name) && return (NE, NE)
    occursin("w1", name) && return (NFF, NE); occursin("w3", name) && return (NFF, NE); occursin("w2", name) && return (NE, NFF)
    error(name)
end
veclen(name) = occursin("conv.conv", name) ? 7NE : NE

# ---- the request, all the kernel sees: rows [P][4] segment, position, token, target ----
struct RQ
    P::Int32; PL::Int32; NU::Int32; NS::Int32; wanted::Int32; nblk::Int32; card::Int32; ncard::Int32; gen::UInt32; nn::Int32; ne::Int32; cap::Int32
    f2::Float32; iacc::Float32; lossR::Float32
    rows::LP{Int32}; lrows::LP{Int32}; linv::LP{Int32}; wgt::LP{Float32}; uniq::LP{Int32}; urows::LP{Int32}; segs::LP{Int32}
    node::LP{Int32}; nodef::LP{Float32}; edge::LP{Int32}; mat::LP{Int64}; vec::LP{Int64}; rope::LP{Float32}
    plan::LP{Int64}; cnt::LP{UInt32}; cntH::LP{UInt32}; q::LP{UInt32}
    aV::LP{UInt8}; nV::Int64; aH::LP{UInt8}; nH::Int64; slots::LP{UInt8}; slotsz::Int64; dbg::Int32; nflag::Int64
end
# the plan: a header; per edge; per node; per segment; the holes
const PHN, PL_LOC, PL_OFF, PL_MOD, PL_LAST, PL_CA, PL_CB, PL_SZ = 16, 0, 1, 2, 3, 4, 5, 6
const TILEA = 256 * 32   # a tile's accumulators (smem: a 132 stride a slice row; line tables from 8448)
const NP_F0, NP_NT, NP_NC, NP_RT, NP_CT, NP_T0, NP_ON, NP_FL, NP_WAR = 0, 1, 2, 3, 4, 5, 6, 7, 8   # (WAR: the last reader of reused memory)
const PH_NT, PH_NFLAG, PH_STAT, PH_ORDER, PH_S0, PH_S1, PH_SECT, PH_T0S, PH_T1S, PH_VUSE, PH_HUSE, PH_SLOT = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
const NSEG, NHOLE = 64, 512
@inline nd(R::RQ, n, f) = ld(R.node, Int64(n - 1) * NF + f)
@inline ndf(R::RQ, n, i) = ld(R.nodef, Int64(n - 1) * 4 + i)
@inline ed(R::RQ, e, f) = ld(R.edge, Int64(e - 1) * EF + f)
@inline mt(R::RQ, m, f) = ld(R.mat, Int64(m - 1) * MF + f)
@inline vp(R::RQ, v) = lp(Float32, ld(R.vec, Int64(v - 1) * VF)); @inline vmo(R::RQ, v) = lp(Float32, ld(R.vec, Int64(v - 1) * VF + 2))
@inline rw(R::RQ, r, f) = ld(R.rows, Int64(r) * 4 + f)
@inline pl(R::RQ, i) = ld(R.plan, Int64(i)); @inline pl!(R::RQ, i, v) = st!(R.plan, Int64(v), Int64(i))
@inline PE(R::RQ, e, f) = PHN + 8 * (Int64(e) - 1) + f; @inline PN(R::RQ, n, f) = PHN + 8 * Int64(R.ne) + 9 * (Int64(n) - 1) + f
@inline PS(R::RQ, s, f) = PHN + 8 * Int64(R.ne) + 9 * Int64(R.nn) + 2 * Int64(s) + f; @inline PO(R::RQ, h, f) = PS(R, NSEG, 0) + 4 * Int64(h) + f
@inline ple(R::RQ, e, f) = pl(R, PE(R, e, f)); @inline pln(R::RQ, n, f) = pl(R, PN(R, n, f)); @inline pls(R::RQ, s, f) = pl(R, PS(R, s, f))
@inline ebytes(R::RQ, e) = Int64(ed(R, e, E_W)) * (ed(R, e, E_DT) == BF ? 2 : 4)
@inline ebase(R::RQ, e) = (ple(R, e, PL_LOC) == 1 ? R.aV : R.aH) + ple(R, e, PL_OFF)
@inline caddr(R::RQ, e, s, pos) = lp(UInt8, pls(R, s, 0)) + ple(R, e, PL_CA) * pls(R, s, 1) + ple(R, e, PL_CB) + (Int64(pos) & ple(R, e, PL_MOD)) * ebytes(R, e)
@inline function eaddr(R::RQ, e, r)                        # row r (a request row) of edge e
    ed(R, e, E_CACHE) != 0 && return caddr(R, e, rw(R, r, 0), rw(R, r, 1))
    i = ed(R, e, E_ROWS) == LOSS ? ld(R.linv, r) : r
    ebase(R, e) + (Int64(i) & ple(R, e, PL_MOD)) * ebytes(R, e)
end
@inline waddr(R::RQ, e, m) = ebase(R, e) + Int64(m) * ebytes(R, e)   # row m of a weight-shaped edge
@inline rowof(R::RQ, kind, i) = kind == LOSS ? ld(R.lrows, i) : Int32(i)   # a node's i-th row as a request row
@inline function nin(R::RQ, n, i)                        # the i-th edge node n reads (0..7), 0 for none
    i < 4 && return nd(R, n, N_I1 + i)
    i == 4 && return nd(R, n, N_RS)
    k = nd(R, n, N_KIND); att = k == ATTBKV || k == ATTBQ
    i == 5 && return (att || k == ROTNB) ? nd(R, n, N_M) : Int32(0)
    i == 6 && return att ? nd(R, n, N_NM) : Int32(0)
    att ? nd(R, n, N_D) : Int32(0)
end
@inline nrows(R::RQ, n) = (k = nd(R, n, N_ROWS); k == ALL ? R.P : k == LOSS ? R.PL : (nd(R, n, N_KIND) == EMBB ? R.NU : nd(R, n, N_WR)))
@inline cellp(R::RQ, p, t, c) = (pln(R, p, NP_FL) == 1 ? R.cnt : R.cntH) + 4 * (pln(R, p, NP_F0) + Int64(t) * pln(R, p, NP_NC) + c)
@inline function waittile(R::RQ, p, t)
    nc = pln(R, p, NP_NC); c = Int64(0); i = Int64(0)
    while c < nc                                           # (status 3 past 2^24 polls)
        atomldr(cellp(R, p, t, c)) == R.gen && (c += 1); i += 1
        if i == Int64(1) << 24; pl!(R, PH_STAT, 3); pl!(R, PH_S0, p); pl!(R, PH_S1, t + (c << 20) + (Int64(atomld(cellp(R, p, t, c))) << 40)); return; end
    end
    fence()
end
@inline function pow2ge(x::Int64); p = Int64(1); while p < x; p <<= 1; end; p; end
# the derivation (one thread); status 1: no room
@inline function derive!(R::RQ)
    P = R.P; w = R.wanted; nn = R.nn; ne = R.ne
    for e in 1:ne; pl!(R, PE(R, e, PL_LAST), 0); pl!(R, PE(R, e, PL_LOC), 0); pl!(R, PE(R, e, PL_MOD), -1); pl!(R, PE(R, e, PL_SZ), 0); end
    persist = false; for s in 0:R.NS-1; ld(R.segs, 2s) >= 0 && (persist = true); end
    for n in 1:nn
        lvlmask(nd(R, n, N_LVL)) & w != 0 || continue
        for f in 0:7; e = nin(R, n, f); e > 0 && pl(R, PE(R, e, PL_LAST)) < n && pl!(R, PE(R, e, PL_LAST), n); end
    end
# the slots
    ca = Int64(0); cb = Int64(0)                           # a slot: the rings, then the per-position edges
    for e in 1:ne
        ed(R, e, E_CACHE) != 0 || continue; persist && pl!(R, PE(R, e, PL_LAST), typemax(Int32))
        reach = ed(R, e, E_REACH); bytes = ebytes(R, e)
        reach > 0 || continue; ring = max(Int64(16), pow2ge(persist ? Int64(reach) + 64 : Int64(P)))   # (a transient request holds every position)
        pl!(R, PE(R, e, PL_MOD), ring - 1); pl!(R, PE(R, e, PL_CA), 0); pl!(R, PE(R, e, PL_CB), cb); cb += ring * bytes
    end
    for e in 1:ne
        (ed(R, e, E_CACHE) != 0 && ed(R, e, E_REACH) == 0) || continue; pl!(R, PE(R, e, PL_CA), ca); pl!(R, PE(R, e, PL_CB), cb); ca += ebytes(R, e)
    end
    pl!(R, PH_SLOT, ca * 1000000 + cb)
    vtop = Int64(0); htop = Int64(0); nh = 0
    for s in 0:R.NS-1                                      # transient segments: a pool slot
        slot = ld(R.segs, 2s); len = ld(R.segs, 2s + 1)
        if slot >= 0; pl!(R, PS(R, s, 0), Core.Intrinsics.bitcast(Int64, R.slots) + slot * R.slotsz); pl!(R, PS(R, s, 1), R.cap)
        else; sz = ca * len + cb; pl!(R, PS(R, s, 0), Core.Intrinsics.bitcast(Int64, R.aV) + vtop); pl!(R, PS(R, s, 1), len); vtop += (sz + 255) & ~Int64(255); end
    end
# tiles and flag cells of every node
    f0 = Int64(0); t0 = Int64(0); s0 = 0; s1 = 0
    for n in 1:nn
        k = nd(R, n, N_KIND); rk = nd(R, n, N_ROWS); on = lvlmask(nd(R, n, N_LVL)) & w != 0 && nd(R, n, N_CARD) == R.card
        nr = Int64(nrows(R, n)); att = k == ATTN || k == ATTBKV || k == ATTBQ; ct = 64; no = Int64(nd(R, n, N_NO)); nb = Int64(R.nblk)
        rows2 = max(Int64(8), pow2ge(cld(nr, 2 * nb)))     # two tasks a block
        rt = att ? 16 : k in (NRM, STAGE, STEP, MOMU) ? Int(rows2) : rk == WT ? 64 : Int(min(rows2, Int64(64)))
        nc = att && k != ATTBKV ? Int64(NH) : k == ATTBKV ? Int64(NKV) : k == DLOG ? Int64(cld(NV, 4096)) : Int64(1)
        if k == GEMM                                       # the tile with the least time
            best = typemax(Int64); nrp = max(Int64(8), pow2ge(nr)); nop = max(Int64(16), pow2ge(no))
            for lr in 3:7, lc in 4:7
                r_ = 1 << lr; c_ = 1 << lc; (r_ * c_ <= TILEA && r_ <= nrp && c_ <= nop) || continue
                sc = cld(((nr + r_ - 1) >> lr) * ((no + c_ - 1) >> lc), nb) * (4096 + 2 * r_ * c_)   # waves x a slice's cost
                (sc < best || (sc == best && r_ > rt)) && (best = sc; rt = r_; ct = c_)
            end
            nc = cld(no, Int64(ct))
        end
        nt = max(Int64(1), (nr + rt - 1) ÷ rt)
        fl = 1                                             # read on another card: the cells in host memory
        for f in (N_O1, N_O2, N_O3); e = nd(R, n, f); e > 0 && ed(R, e, E_CROSS) != 0 && (fl = 2); end
        pl!(R, PN(R, n, NP_F0), f0); pl!(R, PN(R, n, NP_NT), nt); pl!(R, PN(R, n, NP_NC), nc); pl!(R, PN(R, n, NP_RT), rt); pl!(R, PN(R, n, NP_CT), ct)
        pl!(R, PN(R, n, NP_ON), on ? 1 : 0); pl!(R, PN(R, n, NP_FL), fl); pl!(R, PN(R, n, NP_T0), t0); pl!(R, PN(R, n, NP_WAR), 0)
        f0 += nt * nc + 1; on && (t0 += nt * nc)
        nd(R, n, N_LVL) == LOGITS && rk == ALL && (s0 == 0 && (s0 = n); s1 = n)
    end
    pl!(R, PH_NT, t0); pl!(R, PH_NFLAG, f0); pl!(R, PH_ORDER, 0); pl!(R, PH_S0, s0); pl!(R, PH_S1, s1); pl!(R, PH_STAT, f0 > R.nflag ? 2 : 0)
    f0 > R.nflag && return
# locations
    for h in 0:NHOLE-1; pl!(R, PO(R, h, 1), 0); end; hmax = 0
    for n in 1:nn
        lvlmask(nd(R, n, N_LVL)) & w != 0 || continue
        for f in 0:7                                        # free what nothing after n - 1 reads
            e = n > 1 ? nin(R, n - 1, f) : Int32(0)
            (e > 0 && pl(R, PE(R, e, PL_LAST)) == n - 1 && pl(R, PE(R, e, PL_SZ)) > 0 && ed(R, e, E_CACHE) == 0) || continue
            for h in 0:NHOLE-1
                pl(R, PO(R, h, 1)) == 0 || continue
                pl!(R, PO(R, h, 0), pl(R, PE(R, e, PL_OFF))); pl!(R, PO(R, h, 1), pl(R, PE(R, e, PL_SZ))); pl!(R, PO(R, h, 2), pl(R, PE(R, e, PL_LOC))); pl!(R, PO(R, h, 3), pl(R, PE(R, e, PL_LAST))); hmax = max(hmax, h + 1); break
            end
            pl!(R, PE(R, e, PL_SZ), 0)
        end
        war = Int64(0)
        for f in (N_O1, N_O2, N_O3)
            e = nd(R, n, f); (e > 0 && ed(R, e, E_CACHE) == 0 && pl(R, PE(R, e, PL_SZ)) == 0) || continue
            rk = ed(R, e, E_ROWS); nr = rk == ALL ? Int64(P) : rk == LOSS ? Int64(R.PL) : Int64(ed(R, e, E_R))
            sz = (nr * ebytes(R, e) + 255) & ~Int64(255); sz == 0 && (sz = 256)
            cross = ed(R, e, E_CROSS) != 0                 # read on another card: host memory
            loc = 0; off = Int64(0); best = typemax(Int64); hb = -1
            for want in (1, 2)                              # a VRAM hole, fresh VRAM, a host hole, fresh host
                (want == 2 || !cross) || continue; best = typemax(Int64)
                for h in 0:hmax-1; sz2 = pl(R, PO(R, h, 1)); (sz2 >= sz && sz2 < best && pl(R, PO(R, h, 2)) == want) && (best = sz2; hb = h); end
                hb < 0 && want == 1 && vtop + sz <= R.nV && (loc = 1; off = vtop; vtop += sz; hb = -2)
                hb != -1 && break
            end
            if hb >= 0
                loc = pl(R, PO(R, hb, 2)); off = pl(R, PO(R, hb, 0)); pl!(R, PO(R, hb, 0), off + sz); pl!(R, PO(R, hb, 1), best - sz); war = max(war, pl(R, PO(R, hb, 3)))
            elseif hb == -1
                htop + sz <= R.nH || (pl!(R, PH_STAT, 1); return); loc = 2; off = htop; htop += sz
            end
            pl!(R, PE(R, e, PL_LOC), loc); pl!(R, PE(R, e, PL_OFF), off); pl!(R, PE(R, e, PL_SZ), sz)
        end
        pl!(R, PN(R, n, NP_WAR), war)
    end
    pl!(R, PH_VUSE, vtop); pl!(R, PH_HUSE, htop)
end
# task index -> (node, tile, column): the nodes in order, their tiles in order
@inline function decode(R::RQ, idx::Int64)
    lo = 1; hi = Int(R.nn)
    while lo < hi                                          # the last node whose first task <= idx
        mid = (lo + hi + 1) >> 1
        pln(R, mid, NP_T0) <= idx ? (lo = mid) : (hi = mid - 1)
    end
    n = lo
    while pln(R, n, NP_ON) == 0 || pln(R, n, NP_NT) * pln(R, n, NP_NC) == 0; n -= 1; end
    r = idx - pln(R, n, NP_T0); nc = pln(R, n, NP_NC)
    Int32(n), Int32(r ÷ nc), Int32(r % nc)
end
# a task's waits (thread 0)
@inline function waitnode(R::RQ, p)                          # every task of node p done
    ntp = pln(R, p, NP_NT); d = cellp(R, p, ntp, 0); nn = ntp * pln(R, p, NP_NC); i = Int64(0)
    while atomldr(d) < nn; i += 1; i == Int64(1) << 24 && (pl!(R, PH_STAT, 3); pl!(R, PH_S0, p); return); end
    fence()
end
@inline function waitin!(R::RQ, n, t)
    dep = nd(R, n, N_DEP); rt = pln(R, n, NP_RT); rk = nd(R, n, N_ROWS); nr = Int64(nrows(R, n))
    for f in 0:7
        e = nin(R, n, f); e > 0 || continue; p = ed(R, e, E_PROD); (p > 0 && p != n && lvlmask(nd(R, p, N_LVL)) & R.wanted != 0) || continue
        rtp = pln(R, p, NP_RT)
        if dep == DALL || rk == WT || nd(R, p, N_ROWS) != rk
            waitnode(R, p)
        else
            a = Int64(t) * rt; b = min(nr, (Int64(t) + 1) * rt) - 1
            dep == SPREV && (a = max(Int64(0), a - 7)); dep == SNEXT && (b = min(nr - 1, b + 6)); dep == UPTO && (a = 0); dep == FROM && (b = nr - 1)
            for t2 in a ÷ rtp:b ÷ rtp; waittile(R, p, t2); end
        end
    end
    w = pln(R, n, NP_WAR); (w > 0 && w != n && pln(R, w, NP_ON) == 1) && waitnode(R, w)   # the memory reused: its last reader done
end

# ---- the matrix product, written once: out[m, n] = Σ_k a(m, k) b(n, k); an operand a row edge (or transposed), a strided matrix, or weights ----
struct OPD; p::LP{UInt8}; dt::Int32; mode::Int32; e::Int32; nk::Int32; ms::Int64; ks::Int64; off::Int32; m0::Int32; nm::Int32; end
@inline sp!(sm, i, v::Int64) = (sm[i + 1] = reinterpret(Float32, UInt32(v & 0xffffffff)); sm[i + 2] = reinterpret(Float32, UInt32((v >>> 32) & 0xffffffff)))
@inline sp(sm, i) = Int64(reinterpret(UInt32, sm[i + 1])) | (Int64(reinterpret(UInt32, sm[i + 2])) << 32)
@inline pint(p::LP{UInt8}) = reinterpret(Int64, Core.Intrinsics.bitcast(UInt64, p))
@inline bfp(u::UInt64, i) = reinterpret(Float32, UInt32((u >> (16i)) & 0xffff) << 16)
@inline f32p(u::UInt64, i) = reinterpret(Float32, UInt32((u >> (32i)) & 0xffffffff))
@inline ld16(p::LP{UInt8}, i) = unsafe_load(reinterpret(LP{NTuple{2,VecElement{UInt64}}}, p), i + 1, Val(16))   # 16 bytes, the i-th
@inline function rd8(p::LP{UInt8}, dt::Int32, j::Int64)   # 8 elements from j: 16-byte loads
    if dt == BF; u = ld16(p, j >> 3); a = u[1].value; b = u[2].value; return (bfp(a, 0), bfp(a, 1), bfp(a, 2), bfp(a, 3), bfp(b, 0), bfp(b, 1), bfp(b, 2), bfp(b, 3)); end
    u = ld16(p, j >> 2); w = ld16(p, (j >> 2) + 1); a = u[1].value; b = u[2].value; c = w[1].value; d = w[2].value
    (f32p(a, 0), f32p(a, 1), f32p(b, 0), f32p(b, 1), f32p(c, 0), f32p(c, 1), f32p(d, 0), f32p(d, 1))
end
@inline ld4(sm, i) = unsafe_load(reinterpret(Core.LLVMPtr{NTuple{4,VecElement{Float32}},3}, pointer(sm)), (i >> 2) + 1, Val(16))
@inline isk(o::OPD) = o.mode == 1 || o.mode == 4 || (o.mode == 3 && o.ks == 1)
# a 32-k slice into smem, k-major, each k row rotated by 8 (k >> 3); a line is a row along k (kfast: per m, held for the task) or along m (per k)
@inline function lin(R::RQ, o::OPD, i, lmax, kfast)          # the address of line i (0 past lmax)
    eb = o.dt == BF ? Int64(2) : Int64(4); a = Int64(0)
    if i < lmax
        if o.mode == 3; a = pint(o.p) + i * (kfast ? o.ms : o.ks) * eb
        elseif o.mode >= 4; j = Int64(o.m0); while j < o.m0 + o.nm - 1 && i >= mt(R, j, 1); i -= mt(R, j, 1); j += 1; end; a = mt(R, j, 0) + i * mt(R, j, 2) * 2
        else; r = rowof(R, o.nk, i); a = pint(eaddr(R, o.e, r)) + Int64(o.off) * eb; end
    end
    a
end
@inline ltab!(R::RQ, sm, so, o::OPD, base, lmax, nlines, kfast, tid) = (tid < nlines && sp!(sm, (so == 0 ? 8448 : 8704) + 2tid, lin(R, o, base + tid, lmax, kfast)); bar())
@inline function kdesc(sm, so, o::OPD, nl, tid, st)          # a thread's two groups
    sp0 = so == 0 ? 8448 : 8704; eb = o.dt == BF ? Int64(2) : Int64(4); g = (Int64(tid) & 3) << 3; l1 = Int64(tid) >> 2
    ok1 = Int64(tid) < (nl << 2); p1 = ok1 ? sp(sm, sp0 + 2l1) : Int64(0); ok2 = Int64(tid) + 256 < (nl << 2); p2 = ok2 ? sp(sm, sp0 + 2l1 + 128) : Int64(0)
    (p1 != 0 ? p1 + g * eb : Int64(0), ok1 ? g * st + ((l1 + g) & (nl - 1)) : Int64(-1), p2 != 0 ? p2 + g * eb : Int64(0), ok2 ? g * st + ((l1 + 64 + g) & (nl - 1)) : Int64(-1))
end
@inline function kld!(sm, so, dt, d, k0, K, g, st)            # the held lines' next 32 k
    eb = dt == BF ? Int64(2) : Int64(4); jj = k0 + g
    Base.Cartesian.@nexprs 2 q -> begin
        v = (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
        if d[2q - 1] != 0 && jj < K
            a = lp(UInt8, d[2q - 1] + k0 * eb)
            v = jj + 7 < K ? rd8(a, dt, Int64(0)) : (Base.Cartesian.@ntuple 8 e -> (jj + e - 1 < K ? rdv(a, dt, Int64(e - 1)) : 0.0f0))
        end
        d[2q] >= 0 && Base.Cartesian.@nexprs 8 e -> (sm[so + d[2q] + (e - 1) * st + 1] = v[e])
    end
end
@inline function ldtile!(R::RQ, sm, so, o::OPD, nrow, mmax, m0, k0, K, tid, sa)   # a transposed operand's slice: its lines are k's
    t3 = o.mode == 3; t3 || ltab!(R, sm, so, o, k0, K, 32, false, tid); sp0 = so == 0 ? 8448 : 8704; eb = o.dt == BF ? Int64(2) : Int64(4)
    es = t3 ? o.ms : Int64(1); sh = trailing_zeros(Int64(nrow)) - 3; ng = Int64(32) << sh
    Base.Cartesian.@nexprs 2 q -> begin
        i_q = Int64(tid) + 256 * (q - 1); l_q = i_q >> sh; g_q = (i_q & ((Int64(1) << sh) - 1)) << 3; jj_q = m0 + g_q; ok_q = i_q < ng && k0 + l_q < K && jj_q < mmax
        a_q = lp(UInt8, ok_q ? (t3 ? pint(o.p) + (k0 + l_q) * o.ks * eb : sp(sm, sp0 + 2l_q)) : Int64(0)); v_q = (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)   # (unwritten: any bits)
        if ok_q
            if es == 1 && jj_q + 7 < mmax; v_q = rd8(a_q, o.dt, jj_q)
            else; v_q = Base.Cartesian.@ntuple 8 e -> (jj_q + e - 1 < mmax ? rdv(a_q, o.dt, (jj_q + e - 1) * es) : 0.0f0); end
        end
    end
    Base.Cartesian.@nexprs 2 q -> (i_q < ng && Base.Cartesian.@nexprs 8 e -> (sm[so + l_q * sa + ((g_q + 8 * (l_q >> 3)) & (nrow - 1)) + e] = v_q[e]))
end
@inline function kloop(R::RQ, sm, oA, oB, rt, ct, nr, no, m0, n0, K, tid, ntm, ntn, ntk, t8::Bool)
    act = tid < ntm * ntn * ntk; tk = tid ÷ (ntm * ntn); rem = tid % (ntm * ntn); tm = rem ÷ ntn; tn = rem % ntn; sa = Int64(132); sb = Int64(4224); sbs = Int64(132)   # (constant strides)
    Base.Cartesian.@nexprs 32 i -> (c_i = 0.0f0)
    k0 = Int64(0); ka = isk(oA); kb = isk(oB)
    g = (Int64(tid) & 3) << 3; ka && ltab!(R, sm, 0, oA, m0, nr, rt, true, tid); kb && ltab!(R, sm, sb, oB, n0, no, ct, true, tid)
    dA = ka ? kdesc(sm, 0, oA, Int64(rt), tid, sa) : (Int64(0), Int64(-1), Int64(0), Int64(-1)); dB = kb ? kdesc(sm, sb, oB, Int64(ct), tid, sbs) : (Int64(0), Int64(-1), Int64(0), Int64(-1))
    while k0 < K
        bar(); if ka; kld!(sm, 0, oA.dt, dA, k0, K, g, sa) else ldtile!(R, sm, 0, oA, rt, nr, m0, k0, K, tid, sa) end; if kb; kld!(sm, sb, oB.dt, dB, k0, K, g, sbs) else ldtile!(R, sm, sb, oB, ct, no, n0, k0, K, tid, sbs) end; bar()
        if act                                              # a k: 8 (or 4) rows, 4 columns
            kend = Int32(32) + Int32(K < 0)                  # (opaque: the loop stays a loop)
            if t8
                kq = Int32(0)
                while kq < kend
                    rk = Int64((kq >> 3) << 3); pa = (tm * 8 + rk) & (rt - 1); pb = (tn * 4 + rk) & (ct - 1)
                    Base.Cartesian.@nexprs 2 k -> (a = ld4(sm, (kq + k - 1) * sa + pa); a2 = ld4(sm, (kq + k - 1) * sa + pa + 4); b = ld4(sm, sb + (kq + k - 1) * sbs + pb); Base.Cartesian.@nexprs 4 i -> Base.Cartesian.@nexprs 4 j -> (c_{(i - 1) * 4 + j} = fma(a[i].value, b[j].value, c_{(i - 1) * 4 + j}); c_{16 + (i - 1) * 4 + j} = fma(a2[i].value, b[j].value, c_{16 + (i - 1) * 4 + j})))
                    kq += Int32(2)
                end
            elseif ntk == 1
                kq = Int32(0)
                while kq < kend
                    rk = Int64((kq >> 3) << 3); pa = (tm * 4 + rk) & (rt - 1); pb = (tn * 4 + rk) & (ct - 1)
                    Base.Cartesian.@nexprs 2 k -> (a = ld4(sm, (kq + k - 1) * sa + pa); b = ld4(sm, sb + (kq + k - 1) * sbs + pb); Base.Cartesian.@nexprs 4 i -> Base.Cartesian.@nexprs 4 j -> (c_{(i - 1) * 4 + j} = fma(a[i].value, b[j].value, c_{(i - 1) * 4 + j})))
                    kq += Int32(2)
                end
            else
                k = Int32(tk)
                while k < 32
                    rk = Int64((k >> 3) << 3); a = ld4(sm, k * sa + ((tm * 4 + rk) & (rt - 1))); b = ld4(sm, sb + k * sbs + ((tn * 4 + rk) & (ct - 1))); Base.Cartesian.@nexprs 4 i -> Base.Cartesian.@nexprs 4 j -> (c_{(i - 1) * 4 + j} = fma(a[i].value, b[j].value, c_{(i - 1) * 4 + j})); k += ntk
                end
            end
        end
        k0 += 32
    end
    if ntk > 1                                             # the k slices of a thread group added
        bar(); act && (Base.Cartesian.@nexprs 16 i -> (sm[(tk * ntm * ntn + rem) * 16 + i] = c_i)); bar()
        if act && tk == 0
            for t2 in 1:ntk-1; Base.Cartesian.@nexprs 16 i -> (c_i += sm[(t2 * ntm * ntn + rem) * 16 + i]); end
        end
    end
    Base.Cartesian.@ntuple 32 i -> c_i
end
@inline function put!(a::LP{UInt8}, d::Int32, v::NTuple{4,Float32}, i::Int64, nv)   # 4 elements from i: one vector store if whole
    if nv == 4 && (pint(a) + i * (d == BF ? 2 : 4)) & (d == BF ? 7 : 15) == 0
        if d == BF
            h(x) = UInt64(reinterpret(UInt32, bfr(x)) >> 16); unsafe_store!(reinterpret(LP{UInt64}, a), h(v[1]) | (h(v[2]) << 16) | (h(v[3]) << 32) | (h(v[4]) << 48), (i >> 2) + 1, Val(8))
        else; unsafe_store!(reinterpret(LP{NTuple{4,VecElement{Float32}}}, a), (VecElement(v[1]), VecElement(v[2]), VecElement(v[3]), VecElement(v[4])), (i >> 2) + 1, Val(16)); end
    else; Base.Cartesian.@nexprs 4 j -> (j <= nv && wrv!(a, d, v[j], i + j - 1)); end
    nothing
end
@inline function gemm!(R::RQ, sm, n, tile, col, tid)
    fl = nd(R, n, N_FL); epi = nd(R, n, N_EPI); rt = Int32(pln(R, n, NP_RT)); ct = Int32(pln(R, n, NP_CT)); m0 = Int64(tile) * rt; n0 = Int64(col) * ct
    K = Int64(nd(R, n, N_K)); e1 = nd(R, n, N_I1); e2 = nd(R, n, N_I2); rk = nd(R, n, N_ROWS); kk = rk; nr = Int64(nrows(R, n)); no = Int64(nd(R, n, N_NO))
    K < 0 && (kk = ed(R, e1, E_ROWS); K = kk == LOSS ? Int64(R.PL) : Int64(R.P))
    rs = nd(R, n, N_RS); m = nd(R, n, N_M)
    if fl & A_ROW != 0
        oA = fl & A_TR != 0 ? OPD(LP0, ed(R, e1, E_DT), Int32(2), e1, kk, 0, 0, nd(R, n, N_WIN), 0, 0) : OPD(LP0, ed(R, e1, E_DT), Int32(1), e1, rk, 0, 0, 0, 0, 0)
    else
        W = Int64(ed(R, e1, E_W)); tr = fl & A_TR != 0; oA = OPD(ebase(R, e1), ed(R, e1, E_DT), Int32(3), 0, 0, tr ? 1 : W, tr ? W : 1, 0, 0, 0)
    end
    nB = epi == EP_SILU ? 2 : 1
    if fl & B_MAT != 0
        tr = fl & B_TR != 0; nm = nB == 2 ? Int32(1) : nd(R, n, N_NM); Wm = mt(R, m, 2)
        oB = nm == 1 ? OPD(lp(UInt8, mt(R, m, 0)), BF, Int32(3), 0, 0, tr ? Int64(1) : Wm, tr ? Wm : Int64(1), 0, 0, 0) : OPD(LP0, BF, tr ? Int32(5) : Int32(4), 0, 0, 0, 0, 0, m, nm)
    elseif fl & B_ROWK != 0
        oB = OPD(LP0, ed(R, e2, E_DT), Int32(2), e2, kk, 0, 0, 0, 0, 0)
    else
        W = Int64(ed(R, e2, E_W)); tr = fl & B_TR != 0; oB = OPD(ebase(R, e2), ed(R, e2, E_DT), Int32(3), 0, 0, tr ? 1 : W, tr ? W : 1, 0, 0, 0)
    end
    t8 = rt * ct >= TILEA; tr = t8 ? 8 : 4; ntn = Int32(ct ÷ 4); ntm = Int32((rt + tr - 1) ÷ tr); ntk = Int32(1); while ntk * 2 * ntm * ntn <= 256; ntk *= Int32(2); end
    c1 = kloop(R, sm, oA, oB, rt, ct, nr, no, m0, n0, K, tid, ntm, ntn, ntk, t8)
    c2 = nB == 2 ? kloop(R, sm, oA, OPD(oB.mode == 3 ? lp(UInt8, mt(R, m + Int32(1), 0)) : oB.p, oB.dt, oB.mode, oB.e, oB.nk, oB.ms, oB.ks, oB.off, m + Int32(1), oB.nm), rt, ct, nr, no, m0, n0, K, tid, ntm, ntn, ntk, t8) : c1
    act = tid < ntm * ntn * ntk; tk = tid ÷ (ntm * ntn); rem = tid % (ntm * ntn); tm = rem ÷ ntn; tn = rem % ntn
    o1 = nd(R, n, N_O1); o2 = nd(R, n, N_O2); e3 = nd(R, n, N_I3); f1 = ndf(R, n, 0); f2 = ndf(R, n, 1)
    epi == EP_LSE && bar()
    d1 = ed(R, o1, E_DT); d2 = epi == EP_SILU ? ed(R, o2, E_DT) : d1; d3 = e3 > 0 ? ed(R, e3, E_DT) : d1
    mw = epi == EP_AXPY && e3 <= 0; mom = lp(Float32, mw ? mt(R, m, 4) : Int64(0)); msc = mw ? ld(lp(Float32, waddr(R, rs, 0)), 0) : 0.0f0   
    if act && tk == 0                                      # a row's address once, then its 4 columns
        Base.Cartesian.@nexprs 8 i -> begin
            mm = Int64(tm * tr + i - 1); mr = m0 + mm
            if i <= tr && mm < rt && mr < nr
                r = rowof(R, rk, mr); a1 = epi == EP_AXPY || rk == WT ? waddr(R, o1, mr) : eaddr(R, o1, r); a2 = epi == EP_SILU ? eaddr(R, o2, r) : a1; a3 = epi == EP_AXPY && e3 > 0 ? waddr(R, e3, mr) : a1
                nc0 = n0 + Int64(tn) * 4; nv = min(Int64(4), no - nc0)
                if nv > 0
                    v4 = Base.Cartesian.@ntuple 4 j -> c1[(i - 1) * 4 + j]
                    if epi == EP_SILU
                        w4 = Base.Cartesian.@ntuple 4 j -> c2[(i - 1) * 4 + j]
                        put!(a1, d1, v4, nc0, nv); put!(a1, d1, w4, Int64(NFF) + nc0, nv); put!(a2, d2, Base.Cartesian.@ntuple(4, j -> v4[j] * sigm(v4[j]) * w4[j]), nc0, nv)
                    elseif epi == EP_AXPY                     # f1 acc + f2 other
                        put!(a1, d1, Base.Cartesian.@ntuple(4, j -> (j <= nv ? f1 * v4[j] + f2 * (e3 > 0 ? rdv(a3, d3, nc0 + j - 1) : ld(mom, mr * no + nc0 + j - 1) * msc) : 0.0f0)), nc0, nv)
                    elseif epi == EP_LSE; put!(a1, F32, v4, nc0, nv); Base.Cartesian.@nexprs 4 j -> (j <= nv && (sm[mm * (ct + 1) + Int64(tn) * 4 + j] = v4[j]))
                    else; put!(a1, d1, v4, nc0, nv)
                    end
                end
            end
        end
    end
    if epi == EP_LSE                                       # a row's max and sum over this column tile
        bar()
        if tid < rt && m0 + tid < nr
            mx = -Inf32; for j in 0:ct-1; n0 + j < no && (mx = max(mx, sm[tid * (ct + 1) + j + 1])); end
            s = 0.0f0; for j in 0:ct-1; n0 + j < no && (s += exp(sm[tid * (ct + 1) + j + 1] - mx)); end
            p = lp(Float32, eaddr(R, o2, rowof(R, rk, m0 + tid))); st!(p, mx, 2 * Int64(col)); st!(p, s, 2 * Int64(col) + 1)
        end
    end
end
# ---- the row methods: a thread strides over the elements of a tile's rows ----
@inline function foreach_el(f, R::RQ, n, tile, w, tid)     # f(r, k) over the tile's rows r and k < w
    rt = pln(R, n, NP_RT); nr = nrows(R, n); rk = nd(R, n, N_ROWS); i = Int64(tid); tot = min(rt, nr - Int64(tile) * rt) * Int64(w)
    while i < tot
        f(rowof(R, rk, Int64(tile) * rt + i ÷ w), i % w); i += 256
    end
end
@inline rsv(R::RQ, e, r, b) = ld(lp(Float32, eaddr(R, e, r)), Int64(b))
@inline ev(R::RQ, e, r, k) = rdv(eaddr(R, e, r), ed(R, e, E_DT), Int64(k))
@inline ev!(R::RQ, e, r, v, k) = wrv!(eaddr(R, e, r), ed(R, e, E_DT), v, Int64(k))
@inline rope(R::RQ, tbl, pos, q) = ld(R.rope, (Int64(tbl) * 64 + (q & 63)) * 1) * Float32(pos)   # the angle of pair q at position pos
@inline function rotv(R::RQ, tbl, pos, q, a, b)             # the rotation of (x[q], x[q +- 64])
    an = rope(R, tbl, pos, q); cs = cos(an); sn = sin(an); q < 64 ? a * cs - b * sn : a * cs + b * sn
end
@inline function norm!(R::RQ, sm, n, tile, tid)            # rs[r][b] = rsqrt(mean x^2 + eps)
    rt = pln(R, n, NP_RT); nr = nrows(R, n); rk = nd(R, n, N_ROWS); e = nd(R, n, N_I1); o = nd(R, n, N_O1); o2 = nd(R, n, N_O2); D = nd(R, n, N_D); nb = nd(R, n, N_K) ÷ D; off = nd(R, n, N_WIN); wv = nd(R, n, N_WV) > 0 ? vp(R, nd(R, n, N_WV)) : lp(Float32, 0)
    g = tid ÷ 16; lane = tid % 16; tot = min(rt, nr - Int64(tile) * rt) * nb; i = Int64(g)
    while i - g < tot
        s = 0.0f0; r = Int32(0); b = Int64(0)
        if i < tot
            r = rowof(R, rk, Int64(tile) * rt + i ÷ nb); b = i % nb
            for q in lane:16:D-1; v = ev(R, e, r, off + b * D + q); s = fma(v, v, s); end
        end
        sm[tid + 1] = s; bar()
        if lane == 0 && i < tot
            t = 0.0f0; for j in 0:15; t += sm[g * 16 + j + 1]; end
            sm[300 + g] = 1.0f0 / sqrt(t / Float32(D) + EPS); wrv!(eaddr(R, o, r), F32, sm[300 + g], b)
        end
        bar()
        if o2 > 0 && i < tot                                 # the normalized row x rs w
            for q in lane:16:D-1; c = off + b * D + q; ev!(R, o2, r, ev(R, e, r, c) * sm[300 + g] * ld(wv, c), c); end
        end
        bar(); i += 16
    end
end
@inline function resid!(R::RQ, n, tile, tid)                # x1 = x + w rs o
    x = nd(R, n, N_I1); o = nd(R, n, N_I2); rs = nd(R, n, N_I3); w = vp(R, nd(R, n, N_WV)); out = nd(R, n, N_O1)
    foreach_el(R, n, tile, NE, tid) do r, k
        ev!(R, out, r, ev(R, x, r, k) + ld(w, k) * rsv(R, rs, r, 0) * ev(R, o, r, k), k)
    end
end
@inline function silub!(R::RQ, n, tile, tid)                # gu1 = gs u3 silu'(u1), gu3 = gs silu(u1)
    gs = nd(R, n, N_I1); u = nd(R, n, N_I2); gu = nd(R, n, N_O1)
    foreach_el(R, n, tile, NFF, tid) do r, k
        g = ev(R, gs, r, k); a = ev(R, u, r, k); b = ev(R, u, r, NFF + k); s = sigm(a)
        ev!(R, gu, r, g * b * (s + a * s * (1.0f0 - s)), k); ev!(R, gu, r, g * a * s, NFF + k)
    end
end
@inline function conv!(R::RQ, n, tile, tid)                 # win 0: h = B x; win 1: cv = taps * h, y = C cv
    bcx = nd(R, n, N_I1); h = nd(R, n, N_WIN) == 0 ? nd(R, n, N_O1) : nd(R, n, N_I2)
    if nd(R, n, N_WIN) == 0
        foreach_el(R, n, tile, NE, tid) do r, k; ev!(R, h, r, ev(R, bcx, r, k) * ev(R, bcx, r, 2NE + k), k); end
    else
        y = nd(R, n, N_O1); cv = nd(R, n, N_O2); taps = vp(R, nd(R, n, N_WV))
        foreach_el(R, n, tile, NE, tid) do r, k
            s = rw(R, r, 0); pos = rw(R, r, 1); acc = 0.0f0
            for t in 0:6; p = pos - 6 + t; p >= 0 && (acc = fma(ld(taps, Int64(k) * 7 + t), rdv(caddr(R, h, s, p), BF, Int64(k)), acc)); end
            ev!(R, cv, r, acc, k); ev!(R, y, r, ev(R, bcx, r, NE + k) * acc, k)
        end
    end
end
@inline function convb1!(R::RQ, n, tile, tid)               # gc = gy C; gC = gy cv
    gy = nd(R, n, N_I1); bcx = nd(R, n, N_I2); cv = nd(R, n, N_I3); gc = nd(R, n, N_O1); gb = nd(R, n, N_O2)
    foreach_el(R, n, tile, NE, tid) do r, k
        g = ev(R, gy, r, k); ev!(R, gc, r, g * ev(R, bcx, r, NE + k), k); ev!(R, gb, r, g * ev(R, cv, r, k), NE + k)
    end
end
@inline function convb2!(R::RQ, n, tile, tid)               # gh = taps' gc over 6 rows; gB = gh x, gx = gh B
    gc = nd(R, n, N_I1); bcx = nd(R, n, N_I2); gb = nd(R, n, N_O1); taps = vp(R, nd(R, n, N_WV))
    foreach_el(R, n, tile, NE, tid) do r, k
        s = rw(R, r, 0); acc = 0.0f0
        for t in 0:6; r2 = r + 6 - t; (r2 < R.P && rw(R, r2, 0) == s) && (acc = fma(ld(taps, Int64(k) * 7 + t), ev(R, gc, r2, k), acc)); end
        ev!(R, gb, r, acc * ev(R, bcx, r, 2NE + k), k); ev!(R, gb, r, acc * ev(R, bcx, r, k), 2NE + k)
    end
end
@inline function convw!(R::RQ, n, tile, tid)                # the taps' gradient (64 channels a tile)
    gc = nd(R, n, N_I1); h = nd(R, n, N_I2); mo = vmo(R, nd(R, n, N_WV)); i = Int64(tid)
    while i < 64 * 7
        k = Int64(tile) * 64 + i ÷ 7; t = i % 7; acc = 0.0f0
        for r in 0:R.P-1
            s = rw(R, r, 0); p = rw(R, r, 1) - 6 + t; p >= 0 && (acc = fma(ev(R, gc, r, k), rdv(caddr(R, h, s, p), BF, k), acc))
        end
        st!(mo, R.f2 * ld(mo, k * 7 + t) + R.iacc * acc, k * 7 + t); i += 256
    end
end
@inline function normb!(R::RQ, sm, n, tile, tid)            # gx (+)= w gy rs - (rs^3 / D) (Σ gy w x) x
    gy = nd(R, n, N_I1); x = nd(R, n, N_I2); rs = nd(R, n, N_I3); add = nd(R, n, N_I4); out = nd(R, n, N_O1); w = vp(R, nd(R, n, N_WV)); D = nd(R, n, N_D); lossy = nd(R, n, N_WIN) == 1
    rt = pln(R, n, NP_RT); nr = nrows(R, n); g = tid ÷ 64; lane = tid % 64; tot = min(rt, nr - Int64(tile) * rt); i = Int64(g)
    while i - g < tot
        r = Int32(0); has = false; s = 0.0f0
        if i < tot
            r = rowof(R, ALL, Int64(tile) * rt + i); has = !lossy || ld(R.linv, r) >= 0
            if has; for q in lane:64:D-1; s = fma(ev(R, gy, r, q) * ld(w, q), ev(R, x, r, q), s); end; end
        end
        sm[tid + 1] = s; bar()
        if i < tot
            t = 0.0f0; for j in 0:63; t += sm[g * 64 + j + 1]; end
            rv = has ? rsv(R, rs, r, 0) : 0.0f0; c = t * rv * rv * rv / Float32(D)
            for q in lane:64:D-1
                v = has ? ev(R, gy, r, q) * ld(w, q) * rv - c * ev(R, x, r, q) : 0.0f0
                ev!(R, out, r, add > 0 ? ev(R, add, r, q) + v : v, q)
            end
        end
        bar(); i += 4
    end
end
@inline function normw!(R::RQ, sm, n, tile, tid)                # gw[q] = Σ_r Σ_b gy x rs
    gy = nd(R, n, N_I1); x = nd(R, n, N_I2); rs = nd(R, n, N_I3); D = nd(R, n, N_D); nb = nd(R, n, N_K) ÷ D; off = nd(R, n, N_TBL); lossy = nd(R, n, N_WIN) == 1
    mo = vmo(R, nd(R, n, N_WV)); q = Int64(tile) * 64 + tid % 64; part = tid ÷ 64; acc = 0.0f0
    if q < D
        r = Int64(part)
        while r < R.P
            if !lossy || ld(R.linv, r) >= 0
                for b in 0:nb-1; acc = fma(ev(R, gy, r, off + b * D + q) * ev(R, x, r, off + b * D + q), rsv(R, rs, r, b), acc); end
            end
            r += 4
        end
    end
    sm[tid + 1] = acc; bar()
    if part == 0 && q < D; t = sm[tid + 1] + sm[tid + 65] + sm[tid + 129] + sm[tid + 193]; st!(mo, R.f2 * ld(mo, q) + R.iacc * t, q); end
end
@inline function rope!(R::RQ, n, tile, tid)                 # qr, kr rotated; v to its cache
    qkv = nd(R, n, N_I1); rsq = nd(R, n, N_I2); rsk = nd(R, n, N_I3); qr = nd(R, n, N_O1); kr = nd(R, n, N_O2); v = nd(R, n, N_O3)
    wq = vp(R, nd(R, n, N_WV)); wk = vp(R, nd(R, n, N_D)); tbl = nd(R, n, N_TBL)
    foreach_el(R, n, tile, NE + 2NKV * HD, tid) do r, k
        pos = rw(R, r, 1)
        if k < NE
            j = k ÷ HD; q = k % HD; a = ev(R, qkv, r, k) * ld(wq, q) * rsv(R, rsq, r, j); qp = q < 64 ? q + 64 : q - 64
            b = ev(R, qkv, r, j * HD + qp) * ld(wq, qp) * rsv(R, rsq, r, j); ev!(R, qr, r, rotv(R, tbl, pos, q, a, b), k)
        elseif k < NE + NKV * HD
            kk = k - NE; j = kk ÷ HD; q = kk % HD; a = ev(R, qkv, r, k) * ld(wk, q) * rsv(R, rsk, r, j); qp = q < 64 ? q + 64 : q - 64
            b = ev(R, qkv, r, NE + j * HD + qp) * ld(wk, qp) * rsv(R, rsk, r, j); ev!(R, kr, r, rotv(R, tbl, pos, q, a, b), kk)
        else; ev!(R, v, r, ev(R, qkv, r, k), k - NE - NKV * HD)
        end
    end
end
# the attention of a 16-row tile for head j
@inline function attn!(R::RQ, sm, n, tile, j, tid)
    qr = nd(R, n, N_I1); kr = nd(R, n, N_I2); v = nd(R, n, N_I3); qkv = nd(R, n, N_I4); ao = nd(R, n, N_O1); lse = nd(R, n, N_O2); win = nd(R, n, N_WIN)
    r0 = Int64(tile) * 16; nr = min(Int64(16), Int64(R.P) - r0); tr = tid ÷ 16; tc = tid % 16; g = j ÷ 4
    for i in tid:256:16*HD-1; (r = i ÷ HD; r < nr) && (sm[i + 1] = ev(R, qr, r0 + r, j * HD + i % HD)); end
    Base.Cartesian.@nexprs 8 d -> (o_d = 0.0f0)
    if tid < 16; sm[8192 + tid + 1] = -Inf32; sm[8208 + tid + 1] = 0.0f0; end
    bar()
    seg = rw(R, r0, 0); pos = tr < nr ? rw(R, r0 + tr, 1) : Int32(-1); segr = tr < nr ? rw(R, r0 + tr, 0) : Int32(-1)
    s0 = seg; while true                                   # the segments in the tile, one at a time
        lo = Int32(0); hi = Int32(-1); first = typemax(Int32)
        for r in 0:nr-1; rw(R, r0 + r, 0) == s0 && (hi = max(hi, rw(R, r0 + r, 1)); first = min(first, rw(R, r0 + r, 1))); end
        win > 0 && (lo = max(Int32(0), first - win + 1))   # (the tile's earliest row's window start)
        kt = Int64(lo) & ~Int64(15)
        while kt <= hi
            bar()
            kb = caddr(R, kr, s0, kt); vb = caddr(R, v, s0, kt); ebk = ebytes(R, kr); ebv = ebytes(R, v)   
            for i in tid:256:16*HD-1; c = i ÷ HD; ok = kt + c <= hi; sm[2048 + i + 1] = ok ? rdv(kb + c * ebk, BF, g * HD + i % HD) : 0.0f0; sm[4096 + i + 1] = ok ? rdv(vb + c * ebv, BF, g * HD + i % HD) : 0.0f0; end
            bar()
            kp = kt + tc; s = -Inf32
            if segr == s0 && kp <= pos && kp >= pos - (win > 0 ? win - 1 : pos) && kp <= hi
                s = 0.0f0; for d in 0:HD-1; s = fma(sm[tr * HD + d + 1], sm[2048 + tc * HD + d + 1], s); end; s *= SCALE
            end
            sm[6144 + tr * 16 + tc + 1] = s; bar()
            mo = sm[8192 + tr + 1]; mx = mo; for c in 0:15; mx = max(mx, sm[6144 + tr * 16 + c + 1]); end
            sc = mx == -Inf32 ? 1.0f0 : exp(mo - mx); p = s == -Inf32 ? 0.0f0 : exp(s - mx)
            sm[6400 + tr * 16 + tc + 1] = p; bar()
            if tc == 0; sm[8192 + tr + 1] = mx; l = sm[8208 + tr + 1] * sc; for c in 0:15; l += sm[6400 + tr * 16 + c + 1]; end; sm[8208 + tr + 1] = l; end
            Base.Cartesian.@nexprs 8 d -> (o_d *= sc)
            for c in 0:15; pc = sm[6400 + tr * 16 + c + 1]; Base.Cartesian.@nexprs 8 d -> (o_d = fma(pc, sm[4096 + c * HD + tc * 8 + d], o_d)); end
            kt += 16
        end
        nxt = s0; for r in 0:nr-1; rw(R, r0 + r, 0) > s0 && (nxt == s0 || rw(R, r0 + r, 0) < nxt) && (nxt = rw(R, r0 + r, 0)); end
        nxt == s0 && break; s0 = nxt
    end
    bar()
    if tr < nr
        l = sm[8208 + tr + 1]; gt = sigm(ev(R, qkv, r0 + tr, NE + 2NKV * HD + j)); a = eaddr(R, ao, r0 + tr)
        Base.Cartesian.@nexprs 8 d -> wrv!(a, BF, R.dbg & 16 != 0 ? Float32(tc * 8 + d) : R.dbg & 32 != 0 ? sm[4096 + tc * 8 + d] : o_d / l * gt, Int64(j) * HD + tc * 8 + d - 1)
        tc == 0 && wrv!(eaddr(R, lse, r0 + tr), F32, sm[8192 + tr + 1] + log(l), Int64(j))
    end
end
# its adjoint, two passes: dK, dV; then dQ
# a query tile's dQ and the gate's gradient over the key tiles before it
@inline function attb!(R::RQ, sm, n, tile, col, tid, kvpass::Bool)
    qr = nd(R, n, N_I1); kr = nd(R, n, N_I2); v = nd(R, n, N_I3); qkv = nd(R, n, N_I4); ao = nd(R, n, N_M); gao = nd(R, n, N_NM); lse = nd(R, n, N_D); out = nd(R, n, N_O1); win = nd(R, n, N_WIN)
    t0 = Int64(tile) * 16; nt = (Int64(R.P) + 15) ÷ 16; tr = tid ÷ 16; tc = tid % 16
    Base.Cartesian.@nexprs 8 d -> (k_d = 0.0f0); Base.Cartesian.@nexprs 8 d -> (v_d = 0.0f0)
    jlo = kvpass ? col * 4 : col; jhi = kvpass ? col * 4 + 3 : col
    for j in jlo:jhi
        g = j ÷ 4
        if kvpass                                          # the key tile fixed: K, V in smem
            for i in tid:256:16*HD-1; c = i ÷ HD; r = t0 + c; r < R.P && (sm[i + 1] = ev(R, kr, r, g * HD + i % HD); sm[2048 + i + 1] = ev(R, v, r, g * HD + i % HD)); end
        else                                               # Q, dO in smem
            for i in tid:256:16*HD-1; c = i ÷ HD; r = t0 + c; r < R.P && (sm[4096 + i + 1] = ev(R, qr, r, j * HD + i % HD); sm[6144 + i + 1] = ev(R, gao, r, j * HD + i % HD) * sigm(ev(R, qkv, r, NE + 2NKV * HD + j))); end
        end
        bar()
        if !kvpass && tr < 16 && t0 + tr < R.P          # D_r = Σ dO ao / sigmoid: 16 threads a row
            r = t0 + tr; s = 0.0f0; for d in 0:7; s = fma(ev(R, gao, r, j * HD + tc * 8 + d), ev(R, ao, r, j * HD + tc * 8 + d), s); end
            sm[8448 + tid + 1] = s
        end
        bar()
        if !kvpass && tid < 16 && t0 + tid < R.P
            s = 0.0f0; for c in 0:15; s += sm[8448 + tid * 16 + c + 1]; end
            sm[8704 + tid + 1] = s; sm[8720 + tid + 1] = rsv(R, lse, t0 + tid, j)
            gt = sigm(ev(R, qkv, t0 + tid, NE + 2NKV * HD + j)); wrv!(eaddr(R, out, t0 + tid), F32, s * (1.0f0 - gt), Int64(NE) + j)   # the gate's gradient (ao is gated)
        end
        bar()
        t2 = kvpass ? Int64(tile) : Int64(0); t2end = kvpass ? nt - 1 : Int64(tile)
        while t2 <= t2end                                  # the other tile: queries or keys
            u0 = t2 * 16; bar()
            if kvpass
                for i in tid:256:16*HD-1; c = i ÷ HD; r = u0 + c; r < R.P && (sm[4096 + i + 1] = ev(R, qr, r, j * HD + i % HD); sm[6144 + i + 1] = ev(R, gao, r, j * HD + i % HD) * sigm(ev(R, qkv, r, NE + 2NKV * HD + j))); end
                if tid < 16 && u0 + tid < R.P
                    r = u0 + tid; s = 0.0f0; for d in 0:HD-1; s = fma(ev(R, gao, r, j * HD + d), ev(R, ao, r, j * HD + d), s); end
                    sm[8704 + tid + 1] = s; sm[8720 + tid + 1] = rsv(R, lse, r, j)
                end
            else
                for i in tid:256:16*HD-1; c = i ÷ HD; r = u0 + c; r < R.P && (sm[i + 1] = ev(R, kr, r, g * HD + i % HD); sm[2048 + i + 1] = ev(R, v, r, g * HD + i % HD)); end
            end
            bar()
            qrow = kvpass ? u0 + tr : t0 + tr; krow = kvpass ? t0 + tc : u0 + tc; p = 0.0f0; ds = 0.0f0
            if qrow < R.P && krow < R.P && rw(R, qrow, 0) == rw(R, krow, 0) && rw(R, krow, 1) <= rw(R, qrow, 1) && (win == 0 || rw(R, krow, 1) > rw(R, qrow, 1) - win)
                s = 0.0f0; dp = 0.0f0
                for d in 0:HD-1; s = fma(sm[4096 + tr * HD + d + 1], sm[tc * HD + d + 1], s); dp = fma(sm[6144 + tr * HD + d + 1], sm[2048 + tc * HD + d + 1], dp); end
                p = exp(s * SCALE - sm[8720 + tr + 1]); ds = p * (dp - sm[8704 + tr + 1]) * SCALE
            end
            sm[8192 + tr * 16 + tc + 1] = p; sm[8448 + tr * 16 + tc + 1] = ds; bar()
            if kvpass                                      # thread (key tr, dims tc 8..): dV, dK
                for q in 0:15; pq = sm[8192 + q * 16 + tr + 1]; dq = sm[8448 + q * 16 + tr + 1]
                    Base.Cartesian.@nexprs 8 d -> (v_d = fma(pq, sm[6144 + q * HD + tc * 8 + d], v_d)); Base.Cartesian.@nexprs 8 d -> (k_d = fma(dq, sm[4096 + q * HD + tc * 8 + d], k_d)); end
            else                                           # thread (query r = tr, dims): dQ += ds K
                for c in 0:15; dq = sm[8448 + tr * 16 + c + 1]; Base.Cartesian.@nexprs 8 d -> (k_d = fma(dq, sm[c * HD + tc * 8 + d], k_d)); end
            end
            t2 += 1
        end
        bar()
        if t0 + tr < R.P && (!kvpass || j == jhi)         # (the kv pass sums the group's four heads first)
            a = eaddr(R, out, t0 + tr)
            if kvpass; Base.Cartesian.@nexprs 8 d -> (wrv!(a, F32, k_d, Int64(g) * HD + tc * 8 + d - 1); wrv!(a, F32, v_d, Int64(NKV * HD) + g * HD + tc * 8 + d - 1))
            else; Base.Cartesian.@nexprs 8 d -> wrv!(a, F32, k_d, Int64(j) * HD + tc * 8 + d - 1); end
        end
    end
end
@inline function rotnb!(R::RQ, n, tile, tid)   # the rotations undone, the head norms' adjoint
    gqg = nd(R, n, N_I1); gkv = nd(R, n, N_I2); qkv = nd(R, n, N_I3); rsq = nd(R, n, N_I4); rsk = nd(R, n, N_M); out = nd(R, n, N_O1); gqn = nd(R, n, N_O2)
    wq = vp(R, nd(R, n, N_WV)); wk = vp(R, nd(R, n, N_D)); tbl = nd(R, n, N_TBL)
    foreach_el(R, n, tile, NH + NKV, tid) do r, hh             # a thread a (row, head)
        pos = rw(R, r, 1); isq = hh < NH; j = isq ? hh : hh - NH; w = isq ? wq : wk; rs = rsv(R, isq ? rsq : rsk, r, j); off = isq ? j * HD : NE + j * HD
        src = isq ? gqg : gkv; soff = isq ? j * HD : j * HD; s = 0.0f0
        for q in 0:HD-1                                    # gy = rope'(g): the partner's angle is the same
            a = ev(R, src, r, soff + q); qp = q < 64 ? q + 64 : q - 64; b = ev(R, src, r, soff + qp); an = rope(R, tbl, pos, q)
            gy = q < 64 ? a * cos(an) + b * sin(an) : a * cos(an) - b * sin(an)
            ev!(R, gqn, r, gy, off + q); s = fma(gy * ld(w, q), ev(R, qkv, r, off + q), s)
        end
        c = s * rs * rs * rs / Float32(HD)
        for q in 0:HD-1; ev!(R, out, r, ev(R, gqn, r, off + q) * ld(w, q) * rs - c * ev(R, qkv, r, off + q), off + q); end
    end
    foreach_el(R, n, tile, NKV * HD + NG, tid) do r, k         # v's and the gate's gradients pass through
        k < NKV * HD ? ev!(R, out, r, ev(R, gkv, r, NKV * HD + k), NE + NKV * HD + k) : ev!(R, out, r, ev(R, gqg, r, NE + k - NKV * HD), NE + NKV * HD + k)
    end
end
@inline function emb!(R::RQ, n, tile, tid)                  # x0 = E[token]
    out = nd(R, n, N_O1); E = lp(UInt8, mt(R, nd(R, n, N_M), 0))
    foreach_el(R, n, tile, NE, tid) do r, k; ev!(R, out, r, rdv(E, BF, Int64(rw(R, r, 2)) * NE + k), k); end
end
@inline function embb!(R::RQ, n, tile, tid)              
    gx = nd(R, n, N_I1); mo = lp(Float32, mt(R, nd(R, n, N_M), 4)); i = Int64(tid); u0 = Int64(tile) * 64
    while i < 64 * NE
        u = u0 + i ÷ NE; k = i % NE
        if u < R.NU
            a = ld(R.uniq, 2u); c = ld(R.uniq, 2u + 1); tok = rw(R, ld(R.urows, a), 2); s = 0.0f0
            for q in 0:c-1; s += ev(R, gx, ld(R.urows, a + q), k); end
            p = mo + 4 * (Int64(tok) * NE + k); st!(p, ld(p, 0) + R.iacc * s, 0)
        end
        i += 256
    end
end
@inline function lse!(R::RQ, n, tile, tid)                  # a loss row: lse, the loss
    lpart = nd(R, n, N_I1); lg = nd(R, n, N_I2); out = nd(R, n, N_O1); nc = pln(R, ed(R, lpart, E_PROD), NP_NC)
    foreach_el(R, n, tile, 1, tid) do r, _
        a = lp(Float32, eaddr(R, lpart, r)); mx = -Inf32; for c in 0:nc-1; mx = max(mx, ld(a, 2c)); end
        s = 0.0f0; for c in 0:nc-1; s += exp(ld(a, 2c) - mx) * ld(a, 2c + 1); end
        l = mx + log(s); i = ld(R.linv, r); tgt = rw(R, r, 3)
        ev!(R, out, r, l, 0); ev!(R, out, r, tgt >= 0 ? (l - ev(R, lg, r, tgt)) * ld(R.wgt, i) : 0.0f0, 1)
    end
end
@inline function dlog!(R::RQ, n, tile, col, tid)            # dlogits = (softmax - onehot) weight / R
    lg = nd(R, n, N_I1); lse = nd(R, n, N_I2); out = nd(R, n, N_O1)
    foreach_el(R, n, tile, 4096, tid) do r, kk
        k = Int64(col) * 4096 + kk; k < NV || return
        l = rsv(R, lse, r, 0); tgt = rw(R, r, 3); w = ld(R.wgt, ld(R.linv, r)) * R.lossR
        ev!(R, out, r, tgt >= 0 ? (exp(ev(R, lg, r, k) - l) - (k == tgt ? 1.0f0 : 0.0f0)) * w : 0.0f0, k)
    end
end
@inline function nrm!(R::RQ, sm, n, tile, tid)              # a tile's sum of squares; NRMS: 1 / (|mo| + 1e-7)
    m = nd(R, n, N_M); mo = lp(Float32, mt(R, m, 4)); Rr = mt(R, m, 1); K = mt(R, m, 2); out = nd(R, n, N_O1)
    if nd(R, n, N_KIND) == NRMS
        p = lp(Float32, waddr(R, nd(R, n, N_I1), 0)); nt = pln(R, ed(R, nd(R, n, N_I1), E_PROD), NP_NT); s = 0.0f0
        for i in tid:256:nt-1; s += ld(p, i); end
        sm[tid + 1] = s; bar()
        if tid == 0; t = 0.0f0; for i in 0:255; t += sm[i + 1]; end; wrv!(waddr(R, out, 0), F32, t == 0.0f0 ? 0.0f0 : 1.0f0 / (sqrt(t) + 1.0f-7), 0); end   # (0: no gradient since the last step)
        return
    end
    rt = pln(R, n, NP_RT); i = Int64(tile) * rt * K + tid; hi = min(Int64(tile + 1) * rt, Rr) * K; s = 0.0f0
    while i < hi; v = ld(mo, i); s = fma(v, v, s); i += 256; end
    sm[tid + 1] = s; bar()
    if tid == 0; t = 0.0f0; for q in 0:255; t += sm[q + 1]; end; wrv!(waddr(R, out, 0), F32, t, Int64(tile)); end
end
@inline function step!(R::RQ, n, tile, tid)                 # master -= f X (X: the momentum if no edge)
    m = nd(R, n, N_M); X = nd(R, n, N_I1); ms = lp(Float32, mt(R, m, 3)); mo = lp(Float32, mt(R, m, 4)); W = lp(UInt8, mt(R, m, 0)); W2 = lp(UInt8, mt(R, m, 5)); K = mt(R, m, 2); Rr = mt(R, m, 1); f = ndf(R, n, 0)
    rt = pln(R, n, NP_RT); i = Int64(tile) * rt * K + tid; hi = min(Int64(tile + 1) * rt, Rr) * K
    while i < hi
        v = ld(ms, i) - f * (X > 0 ? rdv(waddr(R, X, i ÷ K), F32, i % K) : ld(mo, i)); st!(ms, v, i); wrv!(W, BF, v, i); W2 != LP0 && wrv!(W2, BF, v, i); i += 256
    end
end
@inline function stage!(R::RQ, n, tile, tid)                # X = s mo
    m = nd(R, n, N_M); X = nd(R, n, N_O1); mo = lp(Float32, mt(R, m, 4)); K = mt(R, m, 2); Rr = mt(R, m, 1); sc = ld(lp(Float32, waddr(R, nd(R, n, N_RS), 0)), 0)
    rt = pln(R, n, NP_RT); i = Int64(tile) * rt * K + tid; hi = min(Int64(tile + 1) * rt, Rr) * K
    while i < hi; wrv!(waddr(R, X, i ÷ K), F32, sc * ld(mo, i), i % K); i += 256; end
end
@inline function momu!(R::RQ, n, tile, tid)                 # mo = f2 mo + dW / acc, the rows contiguous
    m = nd(R, n, N_M); dW = nd(R, n, N_I1); mo = lp(Float32, mt(R, m, 4)); K = mt(R, m, 2); Rr = mt(R, m, 1)
    rt = pln(R, n, NP_RT); i = Int64(tile) * rt * K + tid; hi = min(Int64(tile + 1) * rt, Rr) * K
    while i < hi; st!(mo, R.f2 * ld(mo, i) + R.iacc * rdv(waddr(R, dW, i ÷ K), F32, i % K), i); i += 256; end
end
@inline function stepv!(R::RQ, n, tid)                      # a vector by its momentum
    v = nd(R, n, N_WV); p = vp(R, v); mo = vmo(R, v); len = ld(R.vec, Int64(v - 1) * VF + 1); f = ndf(R, n, 0); i = Int64(tid)
    while i < len; st!(p, ld(p, i) - f * ld(mo, i), i); i += 256; end
end
@inline function exec!(R::RQ, sm, n, tile, col, tid)
    k = nd(R, n, N_KIND)
    if nd(R, n, N_LVL) == WEIGHTS && (k == GEMM || k == STEP || k == STAGE) && nd(R, n, N_RS) > 0   # an untouched momentum: nothing to do
        ld(lp(Float32, waddr(R, nd(R, n, N_RS), 0)), 0) == 0.0f0 && return
    end
    if k == GEMM; gemm!(R, sm, n, tile, col, tid)
    elseif k == NORM; norm!(R, sm, n, tile, tid)
    elseif k == RESID; resid!(R, n, tile, tid)
    elseif k == ATTN; attn!(R, sm, n, tile, col, tid)
    elseif k == ROPE; rope!(R, n, tile, tid)
    elseif k == CONV; conv!(R, n, tile, tid)
    elseif k == EMB; emb!(R, n, tile, tid)
    elseif k == LSE; lse!(R, n, tile, tid)
    elseif k == DLOG; dlog!(R, n, tile, col, tid)
    elseif k == NORMB; normb!(R, sm, n, tile, tid)
    elseif k == SILUB; silub!(R, n, tile, tid)
    elseif k == ATTBKV; attb!(R, sm, n, tile, col, tid, true)
    elseif k == ATTBQ; attb!(R, sm, n, tile, col, tid, false)
    elseif k == ROTNB; rotnb!(R, n, tile, tid)
    elseif k == CONVB1; convb1!(R, n, tile, tid)
    elseif k == CONVB2; convb2!(R, n, tile, tid)
    elseif k == CONVW; convw!(R, n, tile, tid)
    elseif k == NORMW; normw!(R, sm, n, tile, tid)
    elseif k == EMBB; embb!(R, n, tile, tid)
    elseif k == NRM || k == NRMS; nrm!(R, sm, n, tile, tid)
    elseif k == STEP; step!(R, n, tile, tid)
    elseif k == STAGE; stage!(R, n, tile, tid)
    elseif k == MOMU; momu!(R, n, tile, tid)
    elseif k == STEPV; stepv!(R, n, tid)
    end
end
# ---- the kernel: block 0 derives the plan; every block claims tasks in order ----
@kernel unsafe_indices = true function krun!(R::RQ)
    sm = @localmem Float32 (8960,); si = @localmem Int32 (4,)
    tid = Int32(KI.get_local_id(Int32).x - 1); gid = Int32(KI.get_group_id(Int32).x - 1)
    if gid == 0 && tid == 0
        t0 = tick(); derive!(R); fence(); R.dbg & 64 != 0 && atomst!(R.cnt + 4 * (R.nflag - 4 * (Int64(R.nn) + 1)), UInt32((tick() - t0) >> 8))
        atomst!(R.cntH + 4 * (R.nflag - 3), UInt32(pl(R, PH_STAT) + 16 * pl(R, PH_NT))); atomst!(R.q + 4, R.gen)
    end
    if tid == 0; while atomld(R.q + 4) != R.gen; end; end
    bar(); fence()
    if pl(R, PH_STAT) == 0
        nt = pl(R, PH_NT)
        while true
            tid == 0 && (si[1] = Int32(atomadd!(R.q, UInt32(1))))
            bar(); idx = Int64(si[1]); bar()
            idx < nt || break
            n, tile, col = decode(R, idx); t0 = tick(); bs = R.cntH + 4 * (R.nflag - 80 + 2 * Int64(gid))
            tid == 0 && (atomst!(R.cntH + 4 * (R.nflag - 2), UInt32(idx)); atomst!(R.cntH + 4 * (R.nflag - 1), UInt32(n)); atomst!(bs, UInt32(idx)); atomst!(bs + 4, UInt32(1)))
            tid == 0 && waitin!(R, n, tile); bar(); fence(); t1 = tick(); tid == 0 && atomst!(bs + 4, UInt32(2))
            exec!(R, sm, n, tile, col, tid)
            fence(); bar()
            if tid == 0                                    # (debug 64: cycles waited and run per node)
                if R.dbg & 64 != 0; st = R.cnt + 4 * (R.nflag - 4 * (Int64(R.nn) + 1) + 4 * Int64(n)); atomadd!(st, UInt32((t1 - t0) >> 8)); atomadd!(st + 4, UInt32((tick() - t1) >> 8)); atomadd!(st + 8, UInt32(1)); end
                atomst!(cellp(R, n, tile, col), R.gen); atomadd!(cellp(R, n, pln(R, n, NP_NT), 0), UInt32(1))
            end
        end
    end
end

CUDAON && (KernelAbstractions.compiler_options(::KernelAbstractions.Kernel{CUDABackend,KernelAbstractions.NDIteration.StaticSize{(256,)},KernelAbstractions.NDIteration.DynamicSize,typeof(gpu_krun!)}) = (; maxthreads = 256, maxregs = 128))
# ---- the host ----
mutable struct Card
    i::Int; k::Any; nblk::Int; nV::Int
    bufs::Vector{Any}; mat::Vector{Int64}; vec::Vector{Int64}
    aV::Any; node::Any; edge::Any; matd::Any; vecd::Any; rope::Any; plan::Any; cnt::Any; q::Any; rows::Any; lrows::Any; linv::Any; wgt::Any; uniq::Any; urows::Any; segs::Any
end
mutable struct Sys
    G::Graph; g::ST; dir::String; cards::Vector{Card}; train::Bool
    aH::Ptr{UInt8}; aHd::UInt64; nH::Int; cntH::Ptr{UInt8}; cntHd::UInt64; host::Vector{Tuple{String,Ptr{UInt8},Int}}   # host-held weights: name, pointer, bytes
    slots::UInt64; slotsz::Int; nslot::Int; cap::Int; gen::UInt32; plan::Vector{Int64}; nflag::Int
    vram::Dict{Tuple{Int,Int},Any}                         # (matrix, 4 master | 5 momentum) -> VRAM array
end
const NFLAG = 1 << 24
rftab() = Float32[i < (t == 0 ? 32 : 64) ? 1.0f0 / (Float32(t == 0 ? 1.0e6 : 1.0e4)^(Float32(2i) / Float32(HD))) : 0.0f0 for t in 0:1 for i in 0:63]
function hin(G::Graph, n, i)
    nd(f) = G.node[NF*(n-1)+f+1]; i < 4 && return nd(N_I1 + i); i == 4 && return nd(N_RS); k = nd(N_KIND); att = k == ATTBKV || k == ATTBQ
    i == 5 ? ((att || k == ROTNB) ? nd(N_M) : 0) : i == 6 ? (att ? nd(N_NM) : 0) : (att ? nd(N_D) : 0)
end
card_of(G::Graph, n, ncard) = (l = G.node[NF*(n-1)+N_L+1]; k = G.node[NF*(n-1)+N_KIND+1]; k == EMB ? 0 : l >= NL ? ncard - 1 : min(Int(l) ÷ cld(NL, ncard), ncard - 1))
matcard(G::Graph, name, ncard) = occursin("embed", name) ? ncard - 1 : min(parse(Int, match(r"layers\.(\d+)\.", name)[1]) ÷ cld(NL, ncard), ncard - 1)
function load(dir; train = false, ncard = ndev(), hostgb = 6.0, reserve = 3.5e9, nslot = 4, cap = 65536, lr = 0.002, lrv = 0.0002, lre = 0.002)
    t0 = time(); g = sft(joinpath(dir, "model.safetensors")); G = recipe(; lr, lrv, lre); nn = G.nn; ne = G.ne
    for n in 1:nn; G.node[NF*(n-1)+N_CARD+1] = card_of(G, n, ncard); end
    for m in 1:G.nmat                                      # the update nodes go where their matrix is
        c = matcard(G, G.matname[m], ncard)
        for n in 1:nn; G.node[NF*(n-1)+N_M+1] == m && G.node[NF*(n-1)+N_LVL+1] == WEIGHTS && (G.node[NF*(n-1)+N_CARD+1] = c); end
    end
    for n in 1:nn, f in 0:7                                # read on another card
        e = hin(G, n, f); e > 0 || continue; p = G.edge[EF*(e-1)+E_PROD+1]
        p > 0 && G.node[NF*(p-1)+N_CARD+1] != G.node[NF*(n-1)+N_CARD+1] && (G.edge[EF*(e-1)+E_CROSS+1] = 1)
    end
    nH = round(Int, hostgb * 1e9); aH, aHd = pinned(nH); cntH, cntHd = pinned(4NFLAG); ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), cntH, 0, 4NFLAG)
    ca = 0; cb = 0
    for e in 1:ne
        G.edge[EF*(e-1)+E_CACHE+1] != 0 || continue; w = G.edge[EF*(e-1)+E_W+1] * (G.edge[EF*(e-1)+E_DT+1] == BF ? 2 : 4); reach = G.edge[EF*(e-1)+E_REACH+1]
        reach > 0 ? (cb += nextpow(2, reach + 64) * w) : (ca += w)
    end
    slotsz = ca * cap + cb; S = Sys(G, g, dir, Card[], train, aH, aHd, nH, cntH, cntHd, Tuple{String,Ptr{UInt8},Int}[], 0, slotsz, nslot, cap, UInt32(0), Int64[], 0, Dict{Tuple{Int,Int},Any}())
    Ebf = tensor(g, "model.embed_tokens.weight"); Ehost = Ptr{UInt8}(0); Ehd = UInt64(0)
    if ncard > 1 || true; Ehost, Ehd = pinned(2NV * NE); unsafe_copyto!(Ptr{UInt16}(Ehost), pointer(Ebf), NV * NE); end   # the gather's copy, host memory
    for c in 0:ncard-1
        dev!(c + 1); bufs = Any[]; mat = zeros(Int64, MF * G.nmat); vec = zeros(Int64, VF * G.nvec)
        wb = 0; for (m, nm) in enumerate(G.matname); matcard(G, nm, ncard) == c && (wb += prod(matdims(nm)) * 2); end
        W = dalloc(wb); push!(bufs, W); off = 0
        for (m, nm) in enumerate(G.matname)
            matcard(G, nm, ncard) == c || continue; R, K = matdims(nm); src = tensor(g, nm)
            if nm == "model.embed_tokens.weight"
                mat[MF*(m-1)+1] = addr(W) + off; mat[MF*(m-1)+6] = Ehd
            else; mat[MF*(m-1)+1] = addr(W) + off; end
            up!(W, off, src); off += 2R * K
            mat[MF*(m-1)+2] = R; mat[MF*(m-1)+3] = K
        end
        c == 0 && (mat[MF*(1-1)+1] == 0) && (mat[MF*(1-1)+1] = Int64(Ehd); mat[MF*(1-1)+2] = NV; mat[MF*(1-1)+3] = NE)   # the gather on card 0
        vb = sum(veclen(nm) for nm in G.vecname) * 4; Vb = dalloc(vb); push!(bufs, Vb); vmb = train ? dalloc(vb) : nothing; train && push!(bufs, vmb); off = 0
        for (v, nm) in enumerate(G.vecname)
            n = veclen(nm); vec[VF*(v-1)+1] = addr(Vb) + off; vec[VF*(v-1)+2] = n; train && (vec[VF*(v-1)+3] = addr(vmb) + off)
            up!(Vb, off, Float32[bf32(u) for u in tensor(g, nm)]); off += 4n
        end
        train && fill!(vmb, 0)
        if train   # masters and momenta: VRAM, else host
            for which in (5, 4), (m, nm) in enumerate(G.matname)
                matcard(G, nm, ncard) == c || continue; R, K = matdims(nm); nb = 4R * K
                if vfree() - nb > reserve && nm != "model.embed_tokens.weight"
                    b = dalloc(nb); push!(bufs, b); mat[MF*(m-1)+which] = addr(b); S.vram[(m, which)] = b
                    which == 4 ? up!(b, 0, Float32[bf32(u) for u in tensor(g, nm)]) : fill!(b, 0)
                else
                    p, d = pinned(nb); mat[MF*(m-1)+which] = d; push!(S.host, (nm * (which == 4 ? ".master" : ".momentum"), p, nb))
                    if which == 4; a = unsafe_wrap(Array, Ptr{Float32}(p), R * K); src = tensor(g, nm); Threads.@threads for i in 1:R*K; a[i] = bf32(src[i]); end
                    else; ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), p, 0, nb); end
                end
            end
        end
        if c == ncard - 1 && nslot > 0; sl = dalloc(nslot * slotsz); push!(bufs, sl); S.slots = addr(sl); end   
        nV = vfree() - max(round(Int, reserve ÷ 2), 1 << 30); aV = dalloc(nV); push!(bufs, aV)   
        rope = dalloc(4 * 128); up!(rope, 0, rftab())
        node = dalloc(4 * length(G.node)); up!(node, 0, G.node); edge = dalloc(4 * length(G.edge)); up!(edge, 0, G.edge)
        nodef = dalloc(4 * length(G.nodef)); up!(nodef, 0, G.nodef); push!(bufs, nodef)
        matd = dalloc(8 * length(mat)); up!(matd, 0, mat); vecd = dalloc(8 * length(vec)); up!(vecd, 0, vec)
        np = PHN + 8ne + 9nn + 2NSEG + 4NHOLE; plan = dalloc(8np); cnt = dalloc(4NFLAG); fill!(cnt, 0); q = dalloc(16); fill!(q, 0)
        NPM = 1 << 16; rows = dalloc(16NPM); lrows = dalloc(4NPM); linv = dalloc(4NPM); wgt = dalloc(4NPM); uniq = dalloc(8NPM); urows = dalloc(4NPM); segs = dalloc(8NSEG)
        k = krun!(BE, 256); C = Card(c, k, 0, nV, bufs, mat, vec, aV, node, edge, matd, vecd, rope, plan, cnt, q, rows, lrows, linv, wgt, uniq, urows, segs)
        push!(bufs, nodef); C.nblk = resident(C, S); push!(S.cards, C)
    end
    S.plan = Vector{Int64}(undef, PHN + 8ne + 9nn + 2NSEG + 4NHOLE)
    @printf("loaded %s: %d nodes, %d edges, %d cards (%.0f s)\n", dir, nn, ne, ncard, time() - t0); flush(stdout)
    S
end
const DBG = Ref(parse(Int, get(ENV, "LFM3_DBG", "0")))
function rq(S::Sys, C::Card, P, PL, NU, NS, wanted; f2 = 1.0, iacc = 1.0)
    G = S.G; nf = dalloc(4 * length(G.nodef)); up!(nf, 0, G.nodef); push!(C.bufs, nf)   # (nodef: its own copy, the lr may change)
    RQ(Int32(P), Int32(PL), Int32(NU), Int32(NS), Int32(wanted), Int32(C.nblk), Int32(C.i), Int32(length(S.cards)), S.gen, Int32(G.nn), Int32(G.ne), Int32(S.cap),
       Float32(f2), Float32(iacc), Float32(1 / max(PL, 1)),
       lp(Int32, addr(C.rows)), lp(Int32, addr(C.lrows)), lp(Int32, addr(C.linv)), lp(Float32, addr(C.wgt)), lp(Int32, addr(C.uniq)), lp(Int32, addr(C.urows)), lp(Int32, addr(C.segs)),
       lp(Int32, addr(C.node)), lp(Float32, addr(nf)), lp(Int32, addr(C.edge)), lp(Int64, addr(C.matd)), lp(Int64, addr(C.vecd)), lp(Float32, addr(C.rope)),
       lp(Int64, addr(C.plan)), lp(UInt32, addr(C.cnt)), lp(UInt32, S.cntHd), lp(UInt32, addr(C.q)),
       lp(UInt8, addr(C.aV)), Int64(C.nV), lp(UInt8, S.aHd), Int64(S.nH), lp(UInt8, S.slots), Int64(S.slotsz), Int32(DBG[]), Int64(NFLAG))
end
function resident(C::Card, S::Sys)                         # the blocks the card runs at once
    k = C.k; A = rq(S, C, 1, 1, 1, 1, 1)
    nd, wg, it, _ = KernelAbstractions.launch_config(k, 256, nothing)
    launch = KernelAbstractions.select_launch(KernelAbstractions.sizes_only(k), wg, it)
    ctx = KernelAbstractions.mkcontext(k, nd, it, launch; subgroups = KernelAbstractions.kernel_subgroups(k))
    _, tt = KernelAbstractions.prepare_arguments(BE, k.f, ctx, (A,))
    kf = KI.kernel_function(BE, k.f, tt; KernelAbstractions.compiler_options(k)...)
    nb = Ref{Cint}(0)
    if CUDAON; nb[] = CUDA.active_blocks(kf.kern.kernel.fun, 256); at = CUDA.attributes(kf.kern.kernel.fun); @printf("kernel: %d registers, %d B local, %d blocks/SM\n", at[CUDA.FUNC_ATTRIBUTE_NUM_REGS], at[CUDA.FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES], nb[])
    else; AMDGPU.HIP.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(nb, kf.kern.kernel.fun, 256, 0); end
    max(1, Int(nb[])) * KI.multiprocessor_count(BE)
end
# a request: rows, segments, loss rows and weights, wanted
function request!(S::Sys, rows::Matrix{Int32}, segs::Matrix{Int32}, lrows::Vector{Int32}, wgt::Vector{Float32}, wanted; f2 = 1.0, iacc = 1.0)
    P = size(rows, 2); PL = length(lrows); NS = size(segs, 2); S.gen += 1
    linv = fill(Int32(-1), P); for (i, r) in enumerate(lrows); linv[r+1] = i - 1; end
    order = sortperm(rows[3, :]); urows = Int32.(order .- 1); uniq = Int32[]; i = 1
    while i <= P; j = i; while j < P && rows[3, order[j+1]] == rows[3, order[i]]; j += 1; end; push!(uniq, i - 1, j - i + 1); i = j + 1; end
    ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), S.cntH, 0, 4NFLAG)
    for C in S.cards
        dev!(C.i + 1); up!(C.rows, 0, vec(rows)); up!(C.segs, 0, vec(segs)); up!(C.lrows, 0, lrows); up!(C.linv, 0, linv); up!(C.wgt, 0, wgt); up!(C.uniq, 0, uniq); up!(C.urows, 0, urows)
        up!(C.q, 0, UInt32[0, 0, 0, 0]); fill!(C.cnt, 0)
        R = rq(S, C, P, PL, length(uniq) ÷ 2, NS, wanted; f2, iacc); C.k(R; ndrange = 256 * C.nblk)
    end
    await!(S); for C in S.cards; dev!(C.i + 1); sync(); pop!(C.bufs); end
    dev!(1); dn!(S.plan, S.cards[1].plan, 0)
    DBG[] & 64 != 0 && for C in S.cards; prof(S, C); end
    st = S.plan[PH_STAT+1]; st == 0 || (n = S.plan[PH_S0+1]; error("request: status $st, node $n (kind $(S.G.node[NF*(n-1)+N_KIND+1])), VRAM $(S.plan[PH_VUSE+1]), host $(S.plan[PH_HUSE+1])"))
    nothing
end
await!(S::Sys) = nothing; prof(S::Sys, C::Card) = nothing   # (test/lfm3-dev.jl)
# an edge's rows as a matrix (width x rows): from VRAM or host memory
function fetch(S::Sys, e::Int, nr; card = 0, T = Float32)
    p = S.G.edge[EF*(e-1)+E_PROD+1]; card == 0 && (card = p > 0 ? S.G.node[NF*(p-1)+N_CARD+1] + 1 : 1)   # the producer's card holds it: that card's plan
    dev!(card); plan = dn!(Vector{Int64}(undef, length(S.plan)), S.cards[card].plan, 0)
    w = S.G.edge[EF*(e-1)+E_W+1]; dt = S.G.edge[EF*(e-1)+E_DT+1]; loc = plan[PHN+8(e-1)+PL_LOC+1]; off = plan[PHN+8(e-1)+PL_OFF+1]
    S.G.edge[EF*(e-1)+E_CACHE+1] != 0 && (loc = 1; off = plan[PHN+8(e-1)+PL_CA+1] * nr + plan[PHN+8(e-1)+PL_CB+1])   # (one segment, at the pool's start)
    loc == 0 && error("edge $e: not held"); nb = w * nr * (dt == BF ? 2 : 4)
    raw = Vector{UInt8}(undef, nb)
    if loc == 1; dev!(card); dn!(raw, S.cards[card].aV, off) else; unsafe_copyto!(pointer(raw), S.aH + off, nb); end
    a = dt == BF ? Float32[bf32(u) for u in reinterpret(UInt16, raw)] : collect(reinterpret(Float32, raw))
    reshape(a, w, nr)
end
# the logits of every position of ids (one segment)
function logits(S::Sys, ids::Vector{Int}; lrows = Int32.(0:length(ids)-1))
    P = length(ids); rows = Matrix{Int32}(undef, 4, P); for p in 0:P-1; rows[:, p+1] = Int32[0, p, ids[p+1], -1]; end
    request!(S, rows, reshape(Int32[-1, P], 2, 1), lrows, ones(Float32, length(lrows)), 1)
    fetch(S, S.G.names[:lg], length(lrows))
end
function kl(S::Sys, ids, refpath, n)
    t0 = time(); Lg = logits(S, ids[1:n]); tp = time() - t0
    ref = Mmap.mmap(refpath, Matrix{UInt16}, (NV, n)); kls = Float64[]; top = 0
    for p in 1:n
        r = Float64.(bf32.(view(ref, :, p))); o = Float64.(view(Lg, :, p)); (all(isfinite, r) && all(isfinite, o)) || (push!(kls, Inf); continue)
        lr = r .- (maximum(r) + log(sum(exp.(r .- maximum(r))))); lo = o .- (maximum(o) + log(sum(exp.(o .- maximum(o)))))
        push!(kls, sum(exp.(lr) .* (lr .- lo))); top += argmax(r) == argmax(o)
    end
    m = length(kls)
    @printf("KL vs BF16 over %d positions: mean %.6f, median %.6f, max %.5f; top-1 %.4f; %.1f s (%.1f tok/s)\n", m, sum(kls) / m, sort(kls)[cld(m, 2)], maximum(kls), top / m, tp, n / tp)
end
# ---- serving: flashd's engine protocol (flash-sass kern/flashd.c) on a unix socket; the slot keeps the last chat's cache ----
function slotrun(S::Sys, toks, p0)                           # rows p0.. of the slot's chat; the last logits
    n = length(toks); rows = Int32[r == 1 ? 0 : r == 2 ? p0 + i - 1 : r == 3 ? toks[i] : -1 for r in 1:4, i in 1:n]
    request!(S, rows, reshape(Int32[0, p0 + n], 2, 1), Int32[n - 1], ones(Float32, 1), 1); fetch(S, S.G.names[:lg], 1)[:, 1]
end
function pick(l, temp, topk, topp, rng, seen, pen)
    for t in seen; l[t+1] = l[t+1] > 0 ? l[t+1] / pen : l[t+1] * pen; end
    temp <= 0 && return argmax(l) - 1
    ix = partialsortperm(l, 1:topk; rev = true); p = exp.((l[ix] .- l[ix[1]]) ./ temp); c = cumsum(p); k = something(findfirst(>=(topp * c[end]), c), topk)
    ix[findfirst(>=(rand(rng) * c[k]), c[1:k])] - 1
end
function chat(S::Sys, c, n, ids, stops, (temp, topk, topp), seed, pen, held, cap)
    t0 = time(); P = length(ids); gone = false; (P < cap && P > 0) || (println(c, "error: ", P, " positions, the engine holds ", cap); return)
    @async (try; eof(c); catch; end; gone = true)            # (the client closed its side)
    p0 = 0; while p0 < min(P - 1, length(held)) && held[p0+1] == ids[p0+1]; p0 += 1; end   # the cache's prefix; the last token always runs
    length(held) - p0 > 100 && (p0 = 0); resize!(held, p0); l = Float32[]
    for a in p0+1:64:P                                       # (64 rows a request: the rings)
        b = min(P, a + 63); l = slotrun(S, ids[a:b], a - 1); append!(held, ids[a:b]); println(c, "prog ", b, " ", P, " ", p0); flush(c)
    end
    rng = Random.Xoshiro(seed); seen = pen == 1 ? Int[] : copy(ids); gen = 0; ended = false; t1 = t0; n = clamp(n, 1, cap - P)
    while !gone
        t = pick(l, temp, topk, topp, rng, seen, pen); gen += 1; gen == 1 && (t1 = time()); println(c, "tok ", t); flush(c); push!(seen, t); ended = t in stops
        (gen >= n || ended) && break
        l = slotrun(S, [t], P + gen - 1); push!(held, t)
    end
    println(c, "done ", gen, " ", Int(ended), " ", P, " ", round(1e3 * (time() - t0); digits = 1), " ", round(1e3 * (t1 - t0); digits = 1), " ", p0, " 0"); flush(c)
end
function session(S::Sys, c, lk, held, cap)
    stops = Int[]; smp = (0.0, 32, 1.0); seed = UInt64(1); pen = 1.0
    while !eof(c)
        a = split(readline(c)); isempty(a) && continue
        if a[1] in ("timings", "cancel_on_eof"); println(c, "ok: ", join(a, " ")); flush(c)
        elseif a[1] == "stop"; stops = [t for t in parse.(Int, a[2:end]) if t != 0xffffffff]
        elseif a[1] == "sample"; seed = parse(UInt64, a[2]); smp = (parse(Float64, a[3]), parse(Int, a[4]), parse(Float64, a[5]))
        elseif a[1] == "penalty"; pen = parse(Float64, a[2])
        elseif a[1] == "info"; println(c, "info ", cap, " 0"); flush(c)
        elseif a[1] == "chat"; lock(lk) do; chat(S, c, parse(Int, a[2]), parse.(Int, a[3:end]), stops, smp, seed, pen, held, cap); end; return
        end
    end
end
function serve(dir, sock; cap = 4096)
    S = load(dir; nslot = 1, cap); lk = ReentrantLock(); held = Int[]; ispath(sock) && rm(sock); srv = listen(sock); println("serving on ", sock); flush(stdout)
    while true
        c = accept(srv)
        @async try; session(S, c, lk, held, cap); catch e; try; println(c, "error: ", sprint(showerror, e)); catch; end; finally; close(c); end
    end
end
function main(args)
    arg(k, d) = (i = findfirst(==(k), args); i === nothing ? d : args[i+1])
    dir = arg("--model", expanduser("~/models/lfm3-1.7b-base"))
    if (i = findfirst(==("--kl"), args)) !== nothing
        n = parse(Int, args[i+1]); ref = args[i+2]; S = load(dir; nslot = 0)
        ids = Int.(reinterpret(Int32, read(replace(ref, ".logits.bf16" => ".ids"))))
        kl(S, ids, ref, n)
    elseif (i = findfirst(==("--serve"), args)) !== nothing; serve(dir, args[i+1])
    elseif "--train" in args; train(args)
    end
end
# ---- training: what lfm3-data wrote ----
const NPMAX = 1568
struct Data; tok::Vector{Int32}; tr::Vector{NTuple{5,UInt32}}; ex::Vector{NTuple{4,UInt32}}; end
function Data(dir)
    tok = reinterpret(Int32, read(joinpath(dir, "tokens.bin"))) |> collect
    t = reinterpret(UInt32, read(joinpath(dir, "trans.bin"))); e = reinterpret(UInt32, read(joinpath(dir, "examples.bin")))
    Data(tok, [(t[6i+1], t[6i+2], t[6i+3], t[6i+4], t[6i+5]) for i in 0:length(t)÷6-1], [(e[4i+1], e[4i+2], e[4i+3], e[4i+4]) for i in 0:length(e)÷4-1])
end
tfirst(D::Data, t) = Int(D.tr[t+1][1]) | Int(D.tr[t+1][2]) << 32
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
# one window: the loss, its gradient when wanted
function window!(S::Sys, D::Data, w; grad = true, f2 = 1.0, iacc = 1.0)
    t, f0, N, pos, tgt = w
    rows = Matrix{Int32}(undef, 4, N); for p in 0:N-1; rows[:, p+1] = Int32[0, p, D.tok[f0+p+1], -1]; end
    for (p, g) in zip(pos, tgt); rows[4, p+1] = g; end
    request!(S, rows, reshape(Int32[-1, N], 2, 1), Int32.(pos), ones(Float32, length(pos)), grad ? 6 : 2; f2, iacc)
    L = fetch(S, S.G.names[:lse], length(pos)); sum(L[2, :]) / length(pos)
end
step!(S::Sys) = request!(S, reshape(Int32[0, 0, 0, -1], 4, 1), reshape(Int32[-1, 1], 2, 1), Int32[0], Float32[1], 8)
function heldout(S::Sys, D::Data, W)
    t0 = time(); tot = 0.0; n = 0
    for w in W; tot += window!(S, D, w; grad = false) * length(w[4]); n += length(w[4]); end
    @printf("held out: %.4f over %d reply tokens in %d windows (%.0f s)\n", tot / n, n, length(W), time() - t0); flush(stdout)
    tot / n
end
# every trained tensor as a Float32 vector (momenta: name * ".momentum")
function trained(S::Sys; only = nothing)                
    G = S.G; byname = Dict{String,Vector{Float32}}(); want(nm) = only === nothing || (endswith(nm, ".master") ? nm[1:end-7] : nm) in only
    for (nm, p, nb) in S.host; want(nm) && (byname[nm] = copy(unsafe_wrap(Array, Ptr{Float32}(p), nb ÷ 4))); end
    for ((m, which), b) in S.vram; nm = G.matname[m] * (which == 4 ? "" : ".momentum"); want(nm) && (byname[nm] = dn!(Vector{Float32}(undef, length(b) ÷ 4), b, 0)); end
    for C in S.cards
        dev!(C.i + 1)
        for (v, nm) in enumerate(G.vecname)
            C.vec[VF*(v-1)+1] == 0 && continue; n = veclen(nm); off = Int(C.vec[VF*(v-1)+1] - addr(C.bufs[2]))
            byname[nm] = dn!(Vector{Float32}(undef, n), C.bufs[2], off); S.train && (byname[nm*".momentum"] = dn!(Vector{Float32}(undef, n), C.bufs[3], off))
        end
    end
    for k in collect(keys(byname)); endswith(k, ".master") && (byname[k[1:end-7]] = byname[k]; delete!(byname, k)); end
    byname
end
function restore!(S::Sys, byname)                          # the inverse
    G = S.G
    for (nm, p, nb) in S.host; k = endswith(nm, ".master") ? nm[1:end-7] : nm; haskey(byname, k) && copyto!(unsafe_wrap(Array, Ptr{Float32}(p), nb ÷ 4), byname[k]); end
    for ((m, which), b) in S.vram; nm = G.matname[m] * (which == 4 ? "" : ".momentum"); haskey(byname, nm) && up!(b, 0, byname[nm]); end
    for C in S.cards
        dev!(C.i + 1)
        for (v, nm) in enumerate(G.vecname)
            C.vec[VF*(v-1)+1] == 0 && continue; off = Int(C.vec[VF*(v-1)+1] - addr(C.bufs[2]))
            haskey(byname, nm) && up!(C.bufs[2], off, byname[nm]); haskey(byname, nm * ".momentum") && up!(C.bufs[3], off, byname[nm*".momentum"])
        end
    end
end
function bf16(a::AbstractVector{Float32})
    o = Vector{UInt16}(undef, length(a))
    Threads.@threads for i in eachindex(a); u = reinterpret(UInt32, a[i]); o[i] = ((u + 0x00007fff + ((u >> 16) & 0x00000001)) >> 16) % UInt16; end
    o
end
function export_bf16(S::Sys, out)
    mkpath(out)
    for f in readdir(S.dir); (f == "model.safetensors" || endswith(f, ".bak-nobos") || startswith(f, "ref.")) || cp(joinpath(S.dir, f), joinpath(out, f); force = true); end
    byname = trained(S); names = sort(collect(keys(S.g.t))); hdr = IOBuffer(); off = 0; print(hdr, "{")
    for (i, nm) in enumerate(names)
        shape = S.g.t[nm][1]; n = prod(shape); length(byname[nm]) == n || error("$nm: $(length(byname[nm])) values, shape $shape")
        print(hdr, i == 1 ? "" : ",", "\"", nm, "\":{\"dtype\":\"BF16\",\"shape\":[", join(shape, ","), "],\"data_offsets\":[", off, ",", off + 2n, "]}"); off += 2n
    end
    print(hdr, "}"); h = take!(hdr); pad = (8 - length(h) % 8) % 8; append!(h, fill(UInt8(' '), pad))
    open(joinpath(out, "model.safetensors"), "w") do io
        write(io, UInt64(length(h))); write(io, h); for nm in names; write(io, bf16(byname[nm])); end
    end
    println("wrote $out/model.safetensors"); flush(stdout)
end
# a checkpoint: masters, momenta, where the epoch stood; written whole, renamed
function ckpt_write(S::Sys, path, ep, i, best)
    t0 = time(); tmp = path * ".tmp"; byname = trained(S); names = sort(collect(keys(byname)))
    open(tmp, "w") do io
        write(io, Int64(ep), Int64(i), Float64(best), Int64(length(names)))
        for nm in names; write(io, Int64(length(nm))); write(io, nm); write(io, Int64(length(byname[nm]))); write(io, byname[nm]); end
    end
    mv(tmp, path; force = true); @printf("checkpoint: epoch %d window %d (%.0f s)\n", ep, i, time() - t0); flush(stdout)
end
function ckpt_read(S::Sys, path)
    byname = Dict{String,Vector{Float32}}(); ep = 0; i = 0; best = 0.0
    open(path) do io
        ep = read(io, Int64); i = read(io, Int64); best = read(io, Float64); n = read(io, Int64)
        for _ in 1:n; l = read(io, Int64); nm = String(read(io, l)); m = read(io, Int64); byname[nm] = read!(io, Vector{Float32}(undef, m)); end
    end
    restore!(S, byname); step0!(S)
    @printf("resumed: epoch %d window %d, best held out %.4f\n", ep, i, best); flush(stdout)
    Int(ep), Int(i), best
end
step0!(S::Sys) = (for n in 1:S.G.nn; S.G.node[NF*(n-1)+N_KIND+1] == STEP && (S.G.nodef[4(n-1)+1] = 0); end; step!(S))   # W from the masters: a step of 0
# the gradient check: one window (momenta with f2 = 0, acc = 1)
function train(args)
    arg(k, d) = (i = findfirst(==(k), args); i === nothing ? d : args[i+1])
    dir = arg("--model", ""); data = arg("--train", ""); out = arg("--out", ""); epochs = parse(Int, arg("--epochs", "2"))
    lr = parse(Float64, arg("--lr", "0.002")); lrv = parse(Float64, arg("--lrv", "0.0002")); lre = parse(Float64, arg("--lre", "0.002")); beta = parse(Float64, arg("--beta", "0.95"))
    acc = parse(Int, arg("--acc", "8")); nwin = parse(Int, arg("--windows", "0")); seed = parse(Int, arg("--seed", "1")); reserve = parse(Float64, arg("--reserve", "3.5e9"))
    out == "" || mkpath(out); (dir == "" || data == "") && error("usage: lfm3.jl --train DATADIR --model DIR --out OUTDIR [--epochs E --lr L --lrv L --lre L --beta B --acc A --windows W --check PREFIX N --resume]")
    S = load(dir; train = true, nslot = 0, lr, lrv, lre, reserve); D = Data(data)
    Wt = windows(D, false); Wh = windows(D, true)
    nwin > 0 && (Wt = Wt[1:min(nwin, end)]; Wh = Wh[1:min(cld(nwin, 3), end)])
    @printf("%d training windows, %d held out; %d windows a step\n", length(Wt), length(Wh), acc); flush(stdout)
    ck = joinpath(out, "checkpoint.bin"); ep0 = 1; i0 = 0
    if "--resume" in args && isfile(ck); ep0, i0, best = ckpt_read(S, ck) else; best = heldout(S, D, Wh); end
    rng = Random.MersenneTwister(seed); for _ in 1:ep0-1; Random.shuffle(rng, 1:length(Wt)); end
    for ep in ep0:epochs
        order = Random.shuffle(rng, 1:length(Wt)); t0 = time(); tot = 0.0; n = 0
        for (i, wi) in enumerate(order)
            ep == ep0 && i <= i0 && continue
            w = Wt[wi]; t1 = time(); loss = window!(S, D, w; f2 = (i - 1) % acc == 0 ? beta : 1.0, iacc = 1 / acc)
            tot += loss * length(w[4]); n += length(w[4])
            @printf("epoch %d window %d/%d: N %d, %d reply tokens, loss %.4f (%.1f s)\n", ep, i, length(order), w[3], length(w[4]), loss, time() - t1); flush(stdout)
            (i % acc == 0 || i == length(order)) && (t2 = time(); step!(S); @printf("step (%.1f s)\n", time() - t2))
            i % (acc * 32) == 0 && ckpt_write(S, ck, ep, i, best)
        end
        @printf("epoch %d: train loss %.4f over %d reply tokens (%.0f min)\n", ep, tot / n, n, (time() - t0) / 60)
        h = heldout(S, D, Wh)
        h < best ? (best = h; export_bf16(S, out)) : println("not the best ($best): no export")
        ckpt_write(S, ck, ep, length(order), best); flush(stdout)
    end
end
(abspath(PROGRAM_FILE) == @__FILE__) && main(ARGS)
