# Qwen3.6-35B-A3B GSQ, text inference on engi's RX 7700 XT (gfx1101, 12 GiB).
# Native Julia/AMDGPU kernels; no llama.cpp, Python, CUDA, or external tokenizer at runtime.
# One-time dependency setup (outside this checkout):
#   julia --project=~/.julia/environments/qwen36 -e 'using Pkg; Pkg.add("AMDGPU")'
# Run:
#   julia --project=~/.julia/environments/qwen36 qwen3.6-35b-a3b.jl --prompt 'What is 2+2?' --tokens 32
#   ... --download                 fetch the pinned 11.38-GiB text GGUF
#   ... --model FILE --raw         complete an untemplated prompt
# Source checkpoint: https://huggingface.co/ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ
# GGUF: https://huggingface.co/chtisgit/Qwen3.6-35B-A3B-2Bit-GSQ-hybrid-GGUF
# The converter preserves GSQ assignments; BF16->FP16 block scales incur the errors in its model card.
# Weights are allocated once; one cooperative persistent dispatch executes a request.
# The BF16 input embedding stays registered in host memory, with direct GPU reads.
using AMDGPU, Mmap, Printf, Downloads, SHA, Random

const MODEL_REV = "34a912bde355affe80eb1af654475a6fe9feff5f"
const MODEL_FILE = "Qwen3.6-35B-A3B-GSQ-hybrid.gguf"
const MODEL_URL = "https://huggingface.co/chtisgit/Qwen3.6-35B-A3B-2Bit-GSQ-hybrid-GGUF/resolve/$MODEL_REV/$MODEL_FILE"
const MODEL_SHA = "8e50912f5d0703401ca21b09deb2ee8a84d03536ade6fb44815fcb40a0b03596"
const MODEL_BYTES = 12_217_776_832
const DEFAULT_MODEL = joinpath(homedir(), "models-cache", "qwen3.6-gsq", MODEL_FILE)
const TYPES = Dict(0=>(1,4), 1=>(1,2), 8=>(32,34), 30=>(1,2), 42=>(64,18))

struct Tensor
    typ::Int
    dims::Vector{Int}                 # GGML order: contiguous input, output, expert
    off::Int
    nbytes::Int
end
struct GModel
    kv::Dict{String,Any}
    tensors::Dict{String,Tensor}
    map::Vector{UInt8}
end
rowbytes(t::Tensor) = t.dims[1] ÷ TYPES[t.typ][1] * TYPES[t.typ][2]
function gguf(path)
    open(path) do io
        String(read(io, 4)) == "GGUF" || error("$path: not GGUF")
        read(io, UInt32) == 3 || error("GGUF v3 required")
        nt, nk = Int(read(io, UInt64)), Int(read(io, UInt64))
        rdstr() = String(read(io, Int(read(io, UInt64))))
        scalar = Dict(0=>UInt8, 1=>Int8, 2=>UInt16, 3=>Int16, 4=>UInt32, 5=>Int32,
                      6=>Float32, 7=>Bool, 10=>UInt64, 11=>Int64, 12=>Float64)
        function val(t)
            t == 8 && return rdstr()
            if t == 9
                et, n = Int(read(io, UInt32)), Int(read(io, UInt64))
                return et == 8 ? [rdstr() for _ in 1:n] : [read(io, scalar[et]) for _ in 1:n]
            end
            read(io, scalar[t])
        end
        kv = Dict{String,Any}()
        for _ in 1:nk
            k = rdstr(); kv[k] = val(Int(read(io, UInt32)))
        end
        raw = Tuple{String,Int,Vector{Int},Int}[]
        for _ in 1:nt
            n = rdstr(); nd = Int(read(io, UInt32)); dims = [Int(read(io, UInt64)) for _ in 1:nd]
            typ, off = Int(read(io, UInt32)), Int(read(io, UInt64))
            haskey(TYPES, typ) || error("$n: unsupported GGML type $typ; use the GSQ hybrid GGUF")
            push!(raw, (n, typ, dims, off))
        end
        align = Int(get(kv, "general.alignment", 32)); base = cld(position(io), align) * align
        tensors = Dict{String,Tensor}()
        for (n, typ, dims, off) in raw
            block, bytes = TYPES[typ]; prod(dims) % block == 0 || error("$n: invalid block shape")
            nb = prod(dims) ÷ block * bytes
            base + off + nb <= filesize(path) || error("$path: incomplete download ($n is past EOF)")
            tensors[n] = Tensor(typ, dims, base + off, nb)
        end
        GModel(kv, tensors, Mmap.mmap(path))
    end
end
bytes(g, t::Tensor) = view(g.map, t.off+1:t.off+t.nbytes)
bf32(u::UInt16) = reinterpret(Float32, UInt32(u) << 16)
function tensor_f32(g, name)
    t = g.tensors[name]; b = bytes(g, t)
    t.typ == 0 && return copy(reinterpret(Float32, b))
    t.typ == 1 && return Float32.(reinterpret(Float16, b))
    t.typ == 30 && return bf32.(reinterpret(UInt16, b))
    error("$name: expected unquantized tensor")
end
function embedding(g, id)
    t = g.tensors["token_embd.weight"]; 0 <= id < t.dims[2] || error("invalid token $id")
    t.typ == 30 || error("expected BF16 embedding")
    o = t.off + id * rowbytes(t)
    bf32.(reinterpret(UInt16, view(g.map, o+1:o+rowbytes(t))))
end

# Device primitives use 32-lane subgroups, valid on both wave32 and wave64 AMD targets.
@inline tid0() = Int32(workitemIdx().x) - Int32(1)
@inline gid0() = (Int32(workgroupIdx().x)-Int32(1))*Int32(workgroupDim().x)+tid0()
@inline function sum32(x)
    for d in (16, 8, 4, 2, 1)
        x += AMDGPU.Device.shfl_down(x, Int32(d), UInt32(32))
    end
    x
end
@inline sigm(x) = 1.0f0 / (1.0f0 + exp(-x))
@inline clockgpu() = ccall("extern __ockl_steadyctr_u64", llvmcall, UInt64, ())
@inline silu(x) = x * sigm(x)
@inline softplus(x) = max(x, 0.0f0) + log1p(exp(-abs(x)))
@inline function u16(W, o)
    @inbounds UInt16(W[o+1]) | UInt16(W[o+2]) << 8
end
@inline half(W, o) = Float32(reinterpret(Float16, u16(W, o)))
@inline function weight(W, row, k, K, ::Val{T}) where T
    if T == 42 || T == 142
        block=T==42 ? Int32(64) : Int32(128); bs=T==42 ? Int32(18) : Int32(34)
        o = row * (K ÷ block * bs) + (k ÷ block) * bs
        j = k & (block-Int32(1))
        q = @inbounds W[o+Int32(3)+(j>>Int32(2))]
        return half(W, o) * Float32(Int32((q >> (Int32(2)*(j&Int32(3)))) & 0x03)-Int32(1))
    elseif T == 8
        o = row * (K ÷ Int32(32) * Int32(34)) + (k ÷ Int32(32)) * Int32(34)
        q = @inbounds W[o+Int32(3)+(k&Int32(31))]
        return half(W, o) * Float32(reinterpret(Int8, q))
    elseif T == 30
        return bf32(u16(W, Int32(2)*(row*K+k)))
    elseif T == 1
        return half(W, Int32(2)*(row*K+k))
    else
        o = Int32(4)*(row*K+k)
        u = @inbounds UInt32(W[o+1]) | UInt32(W[o+2]) << 8 | UInt32(W[o+3]) << 16 | UInt32(W[o+4]) << 24
        return reinterpret(Float32, u)
    end
end
struct DMat
    W::ROCArray{UInt8,1}
    typ::Int
    K::Int
    rows::Int
end
function DMat(g, name)
    t = g.tensors[name]
    if t.typ==42
        # The source repeats each group-of-128 scale in two 64-value blocks.
        # Remove the identical copy, preserving every weight bit and scale bit.
        t.dims[1]%128==0 || error("Q2_0 row is not a multiple of 128")
        nb=t.nbytes÷36; packed=Vector{UInt8}(undef,nb*34)
        GC.@preserve g packed begin
            src=pointer(g.map)+t.off; dst=pointer(packed)
            for j in 0:nb-1
                a=unsafe_load(Ptr{UInt16}(src+36j)); b=unsafe_load(Ptr{UInt16}(src+36j+18))
                a==b || error("$name: unequal scale pair; cannot losslessly compact")
                unsafe_store!(Ptr{UInt16}(dst+34j),a)
                unsafe_copyto!(dst+34j+2,src+36j+2,16)
                unsafe_copyto!(dst+34j+18,src+36j+20,16)
            end
        end
        W=ROCArray(packed); AMDGPU.synchronize()
        return DMat(W,142,t.dims[1],prod(t.dims[2:end]))
    end
    W=ROCArray{UInt8}(undef,t.nbytes); copyto!(W,1,g.map,t.off+1,t.nbytes)
    DMat(W,t.typ,t.dims[1],prod(t.dims[2:end]))
end
function k_norm!(y, x, w, n::Int32, eps::Float32)
    ti = tid0(); s = @ROCStaticLocalArray(Float32, 256)
    a = 0.0f0
    for i in ti:Int32(256):n-Int32(1)
        @inbounds a = fma(x[i+1], x[i+1], a)
    end
    s[ti+1] = a; sync_workgroup()
    for d in (128,64,32,16,8,4,2,1)
        ti < d && (@inbounds s[ti+1] += s[ti+d+1]); sync_workgroup()
    end
    scale = 1.0f0 / sqrt(s[1]/Float32(n)+eps)
    for i in ti:Int32(256):n-Int32(1)
        @inbounds y[i+1] = x[i+1]*scale*w[i+1]
    end
    return
end
function k_add!(x,y,n::Int32)
    for i in gid0():Int32(workgroupDim().x)*Int32(gridGroupDim().x):n-Int32(1)
        @inbounds x[i+1]+=y[i+1]
    end
    return
end
function k_swiglu!(h,g,u,n::Int32)
    for i in gid0():Int32(workgroupDim().x)*Int32(gridGroupDim().x):n-Int32(1)
        @inbounds h[i+1]=silu(g[i+1])*u[i+1]
    end
    return
end

# DeltaNet: causal depthwise convolution; unit-length q/k; gated delta recurrence;
# per-value-head RMS norm followed by SiLU(z). GGUF value heads are in tiled order.
function k_conv!(u,x,st,w,t::Int32,n::Int32)
    for c in gid0():Int32(workgroupDim().x)*Int32(gridGroupDim().x):n-Int32(1)
        @inbounds begin
            a=w[4c+1]*st[((t+1)&3)*n+c+1]
            a=fma(w[4c+2],st[((t+2)&3)*n+c+1],a)
            a=fma(w[4c+3],st[((t+3)&3)*n+c+1],a)
            a=fma(w[4c+4],x[c+1],a)
            u[c+1]=silu(a); st[(t&3)*n+c+1]=x[c+1]
        end
    end
    return
end
function k_qknorm!(q,k,u,nk::Int32,eps::Float32,h::Int32)
    d=tid0()
    sq=@ROCStaticLocalArray(Float32,256); sk=@ROCStaticLocalArray(Float32,256)
    @inbounds begin
        qv=d<128 ? u[128h+d+1] : 0.0f0; kv=d<128 ? u[128nk+128h+d+1] : 0.0f0
        sq[d+1]=qv*qv; sk[d+1]=kv*kv; sync_workgroup()
        for s in (128,64,32,16,8,4,2,1)
            if d<s; sq[d+1]+=sq[d+s+1]; sk[d+1]+=sk[d+s+1]; end
            sync_workgroup()
        end
        if d<128
            q[128h+d+1]=qv/max(sqrt(sq[1]),eps)*0.08838834764831845f0
            k[128h+d+1]=kv/max(sqrt(sk[1]),eps)
        end
    end
    return
end
function k_delta!(o,state,q,k,u,a,b,dt,aa,nk::Int32,bh::Int32)
    # Eight value rows per workgroup, one wave per row and four keys per lane.
    h=bh÷Int32(16); ti=tid0(); lane=ti&Int32(31)
    c=(bh&Int32(15))*Int32(8)+(ti>>Int32(5)); d=Int32(4)*lane
    @inbounds begin
        kh=h%nk
        decay=exp(aa[h+1]*softplus(a[h+1]+dt[h+1])); beta=sigm(b[h+1])
        base=Int32(16384)*h+Int32(128)*c+d
        sv=ntuple(j->state[base+j]*decay,Val(4))
        kv=ntuple(j->k[Int32(128)*kh+d+j],Val(4))
        qv=ntuple(j->q[Int32(128)*kh+d+j],Val(4))
        r=0.0f0
        for j in 1:4; r=fma(sv[j],kv[j],r); end
        r=AMDGPU.Device.shfl(sum32(r),Int32(0),UInt32(32))
        delta=beta*(u[Int32(256)*nk+Int32(128)*h+c+1]-r)
        nv=ntuple(j->fma(kv[j],delta,sv[j]),Val(4)); v=0.0f0
        for j in 1:4
            state[base+j]=nv[j]; v=fma(nv[j],qv[j],v)
        end
        v=sum32(v); lane==0 && (o[Int32(128)*h+c+1]=v)
    end
    return
end
function k_gatednorm!(o,raw,z,w,eps::Float32,h::Int32)
    d=tid0()
    sh=@ROCStaticLocalArray(Float32,256)
    @inbounds begin
        x=d<128 ? raw[128h+d+1] : 0.0f0; sh[d+1]=x*x; sync_workgroup()
        for s in (128,64,32,16,8,4,2,1)
            d<s && (sh[d+1]+=sh[d+s+1]); sync_workgroup()
        end
        d<128 && (o[128h+d+1]=x/sqrt(sh[1]/128.0f0+eps)*w[d+1]*silu(z[128h+d+1]))
    end
    return
end

# Gated grouped-query attention. FP32 KV cache; partial NeoX RoPE (64 of 256 dims).
# Text positions are equal on all IMRoPE axes, so the multimodal sections coincide here.
function k_attnprep!(q,kc,vc,qg,k,v,qn,kn,t::Int32,nq::Int32,nkv::Int32,theta::Float32,eps::Float32,h::Int32,ropepos::Int32=t)
    d=tid0()
    sh=@ROCStaticLocalArray(Float32,256); ss=@ROCStaticLocalArray(Float32,256)
    isq=h<nq; hh=isq ? h : h-nq
    @inbounds begin
        x=isq ? qg[512hh+d+1] : k[256hh+d+1]
        ss[d+1]=x*x; sync_workgroup()
        for s in (128,64,32,16,8,4,2,1)
            d<s && (ss[d+1]+=ss[d+s+1]); sync_workgroup()
        end
        w=isq ? qn[d+1] : kn[d+1]
        sh[d+1]=x/sqrt(ss[1]/256.0f0+eps)*w; sync_workgroup()
        y=sh[d+1]
        if d<64
            p=d%Int32(32); angle=Float32(ropepos)*theta^(-Float32(p)/32.0f0)
            sn,cs=sincos(angle)
            y=d<32 ? fma(-sh[d+33],sn,sh[d+1]*cs) : fma(sh[d-31],sn,sh[d+1]*cs)
        end
        if isq
            q[256hh+d+1]=y
        else
            ix=(t*nkv+hh)*256+d+1; kc[ix]=y; vc[ix]=v[256hh+d+1]
        end
    end
    return
end
function k_attnscore!(scores,q,kc,t::Int32,nq::Int32,nkv::Int32,ctx::Int32,h::Int32,chunk::Int32)
    ti=tid0(); lane=ti&Int32(31)
    p=chunk*Int32(8)+(ti>>Int32(5))
    if p<=t
        kh=h÷(nq÷nkv); a=0.0f0
        for d in lane:Int32(32):Int32(255)
            @inbounds a=fma(q[256h+d+1],kc[(p*nkv+kh)*256+d+1],a)
        end
        a=sum32(a)*0.0625f0
        lane==0 && (@inbounds scores[h*ctx+p+1]=a)
    end
    return
end
function k_softmax!(p,t::Int32,ctx::Int32,h::Int32)
    d=tid0(); sh=@ROCStaticLocalArray(Float32,256)
    m=-Inf32
    for i in d:Int32(256):t; @inbounds m=max(m,p[h*ctx+i+1]); end
    sh[d+1]=m; sync_workgroup()
    for s in (128,64,32,16,8,4,2,1)
        d<s && (sh[d+1]=max(sh[d+1],sh[d+s+1])); sync_workgroup()
    end
    m=sh[1]; sync_workgroup(); total=0.0f0
    for i in d:Int32(256):t
        @inbounds v=exp(p[h*ctx+i+1]-m); p[h*ctx+i+1]=v; total+=v
    end
    sh[d+1]=total; sync_workgroup()
    for s in (128,64,32,16,8,4,2,1)
        d<s && (sh[d+1]+=sh[d+s+1]); sync_workgroup()
    end
    den=sh[1]
    for i in d:Int32(256):t; @inbounds p[h*ctx+i+1]/=den; end
    return
end
function k_attnout!(o,p,vc,qg,t::Int32,nq::Int32,nkv::Int32,ctx::Int32,h::Int32)
    d=tid0(); kh=h÷(nq÷nkv); a=0.0f0
    for i in Int32(0):t
        @inbounds a=fma(p[h*ctx+i+1],vc[(i*nkv+kh)*256+d+1],a)
    end
    @inbounds o[256h+d+1]=a*sigm(qg[512h+256+d+1])
    return
end

# Selected experts are read directly from resident, packed GPU weights.
function k_moemix!(x,down,weights,shared,sg,n::Int32,ns::Int32)
    for i in gid0():Int32(workgroupDim().x)*Int32(gridGroupDim().x):n-Int32(1)
        @inbounds begin
            a=shared[i+1]*sigm(sg[1])
            for e in Int32(0):ns-Int32(1); a=fma(down[e*n+i+1],weights[e+1],a); end
            x[i+1]+=a
        end
    end
    return
end

# Tokenizer follows the existing flashnext.jl byte BPE and qwen35 Unicode pre-split.
const F_UNDEF, F_NUMBER, F_LETTER, F_SEP, F_MARK, F_PUNCT, F_SYM, F_CTRL, F_WS = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x100
const WS_CPTS = Set{UInt32}([0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x20, 0x85, 0xa0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005,
    0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000])
function cflags(c::UInt32)
    c >= 0x110000 && return 0
    cat = Base.Unicode.category_code(Char(c))
    f = cat == 0 ? F_UNDEF : cat <= 5 ? F_LETTER : cat <= 8 ? F_MARK : cat <= 11 ? F_NUMBER : cat <= 18 ? F_PUNCT :
        cat <= 22 ? F_SYM : cat <= 25 ? F_SEP : F_CTRL
    c in WS_CPTS ? f | F_WS : f
end

struct Tok
    text::Vector{String}
    typ::Vector{Int32}
    ids::Dict{String,Int32}
    ranks::Dict{String,Int32}
    special::Vector{Int32}          # control / user-defined / unknown ids, longest text first
    byte2u::Vector{String}          # byte -> its byte-level character (UTF-8)
    u2byte::Dict{Char,UInt8}
    eos::Int32
end
function Tok(g::GModel)
    kv = g.kv
    kv["tokenizer.ggml.model"] == "gpt2" || error("tokenizer model is not gpt2")
    kv["tokenizer.ggml.pre"] == "qwen35" || error("tokenizer pre is $(kv["tokenizer.ggml.pre"]), only qwen35 is implemented")
    text = kv["tokenizer.ggml.tokens"]::Vector{String}
    typ = Int32.(kv["tokenizer.ggml.token_type"])
    ids = Dict{String,Int32}()
    for (i, s) in enumerate(text)
        haskey(ids, s) || (ids[s] = i - 1)
    end
    ranks = Dict{String,Int32}()
    for (i, s) in enumerate(kv["tokenizer.ggml.merges"]::Vector{String})
        haskey(ranks, s) || (ranks[s] = i - 1)
    end
    special = Int32[i - 1 for i in eachindex(typ) if typ[i] in (2, 3, 4)]
    sort!(special; by = i -> (-ncodeunits(text[i+1]), i))
    byte2u = Vector{String}(undef, 256); u2byte = Dict{Char,UInt8}()
    extra = 0
    for b in 0:255
        keep = (0x21 <= b <= 0x7e) || (0xa1 <= b <= 0xac) || (0xae <= b <= 0xff)
        c = keep ? Char(b) : Char(256 + extra)
        keep || (extra += 1)
        byte2u[b+1] = string(c); u2byte[c] = UInt8(b)
    end
    Tok(text, typ, ids, ranks, special, byte2u, u2byte, Int32(kv["tokenizer.ggml.eos_token_id"]))
end

# llama.cpp cpts_from_utf8: an invalid sequence yields U+FFFD and consumes one byte
function cpts(s::Vector{UInt8})
    out = UInt32[]; i = 1; n = length(s)
    while i <= n
        b = s[i]
        if b < 0x80
            push!(out, b); i += 1; continue
        end
        ok = false
        if (b & 0x40) != 0
            if (b & 0x20) == 0
                if i + 1 <= n && (s[i+1] & 0xc0) == 0x80
                    r = (UInt32(b & 0x1f) << 6) | (s[i+1] & 0x3f)
                    r >= 0x80 && (push!(out, r); i += 2; ok = true)
                end
            elseif (b & 0x10) == 0
                if i + 2 <= n && (s[i+1] & 0xc0) == 0x80 && (s[i+2] & 0xc0) == 0x80
                    r = (UInt32(b & 0x0f) << 12) | (UInt32(s[i+1] & 0x3f) << 6) | (s[i+2] & 0x3f)
                    r >= 0x800 && (push!(out, r); i += 3; ok = true)
                end
            elseif (b & 0x08) == 0
                if i + 3 <= n && (s[i+1] & 0xc0) == 0x80 && (s[i+2] & 0xc0) == 0x80 && (s[i+3] & 0xc0) == 0x80
                    r = (UInt32(b & 0x07) << 18) | (UInt32(s[i+1] & 0x3f) << 12) | (UInt32(s[i+2] & 0x3f) << 6) | (s[i+3] & 0x3f)
                    0x10000 <= r <= 0x10ffff && (push!(out, r); i += 4; ok = true)
                end
            end
        end
        ok || (push!(out, 0xfffd); i += 1)
    end
    out
end

# unicode_regex_split_custom_qwen35 over cp: word lengths in codepoints
function split_qwen35(cp::Vector{UInt32})
    n = length(cp); OUT = 0xffffffff
    C(p) = p < n ? cp[p+1] : OUT
    FL(p) = p < n ? cflags(cp[p+1]) : 0
    lower(c) = (0x41 <= c <= 0x5a) ? c + 0x20 : c
    lens = Int[]; prev = 0
    add(e) = (e > prev && push!(lens, e - prev); prev = e)
    pos = 0
    while pos < n
        c = C(pos); f = FL(pos)
        if c == UInt32('\'') && pos + 1 < n
            c1 = lower(C(pos + 1))
            if c1 in (UInt32('s'), UInt32('t'), UInt32('m'), UInt32('d'))
                add(pos + 2); pos += 2; continue
            end
            if pos + 2 < n
                c2 = lower(C(pos + 2))
                if (c1 == UInt32('r') && c2 == UInt32('e')) || (c1 == UInt32('v') && c2 == UInt32('e')) || (c1 == UInt32('l') && c2 == UInt32('l'))
                    add(pos + 3); pos += 3; continue
                end
            end
        end
        if !(c == 0x0d || c == 0x0a || (f & F_NUMBER) != 0)
            if (f & (F_LETTER | F_MARK)) != 0 || (FL(pos + 1) & (F_LETTER | F_MARK)) != 0
                pos += 1
                while (FL(pos) & (F_LETTER | F_MARK)) != 0
                    pos += 1
                end
                add(pos); continue
            end
        end
        if (f & F_NUMBER) != 0
            pos += 1; add(pos); continue
        end
        f2 = c == 0x20 ? FL(pos + 1) : f
        if (f2 & (F_WS | F_LETTER | F_MARK | F_NUMBER)) == 0 && f != 0
            pos += (c == 0x20)
            while (f2 & (F_WS | F_LETTER | F_MARK | F_NUMBER)) == 0 && f2 != 0
                pos += 1; f2 = FL(pos)
            end
            c2 = C(pos)
            while c2 == 0x0d || c2 == 0x0a
                pos += 1; c2 = C(pos)
            end
            add(pos); continue
        end
        nws = 0; last_rn = 0
        while (FL(pos + nws) & F_WS) != 0
            c2 = C(pos + nws)
            (c2 == 0x0d || c2 == 0x0a) && (last_rn = pos + nws + 1)
            nws += 1
        end
        if last_rn > 0
            pos = last_rn; add(pos); continue
        end
        if nws > 1 && C(pos + nws) != OUT
            pos += nws - 1; add(pos); continue
        end
        if nws > 0
            pos += nws; add(pos); continue
        end
        pos += 1; add(pos)
    end
    lens
end

# BPE of one byte-level word (llm_tokenizer_bpe_session::tokenize)
function bpe_word!(out::Vector{Int32}, tk::Tok, w::String)
    syms = String[]                         # symbol texts (UTF-8 characters of the byte-level word)
    for c in w
        push!(syms, string(c))
    end
    n = length(syms)
    prv = collect(-1:n-2); nxt = [i == n - 1 ? -1 : i + 1 for i in 0:n-1]
    heap = Tuple{Int32,Int,Int,Int}[]       # (rank, left, right, combined length)
    function addb(l, r)
        (l < 0 || r < 0) && return
        rk = get(tk.ranks, syms[l+1] * " " * syms[r+1], Int32(-1))
        rk < 0 && return
        hpush!(heap, (rk, l, r, ncodeunits(syms[l+1]) + ncodeunits(syms[r+1])))
    end
    for i in 1:n-1
        addb(i - 1, i)
    end
    while !isempty(heap)
        rk, l, r, len = hpop!(heap)
        (isempty(syms[l+1]) || isempty(syms[r+1]) || ncodeunits(syms[l+1]) + ncodeunits(syms[r+1]) != len || nxt[l+1] != r) && continue
        syms[l+1] = syms[l+1] * syms[r+1]; syms[r+1] = ""
        nxt[l+1] = nxt[r+1]
        nxt[r+1] >= 0 && (prv[nxt[r+1]+1] = l)
        addb(prv[l+1], l); addb(l, nxt[l+1])
    end
    i = n == 0 ? -1 : 0
    while i != -1
        s = syms[i+1]
        if !isempty(s)
            id = get(tk.ids, s, Int32(-1))
            if id >= 0
                push!(out, id)
            else
                for b in codeunits(s)
                    b1 = get(tk.ids, String([b]), Int32(-1))
                    b1 >= 0 && push!(out, b1)
                end
            end
        end
        i = nxt[i+1]
    end
    out
end
hless(a, b) = a[1] < b[1] || (a[1] == b[1] && a[2] < b[2])
function hpush!(h, x)
    push!(h, x); i = length(h)
    while i > 1
        p = i ÷ 2
        hless(h[i], h[p]) || break
        h[i], h[p] = h[p], h[i]; i = p
    end
end
function hpop!(h)
    top = h[1]; last = pop!(h)
    if !isempty(h)
        h[1] = last; i = 1; n = length(h)
        while true
            l = 2i; r = l + 1; m = i
            l <= n && hless(h[l], h[m]) && (m = l)
            r <= n && hless(h[r], h[m]) && (m = r)
            m == i && break
            h[i], h[m] = h[m], h[i]; i = m
        end
    end
    top
end

function encode_raw!(out::Vector{Int32}, tk::Tok, s::Vector{UInt8})
    isempty(s) && return out
    cp = cpts(s)
    at = 0
    for wl in split_qwen35(cp)
        io = IOBuffer()
        for i in at+1:at+wl
            for b in codeunits(string(Char(cp[i])))
                write(io, tk.byte2u[b+1])
            end
        end
        at += wl
        bpe_word!(out, tk, String(take!(io)))
    end
    out
end
# text -> ids; special tokens written as text become their ids (longest first, left to right)
function encode(tk::Tok, text::AbstractString; parse_special = true)
    frags = Any[Vector{UInt8}(codeunits(text))]
    for id in tk.special
        (!parse_special && tk.typ[id+1] in (2, 3)) && continue
        st = Vector{UInt8}(codeunits(tk.text[id+1]))
        isempty(st) && continue
        nf = Any[]
        for f in frags
            if f isa Int32
                push!(nf, f); continue
            end
            rest = f
            while true
                r = findfirst(st, rest)
                if r === nothing
                    isempty(rest) || push!(nf, rest)
                    break
                end
                first(r) > 1 && push!(nf, rest[1:first(r)-1])
                push!(nf, id)
                rest = rest[last(r)+1:end]
            end
        end
        frags = nf
    end
    out = Int32[]
    for f in frags
        f isa Int32 ? push!(out, f) : encode_raw!(out, tk, f)
    end
    out
end
function piece(tk::Tok, id)
    t = tk.text[id+1]
    tk.typ[id+1] != 1 && return Vector{UInt8}(codeunits(t))
    out = UInt8[]
    for c in t
        b = get(tk.u2byte, c, nothing)
        b === nothing ? append!(out, codeunits(string(c))) : push!(out, b)
    end
    out
end
hidden(tk::Tok, id) = tk.typ[id+1] in (2, 3)
chat_prompt(user) = "<|im_start|>user\n" * user * "<|im_end|>\n<|im_start|>assistant\n<think>\n"

struct Config
    hidden::Int
    layers::Int
    experts::Int
    used::Int
    ff::Int
    nq::Int
    nkv::Int
    nkh::Int
    nvh::Int
    eps::Float32
    theta::Float32
end
function Config(g)
    get(g.kv,"general.architecture","") == "qwen35moe" || error("expected qwen35moe architecture")
    v(k)=Int(g.kv["qwen35moe."*k])
    c=Config(v("embedding_length"),v("block_count"),v("expert_count"),v("expert_used_count"),
             g.tensors["blk.0.ffn_gate_exps.weight"].dims[2],v("attention.head_count"),v("attention.head_count_kv"),
             v("ssm.group_count"),v("ssm.time_step_rank"),Float32(g.kv["qwen35moe.attention.layer_norm_rms_epsilon"]),
             Float32(g.kv["qwen35moe.rope.freq_base"]))
    (c.hidden,c.layers,c.experts,c.used,c.ff,c.nq,c.nkv,c.nkh,c.nvh)==(2048,40,256,8,512,16,2,16,32) ||
        error("this kernel targets Qwen3.6-35B-A3B; unexpected configuration $c")
    Int(g.kv["qwen35moe.rope.dimension_count"])==64 || error("expected 64 rotary dimensions")
    c
end

# The only inference path is one HIP cooperative dispatch for the whole request.
# Descriptors point to weights allocated once at load time. The input embedding is
# a permanently registered host mapping, read directly by the GPU (4 KiB/token).
struct PMat
    ptr::UInt64
    K::Int32
    rows::Int32
    typ::Int32
end
const MNAMES=("attn_qkv.weight","attn_gate.weight","ssm_alpha.weight","ssm_beta.weight","ssm_out.weight",
              "attn_q.weight","attn_k.weight","attn_v.weight","attn_output.weight","ffn_gate_inp.weight",
              "ffn_gate_inp_shexp.weight","ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight",
              "ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight")
const VNAMES=("attn_norm.weight","post_attention_norm.weight","ssm_a","ssm_dt.bias",
              "ssm_conv1d.weight","ssm_norm.weight","attn_q_norm.weight","attn_k_norm.weight")
struct PBlock
    mat::NTuple{17,PMat}
    vec::NTuple{8,UInt64}
    state::UInt64
    conv::UInt64
    kc::UInt64
    vc::UInt64
    linear::Bool
end
PMat(m::DMat)=PMat(UInt(pointer(m.W)),Int32(m.K),Int32(m.rows),Int32(m.typ))
@inline av(::Type{T},p::UInt64,n::Integer) where T =
    AMDGPU.ROCDeviceArray{T,1,1}((Int(n),),Core.LLVMPtr{T,1}(p))
@inline fv(p::UInt64,n::Integer=16384)=av(Float32,p,n)
@inline function p_mul_t!(y,m,x,ids,mode::Int32,::Val{T}) where T
    ti=tid0(); lane=ti&Int32(31); bid=Int32(workgroupIdx().x)-Int32(1)
    ng=Int32(gridGroupDim().x); K=m.K
    # mode 0=dense; 1=selected expert gate/up; 2=selected expert down.
    rpe=mode==Int32(1) ? Int32(512) : Int32(2048)
    R=mode==Int32(0) ? m.rows : Int32(8)*rpe
    W=av(UInt8,m.ptr,Int64(typemax(Int32)))
    for r in bid*Int32(8)+(ti>>Int32(5)):ng*Int32(8):R-Int32(1)
        slot=r÷rpe; row=mode==Int32(0) ? r : (@inbounds ids[slot+1])*rpe+r%rpe
        xo=mode==Int32(2) ? slot*K : Int32(0); a=0.0f0
        for k in lane:Int32(32):K-Int32(1)
            @inbounds a=fma(weight(W,row,k,K,Val(T)),x[xo+k+1],a)
        end
        a=sum32(a); lane==0 && (@inbounds y[r+1]=a)
    end
    return
end
@inline function p_mul!(y,m,x,ids,mode::Int32=Int32(0))
    if m.typ==142; p_mul_t!(y,m,x,ids,mode,Val(142))
    elseif m.typ==42; p_mul_t!(y,m,x,ids,mode,Val(42))
    elseif m.typ==8; p_mul_t!(y,m,x,ids,mode,Val(8))
    elseif m.typ==30; p_mul_t!(y,m,x,ids,mode,Val(30))
    elseif m.typ==1; p_mul_t!(y,m,x,ids,mode,Val(1))
    else; p_mul_t!(y,m,x,ids,mode,Val(0))
    end
    AMDGPU.Device.sync_grid()
    return
end
@inline function p_route!(ids,w,r)
    if gid0()==0
        @inbounds begin
            for s in Int32(0):Int32(7)
                best=-Inf32; ix=Int32(0)
                for e in Int32(0):Int32(255)
                    selected=false
                    for j in Int32(0):s-Int32(1); selected|=ids[j+1]==e; end
                    if !selected && r[e+1]>best; best=r[e+1]; ix=e; end
                end
                ids[s+1]=ix; w[s+1]=best
            end
            den=0.0f0; mx=w[1]
            for s in 1:8; w[s]=exp(w[s]-mx); den+=w[s]; end
            for s in 1:8; w[s]/=den; end
        end
    end
    AMDGPU.Device.sync_grid()
    return
end
@inline function p_argmax!(logits,part,ix,control,output,n::Int32,gen::Int32,eos::Int32,imend::Int32)
    bid=Int32(workgroupIdx().x)-Int32(1); ti=tid0(); stride=Int32(gridItemDim().x)
    sh=@ROCStaticLocalArray(Float32,256); si=@ROCStaticLocalArray(Int32,256)
    best=-Inf32; id=typemax(Int32)
    @inbounds begin
        for j in gid0():stride:n-Int32(1)
            v=logits[j+1]
            if v>best || (v==best && j<id); best=v; id=j; end
        end
        sh[ti+1]=best; si[ti+1]=id; sync_workgroup()
        for s in (128,64,32,16,8,4,2,1)
            if ti<s
                v=sh[ti+s+1]; j=si[ti+s+1]
                if v>sh[ti+1] || (v==sh[ti+1] && j<si[ti+1]); sh[ti+1]=v; si[ti+1]=j; end
            end
            sync_workgroup()
        end
        if ti==0; part[bid+1]=sh[1]; ix[bid+1]=si[1]; end
    end
    AMDGPU.Device.sync_grid()
    if gid0()==0
        @inbounds begin
            best=part[1]; id=ix[1]
            for j in 2:Int32(gridGroupDim().x)
                if part[j]>best || (part[j]==best && ix[j]<id); best=part[j]; id=ix[j]; end
            end
            output[gen+1]=id; control[2]=gen+Int32(1); control[3]=id
            control[1]=(id==eos || id==imend || !isfinite(best)) ? Int32(1) : Int32(0)
        end
    end
    AMDGPU.Device.sync_grid()
    return
end
function k_request!(blocks,head,normout,embedding,b,ids,output,control,ticks,part,ix,expertids,
                    np::Int32,nt::Int32,ctx::Int32,eos::Int32,imend::Int32)
    bid=Int32(workgroupIdx().x)-Int32(1); ng=Int32(gridGroupDim().x)
    if gid0()==0; @inbounds ticks[1]=clockgpu(); control[1]=Int32(0); control[2]=Int32(0); end
    AMDGPU.Device.sync_grid()
    for pos in Int32(0):np+nt-Int32(2)
        id=pos<np ? (@inbounds ids[pos+1]) : (@inbounds control[3])
        @inbounds for j in gid0():Int32(gridItemDim().x):Int32(2047)
            b.x[j+1]=bf32(embedding[Int64(id)*Int64(2048)+j+1])
        end
        AMDGPU.Device.sync_grid()
        @inbounds for l in Int32(1):Int32(40)
            B=blocks[l]; M=B.mat; V=B.vec
            bid==0 && k_norm!(b.xn,b.x,fv(V[1]),Int32(2048),1.0f-6)
            AMDGPU.Device.sync_grid()
            if B.linear
                p_mul!(b.qkv,M[1],b.xn,expertids); p_mul!(b.z,M[2],b.xn,expertids)
                p_mul!(b.alpha,M[3],b.xn,expertids); p_mul!(b.beta,M[4],b.xn,expertids)
                k_conv!(b.u,b.qkv,fv(B.conv,32768),fv(V[5],32768),pos,Int32(8192)); AMDGPU.Device.sync_grid()
                for h in bid:ng:Int32(15); k_qknorm!(b.qdn,b.kdn,b.u,Int32(16),1.0f-6,h); end
                AMDGPU.Device.sync_grid()
                for task in bid:ng:Int32(511)
                    k_delta!(b.rawdn,fv(B.state,524288),b.qdn,b.kdn,b.u,b.alpha,b.beta,fv(V[4]),fv(V[3]),Int32(16),task)
                end
                AMDGPU.Device.sync_grid()
                for h in bid:ng:Int32(31); k_gatednorm!(b.on,b.rawdn,b.z,fv(V[6]),1.0f-6,h); end
                AMDGPU.Device.sync_grid(); p_mul!(b.y,M[5],b.on,expertids)
            else
                p_mul!(b.qg,M[6],b.xn,expertids); p_mul!(b.k,M[7],b.xn,expertids); p_mul!(b.v,M[8],b.xn,expertids)
                for h in bid:ng:Int32(17)
                    k_attnprep!(b.q,fv(B.kc,512ctx),fv(B.vc,512ctx),b.qg,b.k,b.v,fv(V[7]),fv(V[8]),pos,Int32(16),Int32(2),1.0f7,1.0f-6,h)
                end
                AMDGPU.Device.sync_grid()
                nch=cld(pos+Int32(1),Int32(8))
                for task in bid:ng:Int32(16)*nch-Int32(1)
                    k_attnscore!(b.scores,b.q,fv(B.kc,512ctx),pos,Int32(16),Int32(2),ctx,task%Int32(16),task÷Int32(16))
                end
                AMDGPU.Device.sync_grid()
                for h in bid:ng:Int32(15); k_softmax!(b.scores,pos,ctx,h); end
                AMDGPU.Device.sync_grid()
                for h in bid:ng:Int32(15); k_attnout!(b.on,b.scores,fv(B.vc,512ctx),b.qg,pos,Int32(16),Int32(2),ctx,h); end
                AMDGPU.Device.sync_grid(); p_mul!(b.y,M[9],b.on,expertids)
            end
            k_add!(b.x,b.y,Int32(2048)); AMDGPU.Device.sync_grid()
            bid==0 && k_norm!(b.xn,b.x,fv(V[2]),Int32(2048),1.0f-6)
            AMDGPU.Device.sync_grid()
            p_mul!(b.route,M[10],b.xn,expertids); p_mul!(b.sg,M[11],b.xn,expertids)
            p_mul!(b.shg,M[12],b.xn,expertids); p_mul!(b.shu,M[13],b.xn,expertids)
            k_swiglu!(b.shh,b.shg,b.shu,Int32(512)); AMDGPU.Device.sync_grid()
            p_mul!(b.shout,M[14],b.shh,expertids); p_route!(expertids,b.weights,b.route)
            p_mul!(b.eg,M[15],b.xn,expertids,Int32(1)); p_mul!(b.eu,M[16],b.xn,expertids,Int32(1))
            k_swiglu!(b.eh,b.eg,b.eu,Int32(4096)); AMDGPU.Device.sync_grid()
            p_mul!(b.ed,M[17],b.eh,expertids,Int32(2))
            k_moemix!(b.x,b.ed,b.weights,b.shout,b.sg,Int32(2048),Int32(8)); AMDGPU.Device.sync_grid()
        end
        if pos>=np-Int32(1)
            bid==0 && k_norm!(b.xn,b.x,normout,Int32(2048),1.0f-6)
            AMDGPU.Device.sync_grid(); p_mul!(b.logits,head,b.xn,expertids)
            p_argmax!(b.logits,part,ix,control,output,Int32(length(b.logits)),pos-np+Int32(1),eos,imend)
            if pos==np-Int32(1) && gid0()==0; ticks[2]=clockgpu(); end
            AMDGPU.Device.sync_grid()
            @inbounds control[1]!=0 && break
        end
    end
    if gid0()==0; @inbounds ticks[3]=clockgpu(); end
    return
end

mutable struct Engine
    g::GModel
    tk::Tok
    cfg::Config
    weights::Vector{DMat}
    floats::Vector{ROCArray{Float32,1}}
    blocks::ROCArray{PBlock,1}
    head::PMat
    output_norm::ROCArray{Float32,1}
    embedding::ROCArray{UInt16,1}
    b::NamedTuple
    ids::ROCArray{Int32,1}
    output::ROCArray{Int32,1}
    control::ROCArray{Int32,1}
    ticks::ROCArray{UInt64,1}
    part::ROCArray{Float32,1}
    ix::ROCArray{Int32,1}
    expertids::ROCArray{Int32,1}
    ctx::Int
    groups::Int
end
z(n)=AMDGPU.zeros(Float32,n)
function Engine(g,ctx;compileonly=false)
    AMDGPU.functional() || error("AMDGPU is not functional")
    c=Config(g); tk=Tok(g); weights=DMat[]; floats=ROCArray{Float32,1}[]; bs=PBlock[]
    keepvec(v)=(push!(floats,v); UInt(pointer(v)))
    for l in 0:39
        p="blk.$l."; linear=haskey(g.tensors,p*"ssm_a")
        ms=ntuple(17) do i
            n=p*MNAMES[i]
            haskey(g.tensors,n) || return PMat(0,0,0,0)
            t=g.tensors[n]
            m=compileonly ? DMat(AMDGPU.zeros(UInt8,1),t.typ==42 ? 142 : t.typ,t.dims[1],prod(t.dims[2:end])) : DMat(g,n)
            push!(weights,m); PMat(m)
        end
        vs=ntuple(i->haskey(g.tensors,p*VNAMES[i]) ? keepvec(compileonly ? z(1) : ROCArray(tensor_f32(g,p*VNAMES[i]))) : UInt64(0),8)
        push!(bs,PBlock(ms,vs,keepvec(z(compileonly ? 1 : linear ? 524288 : 0)),keepvec(z(compileonly ? 1 : linear ? 32768 : 0)),
                       keepvec(z(compileonly ? 1 : linear ? 0 : 512ctx)),keepvec(z(compileonly ? 1 : linear ? 0 : 512ctx)),linear))
    end
    ht=g.tensors["output.weight"]
    head=compileonly ? DMat(AMDGPU.zeros(UInt8,1),ht.typ,ht.dims[1],ht.dims[2]) : DMat(g,"output.weight"); push!(weights,head)
    et=g.tensors["token_embd.weight"]
    emb=compileonly ? AMDGPU.zeros(UInt16,1) : GC.@preserve g unsafe_wrap(ROCArray,Ptr{UInt16}(pointer(g.map)+et.off),(prod(et.dims),))
    sizes=(:x=>2048,:xn=>2048,:y=>2048,:qkv=>8192,:u=>8192,:z=>4096,:alpha=>32,:beta=>32,:qdn=>2048,:kdn=>2048,
           :rawdn=>4096,:on=>4096,:qg=>8192,:q=>4096,:k=>512,:v=>512,:scores=>16ctx,:route=>256,:sg=>1,
           :shg=>512,:shu=>512,:shh=>512,:shout=>2048,:eg=>4096,:eu=>4096,:eh=>4096,:ed=>16384,:weights=>8,:logits=>length(tk.text))
    b=NamedTuple{Tuple(first.(sizes))}(Tuple(z(last(s)) for s in sizes))
    groups=Int(AMDGPU.HIP.properties(AMDGPU.device()).multiProcessorCount)
    Engine(g,tk,c,weights,floats,ROCArray(bs),PMat(head),ROCArray(tensor_f32(g,"output_norm.weight")),emb,b,
           AMDGPU.zeros(Int32,ctx),AMDGPU.zeros(Int32,ctx),AMDGPU.zeros(Int32,4),AMDGPU.zeros(UInt64,3),
           z(groups),AMDGPU.zeros(Int32,groups),AMDGPU.zeros(Int32,8),ctx,groups)
end
function reset!(E)
    # Every state buffer belongs to the final four floats entries of its block.
    for B in Array(E.blocks)
        for p in (B.state,B.conv,B.kc,B.vc)
            for v in E.floats
                UInt(pointer(v))==p && fill!(v,0.0f0)
            end
        end
    end
    AMDGPU.synchronize(); E
end
function request_kernel(E,np,nt;compile=false)
    args=(E.blocks,E.head,E.output_norm,E.embedding,E.b,E.ids,E.output,E.control,E.ticks,E.part,E.ix,E.expertids,
          Int32(np),Int32(nt),Int32(E.ctx),E.tk.eos,get(E.tk.ids,"<|im_end|>",E.tk.eos))
    k=@roc launch=false maxthreads=256 k_request!(args...)
    compile || k(args...;groupsize=256,gridsize=E.groups,cooperative=true)
    k
end
function request!(E,ids,nt)
    length(ids)+nt<=E.ctx || error("context too small")
    copyto!(E.ids,1,ids,1,length(ids)); request_kernel(E,length(ids),nt); AMDGPU.synchronize()
    ctl=Array(E.control); out=Array(E.output)[1:ctl[2]]; ticks=Array(E.ticks)
    out, ticks, ctl
end
function download_model(path)
    mkpath(dirname(abspath(path))); tmp=path*".part"; Downloads.download(MODEL_URL,tmp)
    filesize(tmp)==MODEL_BYTES || error("incomplete download")
    bytes2hex(open(sha256,tmp))==MODEL_SHA || error("SHA-256 mismatch")
    mv(tmp,path;force=true)
end
function main(args)
    if "--help" in args
        println("julia --project=~/.julia/environments/qwen36 qwen3.6-35b-a3b.jl [--model FILE] [--prompt TEXT] [--tokens N] [--ctx N] [--raw] [--think] [--download] [--compile]")
        return
    end
    opts=Dict{String,String}(); flags=Set{String}(); i=1
    valued=Set(["--model","--prompt","--prompt-file","--tokens","--ctx"])
    while i<=length(args)
        a=args[i]
        if a in valued
            i<length(args) || error("$a needs a value"); opts[a]=args[i+1]; i+=2
        elseif a in ("--raw","--think","--download","--compile")
            push!(flags,a); i+=1
        else; error("unknown option $a")
        end
    end
    path=expanduser(get(opts,"--model",DEFAULT_MODEL)); "--download" in flags && return download_model(path)
    isfile(path) || error("missing model; run with --download")
    isfile(path*".aria2") && error("model download is still running")
    nt=parse(Int,get(opts,"--tokens","32")); ctx=parse(Int,get(opts,"--ctx","2048"))
    nt>0 && ctx>0 || error("token/context counts must be positive")
    prompt=haskey(opts,"--prompt-file") ? read(opts["--prompt-file"],String) : get(opts,"--prompt","What is 2+2? Answer briefly.")
    E=Engine(gguf(path),ctx;compileonly="--compile" in flags)
    tm=@elapsed request_kernel(E,1,1;compile=true)
    tm<=120 || error("compile time exceeds two minutes: $tm s")
    "--compile" in flags && return
    request!(E,Int32[0],1); reset!(E)
    rate=Ref{Cint}()
    AMDGPU.HIP.hipDeviceGetAttribute(rate,AMDGPU.HIP.hipDeviceAttributeWallClockRate,AMDGPU.HIP.device_id(AMDGPU.device()))
    frequency=Float64(rate[])*1000; frequency>0 || error("no GPU steady clock frequency")
    # A marker covers tokenization, copies, dispatch, synchronization and decoding.
    # rocprofv3 records the same monotonic CPU time domain for all these activities.
    ccall((:roctxRangePushA,"/opt/rocm/lib/librocprofiler-sdk-roctx.so"),Cint,(Cstring,),"qwen36/request")
    text="--raw" in flags ? prompt : chat_prompt(prompt)*("--think" in flags ? "" : "\n</think>\n\n")
    ids=encode(E.tk,text); isempty(ids) && error("empty prompt")
    out,ticks,ctl=request!(E,ids,nt)
    decoded=UInt8[]
    for id in out
        id in (E.tk.eos,get(E.tk.ids,"<|im_end|>",E.tk.eos)) && break
        append!(decoded,piece(E.tk,id))
    end
    get(ENV,"QWEN36_OUTPUT","")=="" || write(ENV["QWEN36_OUTPUT"],decoded)
    ccall((:roctxRangePop,"/opt/rocm/lib/librocprofiler-sdk-roctx.so"),Cint,())
    pre=Float64(ticks[2]-ticks[1])/frequency
    dec=max(Float64(ticks[3]-ticks[2])/frequency,eps(Float64)); steps=max(ctl[2]-1,0)
    @printf("prefill tps: %.2f\ndecode tps: %.2f\nidle time %%: unmeasured\n",length(ids)/pre,steps/dec)
end
if abspath(PROGRAM_FILE)==@__FILE__()
    main(ARGS)
end
