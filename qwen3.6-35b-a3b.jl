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
# Q2_0 experts stay mapped on the host; only the eight selected experts per layer cross PCIe.
# Attention/shared weights stay on the GPU. This leaves VRAM for engi's running desktop.
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
@inline silu(x) = x * sigm(x)
@inline softplus(x) = max(x, 0.0f0) + log1p(exp(-abs(x)))
@inline function u16(W, o)
    @inbounds UInt16(W[o+1]) | UInt16(W[o+2]) << 8
end
@inline half(W, o) = Float32(reinterpret(Float16, u16(W, o)))
@inline function weight(W, row, k, K, ::Val{T}) where T
    if T == 42
        o = row * (K ÷ Int32(64) * Int32(18)) + (k ÷ Int32(64)) * Int32(18)
        j = k & Int32(63)
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
function k_gemv!(y, W, x, K::Int32, R::Int32, rpe::Int32, ::Val{T}) where T
    ti = tid0(); lane = ti & Int32(31)
    r = (Int32(workgroupIdx().x)-Int32(1))*Int32(8)+(ti>>Int32(5))
    if r < R
        xo = rpe == 0 ? Int32(0) : (r ÷ rpe) * K
        a = 0.0f0
        for k in lane:Int32(32):K-Int32(1)
            @inbounds a = fma(weight(W, r, k, K, Val(T)), x[xo+k+1], a)
        end
        a = sum32(a)
        lane == 0 && (@inbounds y[r+1] = a)
    end
    return
end
struct DMat
    W::ROCArray{UInt8,1}
    typ::Int
    K::Int
    rows::Int
end
function DMat(g, name)
    t = g.tensors[name]
    DMat(ROCArray(collect(bytes(g, t))), t.typ, t.dims[1], prod(t.dims[2:end]))
end
function gemv!(y, m::DMat, x; rpe=0)
    @roc groupsize=256 gridsize=cld(m.rows,8) k_gemv!(y,m.W,x,Int32(m.K),Int32(m.rows),Int32(rpe),Val(m.typ))
    y
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
norm!(y,x,w,eps) = (@roc groupsize=256 gridsize=1 k_norm!(y,x,w,Int32(length(x)),eps); y)
function k_add!(x,y,n::Int32)
    i=gid0(); i<n && (@inbounds x[i+1]+=y[i+1]); return
end
add!(x,y) = (@roc groupsize=256 gridsize=cld(length(x),256) k_add!(x,y,Int32(length(x))); x)
function k_swiglu!(h,g,u,n::Int32)
    i=gid0(); i<n && (@inbounds h[i+1]=silu(g[i+1])*u[i+1]); return
end
swiglu!(h,g,u) = (@roc groupsize=256 gridsize=cld(length(h),256) k_swiglu!(h,g,u,Int32(length(h))); h)

# DeltaNet: causal depthwise convolution; unit-length q/k; gated delta recurrence;
# per-value-head RMS norm followed by SiLU(z). GGUF value heads are in tiled order.
function k_conv!(u,x,st,w,t::Int32,n::Int32)
    c=gid0()
    if c<n
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
function k_qknorm!(q,k,u,nk::Int32,eps::Float32)
    h=Int32(workgroupIdx().x)-Int32(1); d=tid0()
    sq=@ROCStaticLocalArray(Float32,128); sk=@ROCStaticLocalArray(Float32,128)
    @inbounds begin
        qv=u[128h+d+1]; kv=u[128nk+128h+d+1]
        sq[d+1]=qv*qv; sk[d+1]=kv*kv; sync_workgroup()
        for s in (64,32,16,8,4,2,1)
            if d<s; sq[d+1]+=sq[d+s+1]; sk[d+1]+=sk[d+s+1]; end
            sync_workgroup()
        end
        q[128h+d+1]=qv/max(sqrt(sq[1]),eps)*0.08838834764831845f0
        k[128h+d+1]=kv/max(sqrt(sk[1]),eps)
    end
    return
end
function k_delta!(o,state,q,k,u,a,b,dt,aa,nk::Int32)
    # One workgroup per (value head, value channel). Threads cover contiguous keys.
    bh=Int32(workgroupIdx().x)-Int32(1); h=bh÷Int32(128); c=bh&Int32(127); d=tid0()
    sh=@ROCStaticLocalArray(Float32,128)
    @inbounds begin
        kh=h%nk; kv=k[128kh+d+1]; qv=q[128kh+d+1]
        decay=exp(aa[h+1]*softplus(a[h+1]+dt[h+1])); beta=sigm(b[h+1])
        ix=16384h+128c+d+1
        sv=state[ix]*decay; sh[d+1]=sv*kv; sync_workgroup()
        for s in (64,32,16,8,4,2,1)
            d<s && (sh[d+1]+=sh[d+s+1]); sync_workgroup()
        end
        delta=beta*(u[256nk+128h+c+1]-sh[1]); sync_workgroup()
        sv=fma(kv,delta,sv); state[ix]=sv; sh[d+1]=sv*qv; sync_workgroup()
        for s in (64,32,16,8,4,2,1)
            d<s && (sh[d+1]+=sh[d+s+1]); sync_workgroup()
        end
        d==0 && (o[128h+c+1]=sh[1])
    end
    return
end
function k_gatednorm!(o,raw,z,w,eps::Float32)
    h=Int32(workgroupIdx().x)-Int32(1); d=tid0()
    sh=@ROCStaticLocalArray(Float32,128)
    @inbounds begin
        x=raw[128h+d+1]; sh[d+1]=x*x; sync_workgroup()
        for s in (64,32,16,8,4,2,1)
            d<s && (sh[d+1]+=sh[d+s+1]); sync_workgroup()
        end
        o[128h+d+1]=x/sqrt(sh[1]/128.0f0+eps)*w[d+1]*silu(z[128h+d+1])
    end
    return
end

# Gated grouped-query attention. FP32 KV cache; partial NeoX RoPE (64 of 256 dims).
# Text positions are equal on all IMRoPE axes, so the multimodal sections coincide here.
function k_attnprep!(q,kc,vc,qg,k,v,qn,kn,t::Int32,nq::Int32,nkv::Int32,theta::Float32,eps::Float32)
    h=Int32(workgroupIdx().x)-Int32(1); d=tid0()
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
            p=d%Int32(32); angle=Float32(t)*theta^(-Float32(p)/32.0f0)
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
function k_attnscore!(scores,q,kc,t::Int32,nq::Int32,nkv::Int32,ctx::Int32)
    h=Int32(workgroupIdx().x)-Int32(1); ti=tid0(); lane=ti&Int32(31)
    p=(Int32(workgroupIdx().y)-Int32(1))*Int32(8)+(ti>>Int32(5))
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
function k_softmax!(p,t::Int32,ctx::Int32)
    h=Int32(workgroupIdx().x)-Int32(1); d=tid0(); sh=@ROCStaticLocalArray(Float32,256)
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
function k_attnout!(o,p,vc,qg,t::Int32,nq::Int32,nkv::Int32,ctx::Int32)
    h=Int32(workgroupIdx().x)-Int32(1); d=tid0(); kh=h÷(nq÷nkv); a=0.0f0
    for i in Int32(0):t
        @inbounds a=fma(p[h*ctx+i+1],vc[(i*nkv+kh)*256+d+1],a)
    end
    @inbounds o[256h+d+1]=a*sigm(qg[512h+256+d+1])
    return
end

# Routing runs over only 256 logits on the host. Quantized expert slices are copied
# without dequantization into reusable GPU buffers; intermediate activations remain on GPU.
function k_moemix!(x,down,weights,shared,sg,n::Int32,ns::Int32)
    i=gid0()
    if i<n
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
struct Block
    linear::Bool
    mat::Dict{String,DMat}
    vec::Dict{String,ROCArray{Float32,1}}
    state::ROCArray{Float32,1}
    conv::ROCArray{Float32,1}
    kc::ROCArray{Float32,1}
    vc::ROCArray{Float32,1}
end
z(n)=AMDGPU.zeros(Float32,n)
function Block(g,c,l,ctx)
    p="blk.$l."; linear=haskey(g.tensors,p*"ssm_a")
    mats=["ffn_gate_inp.weight","ffn_gate_inp_shexp.weight","ffn_gate_shexp.weight",
          "ffn_up_shexp.weight","ffn_down_shexp.weight"]
    vecs=["attn_norm.weight","post_attention_norm.weight"]
    if linear
        append!(mats,["attn_qkv.weight","attn_gate.weight","ssm_alpha.weight","ssm_beta.weight","ssm_out.weight"])
        append!(vecs,["ssm_a","ssm_dt.bias","ssm_conv1d.weight","ssm_norm.weight"])
    else
        append!(mats,["attn_q.weight","attn_k.weight","attn_v.weight","attn_output.weight"])
        append!(vecs,["attn_q_norm.weight","attn_k_norm.weight"])
    end
    Block(linear,Dict(n=>DMat(g,p*n) for n in mats),Dict(n=>ROCArray(tensor_f32(g,p*n)) for n in vecs),
          z(linear ? 128*128*c.nvh : 0),z(linear ? 4*(256*c.nkh+128*c.nvh) : 0),
          z(linear ? 0 : 256*c.nkv*ctx),z(linear ? 0 : 256*c.nkv*ctx))
end
mutable struct Engine
    g::GModel
    cfg::Config
    tk::Tok
    blocks::Vector{Block}
    output::DMat
    output_norm::ROCArray{Float32,1}
    b::Dict{Symbol,ROCArray{Float32,1}}
    expert_gpu::NTuple{3,DMat}
    expert_host::NTuple{3,Vector{UInt8}}
    host_route::Vector{Float32}
    ctx::Int
    pos::Int
end
function Engine(g,ctx)
    AMDGPU.functional() || error("AMDGPU is not functional; check ROCm and /dev/kfd access")
    c=Config(g); ctx>=1 || error("--ctx must be positive")
    tk=Tok(g); blocks=Block[]
    for l in 0:c.layers-1
        push!(blocks,Block(g,c,l,ctx))
        l%10==9 && (println(stderr,"Loaded $(l+1)/$(c.layers) layers");flush(stderr))
    end
    sizes=[:x=>c.hidden,:xn=>c.hidden,:y=>c.hidden,:qkv=>256*c.nkh+128*c.nvh,:u=>256*c.nkh+128*c.nvh,
           :z=>128*c.nvh,:alpha=>c.nvh,:beta=>c.nvh,:qdn=>128*c.nkh,:kdn=>128*c.nkh,:rawdn=>128*c.nvh,
           :on=>128*c.nvh,:qg=>512*c.nq,:q=>256*c.nq,:k=>256*c.nkv,:v=>256*c.nkv,:scores=>c.nq*ctx,
           :route=>c.experts,:sg=>1,:shg=>c.ff,:shu=>c.ff,:shh=>c.ff,:shout=>c.hidden,
           :eg=>c.ff*c.used,:eu=>c.ff*c.used,:eh=>c.ff*c.used,:ed=>c.hidden*c.used,:weights=>c.used,
           :logits=>length(tk.text)]
    b=Dict(k=>z(n) for (k,n) in sizes)
    egpu=DMat[]; ehost=Vector{UInt8}[]
    for n in ("gate","up","down")
        t=g.tensors["blk.0.ffn_$(n)_exps.weight"]
        t.typ==42 || error("expected Q2_0 routed experts")
        nb=t.nbytes÷c.experts*c.used
        push!(egpu,DMat(AMDGPU.zeros(UInt8,nb),42,t.dims[1],t.dims[2]*c.used))
        push!(ehost,Vector{UInt8}(undef,nb))
    end
    Engine(g,c,tk,blocks,DMat(g,"output.weight"),ROCArray(tensor_f32(g,"output_norm.weight")),b,
           Tuple(egpu),Tuple(ehost),zeros(Float32,c.experts),ctx,0)
end
function stream_experts!(E,l,ids)
    c=E.cfg
    for (i,n) in enumerate(("gate","up","down"))
        t=E.g.tensors["blk.$l.ffn_$(n)_exps.weight"]
        t.typ==42 && t.dims[3]==c.experts || error("unexpected expert tensor in layer $l")
        nb=t.nbytes÷c.experts; dst=E.expert_host[i]
        for (slot,id) in enumerate(ids)
            copyto!(dst,(slot-1)*nb+1,E.g.map,t.off+id*nb+1,nb)
        end
        copyto!(E.expert_gpu[i].W,dst)
    end
end
function step!(E,id; logits=true)
    E.pos<E.ctx || error("context exhausted at $(E.pos) tokens (increase --ctx)")
    c=E.cfg; b=E.b; t=Int32(E.pos)
    copyto!(b[:x],embedding(E.g,id))
    for (i,B) in enumerate(E.blocks)
        M=B.mat; V=B.vec
        norm!(b[:xn],b[:x],V["attn_norm.weight"],c.eps)
        if B.linear
            gemv!(b[:qkv],M["attn_qkv.weight"],b[:xn]); gemv!(b[:z],M["attn_gate.weight"],b[:xn])
            gemv!(b[:alpha],M["ssm_alpha.weight"],b[:xn]); gemv!(b[:beta],M["ssm_beta.weight"],b[:xn])
            n=length(b[:qkv])
            @roc groupsize=256 gridsize=cld(n,256) k_conv!(b[:u],b[:qkv],B.conv,V["ssm_conv1d.weight"],t,Int32(n))
            @roc groupsize=128 gridsize=c.nkh k_qknorm!(b[:qdn],b[:kdn],b[:u],Int32(c.nkh),c.eps)
            @roc groupsize=128 gridsize=128*c.nvh k_delta!(b[:rawdn],B.state,b[:qdn],b[:kdn],b[:u],b[:alpha],b[:beta],V["ssm_dt.bias"],V["ssm_a"],Int32(c.nkh))
            @roc groupsize=128 gridsize=c.nvh k_gatednorm!(b[:on],b[:rawdn],b[:z],V["ssm_norm.weight"],c.eps)
            gemv!(b[:y],M["ssm_out.weight"],b[:on])
        else
            gemv!(b[:qg],M["attn_q.weight"],b[:xn]); gemv!(b[:k],M["attn_k.weight"],b[:xn]); gemv!(b[:v],M["attn_v.weight"],b[:xn])
            @roc groupsize=256 gridsize=c.nq+c.nkv k_attnprep!(b[:q],B.kc,B.vc,b[:qg],b[:k],b[:v],V["attn_q_norm.weight"],V["attn_k_norm.weight"],t,Int32(c.nq),Int32(c.nkv),c.theta,c.eps)
            @roc groupsize=256 gridsize=(c.nq,cld(E.pos+1,8)) k_attnscore!(b[:scores],b[:q],B.kc,t,Int32(c.nq),Int32(c.nkv),Int32(E.ctx))
            @roc groupsize=256 gridsize=c.nq k_softmax!(b[:scores],t,Int32(E.ctx))
            @roc groupsize=256 gridsize=c.nq k_attnout!(b[:on],b[:scores],B.vc,b[:qg],t,Int32(c.nq),Int32(c.nkv),Int32(E.ctx))
            gemv!(b[:y],M["attn_output.weight"],b[:on])
        end
        add!(b[:x],b[:y]); norm!(b[:xn],b[:x],V["post_attention_norm.weight"],c.eps)
        gemv!(b[:route],M["ffn_gate_inp.weight"],b[:xn])
        gemv!(b[:sg],M["ffn_gate_inp_shexp.weight"],b[:xn])
        gemv!(b[:shg],M["ffn_gate_shexp.weight"],b[:xn]); gemv!(b[:shu],M["ffn_up_shexp.weight"],b[:xn])
        swiglu!(b[:shh],b[:shg],b[:shu]); gemv!(b[:shout],M["ffn_down_shexp.weight"],b[:shh])
        copyto!(E.host_route,b[:route])
        all(isfinite,E.host_route) || error("nonfinite router logits at layer $(i-1), position $(E.pos)")
        ids=sortperm(E.host_route;by=x->-x,alg=Base.Sort.MergeSort)[1:c.used].-1
        w=exp.(E.host_route[ids.+1].-maximum(E.host_route[ids.+1])); w./=sum(w)
        stream_experts!(E,i-1,ids); copyto!(b[:weights],w)
        gemv!(b[:eg],E.expert_gpu[1],b[:xn]); gemv!(b[:eu],E.expert_gpu[2],b[:xn])
        swiglu!(b[:eh],b[:eg],b[:eu]); gemv!(b[:ed],E.expert_gpu[3],b[:eh];rpe=c.hidden)
        @roc groupsize=256 gridsize=cld(c.hidden,256) k_moemix!(b[:x],b[:ed],b[:weights],b[:shout],b[:sg],Int32(c.hidden),Int32(c.used))
    end
    E.pos+=1
    logits || return nothing
    norm!(b[:xn],b[:x],E.output_norm,c.eps); gemv!(b[:logits],E.output,b[:xn])
    out=Array(b[:logits]); all(isfinite,out) || error("nonfinite output logits"); out
end
function reset!(E)
    for B in E.blocks
        fill!(B.state,0.0f0); fill!(B.conv,0.0f0); fill!(B.kc,0.0f0); fill!(B.vc,0.0f0)
    end
    E.pos=0; E
end
function sample(rng,logits,temperature)
    temperature==0 && return argmax(logits)-1
    p=exp.((Float64.(logits).-maximum(logits))./temperature); target=rand(rng)*sum(p)
    s=0.0
    for i in eachindex(p); s+=p[i]; s>=target && return i-1; end
    length(p)-1
end
function download_model(path)
    mkpath(dirname(abspath(path))); tmp=path*".part"
    println(stderr,"Downloading $MODEL_URL")
    Downloads.download(MODEL_URL,tmp)
    filesize(tmp)==MODEL_BYTES || error("incomplete download: $tmp")
    digest=open(sha256,tmp); bytes2hex(digest)==MODEL_SHA || error("SHA-256 mismatch: $tmp")
    mv(tmp,path;force=true); println(stderr,"Saved and verified $path")
end
function main(args)
    if "--help" in args || "-h" in args
        println("""
        Native AMDGPU.jl Qwen3.6-35B-A3B GSQ text inference (RX 7700 XT).
          --model FILE       GSQ hybrid GGUF (default: $DEFAULT_MODEL)
          --download         download pinned model and verify SHA-256, then exit
          --prompt TEXT      user message (default: What is 2+2? Answer briefly.)
          --prompt-file FILE read the prompt from a UTF-8 text file
          --tokens N         maximum generated tokens (default: 64)
          --ctx N            total prompt + output capacity (default: 2048)
          --raw              plain completion instead of the Qwen chat template
          --think            enable the open thinking prefix (default: closed)
          --temperature X    sampling temperature (default: 0, greedy)
          --seed N           random seed (default: 1)
        Experts stream from RAM; attention/shared weights and activations stay on GPU.
        Prefill uses the same recurrent token step as decode. Text only; no vision/MTP.
        """)
        return
    end
    opts=Dict{String,String}(); flags=Set{String}(); i=1
    valued=Set(["--model","--prompt","--prompt-file","--tokens","--ctx","--temperature","--seed"])
    boolean=Set(["--download","--raw","--think"])
    while i<=length(args)
        a=args[i]
        if a in valued
            i<length(args) || error("$a needs a value"); opts[a]=args[i+1]; i+=2
        elseif a in boolean
            push!(flags,a); i+=1
        else
            error("unknown option $a (see --help)")
        end
    end
    path=expanduser(get(opts,"--model",DEFAULT_MODEL))
    "--download" in flags && return download_model(path)
    isfile(path) || error("missing model $path; use --download or --model FILE")
    isfile(path*".aria2") && error("model download is still running: $path")
    n=parse(Int,get(opts,"--tokens","64")); ctx=parse(Int,get(opts,"--ctx","2048"))
    temp=parse(Float64,get(opts,"--temperature","0")); rng=MersenneTwister(parse(Int,get(opts,"--seed","1")))
    n>=1 || error("--tokens must be positive"); isfinite(temp) && temp>=0 || error("invalid temperature")
    prompt=haskey(opts,"--prompt-file") ? read(expanduser(opts["--prompt-file"]),String) : get(opts,"--prompt","What is 2+2? Answer briefly.")
    g=gguf(path); tk=Tok(g)
    text="--raw" in flags ? prompt : chat_prompt(prompt)*("--think" in flags ? "" : "\n</think>\n\n")
    ids=encode(tk,text); isempty(ids) && error("prompt has no tokens")
    length(ids)+n<=ctx || error("$(length(ids)) prompt + $n output tokens exceed --ctx $ctx")
    println(stderr,"Model: $path\nGPU: $(AMDGPU.device())\nPrompt: $(length(ids)) tokens; context: $ctx")
    tload=time(); E=Engine(g,ctx); @printf(stderr,"Loaded in %.2f s\n",time()-tload)
    # Compile with the actual full model, then reset all recurrent/attention history.
    println(stderr,"Warming all kernels with one complete token..."); step!(E,ids[1]); AMDGPU.synchronize(); reset!(E)
    t0=time(); logits=Float32[]
    for (j,id) in enumerate(ids)
        result=step!(E,id;logits=j==length(ids)); result===nothing || (logits=result)
    end
    AMDGPU.synchronize(); prefill=time()-t0; t0=time(); generated=0; steps=0
    stops=Set([Int(tk.eos)])
    for s in ("<|im_end|>","<|endoftext|>"); haskey(tk.ids,s) && push!(stops,Int(tk.ids[s])); end
    for j in 1:n
        id=sample(rng,logits,temp); id in stops && break
        write(stdout,piece(tk,id)); flush(stdout); generated+=1
        if j<n; logits=step!(E,id); steps+=1; end
    end
    AMDGPU.synchronize(); decode=time()-t0; println()
    @printf(stderr,"prefill: %d tokens, %.3f s, %.2f tok/s\ndecode: %d tokens (%d forward steps), %.3f s, %.2f tok/s\n",
            length(ids),prefill,length(ids)/prefill,generated,steps,decode,steps/max(decode,eps(Float64)))
end
if abspath(PROGRAM_FILE)==@__FILE__()
    main(ARGS)
end

