# gguf-template.jl: a copy of a tokenizer GGUF (no tensors) with its chat template replaced by a file's text and, optionally,
# sampling defaults set (general.sampling.temp / top_k / top_p, which flashd reads; added when absent).
#   julia test/gguf-template.jl IN.gguf TEMPLATE.jinja OUT.gguf [temp top_k top_p]
io = open(ARGS[1]); tpl = read(ARGS[2], String); out = IOBuffer()
samp = length(ARGS) >= 6 ? Dict("general.sampling.temp" => parse(Float32, ARGS[4]), "general.sampling.top_k" => parse(Int32, ARGS[5]), "general.sampling.top_p" => parse(Float32, ARGS[6])) : Dict{String,Any}()
magic = read(io, 4); ver = read(io, UInt32); nt = read(io, UInt64); nkv = read(io, UInt64)
nt == 0 || error("tensors present")
rstr(io) = String(read(io, read(io, UInt64)))
wstr(o, s) = (write(o, UInt64(sizeof(s))); write(o, s))
sz = Dict(0 => 1, 1 => 1, 2 => 2, 3 => 2, 4 => 4, 5 => 4, 6 => 4, 7 => 1, 10 => 8, 11 => 8, 12 => 8)
function copyval(io, out, t)
    if t == 8; wstr(out, rstr(io))
    elseif t == 9; et = read(io, UInt32); n = read(io, UInt64); write(out, et, n); for _ in 1:n; copyval(io, out, et); end
    else; write(out, read(io, sz[t])); end
end
wsamp(o, k, v::Float32) = (wstr(o, k); write(o, UInt32(6), v))
wsamp(o, k, v::Int32) = (wstr(o, k); write(o, UInt32(5), v))
body = IOBuffer(); seen = Set{String}()
for _ in 1:nkv
    k = rstr(io); t = read(io, UInt32)
    if k == "tokenizer.chat_template"; t == 8 || error("template type $t"); rstr(io); wstr(body, k); write(body, t); wstr(body, tpl); println("replaced")
    elseif haskey(samp, k); copyval(io, IOBuffer(), t); wsamp(body, k, samp[k]); push!(seen, k)
    else; wstr(body, k); write(body, t); copyval(io, body, t); end
end
for (k, v) in samp; k in seen || (wsamp(body, k, v); nkv += 1); end
write(out, magic, ver, nt, nkv, take!(body)); write(ARGS[3], take!(out))
