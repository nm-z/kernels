# gguf-template.jl: a copy of a tokenizer GGUF (no tensors) with its chat template replaced by a file's text.
#   julia test/gguf-template.jl IN.gguf TEMPLATE.jinja OUT.gguf
io = open(ARGS[1]); tpl = read(ARGS[2], String); out = IOBuffer()
magic = read(io, 4); ver = read(io, UInt32); nt = read(io, UInt64); nkv = read(io, UInt64)
nt == 0 || error("tensors present")
write(out, magic, ver, nt, nkv)
rstr(io) = String(read(io, read(io, UInt64)))
wstr(o, s) = (write(o, UInt64(sizeof(s))); write(o, s))
sz = Dict(0 => 1, 1 => 1, 2 => 2, 3 => 2, 4 => 4, 5 => 4, 6 => 4, 7 => 1, 10 => 8, 11 => 8, 12 => 8)
function copyval(io, out, t)
    if t == 8; wstr(out, rstr(io))
    elseif t == 9; et = read(io, UInt32); n = read(io, UInt64); write(out, et, n); for _ in 1:n; copyval(io, out, et); end
    else; write(out, read(io, sz[t])); end
end
for _ in 1:nkv
    k = rstr(io); t = read(io, UInt32); wstr(out, k); write(out, t)
    if k == "tokenizer.chat_template"; t == 8 || error("template type $t"); rstr(io); wstr(out, tpl); println("replaced")
    else; copyval(io, out, t); end
end
write(ARGS[3], take!(out))
