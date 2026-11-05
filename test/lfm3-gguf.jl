# lfm3-gguf.jl: a GGUF with no tensors holding LFM3's tokenizer and chat template, for flashd (which reads them from a GGUF;
# the model itself is safetensors, read by lfm3-1.7b.jl). Every id below the vocabulary's 128000 gets a token.
#   julia test/lfm3-gguf.jl DIR OUT.gguf
function jstr(s, i)                                        # the JSON string at s[i] == '"': (the text, the index after it)
    io = IOBuffer(); i += 1
    while true
        c = s[i]; i = nextind(s, i)
        c == '"' && return String(take!(io)), i
        if c == '\\'
            e = s[i]; i += 1
            if e == 'u'
                u = parse(UInt32, s[i:i+3], base = 16); i += 4
                if 0xd800 <= u < 0xdc00
                    v = parse(UInt32, s[i+2:i+5], base = 16); i += 6; u = 0x10000 + ((u - 0xd800) << 10) + (v - 0xdc00)
                end
                print(io, Char(u))
            else
                print(io, e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e == 'b' ? '\b' : e == 'f' ? '\f' : e)
            end
        else
            print(io, c)
        end
    end
end
dir, out = ARGS
s = read(joinpath(dir, "tokenizer.json"), String)
const NV = 128000
tokens = ["<unused_$i>" for i in 0:NV-1]; types = fill(Int32(5), NV)
# model.vocab: "text": id
i = findfirst("\"vocab\":", s).stop + 1; i = findnext('{', s, i) + 1
while true
    global i = findnext(c -> c == '"' || c == '}', s, i)
    s[i] == '}' && break
    t, j = jstr(s, i); k = findnext(c -> isdigit(c), s, j); e = findnext(c -> !isdigit(c), s, k); id = parse(Int, s[k:e-1])
    tokens[id+1] = t; types[id+1] = 1; global i = e
end
vend = i
# added_tokens: {"id":N,"content":"...",...,"special":bool}
a = findfirst("\"added_tokens\":", s).stop + 1; a = findnext('[', s, a) + 1; aend = findnext("\"normalizer\":", s, a).start
for m in eachmatch(r"\{\s*\"id\":\s*(\d+),\s*\"content\":\s*", SubString(s, a, aend))
    o = a + m.offset - 1 + ncodeunits(m.match); t, j = jstr(s, o); id = parse(Int, m[1])
    sp = occursin("\"special\": true", s[j:findnext('}', s, j)]) || occursin("\"special\":true", s[j:findnext('}', s, j)])
    tokens[id+1] = t; types[id+1] = sp ? 3 : 4
end
# model.merges: ["a", "b"] pairs
m = findfirst("\"merges\":", s).stop + 1; m = findnext('[', s, m) + 1; merges = String[]
while true
    global m = findnext(c -> c == '[' || c == ']', s, m)
    s[m] == ']' && break
    a1, j = jstr(s, findnext('"', s, m)); b1, j = jstr(s, findnext('"', s, j)); push!(merges, a1 * " " * b1); global m = findnext(']', s, j) + 1
end
# (the generation tag only marks the assistant's text for training masks; flashd's Jinja renders the same text through an
# if that is always true, with the same whitespace control)
template = replace(read(joinpath(dir, "chat_template.jinja"), String), "{%- generation -%}" => "{%- if true -%}", "{%- endgeneration -%}" => "{%- endif -%}")
findfirst(==("<|im_end|>"), tokens) == 124901 || error("<|im_end|> is not id 124900")
open(out, "w") do io
    wstr(t) = (write(io, UInt64(ncodeunits(t))); write(io, t))
    key(k, ty) = (wstr(k); write(io, UInt32(ty)))
    write(io, "GGUF", UInt32(3), UInt64(0), UInt64(9))
    key("general.architecture", 8); wstr("lfm3")
    key("tokenizer.ggml.model", 8); wstr("gpt2")
    key("tokenizer.ggml.pre", 8); wstr("lfm3")
    key("tokenizer.ggml.tokens", 9); write(io, UInt32(8), UInt64(NV)); foreach(wstr, tokens)
    key("tokenizer.ggml.token_type", 9); write(io, UInt32(5), UInt64(NV)); write(io, types)
    key("tokenizer.ggml.merges", 9); write(io, UInt32(8), UInt64(length(merges))); foreach(wstr, merges)
    key("tokenizer.ggml.bos_token_id", 4); write(io, UInt32(124894))
    key("tokenizer.ggml.eos_token_id", 4); write(io, UInt32(124900))
    key("tokenizer.chat_template", 8); wstr(template)
end
println("$(count(==(Int32(1)), types)) vocabulary tokens, $(count(t -> t == 3 || t == 4, types)) added, $(length(merges)) merges, $(filesize(out)) bytes")
