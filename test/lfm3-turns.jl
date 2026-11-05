# lfm3-turns.jl: multi-turn conversations through the vendor's template and model on the CPU (eager attention, BF16): the
# prompt's ids and G greedy tokens of each, into OUT.N.ids and OUT.N.gen, for comparing with lfm3-1.7b.jl and flashd.
#   JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=VENV/bin/python julia --project=ENV test/lfm3-turns.jl DIR OUT G
using PythonCall
dir, out = ARGS[1:2]; g = parse(Int, ARGS[3])
torch = pyimport("torch"); tr = pyimport("transformers"); torch.set_num_threads(Sys.CPU_THREADS)
tok = tr.AutoTokenizer.from_pretrained(dir, trust_remote_code = true)
model = tr.AutoModelForCausalLM.from_pretrained(dir, trust_remote_code = true, dtype = torch.bfloat16, attn_implementation = "eager"); model.eval()
const U, A = "user", "assistant"
convs = [
    [(U, "hi bich")],
    [(U, "hi bich"), (A, "Hello! How can I help you today?"), (U, "so are you a weak ass ho?")],
    [(U, "hi bich"), (A, "Hello! How can I help you today?"), (U, "so are you a weak ass ho?"),
     (A, "I'm not sure what you're referring to. Could you please provide more context or clarify your question?"), (U, "right so did you know you are a base model?")],
    [(U, "speak french"), (A, "Je comprends. Je vais répondre en français. Comment puis-je vous aider ?"), (U, "no habla espanol")],
]
ctx = torch.no_grad(); ctx.__enter__()
for (n, c) in enumerate(convs)
    m = pylist([pydict(Dict("role" => r, "content" => t)) for (r, t) in c])
    s = pyconvert(String, tok.apply_chat_template(m, tokenize = false, add_generation_prompt = true))
    ids = pyconvert(Vector{Int}, tok(s, add_special_tokens = false).input_ids)
    o = model.generate(torch.tensor(pylist(ids)).unsqueeze(0), max_new_tokens = g, do_sample = false, use_cache = true, eos_token_id = 124900)
    gen = pyconvert(Vector{Int}, o[0].tolist())[length(ids)+1:end]
    write("$out.$n.ids", Int32.(ids)); write("$out.$n.gen", Int32.(gen))
    println("== $n: $(length(ids)) prompt, $(length(gen)) out; prompt tail ", repr(last(s, 30)))
    println(repr(pyconvert(String, tok.decode(pylist(gen)))))
end
