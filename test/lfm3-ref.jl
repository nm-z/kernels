# lfm3-ref.jl: LFM3 reference data from the vendor's model code on the CPU, in BF16 (eager attention), for lfm3-1.7b.jl's
# kl and gen commands. Writes OUT.ids (Int32), OUT.logits.bf16 ([position][vocab], BFloat16) of one forward over the
# first N tokens of TEXT, and OUT.gen (Int32): G greedy tokens after those of the first M (cached decode).
#   JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=VENV/bin/python julia --project=ENV test/lfm3-ref.jl DIR TEXT OUT N M G
using PythonCall

dir, text, out = ARGS[1:3]; n, m, g = parse.(Int, ARGS[4:6])
torch = pyimport("torch"); tr = pyimport("transformers")
torch.set_num_threads(Sys.CPU_THREADS)
tok = tr.AutoTokenizer.from_pretrained(dir, trust_remote_code = true)
model = tr.AutoModelForCausalLM.from_pretrained(dir, trust_remote_code = true, dtype = torch.bfloat16, attn_implementation = "eager")
model.eval()
ids = pyconvert(Vector{Int}, tok(read(text, String)).input_ids)
length(ids) >= max(n, m) || error("$text: $(length(ids)) tokens, need $(max(n, m))")
ctx = torch.no_grad(); ctx.__enter__()
t0 = time()
lg = model(torch.tensor(pylist(ids[1:n])).unsqueeze(0), use_cache = false).logits[0]
println("forward of $n: $(round(time() - t0; digits = 1)) s")
# (the raw bits: a BF16 tensor as int16)
bits = PyArray(lg.view(torch.int16).numpy())
write(out * ".logits.bf16", permutedims(Int16.(bits)))                 # (position, vocab) -> [position][vocab] on disk
write(out * ".ids", Int32.(ids[1:max(n, m)]))
t0 = time()
gen = model.generate(torch.tensor(pylist(ids[1:m])).unsqueeze(0), max_new_tokens = g, do_sample = false, use_cache = true, eos_token_id = pybuiltins.None)
println("generate of $g after $m: $(round(time() - t0; digits = 1)) s")
write(out * ".gen", Int32.(pyconvert(Vector{Int}, gen[0].tolist())[m+1:end]))
