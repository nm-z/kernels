# lfm3-grad.jl: the gradients lfm3-1.7b.jl --check PREFIX N wrote (one window, the loss the mean cross-entropy at PREFIX.pos)
# against the vendor's model in Float32 on the CPU by autograd, on the same tokens. A line a tensor: the relative error and the
# cosine between the two gradients.
#   JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=env/py/bin/python julia test/lfm3-grad.jl DIR PREFIX
using PythonCall, LinearAlgebra
dir, prefix = ARGS[1:2]
torch = pyimport("torch"); tr = pyimport("transformers"); torch.set_num_threads(Sys.CPU_THREADS)
model = tr.AutoModelForCausalLM.from_pretrained(dir, trust_remote_code = true, dtype = torch.float32, attn_implementation = "eager")
model.eval()
ids = collect(reinterpret(Int32, read(prefix * ".ids"))); pos = collect(reinterpret(Int32, read(prefix * ".pos")))
t0 = time()
lg = model(torch.tensor(pylist(ids)).unsqueeze(0), use_cache = false).logits[0]
loss = torch.nn.functional.cross_entropy(lg[pylist(pos)], torch.tensor(pylist(ids[pos.+2])))
loss.backward()
println("torch: loss $(pyconvert(Float64, loss.item())) (kernel $(only(reinterpret(Float32, read(prefix * ".loss"))))), $(round(time() - t0; digits = 1)) s")
params = pydict(model.named_parameters())
for f in sort(filter(f -> endswith(f, ".f32"), readdir(dirname(prefix))))
    startswith(f, basename(prefix) * ".") || continue
    name = f[length(basename(prefix))+2:end-4]
    a = collect(reinterpret(Float32, read(joinpath(dirname(prefix), f))))
    b = Vector{Float32}(PyArray(params[name].grad.detach().numpy().ravel()))
    length(a) > length(b) && (a = a[1:length(b)])   # (a head norm's vector is held padded to NE)
    length(a) == length(b) || (println("$name: $(length(a)) vs $(length(b))"); continue)
    println(rpad(name, 48), " rel ", round(norm(a - b) / norm(b); digits = 4), "  cos ", round(dot(a, b) / (norm(a) * norm(b)); digits = 5), "  |torch| ", round(norm(b); sigdigits = 3))
end
