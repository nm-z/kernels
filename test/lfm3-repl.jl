# lfm3-repl.jl: a resident model; each file dropped into CMD is run with S in scope, its output to FILE.out, then renamed FILE.done
#   julia --project=env/cuda -t 8 test/lfm3-repl.jl DIR CMD [train]
include(joinpath(@__DIR__, "..", "lfm3.jl"))
const CMD = ARGS[2]; mkpath(CMD)
S = load(ARGS[1]; nslot = 0, reserve = 2e9, hostgb = 6.0, train = length(ARGS) > 2); println("ready"); flush(stdout)
while true
    for f in sort(filter(x -> endswith(x, ".jl"), readdir(CMD)))
        p = joinpath(CMD, f); t0 = time(); dead = false
        open(p[1:end-3] * ".out", "w") do io
            redirect_stdout(io) do
                try; include(p); catch e; showerror(stdout, e, catch_backtrace()); dead = e isa CUDA.CuError || (e isa LoadError && e.error isa CUDA.CuError); end
                @printf("\n[%.2f s]\n", time() - t0)
            end
        end
        mv(p, p[1:end-3] * ".done"; force = true); dead && (println("CUDA context lost: restarting"); exit(3))   # (the loop around this script starts it again)
    end
    sleep(0.2)
end
