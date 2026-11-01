# phases.jl: die 0's phase times of the last request (LASTACCT), by phase kind, over the decode forwards.
# In the server: eval include("prof/phases.jl"); phases(io)
function plabels()
    L = String["emb", "emb.x"]
    for l in 0:63
        if isrec(l)
            append!(L, ["qkvza", "conv", "rec"])
        else
            append!(L, ["qkv", "aprep", "attn", "acomb"])
        end
        append!(L, [isrec(l) ? "dn.out" : "at.out", "xwr1", "xwr1.x", "gateup", "act", "down", "xwr2", "xwr2.x"])
    end
    append!(L, ["head", "hmax", "hred", "tok"])
end
function phases(io = stdout; skip = 1)
    ah = LASTACCT[]; hz = PK[].hz; L = plabels(); n = length(L)
    ns = (length(ah) - 3) ÷ 2; nf = 0
    tot = Dict{String,Float64}(); gap = Dict{String,Float64}(); fw = Float64[]
    s0 = 0
    while s0 + n <= ns
        b0 = ah[4+2s0]; b0 == typemax(UInt64) && break
        if nf >= skip
            pe = ah[4+2s0]
            for i in 0:n-1
                b = ah[4+2(s0+i)]; e = ah[5+2(s0+i)]; b == typemax(UInt64) && continue
                tot[L[i+1]] = get(tot, L[i+1], 0.0) + (e - b) / hz
                gap[L[i+1]] = get(gap, L[i+1], 0.0) + max(0.0, (Float64(b) - Float64(pe)) / hz)
                pe = max(pe, e)
            end
            push!(fw, (pe - b0) / hz)
        end
        nf += 1; s0 += n
    end
    m = length(fw); m == 0 && return println(io, "no decode forwards")
    @printf(io, "%d forwards, %.2f ms each (median %.2f)\n", m, 1e3sum(fw) / m, 1e3sort(fw)[cld(m, 2)])
    ks = sort(collect(keys(tot)), by = k -> -(tot[k] + gap[k]))
    for k in ks
        @printf(io, "%-8s busy %7.3f ms  gap before %7.3f ms\n", k, 1e3tot[k] / m, 1e3gap[k] / m)
    end
end
