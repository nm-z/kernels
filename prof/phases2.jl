# phases2.jl: die 0's phase times of the last request by phase sequence index within a forward (whatever the forward's
# kind), summed over the forwards from skip on with the given phase count n (the decode forwards), the top m
#   eval include("prof/phases2.jl"); s = IOBuffer(); phases2(s, n); String(take!(s))
function phases2(io, n; skip = 1, m = 25, die = 1)
    ah = ACCTS[][die]; hz = PK[].hz; ns = (length(ah) - 3) ÷ 2
    # forward boundaries: a forward's first phase is its embedding exchange; find them by the gap pattern is not
    # possible, so take fixed-length forwards of n phases after the prefill's (skip forwards of nskip phases each)
    busy = zeros(n); gap = zeros(n); fw = Float64[]; s0 = skip; c = 0
    while s0 + n <= ns
        b0 = ah[4+2s0]; b0 == typemax(UInt64) && break
        pe = b0
        for i in 0:n-1
            b = ah[4+2(s0+i)]; e = ah[5+2(s0+i)]; b == typemax(UInt64) && continue
            busy[i+1] += (e - b) / hz; gap[i+1] += max(0.0, (Float64(b) - Float64(pe)) / hz); pe = max(pe, e)
        end
        push!(fw, (pe - b0) / hz); s0 += n; c += 1
    end
    c == 0 && return println(io, "none")
    @printf(io, "%d forwards, %.2f ms each; busy %.2f gap %.2f ms\n", c, 1e3sum(fw) / c, 1e3sum(busy) / c, 1e3sum(gap) / c)
    for i in sortperm(gap .+ 0, rev = true)[1:m]
        @printf(io, "phase %4d  busy %7.3f  gap %7.3f us\n", i - 1, 1e6busy[i] / c, 1e6gap[i] / c)
    end
end
