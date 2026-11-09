# chain.jl: K decode steps queued back to back, nothing between them for the host (lfm3.jl's SAMPLE node picks each row's next token on the device).
# The device writes the token into the next launch's row and out through a ring in host memory; the host polls the ring while the card runs. One launch a step, the plan standing.
const CHB = Dict{Any,Any}(); const RING = NFLAG - 8448; const RINGW, KMAX = 128, 64   # the ring: KMAX steps of RINGW rows; the abort word is NFLAG - 5
function chain!(S::Sys, slots, pos, tok, K; seed = 1, samp = 2)       # lane i: its slot, the position of its token tok (the cache holds the positions before)
    C = S.cards[1]; NS = length(slots); (1 <= K <= KMAX && NS <= NSEG) || error("chain: $NS rows, $K steps"); dev!(1)
    rb, sb = get!(CHB, S) do; (dalloc(16 * NSEG * KMAX), dalloc(8 * NSEG * KMAX)); end
    rows = zeros(Int32, 4, NS, K); segs = zeros(Int32, 2, NS, K)
    for k in 1:K, i in 1:NS; rows[:, i, k] = Int32[i - 1, pos[i] + k - 1, k == 1 ? tok[i] : 0, -1]; segs[:, i, k] = Int32[slots[i], pos[i] + k]; end
    lr = Int32.(0:NS-1); up!(rb, 0, vec(rows)); up!(sb, 0, vec(segs)); up!(C.lrows, 0, lr); up!(C.linv, 0, lr); up!(C.wgt, 0, ones(Float32, NS)); up!(C.uniq, 0, Int32[x for i in 0:NS-1 for x in (i, 1)]); up!(C.urows, 0, lr)
    sig = hash((NS, NS, NS, NS, 1, samp, DBG[], [(Int32(s), Int32(0)) for s in slots])); reuse = get(SIGS, S, UInt(0)) == sig ? 1 : 0
    hu0 = get(HUSED, S, -1); nz = 4 * (reuse == 1 && hu0 >= 0 ? min(NFLAG, hu0 + 4096) : NFLAG)   # (the device cells to clear: below the largest plan seen)
    ring = Ptr{Int32}(S.cntH + 4 * RING); ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), ring, 0xff, 4 * RINGW * K); unsafe_store!(Ptr{UInt32}(S.cntH), 0, NFLAG - 4)
    nf = dalloc(4 * length(S.G.nodef)); up!(nf, 0, S.G.nodef); push!(C.bufs, nf)
    for k in 1:K
        S.gen += 1; fill!(C.q, 0); fill!(view(C.cnt, 1:nz), 0)
        C.k(rq(S, C, NS, NS, NS, NS, 1; reuse = k == 1 ? reuse : 1, samp, seed, step = k, rows = addr(rb) + 16NS * (k - 1), segs = addr(sb) + 8NS * (k - 1), rowsn = k < K ? addr(rb) + 16NS * k : 0, ring = S.cntHd + 4 * RING, nf); ndrange = 256 * C.nblk)
    end
    (S = S, NS = NS, K = K, sig = sig, hu0 = hu0, ring = ring)
end
chainabort!(S::Sys) = unsafe_store!(Ptr{UInt32}(S.cntH), 1, NFLAG - 4)     # the launches not yet started do nothing
chainready(ch, k) = all(i -> Core.Intrinsics.atomic_pointerref(ch.ring + 4 * ((k - 1) * RINGW + i - 1), :monotonic) >= 0, 1:ch.NS)   # step k's tokens are out
chaintoks(ch, k) = Int[unsafe_load(ch.ring, (k - 1) * RINGW + i) for i in 1:ch.NS]
chaindone() = CUDA.isdone(CUDA.stream())
function chainend!(ch)                                                  # the card is done with every launch: the plan's status
    S = ch.S; C = S.cards[1]; dev!(1); sync(); pop!(C.bufs); dn!(S.plan, C.plan, 0); st = S.plan[PH_STAT+1]
    st == 0 && (SIGS[S] = ch.sig; HUSED[S] = max(ch.hu0, Int(S.plan[PH_NFLAG+1]))); st == 0 || (delete!(SIGS, S); delete!(HUSED, S); error("chain: status $st"))
end
