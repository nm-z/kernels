# lfm3-slice.jl: one slice of the update a command, so no command passes ~10 s. The cursor and the measured rates live in the resident
# (UPDCUR, UPDRATE); drop this file into the resident's command directory once per slice until it prints "step done".
# test/lfm3-upd2.jl runs the whole step (step! = every slice) against a Float64 Newton-Schulz: the slices must agree with it.
t0 = time(); done = stepslice!(S; seconds = 10.0); dt = time() - t0
@printf("slice %.1f s; next group %d of %d; rates (Muon flops/s, momentum-step elements/s) %.3g %.3g; %s\n", dt, UPDCUR[], length(S.G.upd), UPDRATE[1], UPDRATE[2], done ? "step done" : "more")
