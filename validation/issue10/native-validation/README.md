# Native compact-archive validation

The CPU checks were run on `engi` from a clean checkout of commit [`7d19d09c6ab734847bf737cfa75448161fb092e9`](https://github.com/nm-z/flash-sass/commit/7d19d09c6ab734847bf737cfa75448161fb092e9). [result.json](result.json) records the exact commands, times, exit statuses, source hashes and replay result. [executed-validation.pl](executed-validation.pl) is the exact local orchestration used; its absolute paths identify that execution.

All **77 tests passed**. Both Perl source files passed syntax checks, and the source commit passed `git diff --check`. The archive replay validated **60 traced runs, 20 untraced runs and 360 die records**. Its three regenerated reference summaries had **zero numerical differences**, with a permitted tolerance of `1e-8` in the reported units. The test suite also rejects incomplete pairs, mismatched output IDs, invalid completion/observation fields, missing die records, invalid selected counts, nonfinite timings, truncated gzip, artifact tampering, and coordinated timing changes that disagree with the preserved references.

The replay identifies the input identity manifest at the tested commit. Adding this validation record changes the enclosing archive manifest; it does not change the six historical input/reference files or the tested Perl code. The archived replay outputs and their identities are in [replay/](replay/).

## Re-run

From the repository root, use a new output directory:

```sh
prove -v kern/decode_forensics_test.pl
perl kern/decode_forensics.pl --archive validation/issue10 --out /tmp/issue10-compact-replay
```

This validates compact-archive consistency and numerical reproduction. It does not reread the original phase CSVs, launch GPU inference, recover historical provenance, measure tracing overhead, or complete issue #10.

## Shared GPU availability

The [lock preflight](gpu-lock-preflight.sh) attempted the required `/tmp/flash-sass-engine-validation.lock` on `archy` at **2026-10-18 03:45:22 UTC**, waited 45 seconds, and exited **73**. The [raw result](gpu-lock-preflight.log) identifies the holder as PID `4010221`, running `validation/issue19/run.rb --mode model`. The preflight exits before GPU queries when lock acquisition fails. No GPU measurement was started during this validation turn.

The acceptance inventory remains [INCOMPLETE](../acceptance.json). A fully attributable matched depth campaign, precise compute/wait boundaries, controlled tracing-overhead comparison and actual workload profiler counters remain outstanding. Successful profiler metric enumeration is preserved separately in [counters/](../counters/).
