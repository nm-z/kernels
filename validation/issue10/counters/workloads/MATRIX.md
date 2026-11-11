# Paired 3,100/8,000 gather counter matrix

**Component campaign: PASS. Full issue #10 acceptance: incomplete.**

After the old issue #7 validator released the shared resources, this worker completed the previously blocked component campaign on archy. It entered the blocking lock wait at **2026-10-18T06:10:00Z**, acquired `/tmp/flash-sass-engine-validation.lock` at **06:10:59Z**, and finished at **06:13:27Z**, exit **0**. The finite job released its lease on completion; no resident engine was reloaded or stopped.

## Coverage and measured results

The campaign covers 3,100 and 8,000 visible positions, each with clustered, evenly spread, random and changing selected-block layouts. Every case has a matched VRAM-KV and host-pinned-KV arm for each of four separately executed metric groups: traffic bytes, traffic transactions, occupancy/eligible warps, and memory/synchronization stalls.

That is **8 cases, 64 successful profiler commands, and 208 accepted metric rows**. All four iterations in each native arm passed output verification. Cache, selected-row, and verified output identities agree between placements and across metric groups. Each case uses block budget 512, seed 4242, zero tail, and 2,048 gathered keys/physical slots.

| Visible positions | Layout | VRAM system-memory read bytes/invocation | Host-pinned system-memory read bytes/invocation |
| ---: | --- | ---: | ---: |
| 3,100 | Clustered | 0 | 2,359,296 |
| 3,100 | Evenly spread | 0 | 2,359,296 |
| 3,100 | Random | 0 | 2,359,296 |
| 3,100 | Changing | 0 | 2,359,296 |
| 8,000 | Clustered | 0 | 2,359,296 |
| 8,000 | Evenly spread | 0 | 2,359,296 |
| 8,000 | Random | 0 | 2,359,296 |
| 8,000 | Changing | 0 | 2,359,296 |

These values are reconstructed directly from the raw nvprof `Avg` fields, not inferred from allocation sizes. The complete replayed `metrics.csv` contains all **104 paired metric rows**, preserving each placement's minimum, maximum, average and unit. Original CSVs also preserve each metric's kernel name, device name and invocation count.

The measured system-memory read count is unchanged across these particular capped gather cases. This does **not** establish unchanged full-model token time, exclude other context-dependent work, or explain the historical 16.917 ms/token increment. The cases do not execute the model's score scan, selection pipeline, recurrent layers, complete attention path, host request lifecycle, or inter-die dependency graph.

## Scope that must remain visible

The profiler observes the finite `die` kernel, including gather/dequantization and its dispatch, barriers and publication framework. Each profiler command reports **18 logical kernel invocations**: four native correctness preflights, ten warmups and four measured launches. Internal profiler replay can add physical executions. Its averages are already per logical invocation; dividing them by 18 again would be incorrect.

In the reused native benchmark, clustered, evenly spread and random modes keep their selected row fixed. Only `changing` varies the row each iteration and verifies three selection/output-signature changes. Warmups repeat the first row even in changing mode. No case contains checkpoint-derived selected IDs. Therefore this matrix does not complete issue #10's requirement for changing versions of all representative layouts and real traces.

Metric groups are separate executions. Do not require their independently collected average bytes and transactions to satisfy exact simultaneous identities. Profiled CUDA-event rates are retained as diagnostics only: kernel replay perturbs them and they are **not authoritative throughput or tracing-overhead measurements**. Do not divide these bytes by another run's duration to claim measured bandwidth, and do not add mapped transfer/dequantization time to gather time as independent wall-time components.

The selected GPU was the otherwise unoccupied spare Tesla M60 `GPU-431371bd-b482-96b2-4b70-04f83c6d075b`. Full GPU/process snapshots are preserved. Other model GPUs were not uniformly idle, so this is not a claim of a globally quiescent machine or a matched full-model clock experiment. The collector rejected existing compute processes on its target GPU; it did not interfere with processes on the model dies.

## Provenance and reproduction

All raw captures are in [matrix-3100-8000](matrix-3100-8000). Every case retains its manifest, exact profiler/native command arguments and exits, GPU snapshots, raw native outputs, raw counter CSVs, and accepted projections. `queue.log`, `queue.exit` and `queue-counters.sh` record the executed finite campaign. `decode_counter_capture.pl` in that directory is the **exact collector source executed on archy**.

The native C benchmark and cubin retain the PR #43 identities used by the earlier smoke. The embedded native revision is `7c6f34d4372567fd97f1cef9b609d3dc0c81b0d3`; benchmark SHA256 is `126ba2ed56f7623a0ee1eeb5f658d719e057ef3c3f54eb1d5d2746f8776523b8`, cubin SHA256 is `0fa2a17fa41c07aac507f9bc218a777601719a8e2b6eb8b40ec93867d4768027`, and CUDA 11.4 nvprof SHA256 is `c3e58ebca670c1511c55e1d1420f8fea67d25fba93a4e0d178bc562473ae819d`. These are executable/source artifact identities, not a new checkpoint-conversion attestation.

The collector now accepts an inherited lock descriptor. It verifies that the descriptor references the actual, non-symlink shared lock inode and can acquire its exclusive flock. It duplicates the inherited open-file description rather than reopening the pathname, avoiding a nested-lock deadlock while the outer campaign retains its lease. The recorded campaign used a blocking flock; the currently installed `gpuq` FIFO can provide the same descriptor for subsequent captures:

```sh
gpuq run -- perl /path/to/flash-sass/kern/decode_counter_capture.pl \
  --manifest /path/to/nv775-changing.manifest.json \
  --out /tmp/issue10-new-counter-capture --sudo --lock-fd 9
```

The example command is a reproduction route, not a claim that the archived job used `gpuq`.

Run the complete CPU replay from the repository root:

```sh
perl kern/decode_counter_matrix.pl \
  --archive validation/issue10/counters/workloads/matrix-3100-8000 \
  --out /tmp/issue10-matrix-new-replay
prove -v kern/decode_counter_capture_test.pl kern/decode_counter_matrix_test.pl
```

Replay checks all eight cases and 64 zero command exits, exact metric groups and invocation counts, raw-file byte/hash identities, command/device/workload matching, native row identities, and cross-case executable/GPU identities. It independently rebuilds accepted projections and the complete paired metric table. Rejection tests mutate coverage, queue exits, metadata, device settings, workload output, raw files and accepted records; a coordinated raw-hash mutation still has to pass the actual metric parser. Incomplete or rejected campaigns produce no successful replay directory.

These checks establish consistency of the supplied records. Authenticity is anchored to the reviewed Git revision and `validation/issue10/IDENTITIES.json`, not to an arbitrary self-consistent replacement manifest.

The [native validation record](native-validation-matrix/result.json) records **192 passing CPU tests**: 67 collector checks, 48 complete-matrix checks and 77 legacy archive checks. Both new Perl files and their test files passed syntax checks; both executed Bash scripts passed `bash -n`; the staged patch passed whitespace checks. The legacy replay retained all 60 traced runs, 20 untraced runs and 360 die records, reproducing the three historical summaries with zero numerical differences. Its input-manifest hash describes the corpus before these validation logs were attached; the final root identity manifest also covers the logs. The record includes the exact validator, tested code patch and source identities. None of these CPU replays is a new full-model GPU run.

## Separate full-model readiness failure

A second queued operation acquired the shared lock at **2026-10-18T06:18:12Z** and performed only read-only full-model readiness checks. Its six samples, ending at **06:18:39Z**, each reported **100% utilization on model GPUs 1–5**. GPU 0 reported 0%. The model-GPU free-memory observations were 52, 16, 77, 7, 3 and 65 MiB, respectively. The process inventory included resident `flashcore` PID **3575268** and the existing weight holders/fence.

The readiness operation exited **75** without inference, engine reload, process termination, or holder changes. [Full-model readiness records](full-model-readiness-20261018) preserve its exact script, command log, timestamps, six samples, complete driver state and process inventory. GPU-level utilization does not, by itself, identify which resident process caused it, and this record does not diagnose a deadlock. PID 3575268 is the observed process at that time, not a claim that it continues to exist indefinitely.

The old issue #7 lock holder is no longer the blocker recorded by this campaign: the component campaign and the full-model preflight both obtained the lock. What remains unvalidated is an isolated full-model campaign with complete runtime provenance, supported deeper contexts, precise all-die schema-2 attribution, controlled tracing overhead, and verified checkpoint-selected layouts. The existing historical measurements remain explicitly unreproduced as the original 36/28/19 experiment. PR #38 must remain draft until those acceptance gates are met.
