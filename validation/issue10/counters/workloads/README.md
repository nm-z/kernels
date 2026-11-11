# Actual Maxwell gather counters

This page preserves the initial eight-command smoke and its first blocked follow-ups. The later [completed 64-command component matrix](MATRIX.md) covers 3,100 and 8,000 positions and supersedes those blocked component-matrix statuses. Neither campaign is a full-model acceptance result.

## Accepted measurement and limits

`smoke-8000-changing/status.json` records a successful **single-die component counter capture**, not closure of issue #10. The native C gather benchmark from PR #43 ran on archy under `/tmp/flash-sass-engine-validation.lock`. Its full driver snapshots bracket the capture at **October 17, 2026, 22:34:27–22:34:46 host-local time**. The selected Tesla M60 UUID was `GPU-431371bd-b482-96b2-4b70-04f83c6d075b`.

The case uses 8,000 visible positions, 2,000 complete four-token blocks, budget 512, no tail, and 2,048 gathered keys/physical slots. Seed 4242 generates finite, distinguishable synthetic KV values. Four selected-block rows and their verified output signatures change three times. Each host/VRAM pair has identical cache, selected-row, and output identities.

Four independent profiler groups run in both placements: memory bytes, memory transactions, occupancy/eligible warps, and selected stall categories. All **eight profiler commands exited 0** and returned the complete requested **26 metric rows**. nvprof reports **18 logical kernel invocations per command**: four correctness-preflight launches, ten warmups, and four measured launches. The profiler may replay kernels internally; 18 is not a count of all physical replay executions. Warmups repeat the first row, so this is not an exclusively changing-row sample.

The reported kernel is `die`, including its persistent-kernel dispatch, barriers, and publication framework around gather/dequantization. These counters do not isolate the gather body from that framework. System-memory traffic through mapped GPU loads is measured; no separate cudaMemcpy interval is needed to observe it. The separate native benchmark's raw `hardware_counter_gate: not_run` and null measured-byte field remain unchanged: those fields describe its own CUDA-event-only instrumentation. The external profiler output below supplies the new measurements.

### Measured averages per logical kernel invocation

| Metric | VRAM KV | Host-pinned KV | Unit |
| --- | ---: | ---: | --- |
| System-memory read bytes | 0 | 2,359,296 | bytes |
| System-memory write bytes | 672 | 672 | bytes |
| DRAM read bytes | 7,269,361 | 4,723,447 | bytes |
| DRAM write bytes | 12,675,361 | 15,238,163 | bytes |
| System-memory read transactions | 0 | 73,728 | transactions |
| System-memory write transactions | 21 | 21 | transactions |
| DRAM read transactions | 226,505 | 147,132 | transactions |
| DRAM write transactions | 394,891 | 475,440 | transactions |
| Achieved occupancy | 0.124980 | 0.124999 | ratio |
| Eligible warps per active cycle | 0.143410 | 0.007723 | warps/cycle |
| Memory-dependency stalls | 84.634260 | 97.232118 | percent |
| Memory-throttle stalls | 0.000674 | 0.000034 | percent |
| Synchronization stalls | 3.508835 | 2.429387 | percent |

These are nvprof's `Avg` values, already per invocation; do not divide them by 18 again. Each raw file also preserves `Min` and `Max`. Metric groups are **separate executions**, not simultaneous samples; do not impose exact cross-group byte/transaction identities on their averages. Unique source addresses cover 2,228,224 bytes and destination addresses 8,388,608 bytes, but those address footprints are not substituted for the measured traffic.

**Discard all profiled elapsed rates from authoritative throughput comparisons.** Kernel replay heavily perturbs the native CUDA-event durations preserved in the raw workload JSON/CSV. Do not divide these byte counts by an unrelated unprofiled duration to claim effective bandwidth. Do not add mapped transfer/dequantization time to gather time as separate wall-time costs.

This component sample neither reproduces nor explains the historical 16.917 ms/token full-model increment. It does not supply all-die schema-2 phase attribution, a controlled 129-output depth matrix, tracing-overhead measurements, longer-context capacity validation, or checkpoint-derived selected-row provenance. Issue #10 and PR #38 remain incomplete/draft.

## Identities and independent replay

The native benchmark embeds source revision `7c6f34d4372567fd97f1cef9b609d3dc0c81b0d3`. Its binary SHA256 is `126ba2ed56f7623a0ee1eeb5f658d719e057ef3c3f54eb1d5d2746f8776523b8`; the cubin SHA256 is `0fa2a17fa41c07aac507f9bc218a777601719a8e2b6eb8b40ec93867d4768027`. Both match PR #43's preserved native matrix identities. Current benchmark/reference-source hashes are also retained. These are **artifact identities, not a newly reconstructed compiler invocation or checkpoint-conversion attestation**.

The compatible nvprof is version 11.4.120, SHA256 `c3e58ebca670c1511c55e1d1420f8fea67d25fba93a4e0d178bc562473ae819d`. The private CUDA 11.4 profiler was already installed; this run replaced no system package.

`executed-smoke-collector.pl` preserves the exact initial collector used for the successful GPU capture, SHA256 `1d68e9f3fbabe27a3b630ed30a1e028ac9ff5b3e03a548281495045f34f78a36`. Later hardening in `kern/decode_counter_capture.pl` adds raw-row matching, stricter dimensions, cross-group identity checks, timestamp fields, and a privileged timeout. **The successful smoke did not execute that later version.** The current CPU tests replay all eight raw profiler/benchmark arms, compare reconstructed metrics against accepted records, verify raw-file identities, and check matching host/VRAM output rows.

From the repository root:

```sh
prove -v kern/decode_counter_capture_test.pl
```

All raw profiler CSV, native workload CSV/JSON, exact command arguments/exits, GPU snapshots, active compute-process inventory, input hashes and accepted metrics are in `smoke-8000-changing/`. The archive manifest at `validation/issue10/IDENTITIES.json` binds their content to the reviewed Git revision.

## Current native collection command

The supplied `smoke-manifest.json` names the actual archy paths and GPU UUID. An output directory must be new. Run on archy:

```sh
perl /path/to/flash-sass/kern/decode_counter_capture.pl \
  --manifest /path/to/smoke-manifest.json \
  --out /tmp/issue10-counter-new \
  --sudo --wait-seconds 120 --poll-seconds 30 --timeout-seconds 60
```

The collector acquires the shared lock before any GPU operation, rejects an occupied target GPU, queries metric availability, checks all expected result rows and workload identities, and preserves a non-PASS status on timeout, missing counters, wrong dimensions, changed inputs, or mismatched outputs. `--sudo` uses noninteractive elevation for the profiler and its bounded native child; the collector itself need not run as root. It never stops, reloads or changes a resident model service or another validator.

A supplied `captured` layout requires a selected-ID file, its byte/hash identity, and an explicit provenance document. A raw trace hash anchors that referenced trace, but **does not verify extraction of those selected IDs from a checkpoint execution**. No checkpoint-derived capture is accepted in the present evidence.

## Larger matrix attempt

`matrix-attempt/` preserves manifests for 3,100 and 8,000 visible positions with clustered, evenly spread, random and changing layouts. The attempt began **2026-10-18T05:43:39Z** and ended **05:45:40Z**, with **exit 73** after the shared lock stayed held for 120 seconds. The first case has no measured pairs. The remaining seven manifests are planned cases, not completed runs. No GPU query or workload was launched by this blocked attempt.

Clustered, evenly spread and random modes in the reused native benchmark hold their row fixed; only `changing` varies it each iteration. This matrix would not, by itself, satisfy the stronger requirement for changing versions of each pattern or real checkpoint traces.

A separate 3,100-position changing-layout retry also exited 73 after a 90-second lock wait; `retry-3100-changing/` and `counter-retry.log` retain its failed preflight. The last recorded lock inventory, **2026-10-18T05:52:34Z**, identifies PID **970242**, `bash kern/sequence_validate.sh`, running the issue #7 validation. This is a timestamped observation, not a claim that the same process continues holding the lock indefinitely. Neither blocked attempt supplied additional counter samples or changed the resident model service.

The `native-validation/` record contains **57 counter-collector checks and 77 existing archive checks**, all passing on engi. Compact replay preserves all 60 traced/20 untraced legacy runs and 360 die records and reproduces the three archived summaries with zero numerical differences. This is a CPU replay of saved records, not a new full-model GPU run. Its identity-manifest hash names the archive as tested before the validation logs themselves were attached; the final root manifest includes those additional artifacts.
