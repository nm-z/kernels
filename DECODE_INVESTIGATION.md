# Decode-depth investigation

This report preserves a paired investigation of the historical 36/28/19 tok/s note. **Issue #10 remains incomplete.** The captures support the observations below, but missing launch provenance, a clock-confounded tracing comparison, and missing full-model hardware-traffic attribution prevent acceptance of the full protocol. The original result remains **unreproduced as a matched experiment**: SPEC records the numbers, but the record does not identify the executable, complete configuration, input IDs, clocks or timing window.

The historical 3,100-to-8,000 comparison would be 35.714 to 52.632 ms/token, a 16.917 ms/token increment. It does not establish a linear context cost or a 200k-token throughput prediction.

## Completed experiment

On archy, reuse the six resident weight arenas and compare the same full 48-layer model with budget 512, optional disk PLE, and host-pinned versus VRAM KV. The paired two-core setup uses an 8,192-position engine capacity so both placements fit while preserving the resident holders. The production core normally has a 230,000-position serving limit; its original library, cubin and settings were restored after the experiment.

Each prompt consists of fixed synthetic token IDs from seed 4242. These exercise the actual checkpoint forward pass but do not test language quality. First populate a snapshot through prompt position N-2. Then run five interleaved placement pairs with the same prompt and continuation, greedy sampling and stop tokens disabled. Each measurement generates 129 outputs and measures the 128 intervals between them. Reported N is **prompt length**; the measured visible length grows from N through N+128.

The complete 60-run trace series passed response-count, same-output, snapshot-resume, phase-order, publication, short-read and realized-placement checks. Every measured token includes all 12 production QSA chains: INT8 append, indexer, five selection phases, gather, partial attention and merge. Every die and phase zero is captured.

The engine source used for this series is [bac5b88b665ee6973488dd40b363ded3134136f6](https://github.com/nm-z/flash-sass/commit/bac5b88b665ee6973488dd40b363ded3134136f6). The later PR-head merge of current master is not retroactively labeled as the tested executable. The binary, cubin, GGUF shards, runner, host and device identities are in the archived manifest. The GGUF conversion revision is unknown and remains null.

### Traced results

All rates below include instrumentation. Median is the median of five run means; p95 is the nearest-rank percentile of all 640 individual intervals. No first repetition is discarded.

| Prompt tokens | Host KV median ms/token | Host KV tok/s | VRAM KV median ms/token | VRAM KV tok/s |
|---:|---:|---:|---:|---:|
| 256 | 29.612 | 33.770 | 29.009 | 34.472 |
| 2,048 | 35.000 | 28.572 | 31.910 | 31.339 |
| 2,051 | 34.952 | 28.611 | 31.858 | 31.389 |
| 2,052 | 35.038 | 28.540 | 31.944 | 31.305 |
| 3,100 | 35.011 | 28.563 | 31.924 | 31.325 |
| 8,000 | 35.350 | 28.288 | 32.330 | 30.931 |

The traced median increment from 3,100 to 8,000 is 0.340 ms/token with host KV and 0.406 ms/token with VRAM KV. Those values do not reproduce the historical 16.917 ms/token increment. The 65,000 and 262,000 cases are explicitly capacity-skipped in this controlled setup, not passed. These results do not prove flat decode through 262k.

## Where the time went

Each die has its own timer epoch. The analysis partitions a die's publication-to-publication interval into dispatch, exchange wait, body excluding PLE wait, barrier/send, PLE wait, between-phase gaps, publication footer and between-token gap. These components reconcile to the same die's measured interval. The maximum accepted arithmetic residual in the analysis is 0.00001 microseconds; the recorded residuals are floating-point summation noise below that bound.

These are the historical trace schema's categories. Its body interval begins before opcode dispatch, so `body_excluding_ple_wait` is not a measurement of useful compute alone. Likewise, `barrier_send` combines two activities. An arithmetically complete partition does not by itself reconstruct the critical path across dies or validate the boundaries' semantics. The schema-2 trace work in [PR #30](https://github.com/nm-z/flash-sass/pull/30) addresses these boundaries; its separate validation must not be relabeled as a new run of this depth matrix.

Do not add the six complete die timelines. They overlap. Sender work and the receiver waiting for it are also two observations of the same dependency.

For the illustrative matched repetition 2, the local die-0 timeline gives:

| 3,100→8,000 increment | Host KV, µs/token | VRAM KV, µs/token |
|---|---:|---:|
| Client-observed interval | +339.527 | +413.861 |
| Die-0 publication interval | +340.136 | +413.208 |
| Exchange wait | +379.976 | +461.224 |
| Body excluding PLE wait | +2.624 | −8.640 |
| Barrier/send | −41.168 | −36.960 |
| Other recorded boundaries and waits | −1.296 | −2.416 |

All five repetitions are retained. Repetition 2 is an illustration, not an outlier-exclusion policy.

Summing each distinct attention layer's gather operation once gives about 5.03 ms/token with host KV and 1.96 ms/token with VRAM KV at 3,100 tokens. This is operation work across the 12 serial model layers, not a sum of complete parallel die timelines. In repetition 2, the host gather change at 8,000 tokens is **−2.576 µs/token** across all 12 layers; score scan and five-pass selection increase by 5.848 and 17.416 µs/token respectively. Those measured changes cannot explain an extra 16.917 ms/token.

Die 0 does not own attention in this placement. Its large actq256/exchange-wait bucket includes waiting for the dies that do. Calling that entire bucket “KV quantization time” would misattribute the wait.

### Slow first repetitions are visible in the record

At prompt length 2,048, host repetition 0 takes 44.093 ms/token. Its die-0 PLE wait is 9.090 ms/token, versus 0.00184 ms/token in repetition 2. At 2,051, the corresponding first repetition takes 55.841 ms/token, including 20.898 ms/token of PLE wait; repetition 2 returns to 34.952 ms/token with about 0.00189 ms/token of PLE wait.

This identifies the extra wait in these runs. It does not identify the historical run's PLE configuration, nor isolate NFS service time from operating-system cache state. No storage-cache flush was performed.

At 8,000 tokens, early repetitions also show changing die-4 clocks: the post-run samples include 1,101, 1,126 and 1,151 MHz before returning to 1,177 MHz. Per-layer work on that die and the other dies' receiver waits change with the repetitions. A sampled clock is not a continuous measurement over the entire interval.

## Completed untraced follow-up and remaining gates

The second series completed all 20 measured requests without tracing: five interleaved pairs at each of 3,100 and 8,000 prompt tokens, again with 129 outputs and 128 intervals per request. All continuations matched across KV placement. Both jobs exited 0 and recorded restoration of the normal 230,000-position serving configuration.

| Prompt tokens | Host KV median ms/token | Host KV tok/s | VRAM KV median ms/token | VRAM KV tok/s |
|---:|---:|---:|---:|---:|
| 3,100 | 49.593 | 20.164 | 46.775 | 21.379 |
| 8,000 | 45.664 | 21.899 | 42.747 | 23.394 |

These are the recorded observations, including every repetition. They cannot establish a benefit from longer context or a cost of tracing. During this series die 4 fell to 531 MHz while applications clocks requested 1,177 MHz. The sampled driver report shows **SW Power Cap active**, with **SW Thermal Slowdown inactive**, a temperature of 86°C, and a target temperature of 85°C. The recorded flag does not justify relabeling the cause as thermal throttling. The clock sample also does not continuously measure every token interval.

This clock-confounded series is unsuitable for a clean estimate of tracing overhead against the earlier series. Remaining acceptance includes a matched instrumentation-overhead experiment with controlled clocks, profiler traffic/stall counters, and supported longer-context cases. The original 36/28/19 provenance remains unavailable.

## Metadata audit and limits

The two completed series used the same engine-library SHA256 and core-executable SHA256. The traced and untraced cubins have separately recorded identities. Engine source is pinned to bac5b88; the resident core's source revision was not embedded, so its binary identity is the supported statement. Neither a later source merge nor a later metadata fix changes what these historical runs executed.

The old runner attempted nvidia-smi -q -d ECC,PCIE,CLOCK,TEMPERATURE. Driver 470 rejected that display selector with exit 2. The raw failure remains in the manifests; the ordinary per-run GPU query still supplied UUID, driver, clocks, ECC mode, PCIe link, temperature and memory snapshots. Full before/after driver-counter capture was therefore incomplete. PR #32 now requires a successful full XML query and rejects metadata failures.

The traced runner also captured an empty parent environment. The setup script records the private core's explicit launch settings and the resident engine's reload arguments, but it does not recover every original core launch setting. The untraced series explicitly records the runner/private-core launch environment. These operator setup records are retained separately and are not presented as a retrospectively complete trace manifest.

The resident core was constructed with a larger capacity than the private host core, even though both engine configurations report 8,192. Core-level PLE staging allocation can consequently differ. The weight holders are shared, disk PLE uses shard two at offset 192, and storage cache state was not flushed or independently controlled. Effective configuration equality does not erase those allocation and cache-state limits.

The compact artifacts preserve raw client observation times, generated IDs, every run mean, every die/opcode total and all rejected capacity cases. The analyzer requires successful capture exits, complete five-pair groups, matching configurations apart from KV placement, matching continuations, and exact reconstruction of the 128 client intervals before producing its summaries. These are capture and consistency checks; they do not recover missing provenance or certify every acceptance criterion. In particular, the `PASS` in `review/STATUS.json` concerns the 420-file raw archive check, not closure of issue #10.

## Acceptance status

The machine-readable inventory is [acceptance.json](validation/issue10/acceptance.json). The following distinction is essential when reviewing this evidence:

| Issue #10 requirement | Evidence available | Remaining gap |
| --- | --- | --- |
| Complete run provenance | Engine and artifact identities, token IDs, realized configs, run records | Empty traced parent environment, failed full driver query, incomplete core/conversion and launch records |
| Matched depth matrix | Five host/VRAM pairs at 256, 2,048, 2,051, 2,052, 3,100 and 8,000 | 65,000 and 262,000 have capacity-skip records only; the controlled capacity is not a measured global hardware limit |
| Request and steady decode timing | 129 outputs and 128 recorded intervals per measured request; all repetitions retained | Complete startup/prefill record and an equivalent-clock tracing-overhead comparison |
| All-die phase boundaries | Legacy 129-token traces, phase zero, all-die opcode aggregates | Repeat with the precise schema-2 compute/dispatch boundaries |
| Sender work and receiver waits | Per-die arithmetic reconciliation and illustrative wait deltas | Cross-die causal reconciliation for this depth matrix |
| Hardware traffic and execution counters | CUDA 11.4 enumeration, an initial eight-command smoke, and a completed 64-command host/VRAM component counter matrix at 3,100/8,000 positions | Full-model counter attribution and matched full-model timing |
| Changing selected-block layouts | Related PR #43 gather work and matched component counters for four layout modes at both contexts; changing mode verifies four distinct output rows | Changing clustered/spread/random variants and matched counter arms with verified checkpoint-derived layout provenance |
| Historical slowdown attribution | Historical rates explicitly marked unreproduced; measured replacement deltas retained | A replacement campaign satisfying the complete protocol |

[PR #30](https://github.com/nm-z/flash-sass/pull/30) now contains schema-2 traces and a separate matched, short-generation overhead experiment. [PR #78](https://github.com/nm-z/flash-sass/pull/78) adds stricter live provenance and campaign acceptance checks to the runner from merged [PR #32](https://github.com/nm-z/flash-sass/pull/32). [PR #43](https://github.com/nm-z/flash-sass/pull/43) supplies representative gather workloads. Their source identities and measurement scopes differ from the legacy captures here. They are dependencies and additional evidence, not interchangeable samples for this experiment.

The [profiler discovery record](validation/issue10/counters/README.md) preserves the failed CUDA 12.9 probes and successful CUDA 11.4 inventory. It is explicitly classified as enumeration, not measured hardware traffic.

## New component counter capture

On October 17, 2026, the shared lock became available and a separate native C gather workload completed on archy's spare M60. Its full GPU snapshots bracket the capture at 22:34:27–22:34:46 host-local time. Four independently profiled metric groups in each KV placement produced eight successful command exits and 26 accepted metric rows. At 8,000 visible positions, budget 512 and zero tail, four changing selected-block rows gather 2,048 keys. Host/VRAM cache, selection and verified output identities match.

The raw profiler's per-invocation average `sysmem_read_bytes` is **2,359,296 bytes with host-pinned KV versus 0 with VRAM KV**. Actual DRAM traffic, occupancy, eligible-warps and memory/synchronization stall measurements are retained alongside those system-memory counters. This is measured mapped-memory traffic, not an allocation-size estimate or a substituted logical address footprint.

The [component report, exact commands and raw data](validation/issue10/counters/workloads/README.md) state the narrower scope. The profiled `die` kernel includes gather/dequantization and its persistent-kernel framework; the reported 18 logical invocations include correctness preflight and ten repeated-row warmups. Profiling may replay kernels internally. Profiled CUDA-event rates are retained only as raw diagnostics and are **not authoritative throughput or tracing-overhead measurements**. The native binary's older embedded source identity is kept distinct from the collector and PR revisions.

This sample does not explain the historical 16.917 ms/token increment or validate the full-model depth matrix. A follow-up 3,100/8,000 layout matrix waited 120 seconds for the shared lock and exited 73 before its first GPU query or workload. Full-model schema-2 attribution, complete launch/conversion provenance, matched 129-output runs, supported longer contexts, and actual checkpoint-derived selected rows remain required. PR #38 remains a draft.

### Completed component matrix after the old issue #7 validator

The previously blocked 3,100/8,000 component matrix subsequently completed under the shared lock on **2026-10-18, 06:10:59–06:13:27 UTC**. Eight context/layout cases contain **64 successful profiler commands and 208 accepted metric rows**. Each case has matched host/VRAM payloads and verified outputs. The measured average system-memory read count is 2,359,296 bytes per logical kernel invocation for host-pinned KV and zero for VRAM KV in all eight cases.

[The completed matrix report](validation/issue10/counters/workloads/MATRIX.md) and raw data supersede the earlier claim that this component matrix was still absent. They do not replace the missing full-model experiment. Only the native `changing` layout varies its selected row per iteration; the other three patterns and ten warmups use fixed rows. No checkpoint-derived selection is claimed.

A separate read-only full-model preflight also acquired the shared lock, at **06:18:12 UTC**, but its six samples each observed model GPUs 1–5 at 100% utilization. It exited **75** without inference or service mutation. The recorded GPU/process observations are timestamped evidence, not a claim that the same processes persist. Full-model readiness and the remaining acceptance requirements must be established before this PR is marked ready.

## Reproduction and evidence

Run the resident-core benchmark from PR #32 using the supplied model/placement manifests. The runner itself does not load, reload or kill cores. The setup jobs and restoration records are kept separately so their mutations and effective settings can be reviewed.

Analyze the captured outputs:

~~~bash
perl kern/decode_forensics.pl \
  --traced /path/to/paired-depth-trace3/runs \
  --untraced /path/to/paired-depth-untraced4/runs \
  --out /tmp/issue10-from-captures
~~~

The output directory must be new. For a CPU-only numerical replay using the compact artifacts already committed in this PR, run:

~~~bash
perl kern/decode_forensics.pl \
  --archive validation/issue10 \
  --out /tmp/issue10-compact-replay
prove -v kern/decode_forensics_test.pl
~~~

Archive mode verifies artifact sizes and SHA256 identities before reading strict gzip/JSON inputs. It requires the complete 60-run traced and 20-run untraced matrices, exact paired continuations, 129 publications and 128 positive observation intervals, consistent request/completion fields, and all 360 die records. It checks the per-position selected-key cap/tail counts, phase/opcode partitions, die-0 projections and host/device arithmetic. It independently regenerates all three saved summary files, rejecting differences greater than 1e-8 in their reported units. Outputs identify both Perl source files and the input identity manifest. A rejected archive produces no PASS directory, and an existing output directory is never overwritten.

These checks establish consistency of the supplied compact archive. They do not reread the original phase CSVs, recover missing provenance, or certify the GPU protocol. Artifact identities must still be anchored to the reviewed Git revision; a newly supplied, self-consistent manifest is not proof of historical authenticity.

- [All run intervals, manifests, outputs and die-0 timelines](validation/issue10/traced.json.gz)
- [Untraced intervals, manifests and all 20 runs](validation/issue10/untraced.json.gz)
- [Untraced per-depth summary and every run mean](validation/issue10/untraced-summary.json)
- [All die/opcode totals for all 60 runs](validation/issue10/traced-all-die-phase-totals.json.gz)
- [Per-depth summary and every run mean](validation/issue10/traced-summary.json)
- [Illustrative matched phase deltas](validation/issue10/matched-phase-deltas.json)
- [Later die-4 power/clock snapshot](validation/issue10/die4-power-clock-snapshot.txt)
- [Raw two-position phase excerpts and complete illustrative host traces](validation/issue10/review/raw-excerpts)
- [Full raw archive identities](validation/issue10/review/raw-trace-archive.json)
- [Setup scripts, commands and restoration records](validation/issue10/review/setup)
- [Artifact SHA256 identities](validation/issue10/IDENTITIES.json)

The complete raw per-phase CSV collection is retained at /home/nate/claude/flash-sass-prs/paired-depth-trace3/runs on archy. Its full content is not duplicated in git. The compact review artifacts preserve every run and opcode total. Two-position phase excerpts are attached for dies 0 and 4 at 3,100/8,000 in both KV placements, together with the complete corresponding host traces. The archive manifest identifies every original CSV stream by size and SHA256.

Refs #10. Trace format and lifecycle work are in #13; reproducible capture in #14; shared timing API in #15; representative gather inputs in #16.

## Native reanalysis

The original Perl port replayed all 60 traced and 20 untraced records. Its summary statistics and matched phase deltas agree with the recorded analysis within 1e-8 in the reported units. [Original replay commands, driver hash and comparison](validation/issue10/perl-replay/result.json) retain that driver's identity and measured maximum difference. This older record applies to the exact driver hash it names.

The current compact archive replay and rejection tests are recorded separately in [the native validation record](validation/issue10/native-validation). They exercise successful numerical reproduction and deliberately incomplete, inconsistent, nonfinite, truncated and tampered inputs. Neither replay is a new inference or tracing-overhead experiment.

The original orchestration scripts remain identified by exact Git revision, path, bytes and SHA256 in [historical runner provenance](validation/issue10/review/setup/historical-runners.json). Their executed commands and restoration records remain in the setup JSON files.
