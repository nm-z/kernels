# Maxwell profiler discovery and workload counters

The `profiler/` files preserve earlier profiler discovery on `archy`: **metric enumeration, not measured workload counters**. The separate [initial workload capture](workloads/README.md) contains eight successful real profiler arms, and the later [3,100/8,000 matrix](workloads/MATRIX.md) adds 64 successful profiler commands across eight matched context/layout cases. These advance, but do not complete, issue #10's full-model attribution requirements.

`profiler/discovery.json` records the GPU UUID, driver, profiler paths, timestamps, command arguments, exit codes, and package hashes. The selected device was Tesla M60 `GPU-431371bd-b482-96b2-4b70-04f83c6d075b` (then ordinal 6), compute capability 5.2, driver 470.256.02.

## Recorded results

| Executable | Probe | Recorded exit | Interpretation |
| --- | --- | ---: | --- |
| CUDA 12.9.79 `/opt/cuda/bin/nvprof` | Metric query, both user and root | 12 | CUDA profiling error; no metric values were collected. |
| Private CUDA 11.4.120 `nvprof` | Metric query | 0 | Maxwell metric names were enumerated. |
| Private CUDA 11.4.120 `nvprof` | Event query | 0 | Maxwell event names were enumerated. |

The 11.4 executable's recorded SHA256 is `c3e58ebca670c1511c55e1d1420f8fea67d25fba93a4e0d178bc562473ae819d`. NVIDIA's package records and the two extracted-package SHA256 values are retained. The discovery did not replace system packages. Its metric inventory includes:

- System memory: `sysmem_read_bytes`, `sysmem_write_bytes`, `sysmem_read_transactions`, `sysmem_write_transactions`.
- Device memory: `dram_read_bytes`, `dram_write_bytes`, `dram_read_transactions`, `dram_write_transactions`.
- Execution: `achieved_occupancy`, `eligible_warps_per_cycle`.
- Stalls: `stall_memory_dependency`, `stall_memory_throttle`, `stall_sync`, plus the other enumerated stall categories.

## Remaining collection

A workload run must hold `/tmp/flash-sass-engine-validation.lock`, record `nvidia-smi` and `ps`, verify the target GPU by UUID and available capacity, then use the compatible profiler in a separate experiment. Keep exact source, binary/cubin, profiler, input, environment, command, output-validation, and exit records. Matched host/VRAM arms need the same selected blocks and payloads, including changing clustered, evenly spread, random, and checkpoint-derived layouts. Imported IDs alone do not establish checkpoint provenance.

Retain raw profiler output and check that required values were actually returned. A zero query exit, elapsed CUDA-event time, allocation size, or logical byte count must not be recorded as a hardware traffic sample. The finite operation's profiling scope must also distinguish kernel dispatch/barriers from the operation itself. Workload replay with external peer or host progress dependencies requires separate validation before its metrics can be accepted.

The accepted component captures record actual system-memory/DRAM bytes and transactions, occupancy, eligible warps and stalls. The larger layout/context matrix completed after its earlier blocked attempts, but its raw profiler output is not a full-model counter campaign. Current full-model depth-matrix tracing-overhead measurements remain absent. The legacy decode captures, component counters and newer trace work use different runtime identities; their results cannot be combined into a purported matched full-model comparison.
