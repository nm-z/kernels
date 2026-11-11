#!/usr/bin/env bash
set -euo pipefail
root=$1
mkdir "$root/readiness"
exec > >(tee "$root/readiness/readiness.log") 2>&1
finish() { local rc=$?; trap - EXIT; printf '%s\n' "$rc" > "$root/readiness/exit"; date -u +%FT%TZ; printf 'READINESS_EXIT=%s\n' "$rc"; }
trap finish EXIT
printf 'QUEUED: issue-10 full-model readiness PID=%s\n' "$$"
date -u +%FT%TZ
exec 9<>/tmp/flash-sass-engine-validation.lock
flock -w 600 -E 73 9
printf 'ACQUIRED: read-only full-model readiness\n'
date -u +%FT%TZ
nvidia-smi -q -x > "$root/readiness/before.xml"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv > "$root/readiness/compute-apps.csv"
ps -eLo pid,tid,ppid,stat,etime,comm > "$root/readiness/processes.txt"
for pid in $(pgrep -x flashcore || :); do
  {
    date -u +%FT%TZ
    readlink "/proc/$pid/exe" "/proc/$pid/cwd"
    tr '\0' '\n' < "/proc/$pid/cmdline"
    tr '\0' '\n' < "/proc/$pid/environ" | grep -E '^(FLASH_|DIE_|CUDA_|ATTN_|SPLIT_|PLE_|MODEL_)' || :
    cat "/proc/$pid/stat"
    cat "/proc/$pid/wchan"
  } > "$root/readiness/core-$pid.txt"
done
for attempt in 1 2 3 4 5 6; do
  date -u +%FT%TZ > "$root/readiness/sample-$attempt.utc"
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.free,clocks.current.sm --format=csv,noheader,nounits > "$root/readiness/sample-$attempt.csv"
  busy=$(awk -F, '$1+0<6 {gsub(/ /,"",$3); if($3+0!=0) n++} END{print n+0}' "$root/readiness/sample-$attempt.csv")
  printf 'MODEL_READINESS sample=%s busy_model_gpus=%s\n' "$attempt" "$busy"
  if test "$busy" -eq 0; then
    printf 'IDLE: all six model GPUs report zero utilization; no inference or reload performed\n'
    exit 0
  fi
  if test "$attempt" -lt 6; then sleep 5; fi
done
printf 'BLOCKED: six-sample full-model preflight observed active model GPUs; no resident service was modified\n'
exit 75
