#!/usr/bin/env bash
set -euo pipefail
root=$1
exec > >(tee "$root/queue.log") 2>&1
finish() { local rc=$?; trap - EXIT; printf '%s\n' "$rc" > "$root/queue.exit"; date -u +%FT%TZ; printf 'QUEUE_EXIT=%s\n' "$rc"; }
trap finish EXIT
printf 'QUEUED_ISSUE=10 PID=%s HOST=%s\n' "$$" "$(hostname)"
date -u +%FT%TZ
exec 9<>/tmp/flash-sass-engine-validation.lock
flock -w 600 -E 73 9
printf 'ACQUIRED: finite issue-10 counter campaign\n'
date -u +%FT%TZ
nvidia-smi --query-gpu=index,uuid,name,memory.free,memory.used,memory.total,utilization.gpu,clocks.current.sm --format=csv > "$root/hardware.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv > "$root/compute-apps.csv"
ps -eo pid,ppid,etime,comm > "$root/processes.txt"
{
  date -u +%FT%TZ
  for pid in $(pgrep -x flashcore || :); do
    ps -p "$pid" -o pid,ppid,etime,comm
    readlink "/proc/$pid/exe" "/proc/$pid/cwd"
    tr '\0' '\n' < "/proc/$pid/cmdline"
    tr '\0' '\n' < "/proc/$pid/environ" | grep -E '^(FLASH_|DIE_|CUDA_|ATTN_|SPLIT_|PLE_|MODEL_)' || :
  done
  ss -xlp | grep -E 'flash|core' | cut -c1-300 || :
} > "$root/resident-preflight.txt"
if command -v socat >/dev/null && test -S /tmp/flash.sock; then
  set +e
  printf 'config\n' | timeout --kill-after=1s 8s socat -T 5 - UNIX-CONNECT:/tmp/flash.sock > "$root/resident-config.txt" 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$root/resident-config.exit"
fi
for nv in 775 2000; do
  for pattern in clustered even random changing; do
    name="nv${nv}-${pattern}"
    printf 'CASE=%s\n' "$name"
    perl "$root/decode_counter_capture.pl" --manifest "$root/$name.manifest.json" --out "$root/$name" --sudo --lock-fd 9 --timeout-seconds 60
  done
done
printf 'COMPLETE: finite issue-10 counter campaign; no resident-service mutations\n'
