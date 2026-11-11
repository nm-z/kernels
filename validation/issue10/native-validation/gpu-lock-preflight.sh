#!/usr/bin/env bash
set -u
date -u +%Y-%m-%dT%H:%M:%SZ
ssh archy 'bash -s' <<'REMOTE'
date -u +%Y-%m-%dT%H:%M:%SZ
exec 9<>/tmp/flash-sass-engine-validation.lock
if ! flock -w 45 9; then
  printf '%s\n' 'BLOCKED: shared GPU lock remained held after 45 seconds.'
  lslocks -o PID,COMMAND,PATH | rg flash-sass-engine-validation.lock
  ps -p 4010221 -o pid,etime,pcpu,stat,args
  exit 73
fi
printf '%s\n' 'ACQUIRED: read-only GPU/workload preflight.'
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,clocks.current.sm,temperature.gpu --format=csv
ps -eo pid,etime,args | rg 'flashcore|run.rb|paired_depth|nvprof' | head -30
REMOTE
