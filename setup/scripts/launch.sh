#!/usr/bin/env bash
# Detached job launcher for oz.gccis.rit.edu (no batch scheduler). Adapted from the DANCE Stage 1 launcher.
#
# Shared-server etiquette: the job gets ONE idle GPU (no compute processes, <500 MiB used, <5% util) with at
# least --min-mem-mib of memory, pinned by UUID. It never takes a busy GPU, never retries and never resubmits.
# Each job writes job.json, stdout/stderr, /usr/bin/time -v, 1 Hz nvidia-smi samples and exit_status into
# <log-dir>/<JOB_ID>/ and appends one line to setup/jobs/RESOURCE_LOG.tsv.
#
# Usage (on oz):
#   bash ~/vera_research/setup/scripts/launch.sh --name NAME [options] -- <command> [args...]
# Options:
#   --log-dir DIR        parent directory for the job directory (default setup/jobs)
#   --gpu auto|none|UUID GPU selection (default auto)
#   --min-mem-mib N      auto mode: only GPUs with at least N MiB total (default 40000: an RTX 6000 Ada)
#   --time-limit SEC     optional hard wall-clock bound (TERM at SEC-30, KILL at SEC); default none
#   --threads N          OMP/MKL thread cap (default 16)
#   --dry-run            print the job record, launch nothing
# In the command, {JOB_DIR} is replaced by the job directory (also exported as $JOB_DIR).
set -Eeuo pipefail

ROOT="${VERA_PROJECT_ROOT:-$HOME/vera_research}"
PY="$ROOT/setup/environment/env/bin/python"
JOBS="$ROOT/setup/jobs"
RESLOG="$JOBS/RESOURCE_LOG.tsv"

name=""; log_dir="$JOBS"; gpu_mode="auto"; min_mem=40000; limit_s=0; threads=16; dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) name="$2"; shift 2 ;;
    --log-dir) log_dir="$2"; shift 2 ;;
    --gpu) gpu_mode="$2"; shift 2 ;;
    --min-mem-mib) min_mem="$2"; shift 2 ;;
    --time-limit) limit_s="$2"; shift 2 ;;
    --threads) threads="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --) shift; break ;;
    *) echo "unknown option $1 (put the command after --)" >&2; exit 2 ;;
  esac
done
[[ -n "$name" && "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "--name is required ([A-Za-z0-9_.-])" >&2; exit 2; }
[[ $# -gt 0 ]] || { echo "no command given after --" >&2; exit 2; }
[[ "$limit_s" =~ ^[0-9]+$ ]] && { (( limit_s == 0 || limit_s > 30 )); } || { echo "--time-limit must be 0 or > 30" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "env missing: run setup/scripts/setup_env.sh first" >&2; exit 4; }

mkdir -p "$JOBS" "$log_dir"
exec 9>"$JOBS/.submit.lock"
flock -n 9 || { echo "another submission is in progress" >&2; exit 3; }

case "$gpu_mode" in
  none) gpu_uuid=""; gpu_line="none (CPU-only)" ;;
  auto)
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u | tr '\n' ' ')
    gpu_uuid=$(nvidia-smi --query-gpu=uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F', *' -v busy="$busy" -v min="$min_mem" 'index(busy, $1) == 0 && $2+0 >= min && $3+0 < 500 && $4+0 < 5 {print $1}' |
      head -1)
    [[ -n "$gpu_uuid" ]] || { echo "no idle GPU with >= $min_mem MiB right now; not consuming a busy GPU" >&2; exit 8; }
    gpu_line=$(nvidia-smi --id="$gpu_uuid" --query-gpu=index,name,driver_version,memory.total --format=csv,noheader) ;;
  *) gpu_uuid="$gpu_mode"
     gpu_line=$(nvidia-smi --id="$gpu_uuid" --query-gpu=index,name,driver_version,memory.total --format=csv,noheader) ;;
esac

JOB_ID="${name}_$(date +%Y%m%dT%H%M%S)_$$"
JD="$log_dir/$JOB_ID"
cmd=()
for a in "$@"; do cmd+=("${a//\{JOB_DIR\}/$JD}"); done

record=$("$PY" - "$JOB_ID" "$gpu_uuid" "$gpu_line" "$limit_s" "$threads" "${cmd[@]}" <<'PY'
import json, socket, sys, getpass
from datetime import datetime
(job_id, gpu_uuid, gpu_line, limit_s, threads), cmd = sys.argv[1:6], sys.argv[6:]
print(json.dumps({
    "job_id": job_id,
    "mechanism": "direct execution on shared GPU server (no scheduler on oz); setsid+nohup, detached",
    "submitted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "host": socket.getfqdn(), "user": getpass.getuser(),
    "gpu_uuid": gpu_uuid or None, "gpu": gpu_line,
    "time_limit_s": int(limit_s) or None, "threads": int(threads),
    "command": cmd,
}, indent=2))
PY
)
if (( dry_run )); then
  echo "DRY RUN: nothing launched. Would submit:"; echo "$record"; exit 0
fi
mkdir -p "$JD"
printf '%s\n' "$record" > "$JD/job.json"
[[ -f "$RESLOG" ]] || printf 'job_id\tgpu\tstarted\tfinished\texit_status\twall_s\tmax_rss_kib\tpeak_gpu_mem_mib\tpeak_gpu_util_pct\tjob_dir\n' > "$RESLOG"

cd "$ROOT"
setsid nohup bash -c '
  JD="$1"; GPU="$2"; LIMIT="$3"; THREADS="$4"; RESLOG="$5"; ROOT="$6"; shift 6
  echo $$ > "$JD/pid"
  start=$(date -Is); echo "$start" > "$JD/started_at"
  sampler=""
  if [[ -n "$GPU" ]]; then
    nvidia-smi --id="$GPU" --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -lms 1000 > "$JD/gpu_samples.csv" 2>/dev/null &
    sampler=$!
  fi
  tl=(); (( LIMIT > 0 )) && tl=(timeout --signal=TERM --kill-after=30s "$((LIMIT - 30))s")
  export JOB_DIR="$JD" CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 \
         OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" TOKENIZERS_PARALLELISM=false \
         HF_HOME="$ROOT/shared/models/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERA_PROJECT_ROOT="$ROOT"
  /usr/bin/time -v -o "$JD/time_v.txt" "${tl[@]}" "$@" > "$JD/stdout.log" 2> "$JD/stderr.log"
  rc=$?
  [[ -n "$sampler" ]] && kill "$sampler" 2>/dev/null
  echo "$rc" > "$JD/exit_status"
  end=$(date -Is); echo "$end" > "$JD/finished_at"
  wall=$(( $(date -d "$end" +%s) - $(date -d "$start" +%s) ))
  rss=$(awk -F": " "/Maximum resident/ {print \$2}" "$JD/time_v.txt" 2>/dev/null)
  peak="-"; util="-"
  [[ -s "$JD/gpu_samples.csv" ]] && read -r peak util < <(awk -F", *" "NR>1 {m=\$2+0; if (m>mx) mx=m; u=\$3+0; if (u>ux) ux=u} END {print mx+0, ux+0}" "$JD/gpu_samples.csv")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(basename "$JD")" "${GPU:-none}" "$start" "$end" "$rc" "$wall" "${rss:--}" "$peak" "$util" "$JD" >> "$RESLOG"
' _ "$JD" "$gpu_uuid" "$limit_s" "$threads" "$RESLOG" "$ROOT" "${cmd[@]}" < /dev/null > "$JD/wrapper.log" 2>&1 9>&- &

sleep 1
echo "submitted $JOB_ID on GPU: $gpu_line${gpu_uuid:+ ($gpu_uuid)}; time limit: ${limit_s/#0/none}"
echo "job dir: $JD"
