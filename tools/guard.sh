#!/usr/bin/env bash
# Run any long command under a resource watchdog so it can't take the machine down.
#   bash tools/guard.sh NAME cmd args...
# The command runs in its own process group; every 15s we check, and kill the whole
# group if any limit is hit:
#   RAM available < MIN_RAM_GB, swap used > MAX_SWAP_GB, disk free < MIN_DISK_GB,
#   GPU temp > MAX_GPU_C, GPU memory > MAX_GPU_MIB (above this WSL spills VRAM into
#   Windows RAM, which crawls and eats host commit), Windows host free commit
#   < MIN_WIN_COMMIT_GB (checked ~every 30s; Windows has no pagefile here).
# Windows commit is mostly eaten by the Linux page cache (model weights, frames) held in
# vmmemWSL, so before START and whenever free commit drops below MIN_WIN_COMMIT_GB +
# EVICT_MARGIN_GB we first evict our files from the cache (tools/evict_cache.py, no root
# needed) and only kill if commit is still under the limit after that.
# If GPU temp was the only limit hit, we wait until the GPU is at or below COOL_C (at
# least COOL_MIN_S, at most COOL_MAX_S) and run the command again, up to
# MAX_TEMP_RETRIES times. Only use this for commands that resume (all pipeline steps do).
# Output: logs/guard/NAME.log (command output), one-line summary in logs/guard/guard.log.
# Exit code: the command's, or 137 if the watchdog killed it (and no retry was left).
set -u
cd "$(dirname "$0")/.."
NAME=$1; shift
MIN_RAM_GB=${MIN_RAM_GB:-5}; MAX_SWAP_GB=${MAX_SWAP_GB:-6}; MIN_DISK_GB=${MIN_DISK_GB:-8}
MAX_GPU_C=${MAX_GPU_C:-88}; MAX_GPU_MIB=${MAX_GPU_MIB:-47000}; MIN_WIN_COMMIT_GB=${MIN_WIN_COMMIT_GB:-15}
EVICT_MARGIN_GB=${EVICT_MARGIN_GB:-10}
MAX_TEMP_RETRIES=${MAX_TEMP_RETRIES:-3}; COOL_C=${COOL_C:-70}; COOL_MIN_S=${COOL_MIN_S:-120}; COOL_MAX_S=${COOL_MAX_S:-1800}
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
mkdir -p logs/guard; L=logs/guard/guard.log; OUTLOG=logs/guard/$NAME.log
say() { echo "$(date '+%F %T') [$NAME] $*" | tee -a "$L"; }

win_commit() { (cd /mnt/c && timeout 30 "$PS" -NoProfile -Command '[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)' 2>/dev/null | tr -dc 0-9); }
evict() { python3 tools/evict_cache.py 2>&1 | tail -1; }
gpu_temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1 | tr -dc 0-9; }

run_once() {   # sets rc, killed, hot_only
  say "START $* ($(evict))"
  setsid "$@" >> "$OUTLOG" 2>&1 &
  pid=$!
  killed=""; hot_only=""; wincommit=-1; n=0
  while kill -0 $pid 2>/dev/null; do
    sleep 15
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    swap=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1048576)}' /proc/meminfo)
    disk=$(df -BG --output=avail "$PWD" | tail -1 | tr -dc 0-9)
    read gmem gtemp < <(nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader,nounits | head -1 | tr -d ,)
    gmem=${gmem:-0}; gtemp=${gtemp:-0}
    [ "$gmem" -gt "$peak_gpu" ] && peak_gpu=$gmem
    if [ $((n % 2)) -eq 0 ] && [ -x "$PS" ]; then
      w=$(win_commit)
      [ -n "$w" ] && wincommit=$w
      if [ "$wincommit" -ge 0 ] && [ "$wincommit" -lt $((MIN_WIN_COMMIT_GB + EVICT_MARGIN_GB)) ]; then
        e=$(evict); sleep 5; w=$(win_commit); [ -n "$w" ] && wincommit=$w
        say "low win_commit -> evicted ($e), win_commit now ${wincommit}G"
      fi
    fi
    state="ram_avail=${avail}G swap=${swap}G disk=${disk}G gpu=${gmem}MiB ${gtemp}C win_commit=${wincommit}G"
    other=""
    if [ "$avail" -lt "$MIN_RAM_GB" ] || [ "$swap" -gt "$MAX_SWAP_GB" ] || [ "$disk" -lt "$MIN_DISK_GB" ] \
       || [ "$gmem" -gt "$MAX_GPU_MIB" ] \
       || { [ "$wincommit" -ge 0 ] && [ "$wincommit" -lt "$MIN_WIN_COMMIT_GB" ]; }; then other=1; fi
    if [ -n "$other" ] || [ "$gtemp" -gt "$MAX_GPU_C" ]; then
      killed=$state; [ -z "$other" ] && hot_only=1
      say "LIMIT HIT $state -> killing process group"
      kill -TERM -- -$pid 2>/dev/null; sleep 10; kill -KILL -- -$pid 2>/dev/null
      break
    fi
    n=$((n+1)); [ $((n % 40)) -eq 0 ] && say "ok $state"
  done
  wait $pid 2>/dev/null; rc=$?
}

cool_down() {
  local t0c=$(date +%s) t
  say "GPU too hot -> cooling to <=${COOL_C}C before retry $tries/$MAX_TEMP_RETRIES"
  while :; do
    sleep 30; t=$(gpu_temp); t=${t:-0}
    local dt=$(( $(date +%s) - t0c ))
    { [ "$dt" -ge "$COOL_MIN_S" ] && [ "$t" -le "$COOL_C" ]; } && break
    [ "$dt" -ge "$COOL_MAX_S" ] && { say "still ${t}C after ${dt}s, retrying anyway"; break; }
  done
  say "cooled to ${t}C after $(( $(date +%s) - t0c ))s"
}

t0=$(date +%s); peak_gpu=0; tries=0
while :; do
  run_once "$@"
  [ -n "$hot_only" ] && [ "$tries" -lt "$MAX_TEMP_RETRIES" ] || break
  tries=$((tries+1)); cool_down
done
dt=$(( $(date +%s) - t0 ))
if [ -n "$killed" ]; then say "KILLED after ${dt}s (peak_gpu=${peak_gpu}MiB, temp retries=$tries)"; exit 137; fi
say "EXIT rc=$rc after ${dt}s (peak_gpu=${peak_gpu}MiB, temp retries=$tries)"
exit $rc
