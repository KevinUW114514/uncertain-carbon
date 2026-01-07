#!/usr/bin/env bash
set -euo pipefail

PID="${1:?Usage: $0 <pid> [next_script] [poll_seconds]}"
NEXT_SCRIPT="${2:-./next.sh}"
POLL_SECONDS="${3:-2}"

STAT="/proc/$PID/stat"

# Ensure it exists and we can read it
if [[ ! -r "$STAT" ]]; then
  echo "PID $PID is not running (or /proc not accessible)."
  exit 1
fi

# Field 22 in /proc/<pid>/stat is starttime (ticks since boot)
STARTTIME="$(awk '{print $22}' "$STAT")"

echo "Monitoring PID $PID (starttime=$STARTTIME). Will run: $NEXT_SCRIPT"

while true; do
  # If /proc entry is gone, it's finished
  [[ -r "$STAT" ]] || break

  # If PID got reused, original process is finished
  CUR_STARTTIME="$(awk '{print $22}' "$STAT" 2>/dev/null || true)"
  [[ "$CUR_STARTTIME" == "$STARTTIME" ]] || break

  sleep "$POLL_SECONDS"
done

echo "PID $PID finished; starting $NEXT_SCRIPT"
exec "$NEXT_SCRIPT"

# ./monitor.sh 347044 ./new_start.sh 1