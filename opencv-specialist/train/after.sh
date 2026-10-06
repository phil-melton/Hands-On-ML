#!/usr/bin/env bash
# Queue a job behind the one running in tmux: wait until <log> contains <marker>, then run <command...>.
# Usage (from opencv-specialist/): bash train/after.sh <log> <marker> <name> <command...>
# The queued job logs to ~/opencv-expert/logs/<name>_<time>.log.
set -uo pipefail
WAIT_LOG=$1; MARKER=$2; NAME=$3; shift 3
echo "waiting for '$MARKER' in $WAIT_LOG, then: $*"
until grep -q "$MARKER" "$WAIT_LOG" 2>/dev/null; do sleep 30; done
LOG=~/opencv-expert/logs/${NAME}_$(date +%Y%m%d_%H%M%S).log
echo "starting '$*' (log: $LOG)"
export PYTHONUNBUFFERED=1
("$@") 2>&1 | tee "$LOG"
