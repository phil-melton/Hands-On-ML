#!/usr/bin/env bash
# Launch a long GPU job detached in tmux session "train", so it survives closing any terminal or app.
#   window "job":         the job (top pane, also tee'd to ~/opencv-expert/logs/<name>_<time>.log) + GPU monitor (bottom)
#   window "tensorboard": TensorBoard on http://localhost:6006 for every run under ~/opencv-expert/runs
# Usage (from opencv-specialist/): bash train/tmux_run.sh <name> <command...>
# Watch read-only:  wsl -d Ubuntu-24.04 -e tmux attach -r -t train     (leave: Ctrl-b then d)
# Take control:     wsl -d Ubuntu-24.04 -e tmux attach -t train         (Ctrl-C there stops the job)
set -euo pipefail
NAME=$1; shift
mkdir -p ~/opencv-expert/logs
LOG=~/opencv-expert/logs/${NAME}_$(date +%Y%m%d_%H%M%S).log
if tmux has-session -t train 2>/dev/null; then
  echo "tmux session 'train' already exists; attach to it or kill it first (tmux kill-session -t train)"; exit 1
fi
tmux new-session -d -s train -n job -x 220 -y 55
tmux set-option -t train remain-on-exit on
tmux send-keys -t train:job "cd $(pwd) && export PYTHONUNBUFFERED=1 && ($*) 2>&1 | tee $LOG" Enter
tmux split-window -t train:job -v -l 8 \
  "watch -n 2 /usr/lib/wsl/lib/nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv"
tmux select-pane -t train:job.0
tmux new-window -d -t train -n tensorboard \
  "$HOME/venvs/train/bin/tensorboard --logdir $HOME/opencv-expert/runs --host 127.0.0.1 --port 6006"
echo "started '$*' in tmux session 'train'; log: $LOG"
