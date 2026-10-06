#!/usr/bin/env bash
# Stage 1 hydrotest, Path L: for each base, QLoRA 30 steps -> merge -> GGUF Q4_K_M -> Ollama -> generate -> grade.
# Tests plumbing, not quality. One base failing does not stop the others.
# Usage: bash opencv-specialist/train/hydrotest.sh [BASE ...]   (default: all three candidates, default base first)
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
WORK=~/opencv-expert
BASES=("$@")
[ ${#BASES[@]} -eq 0 ] && BASES=(qwen2.5-coder-7b qwen3.5-4b qwen3.5-9b)
. ~/venvs/train/bin/activate
cd "$REPO"
python -m train.make_format_set

for BASE in "${BASES[@]}"; do
  OUT=$WORK/runs/hydro-$BASE
  mkdir -p "$OUT"
  echo "=================== $(date -Is) hydrotest $BASE -> $OUT"
  # Ollama answers a keep_alive:0 request before its runner has released VRAM, and a busy GPU at load time
  # produced the 21:34-21:37 OOMs (LOGBOOK 2026-10-05). Wait for an idle GPU, not just an empty /api/ps.
  for _ in $(seq 1 24); do
    USED=$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    curl -s 127.0.0.1:11434/api/ps | grep -q '"models":\[\]' && [ "$USED" -lt 2500 ] && break
    sleep 5
  done
  if [ "$USED" -ge 2500 ]; then echo "GPU still holds ${USED} MiB after 2 min; skipping $BASE"; continue; fi
  BASE=$BASE OUT=$OUT QLORA=1 MAX_STEPS=30 SAVE_STEPS=15 python -m train.train_lora; RC=$?
  if [ $RC -eq 137 ] || [ $RC -eq 143 ]; then
    # Killed by a signal (the Windows commit watchdog, or an operator): stop the chain instead of starting
    # the next base under the same memory pressure.
    echo "!!! $BASE: killed (exit $RC); stopping the hydrotest chain"; exit $RC
  fi
  if [ $RC -ne 0 ]; then echo "!!! $BASE: train/export FAILED"; continue; fi
  GGUF=$(find "$OUT" -iname "*q4_k_m*.gguf" | head -1)
  if [ -z "$GGUF" ]; then echo "!!! $BASE: no Q4_K_M GGUF found under $OUT"; continue; fi
  python -m train.ollama_create "$GGUF" "opencv-hydro-$BASE" --base "$BASE" || { echo "!!! $BASE: ollama create FAILED"; continue; }
  python -m train.hydro_eval "opencv-hydro-$BASE" --out "$OUT/hydro_eval.json" || echo "!!! $BASE: eval FAILED"
done
echo "=================== $(date -Is) hydrotest finished"
