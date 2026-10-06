#!/usr/bin/env bash
# Stage 0 (Path L): GPU training environment inside WSL2, per COMPUTE.md "Path L".
# venv and logs live under ~ (ext4 on the D:-hosted vhdx), not /mnt/*.
set -uo pipefail
REPO=~/Hands-On-ML/opencv-specialist
WORK=~/opencv-expert
mkdir -p "$WORK/logs"
LOG="$WORK/logs/train_env_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "== creating venv ~/venvs/train"
[ -d ~/venvs/train ] || python3 -m venv ~/venvs/train
. ~/venvs/train/bin/activate
pip install --upgrade pip

echo "== pip install unsloth bitsandbytes -r requirements-cv5.txt"
pip install unsloth bitsandbytes -r "$REPO/requirements-cv5.txt"
echo "pip exit: $?"
pip freeze > "$WORK/logs/train_env_freeze.txt"

echo "== GPU check (fp16 matmul prediction: 20-27 TFLOP/s; spec 25.6 dense with fp32 accumulate)"
python - <<'EOF'
import importlib, time
import unsloth  # must precede transformers/trl/peft
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0), "| bf16", torch.cuda.is_bf16_supported())
free, total = torch.cuda.mem_get_info()
print(f"VRAM free {free/2**30:.2f} / {total/2**30:.2f} GiB")
n = 4096
x = torch.randn(n, n, device="cuda", dtype=torch.float16)
for _ in range(5): x @ x
torch.cuda.synchronize(); t = time.time()
for _ in range(50): x @ x
torch.cuda.synchronize(); dt = time.time() - t
print(f"fp16 matmul measured: {50 * 2 * n**3 / dt / 1e12:.1f} TFLOP/s")
for m in ["unsloth", "transformers", "trl", "peft", "bitsandbytes", "xformers", "triton", "accelerate", "datasets", "cv2", "numpy"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:13s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:13s} IMPORT FAILED: {type(e).__name__}: {e}")
EOF
echo "== done. log: $LOG"
