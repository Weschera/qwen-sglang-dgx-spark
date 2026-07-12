#!/bin/bash
# serve-sglang.sh — SGLang v0.5.15 on DGX Spark (GB10), Qwen3.6-35B-A3B NVFP4 + MTP(NEXTN).
# Usage: MODELS=/path/to/models CTX=66560 ./serve-sglang.sh
#   CTX=66560  for 64k prompts  (64k + gen headroom)
#   CTX=262144 for 256k prompts (model native max)
set -eu
MODELS="${MODELS:-/home/$USER/models}"
MODEL="${MODEL:-qwen36-35b-nvfp4}"   # or qwen36-27b-nvfp4
CTX="${CTX:-66560}"
PORT="${PORT:-8891}"
IMG="lmsysorg/sglang:v0.5.15-cu130"   # sha256:d0a667eca4e6fff64f7758c5fb1720e16faa806f90ea767e018bb8fa1b09dd44

docker rm -f sgl >/dev/null 2>&1 || true
docker run -d --name sgl --gpus all --network host --ipc host --shm-size 8gb \
  --memory 110g --memory-swap 110g -e HF_HUB_OFFLINE=1 -e NVIDIA_DISABLE_REQUIRE=1 \
  -v "$MODELS":/models:ro \
  "$IMG" \
  python3 -m sglang.launch_server \
    --model-path /models/$MODEL --served-model-name m \
    --host 0.0.0.0 --port "$PORT" --tp 1 --trust-remote-code \
    --mem-fraction-static 0.75 --context-length "$CTX" --max-running-requests 64 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4

echo "SGLang starting on :$PORT (ctx $CTX). Poll http://localhost:$PORT/health for 200."
