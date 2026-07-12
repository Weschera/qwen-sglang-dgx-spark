#!/bin/bash
# serve-vllm.sh — vLLM (standard Spark image) on DGX Spark (GB10), Qwen3.6-35B-A3B NVFP4 + MTP-3.
# Usage: MODELS=/path/to/models MML=69632 ./serve-vllm.sh
#   MML=69632  for 64k prompts
#   MML=262144 for 256k prompts
#
# ⚠️ Do NOT raise --max-num-batched-tokens above 8192 to try to interleave long prefills.
#    A 64k+ prefill batch exhausts GB10 unified memory and HANGS THE NODE (hard power cycle to recover).
set -eu
MODELS="${MODELS:-/home/$USER/models}"
MODEL="${MODEL:-qwen36-35b-nvfp4}"   # or qwen36-27b-nvfp4
MML="${MML:-69632}"
PORT="${PORT:-8891}"
IMG="eugr/spark-vllm:latest"   # sha256:e557b53d549fdea4588a0d0b4de7573f5679d8a0250408a92572802ce3b301b9

cat > /tmp/vllm_inner.sh <<EOS
#!/bin/bash
export VLLM_MARLIN_USE_ATOMIC_ADD=1
exec vllm serve /models/$MODEL --served-model-name m --host 0.0.0.0 --port $PORT \\
  --tensor-parallel-size 1 --kv-cache-dtype fp8 --attention-backend flashinfer \\
  --gpu-memory-utilization 0.7 --max-model-len $MML --max-num-seqs 64 \\
  --max-num-batched-tokens 8192 --enable-chunked-prefill --async-scheduling \\
  --no-enable-prefix-caching \\
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \\
  --load-format fastsafetensors --reasoning-parser qwen3 \\
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \\
  --language-model-only --skip-mm-profiling
EOS

docker rm -f vllm >/dev/null 2>&1 || true
docker run -d --name vllm --gpus all --network host --ipc host --shm-size 8gb \
  --memory 110g --memory-swap 110g --entrypoint /bin/bash \
  -e HF_HUB_OFFLINE=1 -e NVIDIA_DISABLE_REQUIRE=1 \
  -v "$MODELS":/models:ro -v /tmp/vllm_inner.sh:/serve.sh:ro \
  "$IMG" /serve.sh

echo "vLLM starting on :$PORT (max-model-len $MML). Poll http://localhost:$PORT/v1/models for 200."
