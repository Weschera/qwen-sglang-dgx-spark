# SGLang vs vLLM — serving many agents at long context on a DGX Spark (GB10)

Reproducible recipe + results for the long-context, multi-agent serving comparison of **SGLang v0.5.15** vs **vLLM** on a single NVIDIA DGX Spark (GB10, 121 GB unified memory).

**Headline:** at long context with many concurrent agents, SGLang scales and vLLM does not. At 64k context / 64 agents, SGLang finishes all 64 responses before vLLM finishes one. At the model's native 256k, the aggregate gap is ~40×.

**This is NOT "vLLM is slow."** At 1k context / single stream, vLLM is equal-or-faster (it *won* our short-context playground, 450 vs 427 tok/s @ 16 agents). The collapse is specific to **many concurrent long-context prefills** — see [Root cause](#root-cause). Claim is scoped to the **image tags below**; a future vLLM may implement the missing feature.

---

## TL;DR results (1× DGX Spark, Qwen3.6-35B-A3B NVFP4 + MTP)

### 64k context — aggregate decode tok/s

| agents | SGLang v0.5.15 | vLLM (Spark image) |
|---:|---:|---:|
| 2  | 89  | 27 |
| 4  | 178 | 29 |
| 8  | 241 | 30 |
| 16 | 298 | 31 |
| 32 | **324** | **31** |
| 64 | **332** (64/64 complete) | 31 (57/64, rest timed out) |

At 32 agents: **~10× aggregate**, and **first token 6s (SGLang) vs 262s (vLLM)**.

### 256k context (model native max) — the gap widens

| agents | SGLang tok/s (TTFT, done) | vLLM tok/s (TTFT, done) |
|---:|---|---|
| 2 | 85 (1.8s, 2/2) | 3.7 (134s, 2/2) |
| 4 | 109 (4.0s, 4/4) | 3.7 (~6.7 min, 4/4) |
| 8 | **134 (8.2s, 8/8)** | **3.3 (~6.7 min, 6/8 — 2 timed out)** |

**~40× aggregate at 256k.** Single-stream on both engines is fine (vLLM 93 tok/s @64k, 71 @256k) — the collapse is multi-user long-prefill only.

Every number is from `spark_bench.csv` in the [spark-bench](https://github.com/Weschera/spark-bench) harness; the concurrency sweep command is [below](#4-run-the-concurrency-sweep).

---

## Hardware & model

- **1× NVIDIA DGX Spark** (GB10 Grace-Blackwell, SM121a, 121 GB unified memory)
- **Model:** `Qwen3.6-35B-A3B` NVFP4 (MoE, MLA, built-in MTP heads) — same weights served by both engines
- **Drafter:** MTP-3 (vLLM) / NEXTN-3 (SGLang) — the model's own multi-token-prediction heads

## Pinned images (reproducibility)

| Engine | Image | Digest |
|---|---|---|
| vLLM | `eugr/spark-vllm:latest` | `sha256:e557b53d549fdea4588a0d0b4de7573f5679d8a0250408a92572802ce3b301b9` |
| SGLang | `lmsysorg/sglang:v0.5.15-cu130` | `sha256:d0a667eca4e6fff64f7758c5fb1720e16faa806f90ea767e018bb8fa1b09dd44` |

---

## Reproduce

Both engines serve the **same** NVFP4 weights at `/models/qwen36-35b-nvfp4`. Serve one, run the sweep, tear down, serve the other, repeat. Pick a context: `--max-model-len`/`--context-length` sized to the rung (69632 for 64k, 262144 for 256k).

### 1. Serve — SGLang v0.5.15
```bash
docker run -d --name sgl --gpus all --network host --ipc host --shm-size 8gb \
  --memory 110g --memory-swap 110g -e HF_HUB_OFFLINE=1 -e NVIDIA_DISABLE_REQUIRE=1 \
  -v /path/to/models:/models:ro \
  lmsysorg/sglang:v0.5.15-cu130 \
  python3 -m sglang.launch_server \
    --model-path /models/qwen36-35b-nvfp4 --served-model-name m \
    --host 0.0.0.0 --port 8891 --tp 1 --trust-remote-code \
    --mem-fraction-static 0.75 --context-length 66560 --max-running-requests 64 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```
See [`serve-sglang.sh`](serve-sglang.sh). For 256k use `--context-length 262144`.

### 2. Serve — vLLM (standard Spark image)
```bash
docker run -d --name vllm --gpus all --network host --ipc host --shm-size 8gb \
  --memory 110g --memory-swap 110g --entrypoint /bin/bash \
  -e HF_HUB_OFFLINE=1 -e NVIDIA_DISABLE_REQUIRE=1 \
  -v /path/to/models:/models:ro -v ./serve-vllm-inner.sh:/serve.sh:ro \
  eugr/spark-vllm:latest /serve.sh
```
where `serve-vllm-inner.sh` runs (see [`serve-vllm.sh`](serve-vllm.sh)):
```bash
export VLLM_MARLIN_USE_ATOMIC_ADD=1
vllm serve /models/qwen36-35b-nvfp4 --served-model-name m --host 0.0.0.0 --port 8891 \
  --tensor-parallel-size 1 --kv-cache-dtype fp8 --attention-backend flashinfer \
  --gpu-memory-utilization 0.7 --max-model-len 69632 --max-num-seqs 64 \
  --max-num-batched-tokens 8192 --enable-chunked-prefill --async-scheduling \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --load-format fastsafetensors --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --language-model-only --skip-mm-profiling
```

> ⚠️ **GB10 memory warning:** do NOT raise `--max-num-batched-tokens` to interleave long prefills — a 64k+ prefill batch exhausts the GB10 unified-memory budget and **hangs the whole node** (driver OOM holds the RM global lock; the box answers ping but SSH/docker freeze; only a hard power cycle recovers it). We hit this at `--max-num-batched-tokens 65536`. Keep it at 8192.

### 3. Warm up (discard)
One single-stream request at the target context to page in weights before timing.

### 4. Run the concurrency sweep
Using the [spark-bench](https://github.com/Weschera/spark-bench) `tier2` harness (any OpenAI-compatible load client works; match these params):
```bash
python3 spark_bench.py tier2 --label sweep \
  --endpoint http://HOST:8891/v1 --model m \
  --contexts 65536 --concurrency 2,4,8,16,32 --conc-context 65536 \
  --gen-tokens 512 --topology single --spec-decode mtp3
```
`--conc-context` = the prompt length per stream; `--gen-tokens 512` = output length; each `--concurrency` value fires N simultaneous streams.

---

## Root cause

vLLM (this image) **serializes long prefills**. In `vllm/config/scheduler.py`:

```
max_num_partial_prefills:  int = Field(default=1, ge=1)
max_long_partial_prefills: int = Field(default=1, ge=1)
```

Both default to **1**, and they are **hard-gated** — set either ≠ 1 and `arg_utils.py::_check_feature_supported` raises:

```
NotImplementedError: Concurrent Partial Prefill is not supported.
```

So with N concurrent long-context requests, vLLM prefills them **one at a time**: with 64 agents at 64k, the last agent waits behind 63 full prefills (median time-to-first-token in our run: **~9 minutes**). SGLang **interleaves** long prefills by default, so all agents reach first token within seconds and decode overlaps.

If asked *"did you set `--max-num-partial-prefills`?"* — yes; it throws `NotImplementedError`. The feature isn't implemented in this build.

## Scope / honesty

- Claim is scoped to the **image digests above**. Mainline vLLM may implement concurrent partial prefill later; when it does, this gap should close.
- vLLM is **equal-or-faster** at short context / single stream (450 vs 427 tok/s @ 16 agents, 1k ctx). This finding is specific to **many concurrent agents at long context** — the "agent fleet" regime.
- Both engines served the identical NVFP4 checkpoint; only the engine differs.

## License

MIT. Model and engines are under their own licenses.

Harness & full CSV: [Weschera/spark-bench](https://github.com/Weschera/spark-bench) · Leaderboard: [wesche.com/dgx](https://wesche.com/dgx)
