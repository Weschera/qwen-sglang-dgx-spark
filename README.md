# Deploy Qwen3.6 (27B / 35B) NVFP4 on a DGX Spark with SGLang

A working recipe for serving **Qwen3.6 NVFP4 + MTP** — both the **27B** (dense) and **35B-A3B** (MoE) — on a single **NVIDIA DGX Spark (GB10)** with **SGLang v0.5.15**. Same launch command for either model; just point `--model-path` at the checkpoint you want. The recipe is tuned for the case that matters on a desk-sized box: **many agents at long context**.

The commands below use the 35B; swap `qwen36-35b-nvfp4` → `qwen36-27b-nvfp4` for the 27B (smaller weights, so it leaves more room for KV / more concurrent agents, but decodes slower per token since it's dense, not MoE).

> **Scope, up front:** the **deploy recipe (this whole guide) works for both the 27B and the 35B** — same command, just change `--model-path`. **Every benchmark and comparison number in this repo was measured on the 35B MoE.** The 27B deploys identically; we didn't run the long-context agent sweep on it, so no 27B numbers are claimed here.

SGLang v0.5.15 (shipped 2026-07-10) is the first release with GB10-native Qwen support — earlier images couldn't load Qwen NVFP4 at all. This repo documents the config that works, and — [secondarily](#why-sglang-here-the-vllm-comparison) — why we picked it over vLLM for this workload.

---

## Quick start

**Requirements: exactly ONE DGX Spark.** Everything in this repo — the deploy, all 64 concurrent agents, both context sizes — runs on a **single** DGX Spark (GB10, 121 GB unified). No cluster, no multi-node, nothing to network together. You also need Docker with GPU access and the `Qwen3.6-35B-A3B` NVFP4 weights at `/path/to/models/qwen36-35b-nvfp4`.

```bash
# 64k context, up to 64 concurrent agents
docker run -d --name sgl --gpus all --network host --ipc host --shm-size 8gb \
  --memory 110g --memory-swap 110g -e HF_HUB_OFFLINE=1 -e NVIDIA_DISABLE_REQUIRE=1 \
  -v /path/to/models:/models:ro \
  lmsysorg/sglang:v0.5.15-cu130 \
  python3 -m sglang.launch_server \
    --model-path /models/qwen36-35b-nvfp4 --served-model-name qwen \
    --host 0.0.0.0 --port 8891 --tp 1 --trust-remote-code \
    --mem-fraction-static 0.75 --context-length 66560 --max-running-requests 64 \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

Wait for `http://localhost:8891/health` → `200` (boots in ~5-6 min, and — nice property — that boot time is roughly constant whether you set 64k or 256k context). Then hit it like any OpenAI endpoint:

```bash
curl -s http://localhost:8891/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```

Scripted version: [`serve-sglang.sh`](serve-sglang.sh) (`MODELS=/path CTX=66560 ./serve-sglang.sh`). For the model's native max context use `--context-length 262144`.

### Config notes
- **`--context-length 66560`** = 64k prompt + generation headroom. Use `262144` for full 256k.
- **`--max-running-requests 64`** sizes the KV pool for 64 concurrent agents. Lower it if you serve fewer (frees KV / lets each go faster).
- **`--speculative-algorithm NEXTN`** uses the checkpoint's built-in MTP heads — free speedup, no external drafter.
- **`--tool-call-parser qwen3_coder --reasoning-parser qwen3`** — required for tool calls / thinking. On this build, `--tool-call-parser auto` also works; the named `qwen*` parsers did not emit structured tool calls in our tests, so use `qwen3_coder`.
- **`--mem-fraction-static 0.75`** is safe on GB10's 121 GB. Going higher risks the unified-memory node-hang (see warning below).

---

## What one Spark actually holds (SGLang, measured — 35B MoE)

Qwen3.6-**35B**-A3B NVFP4 + MTP, aggregate decode tok/s by concurrent agents:

| agents | 64k context | 256k context |
|---:|---:|---:|
| 2  | 89  | 85  |
| 4  | 178 | 109 |
| 8  | 241 | 134 |
| 16 | 298 | — |
| 32 | 324 | — |
| 64 | **332** (all 64 complete) | — |

At 64k, **all 64 agents complete** with first-token ~6s at 32 agents. At the native 256k, 8 concurrent agents stream cleanly (first token ~8s). Single agent: ~93 tok/s @64k, ~71 @256k. Full numbers in [`results/results.csv`](results/results.csv).

---

## Why SGLang here — the vLLM comparison (measured on the 35B MoE)

This is the *secondary* point, but it's why the recipe uses SGLang. **All numbers in this section are the Qwen3.6-35B-A3B MoE** (the deploy recipe above works for both models; only the 35B was benchmarked). We ran the identical NVFP4 weights on both engines. **At short context / single stream, vLLM is equal-or-faster** (it won our 1k playground, 450 vs 427 tok/s @ 16 agents). But for **many agents at long context**, the standard Spark vLLM image collapses:

| 64k context, aggregate tok/s | SGLang v0.5.15 | vLLM (Spark image) |
|---:|---:|---:|
| 32 agents | **324** | 31 |
| 64 agents | **332** (64/64) | 31 (57/64) |

At 64k / 64 agents, SGLang finishes all 64 responses before vLLM finishes one. At 256k it's ~40×.

**Root cause** (`vllm/config/scheduler.py`): `max_num_partial_prefills` and `max_long_partial_prefills` default to **1** and are hard-gated — set either ≠1 and vLLM raises `NotImplementedError: Concurrent Partial Prefill is not supported`. So vLLM prefills long requests **one at a time** (median time-to-first-token with 64 agents: ~9 min). SGLang interleaves them by default.

This is **not a misconfiguration — it's a known, upstream-tracked gap in vLLM's V1 engine** (the current default, and the only one serving NVFP4+MTP on GB10):
- [vLLM #14003](https://github.com/vllm-project/vllm/issues/14003) — *"Implement Concurrent Partial Prefills In V1 Engine"* — **open** feature request
- [vLLM #39737](https://github.com/vllm-project/vllm/issues/39737) — *"max-num-partial-prefills fails on V1 engine start"* — the same `NotImplementedError`, reported independently
- The feature exists in the legacy **V0** engine ([PR #10235](https://github.com/vllm-project/vllm/pull/10235)), which won't serve this stack on GB10

**What we ruled out** (all still ~31 tok/s @ 64k/32 agents, so it's none of these):
- MTP spec decode **off**
- `--max-num-batched-tokens` raised (16384) + `--async-scheduling` **off**
- The setting that *would* fix it (`--max-num-partial-prefills > 1`) — throws `NotImplementedError`

**Verified across every vLLM that runs on a DGX Spark — including v0.25.0, empirically:**
- Community Spark image (vLLM **0.23.1**, digest `e557b53…`): gate present in source; benchmarked (numbers above) and reproduced from this repo's scripts.
- Latest Spark nightly (`nightly-20260711`): same gate.
- **vLLM v0.25.0 on GB10** (via the [NNNtrance community build](https://github.com/NNNtrance/Qwen3.6-35B-A3B-NVFP4-Fast-DGX-Spark), which pins a digest of `vllm/vllm-openai` with sm_121 kernels): same gate in source, **and we ran the 64k sweep on it — c2/8/32 = 26.4 / 29.3 / 30.6 tok/s, TTFT@c32 269s. Identical collapse (0.23.1 was 26.8 / 30.0 / 31.3).** Upgrading vLLM does not change this; [#14003](https://github.com/vllm-project/vllm/issues/14003) is still open.
- The **tagged official image** (`vllm/vllm-openai:v0.25.0-aarch64-cu129`) does not run on GB10 at all (cu129 vs GB10's CUDA 13 — dies on `libnvrtc.so.13`).
- (v0.25.0's "Model Runner V2" is a faster *execution* core for dense models — it doesn't change concurrent-prefill *scheduling*, and our model is the MoE.)

**Scope:** specific to vLLM's **V1 engine** and the many-agent / long-context regime. When [#14003](https://github.com/vllm-project/vllm/issues/14003) lands, this should close. Reproduce: [`serve-vllm.sh`](serve-vllm.sh).

### Pinned images
| Engine | Image | Digest |
|---|---|---|
| SGLang | `lmsysorg/sglang:v0.5.15-cu130` | `sha256:d0a667eca4e6fff64f7758c5fb1720e16faa806f90ea767e018bb8fa1b09dd44` |
| vLLM | `eugr/spark-vllm:latest` | `sha256:e557b53d549fdea4588a0d0b4de7573f5679d8a0250408a92572802ce3b301b9` |

---

## ⚠️ GB10 memory-hang warning

Do **not** raise `--max-num-batched-tokens` (vLLM) to try to interleave long prefills. A 64k+ prefill batch exhausts GB10's unified-memory budget and **hangs the whole node** — the driver OOM holds the RM global lock, the box still answers ping but SSH/docker freeze, and only a hard power cycle recovers it. We hit this at `--max-num-batched-tokens 65536`. On SGLang, keep `--mem-fraction-static ≤ 0.8`.

## Reproduce the sweep

Concurrency numbers were produced with the [spark-bench](https://github.com/Weschera/spark-bench) `tier2` load harness (any OpenAI-compatible concurrent-load client works):

```bash
python3 spark_bench.py tier2 --label sweep \
  --endpoint http://HOST:8891/v1 --model qwen \
  --contexts 65536 --concurrency 2,4,8,16,32,64 --conc-context 65536 \
  --gen-tokens 512 --topology single
```

## License

MIT. Model and engines under their own licenses. Harness + full CSV: [Weschera/spark-bench](https://github.com/Weschera/spark-bench) · Leaderboard: [wesche.com/dgx](https://wesche.com/dgx)
