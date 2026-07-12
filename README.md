# Deploy Qwen3.6 (27B / 35B) NVFP4 on a DGX Spark with SGLang

A working recipe for serving **Qwen3.6 NVFP4 + MTP** — both the **27B** (dense) and **35B-A3B** (MoE) — on a single **NVIDIA DGX Spark (GB10)** with **SGLang v0.5.15**. Same launch command for either model; just point `--model-path` at the checkpoint you want.

> **Scope, up front:** the **deploy recipe works for both the 27B and the 35B** — same command, just change `--model-path`. All benchmark numbers below were measured on the **35B MoE**.

> ## ⚠️ Correction (2026-07-12)
> An earlier version of this README claimed SGLang delivered **~10–40× more aggregate throughput than vLLM at long-context concurrency**. **That comparison was flawed and the claim is withdrawn.** Our benchmark sent an identical prompt to every concurrent stream; SGLang's radix cache (on by default) served the repeated prefills from cache while vLLM ran with prefix caching explicitly disabled. We were measuring cache asymmetry, not the scheduler.
>
> **With caching disabled on both engines, they are equivalent** at 64k-context concurrency on GB10 (SGLang 25.7/28.6/29.8 tok/s at 2/8/32 agents vs vLLM 26.8/30.0/31.3 — within noise; first token ~4.5 min at 32 agents on both). Full corrected data in [`results/results.csv`](results/results.csv).
>
> The real, useful finding is below: **prefix caching is the entire game for concurrent long-context serving** — and SGLang ships with it on by default.

---

## Quick start

**Requirements: exactly ONE DGX Spark.** Everything here runs on a **single** DGX Spark (GB10, 121 GB unified). No cluster. You also need Docker with GPU access and the NVFP4 weights at `/path/to/models/`.

```bash
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

Wait for `http://localhost:8891/health` → 200 (~6 min boot, roughly constant across context sizes). Scripted: [`serve-sglang.sh`](serve-sglang.sh) (`MODELS=/path MODEL=qwen36-35b-nvfp4 CTX=66560 ./serve-sglang.sh`). For 256k use `CTX=262144`. vLLM equivalent: [`serve-vllm.sh`](serve-vllm.sh).

### Config notes
- `--context-length 66560` = 64k prompt + generation headroom; `262144` = model native max.
- `--max-running-requests 64` sizes KV for 64 concurrent agents.
- `NEXTN` = the checkpoint's built-in MTP heads (free speedup, no external drafter).
- **Radix cache is ON by default** — that's usually what you want (see findings). Add `--disable-radix-cache` only for benchmarking honesty.
- `--mem-fraction-static 0.75` is safe on GB10. See the memory-hang warning below before raising anything.

---

## What we measured (35B MoE, 64k-token prompts, 1× Spark)

### 1. With caching disabled on BOTH engines, SGLang and vLLM are equivalent

| agents @64k | SGLang (no radix cache) | vLLM (no prefix cache) |
|---:|---:|---:|
| 1 (single stream) | 92.2 tok/s | 93.7 tok/s |
| 2 | 25.7 | 26.8 |
| 8 | 28.6 | 30.0 |
| 32 | 29.8 · TTFT ~4.5 min | 31.3 · TTFT ~4.4 min |

Both engines process concurrent **unique** long prefills essentially serially on GB10; aggregate flatlines around ~30 tok/s and first-token latency grows linearly with the queue. (vLLM verified on 0.23.1 **and** 0.25.0 — identical. vLLM's V1 engine also hard-rejects `--max-num-partial-prefills` >1: `NotImplementedError`, tracked in [vllm#14003](https://github.com/vllm-project/vllm/issues/14003).)

**Practical takeaway:** on one Spark, a fleet of agents with fully *distinct* long contexts is prefill-compute-bound no matter the engine. Plan for it: stagger agent starts, or share context (below).

### 2. Prefix caching is the entire game — and SGLang ships with it ON

Same test, but letting SGLang's radix cache work (streams share the prompt):

| agents @64k, shared prefix | SGLang (radix cache, default) |
|---:|---:|
| 32 | **324 tok/s** · TTFT ~6s |
| 64 | **332 tok/s** · all 64 complete |

That's **~10× the no-cache number**, from caching alone. If your agents share a large system prompt / document corpus (most real agent fleets do), prefix reuse is worth more than any other serving knob at long context. vLLM has prefix caching too (`--enable-prefix-caching`) — we disabled it for the original flawed test; enable it in production for the same class of win.

### 3. Honest caveats
- The dramatic numbers in row 2 require **shared prompt prefixes**. Fully independent contexts get row 1.
- Quality is engine-neutral in our separate evals (TrueScore 85.4 SGLang vs 84.1 vLLM on the 27B; within noise across engines).
- Short context (1k), 16 agents: vLLM 450 vs SGLang 427 tok/s — vLLM slightly ahead there.

### Pinned images
| Engine | Image | Digest |
|---|---|---|
| SGLang | `lmsysorg/sglang:v0.5.15-cu130` | `sha256:d0a667eca4e6fff64f7758c5fb1720e16faa806f90ea767e018bb8fa1b09dd44` |
| vLLM | `eugr/spark-vllm:latest` | `sha256:e557b53d549fdea4588a0d0b4de7573f5679d8a0250408a92572802ce3b301b9` |

vLLM 0.25.0 cross-check ran on the [NNNtrance GB10 build](https://github.com/NNNtrance/Qwen3.6-35B-A3B-NVFP4-Fast-DGX-Spark). The tagged official `v0.25.0-aarch64-cu129` image does not run on GB10 (CUDA 12.9 vs 13).

---

## ⚠️ GB10 memory-hang warning

Do **not** raise `--max-num-batched-tokens` (vLLM) to try to speed up long prefills. A 64k+ prefill batch exhausts GB10's unified memory and **hangs the whole node** (driver OOM holds the RM lock; ping answers, SSH doesn't; hard power cycle required). We hit this at `65536`. Keep it at 8192. On SGLang keep `--mem-fraction-static ≤ 0.8`.

## Reproduce

Load harness: [spark-bench](https://github.com/Weschera/spark-bench) `tier2`. Note for benchmarkers: `tier2` sends an **identical prompt to all concurrent streams** — disable caching on both engines (`--disable-radix-cache` / omit `--enable-prefix-caching`) or your numbers will measure cache, not serving. That's the mistake we made first.

```bash
python3 spark_bench.py tier2 --label sweep \
  --endpoint http://HOST:8891/v1 --model qwen \
  --contexts 65536 --concurrency 2,8,32 --conc-context 65536 \
  --gen-tokens 512 --topology single
```

## License

MIT. Harness + full CSV: [Weschera/spark-bench](https://github.com/Weschera/spark-bench) · [wesche.com/dgx](https://wesche.com/dgx)
