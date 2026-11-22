---
name: avalon-decode
description: Operate and profile the Avalon fork serving Qwen3.8-Flash-Next on the 2x Navi21 workstation. Use when measuring decode/prefill t/s, capturing kernel traces, tuning hosted-MoE knobs, or optimizing decode in moe-cache.cu.
---

# Avalon decode profiling and optimization

Private llama.cpp fork (`/home/arnaud/Workspace/git/Avalon`) serving Qwen3.8-Flash-Next (qwen4exp arch: 48 blocks, 512 experts/layer, top-10, IQ3_KS experts, HC hyperconnections, shared expert, 2 kv heads x 256 dim) with MTP speculation. All persistent artifacts live in `/home/arnaud/tmp/` (survives reboot, `/tmp` does not).

## Machine and service

- Host: Ryzen 9 7900X, 30 GiB RAM (~10 GiB available at idle), rootless podman, no usable sudo (cannot write sysfs, reload amdgpu, reboot). `iostat` NOT installed, use `/proc/diskstats`. `rocm-smi` is `/opt/rocm/bin/rocm-smi`. Patched kernel ("arnaud-patch") makes `sclk` stick at 2200/2575 and mclk/fclk sysfs readouts garbage under load: cosmetic, do not chase; measured card bandwidth is 405-424 GB/s.
- GPUs: Navi21 x2 = RX 6800 XT (PCI 03:00.0) + RX 6800 (07:00.0), gfx1030, 16 GiB each, through PLX bridges.
- Service: systemd user quadlet `~/.config/systemd/user/avalon.service` + pod `~/.config/pods/avalon.yaml` (live = stock config; pristine backup `avalon.yaml.bak`). Host port 8099. Image `localhost/avalon:rebased` (= `:prof`, md5 `3b81065bdc593e4f33d0fd75f1927a7a`; `:prof` adds rocprofv3). Boot ready 90-300 s.
- Runtime args (stock): `-c 240000 -b 8000 -ub 8000 -fa on -ctk q8_0 -ctv q4_0 --jinja --repeat-penalty 1.0 --parallel 1 -md MTP... --spec-type draft-mtp --spec-draft-n-max 2 -ubd 256 -ctkd q8_0 -ctvd q8_0 --metrics`. Env: `HIP_VISIBLE_DEVICES=0,1`, `LLAMA_STAGES=0`, `GGML_CUDA_MOE_RING_PREFILL=900`, `GGML_CUDA_MOE_HOSTED_THREADS=12`, `GGML_CUDA_MOE_ARENA_READERS=24`.
- Weights: shards 91 GB at `/home/arnaud/Workspace/ai/avalon-models` (host) = `/models` (container). Draft: `MTP/mtp-Qwen3.8-Flash-Next-shared-IQ3K-IQ4NL-FR65k.gguf`. Ranking `moe-cache-ranking.bin` (Oct 31) auto-preloads at boot.
- Expert hosting: VRAM pools 7262 experts (ROCm0) + 5958 (ROCm1); arena 7640 + 7787 MiB in RAM; expert ~2.05 MB, VRAM slot 2.6 MB. Pool hit ~93.6% in decode. CPU expert rows: HOSTED_THREADS=12, arena readers 24.
- Safety: never stop/restart while the model is loading (SIGKILL during load wedged the pod on Nov 19). `systemctl --user restart avalon.service` is safe between requests. Traced-pod swaps are accepted when the user asks, they cost 10-20 min downtime. `LLAMA_MTP_ROWS_ASYNC=1` is FORBIDDEN on the 2-GPU pipeline: draft acceptance collapses to 0.0 mid-session (verified Nov 21, reverted). `--spec-type none` bootloops the fork. After editing avalon.yaml, quadlet may fire a delayed extra restart minutes later (seen Nov 21 17:11, killed a probe run): confirm the pid is stable before starting measurements.

## Measurement recipes

- Journal decode truth: `journalctl --user -u avalon.service -o cat | grep -E 'eval time|tg = |draft acceptance|preloaded'`. `tg_3s` = decode metric. `prompt eval time` on 11-token prompts is noise. Journal stamp = MINUTES.SECONDS (not hours). `[mem]` lines are lifetime-cumulative: window-diff before dividing by tokens.
- Metrics endpoint `:8099/metrics`: `llamacpp:spec_decode_num_drafts_total` (verify steps), `num_draft_tokens_total`, `num_accepted_tokens_total`, per_pos acceptance. tokens/step = predicted_n / delta(drafts).
- `agentic-probe.py` (chat-API harness, 3.5k sys prompt, 400-gen turns, 2k-tok tool outputs): THE standard test, expect ~26-30 t/s decode. Prints ctx/cache/newp/newpp_t/s/gen/dec_t/s/ttft/acc%/miss-per-tok/diskMB-per-tok. Retries 503/429. Results `/home/arnaud/tmp/agentic-results*.txt`. Short-countdown probes read 32-38 (acceptance-inflated, don't use as headline).
- A/B scripts: `env-ab.sh` (env variant from `.bak` + restart + countdown probes, logs `TAG tps:v v v`), `arg-ab.sh` (yaml arg), `cache-ab.sh` (env on live service).
- LLAMA_LOG_INFO from libllama is invisible unless the server runs `-lv 4` (threshold 3 = WARN only). Without it `LLAMA_UBATCH_TIMING` output never reaches the journal and looks like a broken gate. The live yaml carries `-lv 4`.
- `LLAMA_UBATCH_TIMING=2` adds a windowed per-ubatch aggregation for small ubatches (`small ubatch timing` lines, every 100, reset each print): per-ctx, per-token-bucket ms with total/apply/build/alloc/inputs/kv/launch split. Draft ctx = 1-token (pass1) + 4-token (pass0) buckets; target verify = 3-token bucket. `LLAMA_MTP_TIMING=1` (speculative.cpp) logs per-round draft phase split every 200 rounds, windowed.
- Draft stall ROOT CAUSE (Nov 22, GGML_SCHED_TRACE_INPUTS=600 + GGML_SCHED_TIME=1, fork-built-in diagnostics): the draft graph is split across BOTH GPUs (+ a CPU split); every draft pass copies QSA pool data cross-device, staged GPU->host->GPU: `idx_cell_pos (view)` 960 KB (ROCm1->ROCm0, typically 345 us, spikes to 11.4 ms waiting behind the verify queue) + `qsa_pool_layout` 240 KB x2 + 961 KB (ROCm0->ROCm1, ~570 us). That is the whole `launch` 5-8 ms/pass seen in ut/draft timing; graph build/alloc/inputs/kv and hosted-enter plan waits (plan_gpu_wait 4.7 us/layer) are noise. Fix candidate: pin the draft to one device with `-devd` (see next bullet).
- `-devd NAME` (`--spec-draft-device`, needs the NAME from `--list-devices`, index `0` = "invalid device: 0" bootloop) pins the draft ctx to one GPU. It CRASHES with `-devd ROCm0`: "MUL_MAT_ID with host weights outside a hosted MoE block (ffn_moe_gate-48)" at ggml-cuda.cu:7878 - moe-cache registers a layer's expert pool per-device by LAYER RANGE, and the guest MTP layer 48 lands in device 1's pools only; on device 0 `moe_hosted_register` returns nullptr (pool_of -1) and graph capture aborts. `-devd ROCm1` matches the registration. Debug: `GGML_MOE_HOSTED_DEBUG=1` logs find_hosted_moe decisions + capture declines (phases6+ binaries; debug patch is env-gated, harmless).
- FIX DEPLOYED (Nov 22, `-devd ROCm1` in yaml, image phases8 after rebasing on freriadoc/main): draft graph single-backend (ROCm1+CPU), sched inputs 14 us (was 5.8 ms), draft 6.4-6.9 ms/step (was ~22). Countdown probes 45.8-53.2 t/s (was 36.7-41.8), agentic probe turns 0-4 33.9-39.8 dec_t/s avg ~37 (baseline 27.7), acceptance unchanged (countdown 0.99, agent 0.62 per-pos 0.71/0.53). Draft cost is now the lm_head sample (mt `sample` 3.6-5.6 ms/round at short ctx), not the copies.
- Env diagnostics that exist in the fork (INFO level, need -lv 4): `GGML_SCHED_TIME=1` (+ `GGML_SCHED_TIME_EVERY`, default 250) per-scheduler per-backend splits/inputs/compute us per graph, windowed; `GGML_SCHED_TRACE_INPUTS=N` logs every input copy of graphs N..N+29 with tensor name/bytes/us; `GGML_CUDA_MOE_CACHE_STATS=N`. A/B results Nov 22: `GGML_MOE_SHARE_DRAFT=0` draft launch 13.7 ms vs 6.5 baseline (worse); `GGML_CUDA_MOE_CACHE_DEFER=0` probes 33-35 t/s vs 36.7-41.8 (worse; deferred admissions deliberately fill copies during the draft). Both reverted.
- Per-step spec budget breakdown (no code change): env `LLAMA_SPEC_TIMING=1` logs every 200 decode steps: `spec timing: N steps, X ms/step: draft a (+yield), decode b (+), process c (+), accept d (sample, spec accept), ckpt, seq_rm, output, rest`. In journal via `grep 'spec timing'`.
- Host stack sampling (container binary unstripped): `podman cp avalon-avalon:/app/<lib> /tmp/imglib/` for libllama/libllama-common/libllama-server-impl/libggml-{base,hip,cpu}, then host `gdb -p <host_pid> -batch -ex 'set sysroot /proc/<pid>/root' -ex 'set solib-search-path /tmp/imglib' -ex 'thread apply all bt 12'` in a loop during decode. Host pid = `pgrep -f llama-server` (changes on every restart).
- Bandwidth bench: hipcc float4 copy kernel (`/home/arnaud/tmp/bwbench.hip`), 96 MiB x2 (512 MiB faults: the server owns VRAM, ~400 MiB free/card), run inside `localhost/avalon:prof` via `podman run --rm --entrypoint bash --device /dev/kfd --device /dev/dri -v /tmp:/tmp ...`. Healthy = 405-424 GB/s.
- Bash traps: Linux MAX_ARG_STRLEN - never pipe big curl output inline, write to files. `ask`/`wait` polling wastes turns; background long probes with `timeout: 0` + tee.

## Kernel trace procedure (rocprofv3)

1. `systemctl --user stop avalon.service`, `podman rm -f avalon`
2. `podman play kube /home/arnaud/tmp/avalon-traced.yaml.keep` (rocprofv3-wrapped server; flushes CSV only on SIGINT; output `/home/arnaud/avalon-prof-out/workstation/1_kernel_trace.csv`, ~400 MB)
3. wait health (~5 min under trace), warmup x2 countdown/250
4. print clock anchors before/after the decode probe (countdown n_predict=500): `python3 -c "import time; print(int(time.clock_gettime(time.CLOCK_MONOTONIC)*1e0), time.time_ns())"` (trace timestamps = CLOCK_MONOTONIC ns)
5. snapshot `/metrics` before/after for step counts
6. `podman kill --signal SIGINT avalon-avalon`; the pod exits after flush (up to ~5 min; CSV appears first)
7. `podman rm -f avalon-avalon`; `systemctl --user restart avalon.service`; wait health
- Parse: `parse-kernel-trace.py <csv> --top N --since <rel_ns>`, or stream the CSV filtering window `[mono_start_ns, mono_end_ns]`. Columns: `Kind, Agent_Id ("Agent 1"/"Agent 2"), Queue_Id, Stream_Id, ..., Kernel_Name, Start_Timestamp, End_Timestamp, VGPR_Count, Workgroup_Size_X, Grid_Size_X`.
- Live-equivalent step wall = probe tokens / live t/s / verify steps. Trace inflates wall ~1.5x; kernel durations are trace-independent.

## Established facts (verified Nov 2026 - do not re-verify)

- The "80-99 t/s era" was a measurement artifact (prompt-eval noise + one transient 8-min fast window Nov 19 23:24-23:32 with `Runlist oversubscribed` kernel warnings). No regression exists; reboot/knobs cannot restore it.
- Steady state: short probes 32-38 t/s; agentic 24-32 t/s; 80k ctx ~19 t/s (acceptance 0.41 there); prefill 480-570 t/s at 8k-20k ctx; 328 t/s at 80k ctx.
- Decode is NOT bound by disk, RAM bandwidth, PCIe, thermals, zombie GPU contexts, or VRAM bandwidth. Staging copies = ~1% of decode busy time. The 64 GB RAM upgrade helps prefill/TTFT/cold start only.
- MTP n=2 is the right draft depth (n=1 28.6, n=2 30.8, n=3 28.5 in the n-max sweep). Acceptance: countdown 0.92 (2.86 tok/step), agentic 0.59-0.68 (2.0-2.4 tok/step), 80k ctx 0.41.
- Decode step anatomy (CORRECTED Nov 21, phases2 + live phase metrics): step wall 82.7 ms (agentic, 2.29 tok/step, acc 0.56). Per-device chain busy ~17.5 ms (enter live ~0.22 ms x24 after removing trace spin, dense mmvq 5.3, grouped 3, merge 1.3, hc 2, copies 0.3) = ~35 ms both devices; draft+boundary ~45 ms/step is the dominant cost. Live plan relay is FAST (plan_gpu_wait 4.5-6 us/layer, plan_host 1.2 us); `late` (CPU rows after GPU merge-ready) 55-64% of layers but only ~42-57 us/layer; moved_copy ~1 us/layer. The "enter spins 1 ms" and "gather BW 26 GB/s" stories were trace artifacts / bad byte math (trace inflates plan_gpu_wait to 845 us/layer via rocprofv3 mailbox contention).
- Contiguous 48-layer split = strictly serial per-step chain across the 2 devices: single-device "idle" is mostly the other device's turn, not removable overhead. Both-idle blocks (>2 ms, avg 8 ms, 30% of traced wall) = per-step boundary + draft serialization.
- Host is NOT the per-step bottleneck for WAITING: gdb stack sampling during decode shows hosted threads in pread64 (disk fills), HIP threads in ioctl, rest idle. BUT trace step forensics (marker = verify lm_head grid 7946240 on Agent 2): between verify lm_head end and the next step, dev2 runs draft pass 1 (~2 ms GPU: GEMVs ~80 us, hosted enter 494 us trace-inflated, lm_head 432 us), then BOTH devices idle ~18-20 ms (two host stalls ~10 ms + ~8 ms, between draft pass 1 sample and pass 2 launches), then draft pass 2 (~2 ms), then next verify. Draft GPU work is only ~4 ms/step; the ~20 ms is host CPU between draft passes (llama_process setup / graph build / sampler / sampling on 65536 logits). Main thread stacks show deep CPU call chains (11+ frames, unresolved ?? - libc internals). LLAMA_SPEC_TIMING=1 confirms: draft 25.7 ms/step, decode 64-74 ms/step, host glue (accept/ckpt/seq_rm/output/rest) < 0.7 ms.
- Draft: `nextn_shared_target_tensors=True` (draft = 1 layer reusing target weights incl. hosted MoE pool, no own experts) + own 65536-row IQ4NL lm_head (FR65k, ~94 MB/pass) + fusion proj; draft file_type q8_0. Draft pass ~= one target layer + lm_head read.
- Naive 256-thread GEMV on the lm_head shape (152064x2048 q8, 3 cols) hits 424 GB/s on Navi21 (`/home/arnaud/tmp/bench-gemv.hip`, needs ~400 MB free VRAM, run in :prof container) -> hardware BW is never the decode limit; in-server dense mmvq efficiency still unmapped (trace grid != nrows; per-launch bytes unknown).
- Model config (GGUF kv): n_embd 2560, 48 layers, 512 experts x top-10, expert/shared ffn 640, GDN ssm (inner 6144, interval 4 -> full attn 12/48), hyper_connection x4 low_rank 320, QSA indexer top_k 2048, PLE ngram per-layer emb 160, vocab 248320, output.weight dense q8_0.

## Hosted MoE pipeline map (ggml/src/ggml-cuda/moe-cache.cu, 12.4k lines)

Per MoE layer per pass, per device:
- `k_moe_hosted_enter` (:8375, launch :12157): copy activations to pinned host + q8_0-quantize; `moe_router_row` computes next-layer router logits in 512 extra workgroups (deferred routers, `ggml_cuda_router_defer_ok`); block 0 does 512-expert top-k (`topk_moe_warp`), writes ids/weights to pinned `m_ids`/`m_w`; `__threadfence_system` then publishes pinned mailbox `mail->seq` (:8500); thread 0 SPINS on `*flag` (`moe_hosted_pause`) until the host answers, then reads `m_plan -> d_plan` back over PCIe and builds the group list (`moe_group_list`).
- The launch is not graph-captured while pending: `server_pending++` + `server_cv.notify_one` (:12165-12172). Per-device `moe_hosted_server` (:10770) busy-polls the mailbox (no driver calls in the poll loop - cuBLASLt module load deadlock lesson in the comment), calls `moe_hosted_serve_one` (planning: expert -> pool slot / arena / disk fetch, CPU rows assignment, `m_plan` write, `flag` bump).
- Expert rows already in the VRAM pool are "resident" (no fetch). Misses: arena(RAM)->VRAM slot copies by KERNEL copies `k_ggml_cuda_host_copy` - all host<->device traffic on HIP uses kernel copies, never SDMA: one SDMA ring per engine, an output copy blocks the ring and queued fills hang (documented hang at 16k ctx, :12225 comment). Do not reintroduce SDMA/stream copies.
- Async-arrival precedent: `d_expect` + `m_flags` + `k_moe_split_wait` + `k_moe_split_fetch` (:8525, :12235) - the SPLIT path already waits per-expert and patches `d_plan2` after arrival. Model new overlap on this.
- Timeline ring `tl` (MOE_TL_RING, `moe_tl_clock`): slots per seq: +0 publish clock, +1 flag-arrival clock, +2/+3 split-fetch wait. Enabled by `GGML_CUDA_MOE_TIMELINE` but the init is dead and reads `atoi(getenv())` with NULL UB (~:10039) - fix before relying on it.
- CPU expert rows: activations already shipped to host in enter (`h_act`, `h_act8` q8_1); CPU-computed rows land via a "down id -1" plan row; `k_moe_rows_in` copies results back; `GGML_CUDA_MOE_SPLIT_CPU_SCALE` exists.
- Knobs (env): `GGML_CUDA_MOE_HOSTED_THREADS` (12), `GGML_CUDA_MOE_ARENA_READERS` (24), `GGML_CUDA_MOE_RING_PREFILL` (900), `GGML_CUDA_MOE_RING_AHEAD`, `GGML_CUDA_MOE_SHARE_DRAFT`, `GGML_CUDA_MOE_Q8_FUSED`, `GGML_CUDA_MOE_ROUTER_TAG`, `GGML_CUDA_MOE_SPLIT_*`, `GGML_CUDA_MOE_PIN_RESERVE_MB`, `GGML_CUDA_MOE_ARENA_MB`. `moe-cache-mmv-tuning.h` holds mmv row-shape tuning. `moe-ring.h`, `moe-arena.h`, `moe-ram-budget.h` are the supporting headers.

## Known bugs

- llama-server segfault in `libggml-base.so.0.28.0` (`error 4`, wild address) at 03:13:35 Nov 21: an 80k-token prefill finishing while a concurrent request (~30k ctx) decoded. Service auto-restarted. Reproduction lead: 80k prefill + concurrent request. Report upstream (user does commits).
- `GGML_CUDA_MOE_TIMELINE` dead init + `atoi(getenv())` NULL UB ~:10039.
- Journal `prompt eval time` on tiny prompts and the Nov 19 23:24-23:32 fast window are measurement artifacts (see facts above).

## Decode optimization plan (grounded, prioritized)

Budget (corrected Nov 21): 82.7 ms/step = ~35 ms both-device chain busy + ~45 ms draft+boundary serialization. NOT BW-bound (logical ~4-6 GB/step vs 810 GB/s aggregate -> floor ~8 ms). Live relay is fast; P1-as-designed (hide enter spin) and P2 (prefetch fills) are DEAD (trace artifacts; moved_copy ~1 us/layer).

1. DONE P0 metrics: hosted phase counters live in `/metrics` (`avalon:hosted_phase_seconds_total{device,phase}`, `avalon:hosted_layers_total`), prefill/decode shares from `llamacpp:*_seconds_total`. Per-step spec phases via `LLAMA_SPEC_TIMING=1`.
2. P1' (NEW primary): cut the ~45 ms/step draft+boundary. Get the exact split from `LLAMA_SPEC_TIMING=1` (draft vs decode vs accept/ckpt/seq_rm/output/rest + yields). Then attack the largest component (candidates: draft lm_head 94 MB x2/pass, per-step full device syncs before draft rows, graph replay launch gaps, yield_to_queue handoffs, checkpoint copies).
3. P3 dense GEMV: in-server mmvq efficiency UNMAPPED (trace grids != nrows, bytes unknown). Naive 256-thread GEMV = full 424 GB/s, so headroom exists if kernels are shape-limited; RDNA2 `calc_nwarps` falls to 1 warp (wg=32) vs GENERIC 4 / RDNA3 4 for q8_0 - cheap container-build test: bump RDNA2 q8_0 ncols<=4 to 4-8 warps, A/B via agentic probe. Est upside ~5-10 ms/step if kernels are the issue.
4. Acceptance: draft IQ4NL -> q8_0 via llama-quantize (dequant path); `GGML_CUDA_MOE_SHARE_DRAFT=0` A/B. ~5%.
5. Cheap env sweeps: HOSTED_THREADS 12 vs 24, ARENA_READERS via `env-ab.sh` (late is 55-64% of layers but small absolute).

Verification loop for every change: `env-ab.sh`/`arg-ab.sh` + `agentic-probe.py 8` (dec_t/s, acc; baseline 27.7 t/s acc 0.56 results6) + `LLAMA_SPEC_TIMING=1` step breakdown + phase metrics diff. Keep all staging as kernel copies on compute streams; test the 80k-ctx agent path (segfault + SDMA hang live there).

## Repo etiquette

AGENTS.md: no AI commits/pushes/PRs (user commits; `Assisted-by:` only), ASCII-only concise comments (ASD-STE100 style), no fixed-column wrapping, reuse existing mechanisms, blend with surrounding code. Read `skills/code-review/SKILL.md` before proposing changes.

Image build: the host `build/` tree is a HIP/Release/shared build (ROCm 7.2.1, gfx1030) useful for fast iteration, but its binaries link the host glibc/OpenSSL (2.43/3.4) and WILL NOT RUN in the container (Ubuntu 24.04 = glibc 2.39, OpenSSL 3.0). Ship binaries must be built inside the image family. Recipe (works, ~25 min cold on 16 threads):

```sh
# one-time builder image (rebased + cmake + ssl headers):
printf 'FROM localhost/avalon:rebased\nRUN apt-get update -qq && apt-get install -y -qq cmake libssl-dev\n' > /home/arnaud/tmp/Containerfile.builder
podman build -t localhost/avalon:builder -f /home/arnaud/tmp/Containerfile.builder .
# preseed the UI asset cache (container cannot reach the HF bucket; reuse the host build's extracted dist):
mkdir -p /home/arnaud/tmp/build-img/tools/ui/dist
cp build/tools/ui/dist.tar.gz build/tools/ui/dist.tar.gz.sha256 /home/arnaud/tmp/build-img/tools/ui/
cp -r build/tools/ui/dist/. /home/arnaud/tmp/build-img/tools/ui/dist/
# compile (flags match the running image: curl OFF, OpenSSL 3.0, NATIVE OFF (14 cpu variants), NCCL OFF):
podman run --rm --entrypoint bash -v $PWD:/src:ro -v /home/arnaud/tmp/build-img:/out localhost/avalon:builder -c \
 'HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -S /src -B /out -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=ON -DGGML_NATIVE=OFF -DGGML_CUDA_NCCL=OFF \
  && cmake --build /out -j 16 -- llama-server'
# ship: thin COPY layer, then validate in-container BEFORE touching the service (ldd + --version):
printf 'FROM localhost/avalon:rebased\nCOPY bin/ /app/\n' > /home/arnaud/tmp/Containerfile.phases2
podman build -t localhost/avalon:phases2 -f /home/arnaud/tmp/Containerfile.phases2 /home/arnaud/tmp/build-img
podman run --rm --entrypoint bash localhost/avalon:phases2 -c 'ldd /app/llama-server | grep "not found"; /app/llama-server --version'
```
Deploy = edit `~/.config/pods/avalon.yaml` image tag, `systemctl --user daemon-reload && systemctl --user restart avalon.service`. A 503 "Loading model" answer means the binary is alive; a loader error appears as container die + restart loop in `journalctl --user -u avalon.service`. Validate `ldd`/`--version` in-container first - a bad binary takes the service down with it (happened Nov 21, user reverted).
