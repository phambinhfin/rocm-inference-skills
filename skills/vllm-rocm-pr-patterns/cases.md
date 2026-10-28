# Case index

The 30 merged `rocm`-labelled PRs behind [SKILL.md](SKILL.md), grouped by what each one teaches.
Open one with `gh pr view <N> --repo vllm-project/vllm` when you need the full write-up; these
bodies are the best available examples of what ROCm evidence looks like when it merges.

## Kernel and numerics bugs

| PR | Lesson |
|---|---|
| [#63268](https://github.com/vllm-project/vllm/pull/63268) | gfx942 Triton miscompile: with hundreds of SGPR spills the `output_stride_1` argument is never read and the head index multiplies an unrelated spilled SGPR, leaving 263 of 2048 output values written. 32-bit per-head offsets avoid it. The test had been passing on leftover memory — filling the output with NaN made it fail 3/3. Reviewer asked whether `triton_unified_attention.py` was affected too; the answer was a 60-combination sweep, not a guess. |
| [#63735](https://github.com/vllm-project/vllm/pull/63735) | RDNA: Triton uses AMD buffer ops for tensors ≤ 2 GiB, which NaNs and memory-faults on gfx1201. `num_stages=1` plus warps-by-tile-size fixes it and speeds decode 1.1-1.9×. Review: use `on_rdna()` / `get_rdna_version()`, not `on_gfx1x()`. Honest limit stated: "CI has no RDNA runner." |
| [#63689](https://github.com/vllm-project/vllm/pull/63689) | MXFP8 E8M0 scale code taken from exponent bits returns 1 for subnormal `q`, where vLLM's rule gives 0 — so every all-zero group is mis-scaled. Validated by sweeping *every* non-negative finite f32 (2,139,095,040 values) across all four copies: 73,400,322 wrong before, 0 after. |
| [#63576](https://github.com/vllm-project/vllm/pull/63576) | A combine stage read `klen` with a plain load and no wait while an index source rebuilt the key table in the same launch. Body tabulates all three outcomes of the stale read — equal (right by luck), smaller (wrong output), larger (unbounded spin, killed at 420 s). Cost measured against an unchanged control arm. |
| [#64158](https://github.com/vllm-project/vllm/pull/64158) | The model loader calls only the *root* model's `process_weights_after_loading`, so a hook defined on the inner text model never ran and a folded weight stayed `None` — every serve died at the first CUDA graph capture. Five-line fix, full reproduction command, GSM8K to show nothing else moved. |

## AITER

| PR | Lesson |
|---|---|
| [#63409](https://github.com/vllm-project/vllm/pull/63409) | Code assumed AITER's fused MLA q-prep kernel clamps positions into a one-row cos/sin cache. On v0.1.26 it doesn't on the non-DCP path: q_pe silently becomes all zeros for every position ≥ 1. Passing a preallocated zeros buffer removes the dependency on AITER behaviour entirely. Agentic eval 0.740 → 0.988. `ROCm/aiter#6879` filed for the kernel side. |
| [#63558](https://github.com/vllm-project/vllm/pull/63558) | Removing a layout guard in an earlier PR let flat-qkvz models take the AITER GDN decode fast path under `VLLM_SSM_CONV_STATE_LAYOUT=DS`, hitting a stale `conv_state.stride(-2) == 1` assertion. Falls back to the generic path for DS, as the KDA path already does. |
| [#63559](https://github.com/vllm-project/vllm/pull/63559) | How to merge a temporary `skip`: the bug is in AITER (`fused_moe_2stages` leaves `topk_ids == -1` rows unwritten, so the per-tensor a2 quant scales over stale memory), only fp8 per-tensor configs are affected, the AITER fix is linked, and the body says which pin should delete the guard. Reproduced by poisoning the caching allocator with 1e30. |
| [#63270](https://github.com/vllm-project/vllm/pull/63270) | Backport framing: cherry-pick unchanged, author kept, "Not a duplicate: #61889 targets `main`; this carries the same commit to the M3.1 branch." Also the trap that the released image shipped AITER 0.1.26 while the branch's `Dockerfile.rocm_base` pinned 0.1.25. |

## Performance

| PR | Lesson |
|---|---|
| [#63342](https://github.com/vllm-project/vllm/pull/63342) | `block_m=8` → `128` for a fused KDA conv1d, bit-identical output, and the metadata builder prebuilds the schedule so AITER does not fall back to its own builder and sync. Test asserts no D2H sync under `set_sync_debug_mode("error")`. Review: a `~2.2x` comment belongs in the commit message, not the code. |
| [#63637](https://github.com/vllm-project/vllm/pull/63637) | Fusing activation quant into a norm epilogue: 1.41-2.06× on the pair, 1.07-1.19× with the GEMM, and *bitwise equal* to the unfused path, so no new env var is needed. Enumerates every configuration that does not fuse (non-matching kernel, `VLLM_BATCH_INVARIANT`, LoRA, mono decode, layer 0). Review asked for a `rocm` platform guard in the shared quantization file. |
| [#63669](https://github.com/vllm-project/vllm/pull/63669) | The template for flipping a ROCm default: rewrite only `"auto"`, bail on DCP/PCP and non-CDNA4, fall back silently, `logger.info_once` the opt-out. Evidence covers throughput, TTFT, TPOT, KV capacity, GSM8K *and* GPQA, with McNemar on the disagreements. |
| [#63833](https://github.com/vllm-project/vllm/pull/63833) | HIP multiplexes streams onto `GPU_MAX_HW_QUEUES` (default 4) queues, so `hipStreamSynchronize` on one slot waits for another slot's later work — CUDA does not. Per-slot `torch.Event` restores the overlap (up to 7.9% on MI300X). Reproduced deterministically with `GPU_MAX_HW_QUEUES=1`. Reports the H100 non-regression, the +0.4-1.6% cost without queue sharing, and the higher reserved memory, all as part of the claim. |
| [#63260](https://github.com/vllm-project/vllm/pull/63260) | Generalizing a ROCm-only default: a fixed 15 s EngineCore cleanup grace on ROCm becomes an explicitly-set `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` on every platform, with defaults unchanged. "Adds no new knob" is the argument that carried it. |

## AMD CI

| PR | Lesson |
|---|---|
| [#63850](https://github.com/vllm-project/vllm/pull/63850) | A shared CI host still holds the previous process's VRAM when the next server runs vLLM's free-memory check. Reuses `wait_for_memory_to_settle()` from `tests/utils.py` — "no new helper code, no changes to shared test infrastructure". |
| [#63828](https://github.com/vllm-project/vllm/pull/63828) | Pod lost 3 of 5 runs to exit -7: ~208 GiB of checkpoints read against ~133 GiB of pod memory, with page cache charged to the cgroup. An autouse fixture frees the previous model's page cache on model change. Peak container memory 102.4 → 28.5 GiB for +36 s. |
| [#63453](https://github.com/vllm-project/vllm/pull/63453), [#63557](https://github.com/vllm-project/vllm/pull/63557) | `get_open_port()` hands back a port inside the ephemeral range, and NIXL/MoRIIO bind it only after the model loads — any other `bind(0)` can take it in between. Take the blocks from below `/proc/sys/net/ipv4/ip_local_port_range`. #63453 quantified the race: 26 collisions in 5000 trials, and 200,000 `bind(0)` calls never returned a port below 32768. |
| [#63760](https://github.com/vllm-project/vllm/pull/63760), [#63269](https://github.com/vllm-project/vllm/pull/63269) | `automatic_shard: true` runs as one unsharded job when the generator can't fetch main's timings, so the mirror's `timeout_in_minutes` has to cover the full run (45 → 90 min). #63269: a step was killed mid-suite with every test that ran passing — 20 → 45 min, matching the equivalent step in `test-amd.yaml`. |
| [#63283](https://github.com/vllm-project/vllm/pull/63283) | Un-gating a CUDA-only test for ROCm: `skip_if_not_cuda` → `skip_if_not_cuda_alike`, `vllm.platforms.rocm` imported lazily because importing it resolves the GPU arch at load time, and proof the test isn't vacuous — reverting the fix it guards makes it fail on MI300X. Review: prefer a CDNA-version check over "MI3xx". |
| [#63452](https://github.com/vllm-project/vllm/pull/63452) | vLLM was right and the HF reference was wrong: on ROCm `varlen_attn` aligns the causal mask top-left when `max_q < max_k`, so the reference saw only the first key with a KV cache. Fixed by generating the reference with `use_cache=False`, keeping real ROCm coverage instead of deleting the test. |
| [#63907](https://github.com/vllm-project/vllm/pull/63907) | Enrolling AMD mirrors in `automatic_shard`: every test still runs exactly once, verified by rendering each shard's commands against a mock test tree and comparing passed/skipped totals to main. |

## Recipes and docs

`recipes/models/**.yaml` is rendered by `vllm-project/recipes`, so validation is the renderer plus
a rendered-command diff, never vLLM's CI alone.

| PR | Lesson |
|---|---|
| [#63332](https://github.com/vllm-project/vllm/pull/63332) | Adding an MI355X Quark MXFP4 variant: `meta.default_variant`, `tp`, `hardware_opt_in_features`, a `notes[]` entry explaining why MTP is opt-in, and a troubleshooting entry for the slow TP2 cold start. 234 of 235 renders byte-identical. A GPU-pair-specific startup failure was bisected and attributed to the hardware, not the recipe. Review: `date_added` was missing. |
| [#63344](https://github.com/vllm-project/vllm/pull/63344), [#63520](https://github.com/vllm-project/vllm/pull/63520), [#63657](https://github.com/vllm-project/vllm/pull/63657), [#63665](https://github.com/vllm-project/vllm/pull/63665), [#63726](https://github.com/vllm-project/vllm/pull/63726) | Scheduling-profile updates aligned to an InferenceX PR. All of them: pin the audited source SHA, diff every generated JSON and state how many stayed identical, distinguish synthetic acceptance (throughput only) from real block verification, say plainly that evidence is source-compatible rather than an exact-head GPU rerun, and claim no throughput gain for an interactivity cap. #63344 got the reviewer note that a cap's *cost* belongs in the body too, with the sweep numbers. |

## Process

| PR | Lesson |
|---|---|
| [#63490](https://github.com/vllm-project/vllm/pull/63490) | Splitting a large ROCm PR: this one *removes* 8776 lines so two pieces can land on their own, and proves it with `git grep` for every removed symbol plus "the 14 engine files return to the branch base byte for byte". |
