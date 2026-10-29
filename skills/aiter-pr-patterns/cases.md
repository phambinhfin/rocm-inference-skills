# Case index

The 30 merged PRs behind [SKILL.md](SKILL.md), grouped by what each one teaches. Open one with
`gh pr view <N> --repo ROCm/aiter` for the full write-up.

## CI runs nothing

| PR | Lesson |
|---|---|
| [#6949](https://github.com/ROCm/aiter/pull/6949) | The clearest case in the sample. `aiter_test.sh` runs `op_tests/test_*.py` as `python3 <file>`, so two pytest-style mha_v4 files had **376 cases that never executed in CI** — including the regression test for a K-mean out-of-bounds read that had caused two reported crashes. The fix derives the script-mode case list from the files' own `parametrize`/`skipif` marks rather than restating it, so script mode and pytest provably cover the same set. Proof it works: reverting the bound fix makes the script entry point exit 1 naming 12 failed cases, where it previously exited 0. |
| [#6934](https://github.com/ROCm/aiter/pull/6934) | Copilot caught the same class on review: "this module has no script entry point, CI reports success without executing any of the 17 cases." Also caught the test skipping gfx942 although the same ASM kernel ships for it. Both fixed in one follow-up commit; the author noted the gfx942 CI job would be the first run on that arch. |
| [#6992](https://github.com/ROCm/aiter/pull/6992) | The multi-GPU job never installed mori, so mori-backed tests either skipped themselves or died at import — and one was moved to the skip list rather than fixed. Adds a dedicated smoke test that checks each mori piece aiter actually uses, with the same calls as its aiter caller, run as a separate step *before* the 2-hour suite so a broken install fails in minutes under its own name. Negative controls included: with mori 1.2.4, four of five cases fail; with no mori, all five. |

## Architecture and JIT builds

| PR | Lesson |
|---|---|
| [#6991](https://github.com/ROCm/aiter/pull/6991) → [#7014](https://github.com/ROCm/aiter/pull/7014) | The revert/reland cycle in full. `module_topk_plain` never built for gfx1250 (wave64 static_asserts) — latent until #5686 routed a shared dispatch path through it for every fp32 call on any GPU, so the first `topk_select` call on gfx1250 JIT-built it and failed. Relanded as two commits: #5686 unchanged, then per-arch gating via `_PLAIN_ARCHES`. The regression test routes *as* gfx1250 with every `module_topk_plain` binding replaced by one that raises, so it needs no gfx1250 GPU and runs in existing CI: 450 routing calls reached the module before, none after. |
| [#7026](https://github.com/ROCm/aiter/pull/7026) | Expose a backend's arch-capability query next to the backend (`pa_decode_gluon_supports_arch()`, `pa_fwd_asm_supports_arch()`) so downstream callers route around unavailable backends without duplicating aiter's supported-arch knowledge. Came from a review request on the ATOM side asking for the check to live in aiter. |
| [#7066](https://github.com/ROCm/aiter/pull/7066) | MI300X PR test shards were turned off, keeping MI35X on PRs and MI300X on main. Practical consequence: **gfx942 coverage now arrives after merge**. Validated with `actionlint` and a note listing the four workflows deliberately left alone. |
| [#6988](https://github.com/ROCm/aiter/pull/6988) | 692 files. Selects A0/B0 code objects from the live HIP device's ASIC revision rather than a node-wide stepping, validates actual ELF processor identity with no cross-profile fallback, and rolls back failed registrations. Merged *with* a section headed "Known A0 validation limitation - not a full correctness certification": the all-variant rerun stalled on one kid and the PR explicitly claims neither full certification nor a speedup. Also records which HIP version loads the strict ELF and which rejects it. |

## Tuned configs

| PR | Lesson |
|---|---|
| [#6939](https://github.com/ROCm/aiter/pull/6939) | Two dispatch traps in one review. The config lookup **ignores output dtype**, so 51 new ASM rows would route supported fp16 calls to a bf16-only ASM kernel. And the shape deliberately left out of the table is not actually preserved, because `get_CKGEMM_config` retries through `get_padded_m`, rounding M=24 to 32 and landing on a new row. |
| [#6977](https://github.com/ROCm/aiter/pull/6977) | The **one-backend-per-PR** rule, raised by Copilot against a Triton retune that also touched a FlyDSL CSV — with the author pushing back that a tuned CSV legitimately holds multiple backends. Also: changed config buckets need correctness coverage at the M values they serve, not just the tuned ones; the author checked M=8/16/512/1024/2048/4096 on gfx950 and said so. |
| [#6942](https://github.com/ROCm/aiter/pull/6942) | A missing shape fails loudly (`no codegen'd instance for shape key ...`), so the codegen instance list and the CSV go together. Documents which M values each row serves given power-of-two key rounding, counts tuned-row hits in the server log (148 lookups, 0 default, 0 warnings), and pairs op-level deltas with GSM8K and `benchmark_serving` on two checkpoints. Also fixes the tuner itself: A16W4 stage2 always atomic-scatters, so reduce-named candidates were accumulating and being rejected as errors. |
| [#7067](https://github.com/ROCm/aiter/pull/7067) | Retuning discipline: rows that measured equal or faster keep the new entries, rows that measured slower stay as they are. Reports that the reduce stage2 is 2–6% slower but halves mean relative error *and* is bit-identical across runs where atomic varies — a correctness argument for a slower path. Ships with two Test Plan boxes unticked. |

## Kernels and numerics

| PR | Lesson |
|---|---|
| [#6954](https://github.com/ROCm/aiter/pull/6954) | Rank-rotated reductions sum FP32 partials in an order that depends on the element's offset, so the same row reduced at a different offset lands on a neighbouring BF16 value — identical partials, one ULP apart. Fixed by reading `ptrs[i] = input[i]` everywhere. The model section is "Remaining communication paths with the same issue (not fixed here)": six more sites with file and line, each with why and a proposed fix, plus a list of paths checked and found already correct. |
| [#7004](https://github.com/ROCm/aiter/pull/7004) | Routers started routing pad rows to expert -1; two gfx1250 MoE kernels took every row as live and faulted ("Page not present") on any batch of 64+ tokens containing a pad row. Root cause names both sites and why the other three paths were already safe. The test gives the counter a leading guard element — 384 before the fix, 0 after — and compares pad-row runs against the kernel's own all-real run, because the file's torch reference is already off by ~0.108 on main. |
| [#7036](https://github.com/ROCm/aiter/pull/7036) | A new tile shape for a head count the existing kernel wastes: at H ≤ 16 the 64-head tile leaves three quarters of its MFMA rows empty. Reports the design (occupancy 2, 218 VGPRs, ~70 KB LDS, no spills), why it can be JIT-built rather than shipped as a code object, and how the split-K constants were fit — including that reusing the old constants would have been 4.1% off instead of 2.5%. Correctness over 180 configs at err 0. |
| [#6943](https://github.com/ROCm/aiter/pull/6943) | The most thorough validation in the sample, and worth reading for that alone: per-optimization measurement, ISA inspection (VGPRs, LDS, zero scratch), 115,200 trait combinations checked to confirm only the intended ones changed selection, a median excluded because an external profiler job overlapped it, and an undiagnosed page128 memory fault reported rather than omitted. |
| [#6966](https://github.com/ROCm/aiter/pull/6966) | A one-file optimisation that skips a redundant counter reset, defaulting to the old behaviour. Review asked for the safety contract in the docstring: the option bypasses initialisation of an accumulating counter, so the caller's obligation has to be written down next to it. |

## Host overhead

| PR | Lesson |
|---|---|
| [#6946](https://github.com/ROCm/aiter/pull/6946) | Caching compiled launchers, 73–86% host-time reduction. Two things to copy: a **global dispatch audit** (all 887 Python files under `aiter/` scanned, six further direct-dispatch paths found and wrapped) and a retraction — earlier figures were wrong because an imported test module had installed `torch.set_default_device("cuda")`, which intercepts even tensor metadata calls, so they measured the Python API rather than the launch. Also states plainly that gfx1250 correctness and multi-GPU execution remain unverified because only gfx950 was available. |
| [#7006](https://github.com/ROCm/aiter/pull/7006) | Split out of a closed PR because review asked to separate the backends — the Triton half was dropped. Host dispatch for six linears with a new M each call: 2962 → 157 µs, outputs bitwise identical. Shows the memoisation key being derived from what the heuristic actually depends on, so the cached result is identical rather than approximate. |
| [#7077](https://github.com/ROCm/aiter/pull/7077) | Cleanup that replaces local copies with helpers aiter already provides, and reduces two partition bounds to the one that is tight. Explains why a bound that can now round *past* the plan cap is still safe. Says which arches were not run and which of the changes apply to them anyway. |

## CI plumbing

| PR | Lesson |
|---|---|
| [#7013](https://github.com/ROCm/aiter/pull/7013), [#7023](https://github.com/ROCm/aiter/pull/7023) | Concurrency groups: cancel stale runs for the same ref, but never cancel a main push run with a newer main push run. #7013 cites the two run IDs it cancelled as evidence. |
| [#6997](https://github.com/ROCm/aiter/pull/6997) | A workflow broke because a model disappeared from the *downstream* repo's catalog. Validation replays the matrix selection against that repo's main to show what it now picks. |
| [#7015](https://github.com/ROCm/aiter/pull/7015) | Pins a `:nightly` tag to a digest once so every matrix job uses the same image, and resolves a branch ref to a full SHA for the same reason. Lists exactly what was *not* run outside GitHub, and flags unrelated broken nodes for a follow-up rather than fixing them here. |
| [#7055](https://github.com/ROCm/aiter/pull/7055) | Rebasing onto a registry migration: match all 221 kernel records exactly, keep CO bytes and exported symbols unchanged, and add a CPU-only consistency regression over all profile tables. |
