---
name: aiter-pr-patterns
description: Conventions for contributing to ROCm/aiter (AI Tensor Engine for ROCm) - which of aiter's eight in-repo skills to open for a given change, automatic PR title component tags, the ROCm PR template, the op_tests CI runner that reports success without running a pytest-style test, per-architecture JIT module build failures (gfx942 / gfx950 / gfx1250), and tuned-config dispatch traps. Use when writing, reviewing or debugging an aiter change, when adding or fixing an op_test or a tuned config CSV, when an aiter GitHub Actions job fails, when a kernel builds on one gfx arch but not another, or when deciding what evidence an aiter PR needs.
---

# Merged PRs in ROCm/aiter

Distilled from the 30 most recently merged PRs in `ROCm/aiter`. aiter is AMD's kernel library
behind vLLM ROCm's `VLLM_ROCM_USE_AITER*` paths; for the vLLM side see `vllm-rocm-pr-patterns`.

Per-PR lessons with the PR numbers behind every rule below: [cases.md](cases.md).

## Open the in-repo skills first

**aiter ships its own skills in `.claude/skills/`, and they are far more detailed than anything
here.** Read the matching one before touching code. This skill does not repeat them.

| Change | Skill |
|---|---|
| Any `aiter/configs/**` CSV, a model's tuned config | `aiter-config-shape` — configs are merged per family at runtime, not read one file at a time, so duplicate shapes collide |
| Any `op_tests/test_*.py` | `aiter-op-test` — required structure; `op_tests/test_quant.py` is the canonical reference |
| `aiter/ops/flydsl/**` | `flydsl-kernel-authoring`, and `flydsl-kernel-code-cleanup` for migrations and dedup |
| `csrc/opus_*`, `csrc/include/opus/` | `opus-kernel-best-practice` (per-kernel compile time), `opus-module-build-optimization` (module JIT build wall) |
| Reviewing a PR | `review-pr` (advisory, never a merge gate) and `validate-kernel-pr` (base-to-head on a locked idle GPU; missing evidence is INCONCLUSIVE, never PASS) |

Also path-scoped and applied automatically by editors that honour them:
`.github/instructions/aiter-ops-flydsl.instructions.md` and `aiter-ops-triton.instructions.md`.
`CONTRIBUTE.md` is the human guide and is the source of truth for the tag set.

## Title: do not write the component tags

Component tags are **recomputed from the changed files on every push** by
`.github/workflows/pr-title-tags.yaml`. Writing them by hand is wasted effort — hand variants
(`[TRITON]`, `[Gluon]`, `[CK_TILE]`, `[Doc]`) get normalised, dropped components disappear, new
ones appear. The title shows at most three; the labels carry the full set.

Managed: `[Triton/Gluon]` `[ASM]` `[HIP]` `[CK]` `[OPUS]` `[FlyDSL]` `[CI]` `[JIT]` `[Build]`
`[Config]` `[Docs]`. The `no-auto-title` label opts the title out; labels sync regardless.

**Type prefixes are yours** and are never touched: `[Bugfix]` `[Feature]` `[Kernel]` `[Perf]`
`[Test]` `[Hardware]` `[Misc]`. So is anything outside the component namespace (`[MLA]`,
`[gfx1250]`). Write the type prefix and the descriptive text; let the bot do the rest.

```
[Triton/Gluon] [Perf] Optimize RMSNorm for MI300X using vec16 loads
[HIP] [Bugfix] Fix numerical instability in FP16 softmax
```

If you add or rename a component tag, update both the workflow and `CONTRIBUTE.md`'s
"PR Title Format" — they are kept in sync by hand.

## Body

The template is **org-level**, from `ROCm/.github`, not in the aiter repo:

```markdown
## Motivation          Purpose and goals.
## Technical Details   The changes, with relevant GitHub links.
## Test Plan           Testing done to verify this PR.
## Test Result         Test outcomes.
## Submission Checklist
- [ ] Look over the contributing guidelines at https://github.com/ROCm/TheRock/blob/main/GOVERNANCE.md#pull-requests.
```

`## Summary` / `## Changes` / `## Test` also merges routinely. Either way, `CONTRIBUTE.md`
requires a kernel PR to state **hardware, baseline, optimized, improvement, bandwidth
utilization and roofline position**.

Conventions that distinguish the merged bodies:

- **Say what you could not run.** `CONTRIBUTE.md` asks for it directly, and merged PRs are blunt:
  "gfx942 and gfx950 were not run on hardware", "Available hardware is gfx950", "Not run on
  gfx1250 hardware". One PR merged with a section headed "Known A0 validation limitation - not a
  full correctness certification". Unticked `- [ ]` boxes in a Test Plan are normal and read as
  "not done", not as an oversight.
- **Name both commits.** "Baseline `ada68ed34`, PR change `b59dc6043`", not "before/after".
- **Enumerate what you did not fix.** A correctness PR that fixes some call sites lists the
  remaining ones with file and line and leaves them to a follow-up, rather than implying the
  class is closed.
- **Separate what was measured from what was excluded.** Merged bodies say "MoE e2e performance
  is intentionally omitted", "planning and compilation are excluded", "this median is excluded
  because an external rocprofv3 job overlapped it".

## The op_tests CI trap

`.github/scripts/aiter_test.sh` runs each test file as **`python3 "$file"`**. Only
`tests/test_*.py` and a hardcoded `case` allowlist get `python3 -m pytest`.

**A pytest-style file under `op_tests/` with no script entry point is imported and nothing
runs. The job exits 0 and CI is green with zero cases executed.** This is not hypothetical: 376
`mha_v4` cases never ran in CI, including the regression test for an out-of-bounds read that had
already caused two reported crashes.

So for a new or converted `op_tests/` file, do one of:

1. Give it a `__main__` guard that runs the cases (what `aiter-op-test` requires anyway), or
2. Register the filename in the runner's pytest branch in `aiter_test.sh`.

Check it, don't assume: a run that reports success proves nothing here. Confirm the case count,
and confirm the suite fails when the fix is reverted.

Two more runner facts worth knowing: files are sharded by index across `SHARD_TOTAL`, so a new
file lands in one shard; and `skip_tests` inside the script silently skips files, so a test can
be present, correct and still never run.

## Architecture

| Arch | Parts | In CI |
|---|---|---|
| gfx942 | MI300X | **main only** — PR runs were moved off MI300X, so gfx942 coverage arrives after merge |
| gfx950 | MI350X / MI355X | PR and main |
| gfx1250 | MI455X | not in the standard matrix; some multi-GPU steps gate on it |

Resolve the arch with `get_gfx()` from `aiter.jit.utils.chip_info`, or `get_cdna_version()` where
the family is what matters. Expose a backend's own capability query next to the backend
(`pa_decode_gluon_supports_arch()`), so callers route around it without copying aiter's
supported-arch knowledge.

### A JIT module can be broken on one arch and fine on the others

`module_topk_plain` never built for gfx1250 — its `>= 2 * get_warp_size()` static_asserts assume
wave64. That was latent and harmless until a PR made a **shared dispatch path** call into that
module for every fp32 call on any GPU. The first `topk_select` call on gfx1250 then JIT-built it
and failed. The PR was reverted and relanded with per-arch gating.

Before routing an existing module into a shared path, check it builds on every arch that path
reaches. Gate with an explicit arch tuple (`_PLAIN_ARCHES`) rather than assuming.

### Testing an arch you do not have

- Cross-compile: `GPU_ARCHS=gfx1250`, `FLYDSL_GPU_ARCH=gfx1250`, `COMPILE_ONLY=1`, and
  `MORI_GPU_ARCHS=gfx1250` so mori does not link CCO bitcode built for the host arch.
- Force routing: `AITER_FORCE_GFX1250=1` exercises another arch's dispatch on the hardware you
  have.
- Test the routing, not the kernel: replace every binding of the module that must not be touched
  with one that raises, then assert the router never reaches it. That needs no GPU of that arch
  and runs in the existing CI.

Compile-only and host probes do not establish correctness, graph replay or multi-GPU behaviour.
Merged PRs say so explicitly instead of letting a green compile imply a pass.

## Tuned configs

`aiter-config-shape` covers duplicate shapes. These are the dispatch traps reviewers catch:

- **The config lookup ignores output dtype.** A tuned row can route an fp16 call to an ASM kernel
  that requires bf16 output, and the supported call then fails. Gate ASM dispatch on the dtype
  the kernel accepts and keep a dtype-compatible fallback.
- **Omitting a row does not preserve the fallback.** `get_CKGEMM_config` retries through
  `get_padded_m`, which rounds M up (24 → 32), and A16W4 keys round M to a power of two
  (129..255 read the 256 row). So a shape you deliberately left out can still land on a new row.
  Say which M values each added row actually serves.
- **Changed buckets need correctness coverage** at the M values they serve, against a torch
  reference — not just at the M values that were tuned.
- **One backend per PR** is the stated rule for retunes. A tuned CSV legitimately holds rows for
  several backends, so this is contested in review; if a change spans backends, say why it has to.

Measure a config change with both arms identical except the CSV, and show the lookup landing on
the tuned row (merged PRs count them: "148 lookups, 0 default, 0 warnings").

## Evidence

- `aiter.test_common` provides `@perftest` / `run_perftest` and `checkAllclose`; an op test is
  both a correctness check against a torch reference and a perf sweep ending in a markdown table.
- Report medians with the sample count and alternate the arms round by round
  ("9 paired rounds alternate baseline/candidate order on identical inputs"). Check the GPU is
  idle and say so.
- For a kernel, inspect the ISA and report **VGPR count, LDS bytes and scratch** (zero scratch is
  the claim reviewers look for), alongside latency, TFLOPS and effective bandwidth. Say what the
  bandwidth figure counts and excludes.
- **Bit-exactness is a stronger claim than tolerance** and merged PRs lead with it: "output
  bitwise identical", "the reduce output is bit-identical across runs, while atomic varies
  between runs on the same input". When a reduction's order depends on element position or on
  which rank owns a row, the same input can round differently — fix the order rather than widen
  the tolerance.
- When the file's own torch reference is already far off on main, compare against the kernel's
  own all-real run instead, and say that is what you did and why.
- `black`, `ruff` and `clang-format` are expected clean on changed files; note when a C++ file was
  already not clang-format clean before your change.

## Revert and reland

Reverting a merged PR that broke an arch is routine and carries no stigma. Reland as **two
commits**: the first reapplies the reverted change unchanged, the second fixes the breakage.
Point at the original PR for its description, review history and measurements rather than
restating them, and add the regression test that would have caught it.
