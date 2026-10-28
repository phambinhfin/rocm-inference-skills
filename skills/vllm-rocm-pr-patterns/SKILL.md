---
name: vllm-rocm-pr-patterns
description: Conventions that get a ROCm/AMD change merged into vllm-project/vllm - PR title and body shape, platform-gating helpers (on_cdna / get_cdna_version / on_rdna, rocm_aiter_ops), AMD Buildkite mirror wiring in .buildkite/test_areas, and the recurring ROCm failure classes (HIP hardware queues, Triton gfx942 miscompiles, AITER version drift, shared-host VRAM, ephemeral port races). Use when writing, reviewing or debugging a ROCm change for upstream vLLM, when adding or fixing an :amd: CI step, when gating code by AMD GPU architecture, when an AMD CI job fails, or when deciding what evidence a ROCm PR needs.
---

# Merged ROCm PRs in vLLM

Distilled from the 30 most recently merged `rocm`-labelled PRs in `vllm-project/vllm`
(#63260 to #64158). This covers what is specific to ROCm; it does not restate vLLM's general
contribution rules, which live in the repo itself.

**Read the repo's own instructions first — they are mandatory and they change:**
`AGENTS.md`, `tests/AGENTS.md`, and the `/pr-checklist` skill at
`.agents/skills/pr-checklist/SKILL.md`. Merged PRs tick "I used vLLM's `/pr-checklist` skill".

Per-PR lessons with the PR numbers behind every rule below: [cases.md](cases.md).

## Title and labels

`[ROCm]` plus the change kind, plus the model when it is model-scoped. Both orders are merged:

```
[ROCm][Bugfix] Fix Triton paged decode leaving output heads unwritten on gfx942
[Bugfix][Perf][ROCm] RDNA launch config for kernel_paged_attention_2d
[CI][ROCm] Raise MI300 Core Operation Kernels timeout to 90 min
[ROCm][Perf][Kimi-K3] Use block_m=128 for the fused KDA prefill conv1d
[Doc][ROCm] Update DeepSeek-V4.1-Flash MI355X AgentX scheduling profiles
```

Name the architecture, not just the part: `gfx942`, `gfx950`, `gfx1201` in a kernel title.

The `rocm` label is applied automatically by `.github/mergify.yml` — a title matching `(?i)ROCm`
or `(?i)AMD`, or a touched path under `csrc/rocm/`, `docker/Dockerfile.rocm*`,
`requirements/rocm*.txt`, `vllm/v1/attention/{backends,ops}/rocm*.py`, `tests/kernels/*_rocm*.py`
or `vllm/platforms/rocm.py`. A ROCm change outside those paths (a shared kernel, a
`vllm/models/*/amd/` file, a `.buildkite` mirror) only gets the label from the title, so put
`[ROCm]` in it. `ready` is a maintainer label; don't ask for it.

## Body

Fill in `.github/PULL_REQUEST_TEMPLATE.md` and keep its sections and checklist block:

```markdown
## Overview      One paragraph: what runs today, what breaks, what this changes.
## Claims        3-6 bullets, each one checkable. Include what does *not* change.
## Validation    Commands, hardware, images, before/after tables.
## Details       Root cause, tradeoffs, limits, follow-ups, what was left out.
```

Older merged PRs use `## Purpose` / `## Test Plan` / `## Test Result`; that is the previous
template. Use the four sections above for new PRs, and keep the checklist `<details>` block with
`/pr-checklist` and AI assistance ticked.

Non-negotiable in either shape:

- **AI assistance disclosure.** Literally, e.g. "AI assistance (Claude Code) was used to draft
  this change. Every line was reviewed by the submitter."
- **Non-duplication.** One sentence naming the PRs you searched and why they don't overlap:
  "No open PR changes this launch config; #45916 and #58156 add split-KV decode for RDNA and
  leave it as is."
- **Negative claims.** State the configurations that are untouched — CUDA, other architectures,
  `VLLM_BATCH_INVARIANT`, LoRA, mono decode — and say which ones you actually ran.
- **Limits.** "Index launches were not exercised on gfx942, which shares this code."
  "CI has no RDNA runner, so the new test only catches regressions on gfx12." Saying what you
  could not measure is respected; quietly implying coverage you don't have is not.
- **Upstream issue links** when the real bug is in AITER, Triton or PyTorch: `ROCm/aiter#6879`,
  `pytorch/pytorch#201930`. Say the vLLM change stays correct with and without that fix.

Keep benchmark numbers out of code comments — a reviewer will ask you to move `~2.2x` from
`kda.py` into the description.

## Platform gating

In shared (non-`amd/`) files, gate with `current_platform.is_rocm()` first. Reviewers ask for
this even on code that can only run on ROCm in practice.

Architecture helpers live in `vllm/platforms/rocm.py`. **Use the family helpers**:

| Use | Not |
|---|---|
| `on_cdna()`, `get_cdna_version()` (gfx90a=2, gfx942=3, gfx950=4, gfx1250=5) | `on_mi3xx()`, "MI300/MI355" |
| `on_rdna()`, `get_rdna_version()` (gfx11=3, gfx12=4) | `on_gfx1x()`, `on_gfx12x()` |
| `on_gfx942()` / `on_gfx950()` only for a genuinely per-chip workaround | — |

Reviewers say it directly: "we are moving away from using `on_gfx` functions", and "we prefer
checking cdna version rather than mixxx". `get_cdna_version() == 4` is how merged code spells
"CDNA4 only".

AITER features gate through `rocm_aiter_ops`, not raw env reads:

```python
from vllm._aiter_ops import rocm_aiter_ops

if not rocm_aiter_ops.is_enabled():        # also is_mla_enabled, is_fused_moe_enabled,
    return                                 # is_shuffle_kv_cache_enabled, is_linear_enabled, ...
```

**Importing `vllm.platforms.rocm` resolves the GPU arch at module load and fails on NVIDIA.**
In tests and shared modules, import it inside the function:

```python
def _supports_online_fp8() -> bool:
    if current_platform.is_rocm():
        # Imported lazily: this module resolves the GPU arch at import time.
        from vllm.platforms.rocm import get_cdna_version

        return get_cdna_version() >= 3
    return current_platform.has_device_capability(89)
```

That also fixes the trap behind `has_device_capability(89)` alone: ROCm reports MI250 as
capability 9.0, so a capability check does not skip a part with no FP8 at all. Gate ROCm tests
with `skip_if_not_cuda_alike` plus an architecture check, not `skip_if_not_cuda`.

### Changing a ROCm default

Rewrite only `"auto"`, bail out on every configuration the new path can't serve, fall back
silently, and log the opt-out once. `_maybe_default_dsv4_mxfp4_indexer` in `vllm/platforms/rocm.py`
is the model to copy: it returns early unless `get_cdna_version() == 4`, DCP/PCP are 1 and
`rocm_aiter_ops.is_enabled()`, then `logger.info_once` names the flag that restores the old
behaviour. A default flip needs end-to-end evidence on both arms — throughput, latency, KV
capacity and accuracy — not a microbenchmark.

## AMD CI

AMD steps are `mirror: amd:` blocks inside the NVIDIA step in `.buildkite/test_areas/*.yaml`:

```yaml
  mirror:
    amd:
      label: ":amd: (MI300) Core Operation Kernels"
      dind: false
      automatic_shard: true
      device: mi300_1            # mi300_{1,2,4}, mi355_{1,2,4,8}, mi355_dpx
      timeout_in_minutes: 90
      working_dir: "/vllm-workspace/tests"
      depends_on:
      - image-build-amd
      source_file_dependencies:  # the mirror has its OWN list
      - csrc/
      - vllm/platforms/rocm.py
      - vllm/_aiter_ops.py
      commands:                  # and its own commands; a bare mirror inherits the base ones
      - >-
        pytest -v -s
        kernels/core
```

- **A mirror no step selects never runs, which looks like passing.** When a ROCm change touches
  `vllm/platforms/rocm.py`, `vllm/_aiter_ops.py` or `vllm/models/*/amd/`, check those paths are
  in the mirror's `source_file_dependencies` and add them if not.
- **`automatic_shard: true` falls back to one unsharded job** when the generator can't fetch
  main's timings. `timeout_in_minutes` must cover that full run, not one shard.
- Multi-path pytest commands go in a `>-` block, `pytest -v -s` on the first line, one path per
  line, so concurrent PRs don't conflict on the same line.
- Touching `docker/Dockerfile.rocm*`, `requirements/rocm.txt`, `csrc/`, `cmake/` or
  `vllm/envs.py` triggers the whole ROCm pipeline via `run_all_patterns` in
  `.buildkite/ci_config_rocm.yaml`.
- Two pipelines: `buildkite.com/vllm/ci` for PRs, `buildkite.com/vllm/amd-ci` for the AMD
  nightly and weekly (including AITER-on builds). Link the failing build in the PR body.
- `tests/AGENTS.md`: never buy a green CI with `skip`, `xfail`, reruns or wider tolerances. A
  temporary ROCm `skip` is merged only when the body names the upstream bug, the pin that will
  carry its fix, and the guard that should then be deleted.

## Recurring ROCm failure classes

Check these before inventing a theory. Each one was a merged fix.

**HIP hardware queues.** HIP multiplexes streams onto `GPU_MAX_HW_QUEUES` queues (default 4).
When two streams share a queue, `hipStreamSynchronize` on one also waits for work enqueued
*later* on the other — CUDA does not. Symptom: lost overlap, host stalls, an ordering assertion
failing only on MI300. Fix: record a `torch.Event` per slot and wait on the event, not the
stream. Reproduce deterministically with `GPU_MAX_HW_QUEUES=1`.

**Triton / LLVM miscompiles.** Two live ones:
- gfx942 with hundreds of SGPR spills: a stride argument is never read and the head offset is
  multiplied by an unrelated spilled SGPR, so most output values are never written. Compute
  per-head offsets in 32 bits and the row offset as a 64-bit scalar.
- RDNA (gfx1201) with tensors ≤ 2 GiB, where Triton uses AMD buffer ops: NaN rows and
  `HSA_STATUS_ERROR_MEMORY_FAULT`. `num_stages=1` avoids it.

Confirm it is a codegen bug, not your logic: it disappears under `TRITON_INTERPRET=1`, with
`AMDGCN_USE_BUFFER_OPS=0`, or above the 2 GiB buffer-op threshold. Say "empirical workaround"
in the PR and keep the root cause open. **Prefill the output buffer with NaN in the test** —
otherwise a skipped row passes on leftover memory and the bug looks intermittent.

**AITER version drift.** AITER is pinned by `AITER_BRANCH` in `docker/Dockerfile.rocm_base`,
and kernels change behaviour between releases (v0.1.25 → v0.1.26 added assertions, changed
dispatch, stopped clamping positions). Released images can ship a different AITER than the
branch pins. Record the AITER version with every number, make the vLLM change correct with and
without the upstream fix, and file the `ROCm/aiter` issue.

**Shared CI hosts.** Two distinct resource problems:
- A just-exited process's VRAM still held when the next server starts vLLM's free-memory check.
  Reuse `wait_for_memory_to_settle()` from `tests/utils.py`; don't write a new helper.
- Checkpoint page cache charged to the pod's memory cgroup, OOM-killing the pod (exit -7).
  Release the previous model's page cache in an autouse fixture when the model changes.

**Ephemeral port races.** `get_open_port()` returns a port inside
`/proc/sys/net/ipv4/ip_local_port_range`, and NIXL/MoRIIO only bind it after the model loads —
a ~30 s window in which any other `bind(0)` (RCCL, gloo, ZMQ) can be handed the same port, and
the listener thread dies with `Address already in use`. Take PD side-channel and handshake
ports from *below* the ephemeral range instead.

**The HF reference, not vLLM.** On ROCm, `torch.nn.attention.varlen.varlen_attn` aligns the
causal mask to the top left when `max_q < max_k`, so an HF reference with a KV cache sees only
the first key and emits garbage while vLLM is correct. Before adjusting a threshold, check which
side is wrong.

**Host sync.** Run the affected configuration with `VLLM_GPU_SYNC_CHECK=error`. A D2H sync on
the steady-state path is a blocking review comment; building the schedule in the metadata
builder from `query_start_loc_cpu` is how merged code avoids a kernel's internal fallback sync.

## Evidence a ROCm PR needs

- **Name the silicon and the stack in every table:** `MI355X (gfx950), ROCm 10.0, torch
  2.13.0+rocm10.0.0, Triton 3.8.0, AITER v0.1.26`, plus the exact image tag
  (`vllm/vllm-openai-rocm:nightly-rocm100-<sha>`) and the vLLM commit both arms ran.
- **Before/after, not just after.** `main @ <sha>` versus `main + this commit`. For a bug:
  "fails 6 of 20 runs on main, 10/10 pass with this PR".
- **Interleave arms, report the spread**, and give the noise floor from an unchanged control
  arm. Call a result significant only past it (`|Δ| > 2 × noise` is the convention used).
- **Pair accuracy with something deterministic.** GSM8K 5-shot over all 1319 questions via
  `lm_eval --model local-completions` or `tests/evals/gsm8k/gsm8k_eval.py`, plus a bitwise or
  per-element check. vLLM is not batch invariant above concurrency 1, so greedy output differs
  run to run even with identical kernels — say so instead of explaining away a one-question gap.
- **Prove the test exercises the change.** A test that stubs the changed function guards
  nothing; reviewers catch this. Show it failing on the base commit with the real error.
- **Recipes** (`recipes/models/**.yaml`) are validated by the renderer in
  `vllm-project/recipes`, not by vLLM's CI: `node scripts/validate-source.mjs --source vllm
  --dir <checkout>/recipes`, `pnpm test:schema`, `pnpm test:overlay`, plus
  `pre-commit run --files <the yaml>`. Set `date_added` / `date_updated`. Diff the rendered
  commands before and after and state how many stayed identical.
