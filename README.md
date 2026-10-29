# rocm-inference-skills

Agent skills for contributing to the two repositories behind LLM inference on AMD GPUs:
[vllm-project/vllm](https://github.com/vllm-project/vllm) on ROCm, and
[ROCm/aiter](https://github.com/ROCm/aiter), the kernel library behind vLLM's
`VLLM_ROCM_USE_AITER*` paths.

Each was written by reading the 30 most recently merged pull requests in its repository and
keeping only what an agent could not guess: the conventions maintainers enforce in review, how
the CI is wired, the failure classes that come back again and again, and the evidence a change
has to carry.

| Skill | Covers |
|---|---|
| [vllm-rocm-pr-patterns](skills/vllm-rocm-pr-patterns/SKILL.md) | ROCm changes to vLLM: architecture-gating helpers, AMD Buildkite mirror wiring, HIP hardware queues, Triton gfx942 miscompiles, AITER version drift, shared-host VRAM, ephemeral port races. |
| [aiter-pr-patterns](skills/aiter-pr-patterns/SKILL.md) | aiter changes: which of aiter's own in-repo skills to open, automatic PR title tags, the op_tests runner that reports success without running a test, per-architecture JIT build failures, tuned-config dispatch traps. |

Each skill has a `cases.md` next to it mapping every rule back to the pull request it came from,
so a claim can be checked rather than taken on trust.

## Install

A skill is a plain directory with a `SKILL.md`, so most agents can use it by copying it to
wherever they look for skills:

```bash
git clone https://github.com/phambinhfin/rocm-inference-skills
cp -r rocm-inference-skills/skills/* ~/.cursor/skills/      # Cursor
cp -r rocm-inference-skills/skills/* ~/.claude/skills/      # Claude Code
```

To scope them to one checkout instead, copy into `.cursor/skills/` or `.claude/skills/` in the
repository you work in. Agents without skill support can read the `SKILL.md` files directly.

## Scope

Unofficial, and not maintained by either project. These summarise observed practice, not policy,
and both repositories move fast — the AITER pin, the ROCm image tags, the CI device pools, the
supported `gfx` targets and the PR templates all change.

**Read each repository's own instructions first.** They are authoritative, and for vLLM they are
mandatory for AI-assisted contributions:

- vLLM: [`AGENTS.md`](https://github.com/vllm-project/vllm/blob/main/AGENTS.md),
  [`tests/AGENTS.md`](https://github.com/vllm-project/vllm/blob/main/tests/AGENTS.md),
  [`.agents/skills/pr-checklist/`](https://github.com/vllm-project/vllm/blob/main/.agents/skills/pr-checklist/SKILL.md)
- aiter: [`CONTRIBUTE.md`](https://github.com/ROCm/aiter/blob/main/CONTRIBUTE.md), the eight
  skills in [`.claude/skills/`](https://github.com/ROCm/aiter/tree/main/.claude/skills), and
  [`.github/instructions/`](https://github.com/ROCm/aiter/tree/main/.github/instructions).
  `aiter-pr-patterns` points at the right one per change rather than duplicating them.

Everything cited here is from public pull requests and public source in the two repositories.

## License

Apache-2.0.
