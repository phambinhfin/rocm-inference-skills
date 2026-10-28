# vllm-rocm-skills

An agent skill describing how a ROCm/AMD change actually gets merged into
[vllm-project/vllm](https://github.com/vllm-project/vllm).

It was written by reading the 30 most recently merged `rocm`-labelled pull requests (#63260 to
#64158) and keeping only what an agent could not guess: the architecture-gating helpers
maintainers ask for in review, how AMD steps are wired into Buildkite, the ROCm failure classes
that come back again and again, and the evidence a ROCm PR has to carry.

| Path | What it is |
|---|---|
| [skills/vllm-rocm-pr-patterns/SKILL.md](skills/vllm-rocm-pr-patterns/SKILL.md) | The skill. Title and body conventions, platform gating, AMD CI wiring, failure classes, evidence standards. |
| [skills/vllm-rocm-pr-patterns/cases.md](skills/vllm-rocm-pr-patterns/cases.md) | One line per source PR, grouped by lesson, so any rule can be traced to the change it came from. |

## Install

The skill is a plain directory with a `SKILL.md`, so most agents can use it by copying it into
wherever they look for skills:

```bash
git clone https://github.com/phambinhfin/vllm-rocm-skills
cp -r vllm-rocm-skills/skills/vllm-rocm-pr-patterns ~/.cursor/skills/      # Cursor
cp -r vllm-rocm-skills/skills/vllm-rocm-pr-patterns ~/.claude/skills/      # Claude Code
```

To scope it to one checkout instead, copy it into `.cursor/skills/` or `.claude/skills/` in the
repository you work in. Agents without skill support can read `SKILL.md` directly.

## Scope

This is unofficial and is not maintained by the vLLM project. It is a summary of observed
practice, not policy, and upstream moves fast — the AITER pin, the ROCm image tags, the
Buildkite device pools and the PR template all change.

**Read vLLM's own instructions first.** They are authoritative and mandatory for AI-assisted
contributions:

- [`AGENTS.md`](https://github.com/vllm-project/vllm/blob/main/AGENTS.md)
- [`tests/AGENTS.md`](https://github.com/vllm-project/vllm/blob/main/tests/AGENTS.md)
- [`.agents/skills/pr-checklist/SKILL.md`](https://github.com/vllm-project/vllm/blob/main/.agents/skills/pr-checklist/SKILL.md)

Everything cited here is from public pull requests and public source in `vllm-project/vllm`.

## License

Apache-2.0, matching vLLM.
