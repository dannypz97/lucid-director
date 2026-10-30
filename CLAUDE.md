# lucid-director

Real-time diffusion reskinning of games, with an LLM as art director. Plan: [PLAN.md](PLAN.md).
Concepts: [docs/CONCEPTS.md](docs/CONCEPTS.md). Current session state (gitignored, may be absent):
@HANDOFF.md

## Working agreement

- Work **one PLAN.md sub-task at a time**, only when asked (e.g. "do 0.4"). Stop and summarize after each.
- **Never commit.** The user commits; suggest a short message when asked.
- **Explain as you go:** why something is needed and what it achieves. The user is new to
  diffusion and agents, so give intro-level, big-picture explanations. At the end of a sub-task,
  list any new essential concepts and ask whether to add them to docs/CONCEPTS.md; add only
  what the user approves.
- Mid-task "why" questions: answer directly; if one exposes a real flaw, fix it.
- New feature ideas go to PLAN.md's **Later** section, not the current phase.
- Before switching chats, update HANDOFF.md: where we are, what's next, open loose ends.

## Code style

- Simple, readable Python: type hints, pydantic/dataclasses, `pytest`.
- **Comments describe the code only.** No project slogans or principles in comments; design
  rationale goes in chat, PLAN.md, or CONCEPTS.md.
- Everything outside the core loop sits behind an interface in `core/interfaces.py` and is selected
  by config through the registry (`core/registry.py`). Nothing game-specific in `core/`.

## Settled design (don't relitigate)

- Config: pydantic, `extra="forbid"`, frozen. Swappable sections = `type` (registry name) +
  `options` (implementation-specific). Resolutions are `(width, height)`.
- `StyleSettings` is also the runtime settings type: immutable, changed via `with_changes()`
  (validates), never `model_copy(update=...)`. `Pipeline.process(frame, settings)` gets a
  per-frame snapshot.
- Images are RGB `uint8` `(H, W, 3)`; `Frame` carries depth/labels alongside the image.
- Implementations construct as `cls(config_section)`. Registry entries are lazy `"module:Class"`
  strings, added only once the class exists.
- `LLMClient.complete()` is blocking (agent has its own thread); LLM message types are provider-neutral.
- `agent.decision_interval_s` sets pace; `max_steps_per_episode` counts decision turns, including "no change".

## Environment

- Native Windows 11, PowerShell. Python 3.13, venv at `.venv`.
- Tests: `.\.venv\Scripts\python.exe -m pytest -q`
- RTX 3060 Laptop (6 GB VRAM), 16 GB RAM. GPU torch comes from the PyTorch `cu130` index; PyPI
  torch is CPU-only on Windows.
