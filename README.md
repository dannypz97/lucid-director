# lucid-director

Real-time neural reskinning of games, with an LLM acting as art director.

A diffusion model restyles live game frames (for example, Doom's pixel art becomes a flooded neon city).
An LLM agent watches performance and quality metrics and tunes the style settings. It works within
fps, latency, and flicker budgets, every change goes through validated tools, and changes that break
a budget are rolled back automatically.
[ViZDoom](https://github.com/Farama-Foundation/ViZDoom) is the reference testbed, but nothing in the
core is Doom-specific.

> **Status:** pre-alpha. Planning is done and Phase 0 (scaffold) is next. See [PLAN.md](PLAN.md).

## How it works

The system has two loops that run at very different speeds:

```
 FrameSource ──frames──▶ Pipeline ──styled──▶ Display
 (stub|vizdoom|capture)   (stub|sdturbo|…)    (window / file / none)
        │                    ▲   │
   scene info             settings│metrics
        │                    │   ▼
        └────────▶  Agent (LLMClient + tools + guard + rollback)
                              │
                         JSONL run log
```

- **Data plane (every frame, milliseconds).** Grab a frame, restyle it with a fast img2img diffusion
  model, show it. It runs on its own thread and never waits on the LLM.
- **Control plane (every few seconds).** The agent reads a metrics snapshot and calls tools such as
  `set_prompt`, `set_strength`, and `set_steps`. A guard layer validates and clamps every call. After
  each change the system measures for a window, and if a budget broke it reverts and tells the agent why.

The two planes share one thread-safe settings object. The agent writes to it, and the render loop
reads it between frames.

### Why an LLM, and not a hand-written controller?

Some goals are fuzzy, like "flooded neon city, photoreal, but keep it playable". These goals mix
taste (prompt wording) with numbers (strength, steps, fps). An LLM can reason about both. The
interesting engineering is in keeping it **safe and honest**: bounded tools, budgets, loop detection,
rollback, and a composite score, so it can't "win" by switching the restyling off.

## Everything is swappable

| Interface     | Implementations                                                       |
|---------------|-----------------------------------------------------------------------|
| `FrameSource` | `StubSource`, `ViZDoomSource`, later `WindowCaptureSource`            |
| `Pipeline`    | `StubPipeline`, `PassthroughPipeline`, `DiffusersSDTurbo`, later `StreamDiffusion` |
| `LLMClient`   | `ScriptedClient` (tests), `AnthropicClient`, `OllamaClient`, OpenAI-compatible |
| `Display`     | `OpenCVWindow`, `FileWriter`, `NullDisplay`                           |
| `Controller`  | `FixedController`, `RandomWalkController`, `HillClimbController`, `BanditController`, `LLMController` |

The non-LLM controllers are baselines. They use the same tools, guard, and rollback as the LLM, so
evals can answer a simple question: does the LLM do better than a simple optimizer?

YAML config selects each implementation (see `configs/`). The stub source and stub pipeline let the
whole agent loop run and be tested with no GPU, no game, and no network.

## Quickstart (planned, not yet working)

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -e ".[dev]"
```

```bash
python -m lucid_director --config configs/stub.yaml
```

## Hardware

Hardware is detected at startup and calibrated. Budgets and tool bounds come from measured
performance, not hardcoded constants. Every run log records the hardware, so results can be compared
across machines.

Dev machine: RTX 3060 Laptop (6 GB VRAM), 16 GB RAM, native Windows 11.

## Licensing and assets

- This repo does **not** include Doom WADs or model weights. ViZDoom works with
  [Freedoom](https://freedoom.github.io/) (free); you may supply your own original WADs.
- `stabilityai/sd-turbo` is under the Stability AI Community License (see the
  [model card](https://huggingface.co/stabilityai/sd-turbo)). Weights download from Hugging Face at
  runtime and are not redistributed here.
- Check each upstream license yourself before publishing derived work.

## Project plan and learning resources

See [PLAN.md](PLAN.md) for phases, decisions, risks, and beginner-friendly reading for each phase.
New to diffusion or LLM agents? [docs/CONCEPTS.md](docs/CONCEPTS.md) explains the ideas behind
the code: frames, depth and labels, latents, tool use, budgets, and every config field.
