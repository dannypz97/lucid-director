# Concepts

A guide to the ideas behind lucid-director, for someone new to real-time diffusion or LLM agents.
[PLAN.md](../PLAN.md) says *what we're building, and when*; this file says *what things are and
why they exist*. It grows as the code grows.

**Contents**
1. [The big picture](#1-the-big-picture)
2. [Why stubs](#2-why-stubs)
3. [Frames: image, depth, labels](#3-frames-image-depth-labels)
4. [How diffusion fits in](#4-how-diffusion-fits-in)
5. [Config fields explained](#5-config-fields-explained)
6. [Settings and thread safety](#6-settings-and-thread-safety)
7. [How the agent talks to the LLM](#7-how-the-agent-talks-to-the-llm)
8. [Decisions, steps, and budgets](#8-decisions-steps-and-budgets)
9. [Performance and playability](#9-performance-and-playability)
10. [Code patterns used here](#10-code-patterns-used-here)
11. [Glossary](#11-glossary)

---

## 1. The big picture

Two loops run at very different speeds:

- **Data plane (every frame, ~10–100 ms):** `FrameSource` → `Pipeline` → `Display`. Grab a frame,
  restyle it, show it. It must never wait on anything slow.
- **Control plane (every few seconds):** a **controller** (the LLM agent, or a simple baseline
  like a hill-climber) reads metrics and changes the style settings through validated **tools**.

They share one thing: the current **settings**. The controller writes new settings; the render
loop reads them once per frame.

Every component is an **interface** with swappable implementations chosen in YAML config:

| Interface | What it is | Examples |
|---|---|---|
| `FrameSource` | Anything that produces a stream of images. A camera you can swap out. | stub shapes, Doom, screen capture |
| `Pipeline` | Turns a source frame into a styled frame | stub, passthrough, SD-Turbo |
| `Display` | Shows, saves, or discards styled frames | window, mp4 file, nothing |
| `LLMClient` | A chat model that can call tools | scripted (tests), Claude, Ollama |

## 2. Why stubs

A **stub** is a fake stand-in with the same interface as the real thing, but simple maths inside.
`StubSource` pretends to be Doom (moving shapes); `StubPipeline` pretends to be diffusion: it
returns the frame roughly unchanged, plus made-up but realistic metrics (more steps → lower fps,
more strength → more style but more flicker).

Why fake it:

1. **The agent can be built before the hard parts exist.** It only needs something that answers
   "I changed X, what happened?".
2. **Bugs stay separate.** On the stub, only the agent and guard can be wrong. When something
   breaks later on real Doom, it must be in the new parts.
3. **Tests are fast and repeatable.** No GPU, no game, no network, and the same result every
   time, so tests also run in CI.
4. **The right answer is known.** We wrote the formulas, so "the agent found the optimum" is
   checkable. That's why the stub needs a real sweet spot, not "set everything to max".

It's the agent's flight simulator, and it stays in the repo for good.

## 3. Frames: image, depth, labels

A `Frame` is one snapshot from a `FrameSource`:

| Field | What it is | Used for |
|---|---|---|
| `image` | The picture: RGB, `uint8`, shape `(height, width, 3)` | what diffusion restyles |
| `index` | Frame number in the episode (0, 1, 2, …) | logs; spotting dropped frames (41, 42, 44) |
| `timestamp` | When it was captured (seconds) | latency = time shown − timestamp |
| `depth` | *Optional.* Same size; each value = **distance** to that pixel | keeping geometry intact |
| `labels` | *Optional.* Same size; each value = **object id** (0 = background) | treating objects differently |

**Depth** (bright = far):
```
image                        depth
┌──────────────────┐         ┌──────────────────┐
│  wall   door     │         │ ▒▒▒▒▒▒ ██████    │  door far → bright
│     [imp]        │         │    ░░░           │  imp close → dark
└──────────────────┘         └──────────────────┘
```
At high strength, diffusion only sees colours, so it can "melt" a corridor into something that no
longer matches the real geometry. Depth conditioning (ControlNet, Phase 4) tells it *"there's a
wall 3 m away here; keep it, just make it neon."* Depth also drives structure-fidelity scoring and
occlusion (a fake object hidden behind a real pillar). Doom computes depth anyway; ViZDoom hands
it over. Screen capture can't, so it leaves `depth=None` and depth tools aren't offered.

**Labels:**
```
image                        labels
┌──────────────────┐         ┌──────────────────┐
│     [imp]  [imp] │         │     333    777   │  two imps: ids 3, 7
│        [barrel]  │         │        55555     │  barrel: id 5
└──────────────────┘         └──────────────────┘
```
Labels let the director give objects their own prompts ("imps → jellyfish"), count enemies for
`scene_summary()` ("2 imps visible"), and keep enemies readable so the game stays playable.

**Why depth and labels are inside `Frame`:** the game keeps moving. Fetching them separately, a
few ms later, could return the *next* tic, so outlines would no longer match. Bundling them
guarantees they describe the same moment, and `Frame` is read-only for the same reason.

**Array order gotcha:** config sizes are **(width, height)**, as in "320×200", but numpy image
shapes are **(height, width, channels)**, because images are stored row by row. A 320×200 frame
has `image.shape == (200, 320, 3)` and `frame.size == (320, 200)`. Mixing these up gives rotated or
squashed images.

**Colour order gotcha:** we use **RGB** everywhere. OpenCV uses **BGR**, so conversion happens
only inside OpenCV-based code.

## 4. How diffusion fits in

Stable Diffusion doesn't work on pixels directly; they're too big. It works on a compressed
**latent**, about 8× smaller on each side:

```
game frame ──VAE encode──▶ latent ──UNet × steps──▶ styled latent ──VAE decode──▶ styled frame
```

- **VAE** (variational autoencoder): the **compressor**. Encodes pixels to a latent and back. It
  doesn't style anything.
- **UNet**: the **artist**. Each step it removes some noise from the latent, steering it toward the
  prompt. More steps means more UNet passes.
- **img2img:** instead of starting from pure noise, start from the *game frame* with some noise
  added. **Strength** sets how much: 0 returns the input unchanged, 1 ignores it entirely.
- **Turbo models** (SD-Turbo) are *distilled* to look good in 1–4 steps instead of 20–50. That's
  what makes real-time possible.
- **Multiple of 8:** because the VAE shrinks each side by 8, resolutions must divide by 8.
- **Prompt limits:** the text encoder (CLIP) reads only ~77 tokens (~200–300 characters); the rest
  is cut off. SD-Turbo runs without "guidance", so **negative prompts have no effect**.

`PipelineResult.stage_ms` times each stage separately because different bottlenecks have different
fixes:

| Slow stage | Fix |
|---|---|
| UNet | fewer steps, lower resolution, `torch.compile`, TensorRT |
| VAE decode | tiny VAE (TAESD) |
| VAE encode | tiny VAE, or reuse latents between frames |

## 5. Config fields explained

Config is YAML, validated by pydantic. Unknown keys and out-of-range values are rejected at
startup, so a typo like `stregth` fails immediately instead of being silently ignored.

**Two resolutions, on purpose:**
- `frame_source.resolution` is what the **game renders at** (Doom: 320×200). Small keeps the game fast.
- `pipeline.resolution` is what the **model works at**, e.g. 384×240 (same 16:10 shape). This is
  the biggest performance setting: cost grows with pixel count, so 384→512 is ~1.8× slower.
- A preprocess step resizes between them; the display can scale up separately.

| Field | What it's for |
|---|---|
| `hardware.profile` | `auto` detects the GPU and picks safe limits; set it by hand to simulate weaker/stronger machines |
| `frame_source.seed` | Makes synthetic frames identical every run, for fair tests and comparisons |
| `pipeline.model` | Hugging Face model to download (e.g. `stabilityai/sd-turbo`); stubs ignore it |
| `pipeline.dtype` | Number precision. `fp16` = half the VRAM of `fp32`, ~2× faster, no visible loss |
| `pipeline.conditioning` | Also give diffusion depth or edges so geometry holds (Phase 4) |
| `initial.strength` | img2img denoise strength. The main restyling setting |
| `initial.steps` | Denoising passes. Turbo: 1–4. More steps = better quality but proportionally slower |
| `initial.seed_policy` | `fixed`: the same starting noise every frame (stable, less flicker). `per_frame`: new noise each frame (livelier, flickers) |
| `agent.goal` | The brief the controller is judged against. Fixed for a run, or evals stop meaning anything |
| `agent.budgets` | Hard limits (fps, p95 latency, flicker). Breaking one triggers automatic rollback |
| `agent.decision_interval_s` | How often the controller decides |
| `agent.max_steps_per_episode` | How many decisions in total (see [section 8](#8-decisions-steps-and-budgets)) |
| `logging.dir` | Where JSONL run logs go; evals read them |
| `type` (each section) | Which implementation to use, looked up in the registry |
| `options` (each section) | Implementation-specific settings (e.g. a Doom scenario), passed straight through and checked by that implementation |

**Config vs runtime:** config holds the **starting values** and is read-only after loading.
Whatever changes during a run (the agent rewriting the prompt, raising strength) lives in runtime
settings, never in config, so "what the run started with" is always true.

## 6. Settings and thread safety

The agent changes settings on its own thread while the render loop runs on another. Two rules
keep that safe without complicated locking:

1. **Settings are immutable.** A change creates a *new* `StyleSettings` (`with_changes(...)`, which
   validates; pydantic's `model_copy(update=...)` does **not**). Swapping one reference is atomic,
   so the render loop sees either the old settings or the new, never a mix. Rollback is just
   swapping the old object back.
2. **`Pipeline.process(frame, settings)` receives settings on every call.** The render loop takes
   one snapshot per frame and passes it in, so a change can only take effect between frames.

## 7. How the agent talks to the LLM

**Tool use:** the LLM never runs anything. It returns a structured *request* to call a tool; our
code decides whether and how to act.

| Type | Role |
|---|---|
| `ToolSpec` | A menu item: tool name, description, JSON schema for its arguments |
| `Message` | One conversation entry. `role` is `user` (our observations), `assistant` (the LLM), or `tool` (a tool's result) |
| `ToolCall` | The LLM asking to run a tool with specific arguments |
| `LLMResponse` | The full reply: assistant message, stop reason, token usage (for cost) |

One turn:
1. We send `ToolSpec`s and a `user` message: *"fps=24 (budget ≥20), style=0.41. Goal: more style."*
2. The LLM replies with a `ToolCall`: `visual.set_strength(value=0.65)`.
3. **The guard** validates and clamps it, checks cooldown and loop detection, and applies it.
4. We reply with a `tool` message (matched by `tool_call_id`): *"ok: 0.50 → 0.65. Measuring for 3 s."*
5. Later: new metrics, and the next turn.

These types are **provider-neutral**. The agent loop speaks only them; each `LLMClient` translates
to its own API. That's how Claude, Ollama, and the scripted test client stay interchangeable.

## 8. Decisions, steps, and budgets

| Field | Controls | Analogy |
|---|---|---|
| `decision_interval_s` | **pace**: one decision every N s | how often the director speaks |
| `max_steps_per_episode` | **total**: at most N decisions | how many notes before the cut |

- **One step = one decision turn**, including "metrics look good, hold". It costs an LLM call, so
  it counts. The number of actual setting changes is tracked separately.
- Together they bound an episode: 30 steps × 5 s ≈ 2.5 min, at most 30 API calls. That's also the
  cost cap.
- You need both: an interval alone never ends; a step limit alone lets everything happen in the
  first second, before any metrics exist.
- **Rule:** the interval must be at least as long as the measurement window after a change, or the
  controller decides on metrics from *before* its last change.

**Budgets and rollback:** after a change, measure for a window. If fps, p95 latency, or flicker
breaks its budget, revert to the last good settings and tell the controller what broke, and by how
much.

**Anti-gaming:** an agent told "minimize flicker" can win by setting strength to 0, which means no
restyling at all. So we score a **composite** (style AND structure fidelity AND budgets), and log
each part separately so gaming is visible.

## 9. Performance and playability

Framework overhead (config, interfaces, guard, agent) is microseconds per frame. The agent
thread adds **zero** per-frame cost. Nearly all the time is diffusion. Rough estimates for an RTX
3060 Laptop at 384×240, 1 step (calibration in Phase 3 gives the real numbers):

| Stage | Est. |
|---|---|
| grab + resize + display | ~2–3 ms |
| UNet | ~40–70 ms |
| VAE decode | ~15–30 ms |
| **Total** | **~60–100 ms ≈ 10–15 fps** |

- **fps** is how smooth it looks. **Latency** is how far behind your input the view is. Shooters
  feel laggy above ~50 ms.
- **p95, not average:** averages hide stutters. p95 latency (95% of frames are faster than this)
  is what you actually feel.
- The game itself doesn't slow down; the restyled *view* lags behind and is choppier. ViZDoom can
  run in **lockstep** (game waits for each styled frame: no lag, slow motion) or **frame-skip**
  (normal speed, choppier).
- Recorded demos can be restyled **offline** at any speed, so demo GIFs can be smooth even when
  live play isn't.

## 10. Code patterns used here

- **Interfaces (ABCs):** `abc.ABC` + `@abstractmethod` define what a component must provide. Core
  code depends only on these.
- **Registry with lazy imports:** config `type: stub` → class. Entries can be strings like
  `"pkg.module:Class"`, imported only when selected, so `type: stub` never loads torch.
- **Frozen config / frozen dataclasses:** read-only after creation, so values never change under you.
- **Validation at boundaries:** `Frame` checks its own shape and dtype when created, so bugs fail
  loudly and early instead of showing up as strange output.
- **`src/` layout + editable install:** tests import the *installed* package (catching packaging
  bugs), and edits take effect without reinstalling.
- **Python names vs objects:** `x[k] = v` changes the object (everyone holding it sees the change);
  `x = ...` only moves the name. That's why a registry entry updated through a local variable
  persists.
- **pytest fixtures:** a test parameter named `tmp_path` gets a new temporary folder per test,
  injected by name. Custom fixtures live in `tests/conftest.py`. Run `pytest -v` to see test names
  and `pytest --fixtures` to list fixtures.

## 11. Glossary

| Term | Meaning |
|---|---|
| **Agent** | An LLM in a loop: observe, decide, call tools, repeat |
| **Baseline** | A simple non-LLM controller to compare against (fixed, random walk, hill-climb, bandit) |
| **Bandit** | Picks among a few options ("arms"), learning which pays off while still exploring |
| **CLIP** | A model that puts images and text in a shared space; used to score "does this look like the prompt?" |
| **ControlNet** | An add-on that guides diffusion with extra inputs like depth or edges |
| **Distillation** | Training a fast "student" model to imitate a slow "teacher" model |
| **Flicker** | Frame-to-frame instability, measured by warping frame t−1 onto t with optical flow and comparing |
| **fp16** | 16-bit floating point; half the memory of fp32 |
| **Guard** | The layer that validates, clamps, and rate-limits tool calls before they touch settings |
| **Hill climbing** | Nudge one setting; keep the change if the score improved, otherwise revert |
| **Latent** | The compressed image representation diffusion works on |
| **Optical flow** | How each pixel moved between two frames |
| **p50 / p95** | Median / 95th percentile; 95% of samples are below p95 |
| **Rollback** | Automatically reverting a change that broke a budget |
| **Stub** | A fake implementation with the real interface, for building and testing |
| **TAESD** | A tiny, much faster VAE decoder; slightly softer output |
| **Tic** | One Doom game update; 35 per second at normal speed |
| **Tool use** | The LLM returning structured function-call requests that our code executes |
| **VRAM** | GPU memory; 6 GB on the dev laptop, the tightest constraint |
