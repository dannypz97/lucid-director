# PLAN

The working plan for lucid-director. The README says *what* this is; this file says *how we get
there*, *why each step exists*, and *what to read* so the concepts make sense.
[docs/CONCEPTS.md](docs/CONCEPTS.md) explains the ideas themselves. New essential concepts are
offered at the end of each sub-task and added only if approved.

## Goals (in priority order)

1. **Control plane:** learn and demonstrate agent loops, tool use, guardrails, and failure handling.
2. **Data plane:** learn and demonstrate real-time inference engineering: profiling, concurrency,
   backpressure, and hardware-aware config.
3. **Fun:** Doom restyled live.

## Working agreement

- Small milestones. Each one runs and has at least a smoke test before we move on.
- Everything outside the core loop sits behind an interface and is selected by config. Doom is never
  hardcoded into core code.
- Simple, readable Python: type hints, pydantic/dataclasses, `pytest`.
- Design choices get explained briefly as we go.
- **The user does all git commits.**

## Decisions

| Decision | Status | Choice / recommendation |
|---|---|---|
| OS | Decided | **Native Windows.** Simpler GPU and display; ViZDoom has Windows wheels. |
| Python version | Decided | **3.13.** Verified 2026-10-25: Windows cp313 wheels exist for torch 2.14.1 (CUDA), vizdoom 1.3.2, opencv-python; diffusers/transformers are pure Python. 3.14 has wheels too, but Phase 6 extras (StreamDiffusion/TensorRT/xformers) usually lag the newest Python. |
| torch install | Note for Phase 3 | PyPI `torch` on Windows is **CPU-only**. Install from the PyTorch index: `--index-url https://download.pytorch.org/whl/cu130` (the driver supports CUDA 13). |
| Dev LLM | Decided | API model (Claude Haiku 5.5) behind `LLMClient`; Sonnet 5.5 for comparison. |
| Local LLM | Later | Ollama 3B–4B Q4 on **CPU**. 16 GB RAM makes this viable now, and it keeps the GPU free for diffusion. |
| Diffusion model | Decided | `stabilityai/sd-turbo` via plain `diffusers` img2img. StreamDiffusion later. |

## Hardware (dev)

RTX 3060 Laptop, **6 GB VRAM** (the tight constraint), **16 GB RAM** (comfortable), Windows 11.
A better laptop comes later, so nothing hardcodes these limits. The `hardware/` module detects them,
and calibration measures what is actually achievable.

Realistic expectation on this GPU: roughly 256–384 px output at single-digit to low-teens fps for
1-step SD-Turbo. Calibration gives the real number, and the agent's budgets come from it.

## Controllers and baselines

**The question that keeps the project honest:** does the LLM actually do better than a dumb
algorithm? If a 30-line hill-climber matches it, the LLM is only adding cost and latency. So the
"director" is an interface, `Controller`, with several implementations:

| Controller | What it does | Why it's here |
|---|---|---|
| `FixedController` | Never changes settings | Floor: what you get with no tuning at all |
| `RandomWalkController` | Random valid tool calls | Shows the guard/rollback catching bad moves; a "no intelligence" floor |
| `HillClimbController` | Nudge one knob (e.g. strength ±0.05), keep the change if the composite score improved, otherwise revert and try another knob | The classic, cheap optimizer; strong on smooth numeric knobs |
| `BanditController` | Treat a small set of preset configs (prompt × strength × steps) as "arms"; pick arms with UCB or Thompson sampling, learning which pays off | Handles noisy rewards and discrete choices like prompts; explores vs. exploits explicitly |
| `LLMController` | The LLM agent | The thing we're testing |

**All controllers use the same tools, guard, cooldown, and rollback.** That makes the comparison
fair: the only thing that varies is the decision-making.

**Where the LLM should win:** goals written in language ("make it feel more underwater"), changing
prompts, reacting to scene info ("imps on screen, so keep enemies readable"), and explaining its
choices. **Where the baselines may win:** pure numeric tuning of strength and steps, at near-zero
cost. Both outcomes are worth showing in the results table.

---

## Phases

Each phase lists **why** it exists, **what you'll learn**, the **tasks**, the **milestone**, and
**read first** (short, intro-level resources).

### Phase 0: Scaffold

**Why:** Build the skeleton and the fake world first. A stub pipeline with plausible fake metrics
lets us build and test the whole agent loop with no GPU, no game, and no network, so later phases
plug real parts into a frame we already trust.

**You'll learn:** interfaces (abstract base classes), config-driven design (a registry that maps
YAML `type: stub` to a class), and project packaging.

Sub-tasks. Each one is small, runs on its own, and ends with a check you can run yourself.
They're done in order; ask for them one at a time.

- [x] **0.0 Repo hygiene.** README.md, PLAN.md, .gitignore.

- [x] **0.1 Packaging and venv.**
  `pyproject.toml` (src layout, runtime deps: numpy, pydantic, pyyaml; dev extras: pytest),
  `src/lucid_director/__init__.py`, a placeholder `__main__.py`, venv instructions.
  *Why:* the "src layout" means tests import the *installed* package, not whatever happens to be
  in the folder. That catches packaging bugs early. An editable install (`-e`) means your edits take
  effect without reinstalling.
  *Done when:* `pip install -e ".[dev]"` works and `python -m lucid_director --help` prints usage.

- [x] **0.2 Config models and loader.**
  Pydantic models for each config section (`hardware`, `frame_source`, `pipeline`, `display`, `llm`,
  `agent`, `logging`), `load_config(path)`, and `configs/stub.yaml`. Each swappable section has
  `type` + `options`, and implementation-specific keys go in `options`.
  *Why:* the config is the contract for "everything swappable". Pydantic turns a typo like
  `stregth: 2` into a clear error at startup instead of a strange bug ten minutes into a run.
  *Done when:* `tests/test_config.py` passes. It covers a valid config, an unknown key, and an
  out-of-range value.

- [x] **0.3 Core types and interfaces.**
  Dataclasses `Frame` (image plus optional depth/labels, timestamp, index), `PipelineResult`
  (image, per-stage ms, pipeline-reported metrics), and provider-neutral LLM message types. Settings
  reuse `StyleSettings` from config. ABCs `FrameSource`, `Pipeline`, `LLMClient`, `Display` in
  `core/interfaces.py`, plus a **registry** with lazy imports (`type: stub` → class).
  *Why:* the interfaces are the whole portability story. Core code talks only to these, so swapping
  Doom for another game, or SD-Turbo for StreamDiffusion, is a config change.
  *Done when:* `test_types.py` and `test_registry.py` pass. The built-in registry starts empty;
  each stub registers itself as it's written (0.5, 0.6, 0.7), so it never names something missing.
  Also added `StyleSettings.with_changes()` (validated copy; `model_copy(update=...)` skips
  validation).

- [ ] **0.4 Clock and timing stats.**
  An injectable `Clock` (real and fake), and a rolling-window `TimingStats` (fps, p50/p95 latency,
  per-stage ms). A `fake_clock` fixture in `tests/conftest.py`.
  *Why:* the fake clock makes tests instant and deterministic, with no `sleep()` and no flaky
  timing. Percentiles matter because averages hide stutters: p95 latency is what you actually *feel*.
  The fake clock is also how `StubPipeline` (0.6) "takes time": it advances the clock by its
  simulated latency instead of sleeping.
  *Done when:* a test feeds known timings into `TimingStats` and checks fps and p95. Offer clock
  injection and percentiles for `docs/CONCEPTS.md`.

- [ ] **0.5 `StubSource`.**
  Synthetic moving frames (shapes drifting over a gradient), optional fake depth, and a canned
  `scene_summary()`.
  *Why:* later flicker and fidelity metrics need frames with real motion and structure, and this
  gives them a reproducible version.
  *Done when:* it yields N frames of the configured resolution, deterministic for a given seed,
  and is registered as `frame_source: stub`.

- [ ] **0.6 `StubPipeline`.**
  Returns the input frame (lightly tinted) and simulates cost and quality from the settings:
  latency grows with steps × resolution², style score rises with strength (with diminishing
  returns), and flicker rises with strength, a lot more under the `per_frame` seed policy. A
  little noise is added, seeded.
  *Why:* this fake world is where the Phase 1 agent and hill-climber get tested, so it needs a
  **real sweet spot**, not an obvious corner. Otherwise "the agent found the optimum" proves nothing.
  *Done when:* `tests/test_stub_pipeline.py` checks the directions: more steps means lower fps,
  more strength means a higher style score and more flicker, and a fixed seed means less flicker.
  Registered as `pipeline: stub`. Stub-specific settings (noise level, base latency) go in
  `pipeline.options`.

- [ ] **0.7 Runtime loop, `NullDisplay`, CLI.**
  `run(config, frames=N)` wires source → pipeline → display through the registry and collects
  timing. A preprocess step resizes from `frame_source.resolution` to `pipeline.resolution`, and
  handles aspect-ratio mismatch (resize-to-fit / letterbox) rather than rejecting it.
  `NullDisplay` discards frames and is registered as `display: none`. The CLI `--config` /
  `--frames` options print a summary table.
  *Why:* this is the first end-to-end data plane. It runs single-threaded for now; the agent
  thread arrives in Phase 1, after the loop is proven.
  *Done when:* `python -m lucid_director --config configs/stub.yaml --frames 300` prints fps,
  p50/p95, flicker, and style score, and a smoke test runs the same thing in-process.

- [ ] **0.8 Wrap-up.**
  Update the README quickstart to match reality, tick the boxes here, and review what was built
  together before Phase 1.

**Milestone:** the CLI runs the stub and prints a metrics summary; `pytest` passes.

**Read first:**
- Real Python, [Python Type Checking](https://realpython.com/python-type-checking/): type hints.
- Pydantic docs, [Models](https://docs.pydantic.dev/latest/concepts/models/): what our config is built on.
- Python docs, [`abc` module](https://docs.python.org/3/library/abc.html): how interfaces work in Python.

### Phase 1: Agent loop on the stub

**Why:** This is goal #1. We build the agent loop where mistakes are cheap: tools, a guard,
rollback, and loop detection, all tested against the stub with a scripted fake LLM.

**You'll learn:** what "tool use" actually is, which is the LLM returning structured function calls
that *your* code executes. You'll also learn why you never trust those calls blindly, and how to
make an agent's failures visible and recoverable.

Tasks:
- [ ] Tool schemas (JSON schema): `set_prompt`, `set_strength`, `set_steps`, `set_seed_policy`,
      `get_metrics`, `get_scene_info`, `revert_last`. `set_conditioning` is offered only when
      the source has depth/edges.
- [ ] Guard: validate, **clamp and report** out-of-range values, write error messages for the model
      to read, enforce a cooldown between changes, a step budget, and loop/oscillation detection
- [ ] Rollback: after each change, measure a full window; if a budget broke, revert to the last good
      settings and give the agent a structured reason
- [ ] Thread-safe settings object; agent on its own thread/task; render loop never waits on the LLM
- [ ] JSONL run log (every observation, tool call, clamp, rollback)
- [ ] `ScriptedClient` tests: clamping, budget-violation rollback, oscillation detection
- [ ] `AnthropicClient` against the stub. Goal: "maximize style intensity while keeping fps >= 20"
- [ ] Composite objective (style AND fidelity AND budgets), logged as separate components
- [ ] **Namespace tools by backend** (e.g. `visual.set_strength`), registered by each configured
      backend, so the agent's tool list is built from config. It's cheap now, and later it lets an
      audio backend (see Later) add `audio.*` tools without touching the agent loop.
- [ ] `Controller` interface; `FixedController`, `RandomWalkController`, `HillClimbController`.
      Selected by a new `agent.controller` field (`llm | hill_climb | bandit | random_walk | fixed`);
      the `llm` section is only read when `controller: llm`.
- [ ] **Step = one decision turn**, including "no change" (each costs an LLM call). Track the
      number of actual setting changes as a separate metric.
- [ ] Validate at config load: `agent.decision_interval_s` ≥ the post-change measurement window,
      so the controller never decides on metrics from before its last change.
- [ ] Hill-climber on the stub finds the optimum. This gives a **reference answer** for checking
      the LLM, and a test that the stub's trade-off is actually optimizable

**Milestone:** the agent converges near the stub's optimum (as found by the hill-climber), and the
log shows it reasoning from metrics.

**Read first:**
- Anthropic, [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents):
  the best short big-picture piece on agents vs. workflows, and when to use which.
- Anthropic docs, [Tool use overview](https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview):
  the request/response shape we'll implement.
- Lilian Weng, [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/):
  a broader survey (planning, memory, tools). Skim it.
- Real Python, [Speed Up Your Python Program With Concurrency](https://realpython.com/python-concurrency/):
  threads vs. asyncio, at the level we need.

### Phase 2: ViZDoom source + display

**Why:** Swap the fake world for a real one, with no diffusion yet. This isolates "can we capture and
display game frames fast?" from "can we restyle them fast?".

**You'll learn:** game frame buffers (screen, depth, labels), measuring per-stage timing, and the
latency between the game and what you see.

Tasks:
- [ ] `ViZDoomSource`: screen, optional depth/labels, `scene_summary()` ("2 imps visible, health low")
- [ ] `PassthroughPipeline`, `OpenCVWindow` with an fps/latency overlay, `NullDisplay`, `FileWriter`
- [ ] Per-stage timing instrumentation
- [ ] Freedoom setup documented; no WADs committed

**Milestone:** play Doom with a live fps/latency overlay; depth and labels available.

**Read first:**
- [ViZDoom docs](https://vizdoom.farama.org/): skim "Introduction" and the Python examples.

### Phase 3: Real diffusion

**Why:** This is the core visual effect. It's also where the hardware constraints show up, so we add
detection and calibration here.

**You'll learn:** what a diffusion model does, what img2img "strength" and "steps" mean (the
agent's main knobs), why distilled "turbo" models can run in 1–4 steps, and fp16 on the GPU.

Tasks:
- [ ] `DiffusersSDTurbo` img2img, fp16, stage timers (preprocess, UNet, VAE decode)
- [ ] Hardware detect (GPU, VRAM, RAM, CPU) + profiles (`low`/`mid`/`high`)
- [ ] Calibration (~10 s): measure achievable fps at a few resolutions; derive budgets and bounds
- [ ] Hardware + calibration recorded in every run log
- [ ] Optional **tiny VAE (TAESD)** for decoding: a drop-in replacement that is many times faster
      than the full VAE and slightly softer. Config switch `pipeline.options.vae: full | tiny`;
      calibration measures both.
- [ ] **Game timing mode** (ViZDoom source option): `lockstep` (the game advances only when a
      styled frame is ready: no lag, slow motion) vs `frameskip` (the game advances N tics per
      styled frame: normal speed, choppier)

**Milestone:** Doom restyled live, with real calibration numbers in the log.

**Read first:**
- Jay Alammar, [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/):
  **the** visual intro. Start here.
- Hugging Face, [Diffusion Models Course](https://github.com/huggingface/diffusion-models-class):
  hands-on; unit 1 is enough for now.
- Hugging Face diffusers docs, [Image-to-image](https://huggingface.co/docs/diffusers/using-diffusers/img2img):
  exactly the mode we use; explains `strength`.
- [SD-Turbo model card](https://huggingface.co/stabilityai/sd-turbo): the `steps * strength >= 1` rule and the license.

### Phase 4: Structure and flicker

**Why:** Running img2img on each frame separately flickers, because each frame is restyled on its
own. Fixing this is most of what makes real-time restyling look good, and measuring it keeps the
agent honest.

**You'll learn:** optical flow (how pixels move between frames), how we measure flicker, ControlNet
(guiding diffusion with depth/edges), and seed policy.

Tasks:
- [ ] **Spike first:** does SD 2.1 depth ControlNet work well with SD-Turbo? Fallback: SD 1.5 +
      LCM-LoRA + depth ControlNet (the StreamDiffusion path)
- [ ] Flicker metric (Farneback flow warp error, normalized against source frames)
- [ ] Structure fidelity (edge/depth agreement); style score (CLIP, low rate, on CPU)
- [ ] Seed policy, latent blending, or flow warping experiments
- [ ] **Keyframes + warping:** diffuse every Nth frame, and warp the last styled frame by optical
      flow in between. Cheap, smoother, and higher fps; `N` becomes another agent knob.
- [ ] Wire all of this into `get_metrics`

**Milestone:** measurable flicker reduction vs. baseline, shown in a plot.

**Read first:**
- OpenCV, [Optical Flow tutorial](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html):
  includes Farneback, which we'll use directly.
- Hugging Face diffusers docs, [ControlNet](https://huggingface.co/docs/diffusers/using-diffusers/controlnet):
  what "conditioning on depth" means in practice.
- OpenAI, [CLIP paper](https://arxiv.org/abs/2103.00020): read only the abstract and figure 1;
  the idea is "a shared space for images and text".

### Phase 5: Eval harness

**Why:** Without fair comparisons, "the agent works" is just an opinion. Replaying recorded Doom demos
gives every config and agent the *same* gameplay.

**You'll learn:** evaluation design, including baselines, controlled replays, and what to measure
beyond "it looks good".

Tasks:
- [ ] `BanditController` (UCB and/or Thompson sampling over preset configs)
- [ ] Record a few demo episodes; replay through every controller: fixed, random walk,
      hill-climb, bandit, LLM agent(s)
- [ ] Compare: composite score, budget violations, steps to converge, rollback count, cost
- [ ] Results table in the README

**Milestone:** a results table in the README.

**Read first:**
- Lilian Weng, [The Multi-Armed Bandit Problem and Its Solutions](https://lilianweng.github.io/posts/2018-01-23-multi-armed-bandit/):
  a clear intro to epsilon-greedy, UCB, and Thompson sampling. Read the first half.
- Wikipedia, [Hill climbing](https://en.wikipedia.org/wiki/Hill_climbing): the intro and the
  "local maxima" section are all you need.
- Anthropic docs, [Define success criteria and build evaluations](https://docs.claude.com/en/docs/test-and-evaluate/develop-tests):
  short and practical.

### Phase 6: Optimize + polish

**Why:** Speed and reach. Once we have measured, honest results, we make it faster and let it run
on other games.

Tasks:
- [ ] `torch.compile`; StreamDiffusion/TensorRT backend
- [ ] `WindowCaptureSource` (other games)
- [ ] Optional audio backend; adaptive stems first (see Later → Audio director)
- [ ] Demo GIF; README writeup (architecture, results, hardware comparison across laptops)

**Read first:**
- [StreamDiffusion repo](https://github.com/cumulo-autumn/StreamDiffusion): the README explains
  the pipeline tricks (batching denoise steps, caching) at a high level.

---

## v0.1 definition of done

Phases 0–3 working, config-driven, guard/rollback tests passing, README with a demo GIF and a
license note.

## Known risks

1. **Install friction:** StreamDiffusion/TensorRT on Windows is painful. Use plain `diffusers` first.
2. **VRAM (6 GB):** diffusion takes most of it. LLM via API, or local on CPU.
3. **Unverified performance:** published numbers are from an RTX 4090; trust calibration.
4. **Latency vs. playability:** the restyled view lags the game. Show this in metrics; demo on slow scenarios.
5. **Metric gaming:** an agent can "minimize flicker" by setting strength to zero. Use a composite
   objective with separately logged components.
6. **Python 3.13 wheel availability:** verified for the core stack (see Decisions). Re-check before adding Phase 6 backends.
7. **Licensing:** no WADs or weights in the repo; link the licenses.
8. **Scope creep:** anything not in v0.1 goes into a `later` issue.

## Big-picture reading (any time)

If you only read three things:
1. [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/): how the image side works.
2. [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents): how the agent side should be designed.
3. 3Blue1Brown, [Neural networks series](https://www.3blue1brown.com/topics/neural-networks): if
   "neural network", "training", or "transformer" still feel fuzzy, these videos are the best intuition builder.

## Later (not v0.1)

### Audio director (adaptive stems)

Extend the director from "what it looks like" to "what it sounds like", **without a generative
audio model**. Use a few pre-made or procedurally generated music layers (ambient, percussion,
tension), and let the agent mix them in response to the game:

- `AudioBackend` interface; `StubAudioBackend` first (logs the mix, plays nothing), then a real
  mixer.
- Tools: `audio.set_music_intensity(level)`, `audio.set_layer(name, on/off, gain)`, driven by
  `get_scene_info` (enemies near, health low).
- The same guard, cooldown, and rollback as the visual tools. That's the point: it exercises the
  control plane again in a new domain, needs no GPU, and is a standard technique in game audio
  engines ("adaptive" or "vertical layering" music).
- Rollback needs a different trigger. Audio doesn't affect fps, so "bad" means something like a
  mismatch between music and scene (calm music in a firefight) or layers changing too often.

**Gate:** build this only after Phase 5 shows whether the agent beats the simple baselines. If it
doesn't, a second domain just adds more of the same problem. The only thing we do earlier is
namespace tools by backend in Phase 1.

### Phase 7 (candidate): distilled fast pipeline

Live diffusion is slow on this laptop (~10–15 fps estimated). Use it as a **teacher**, offline,
and train a small fast **student** to copy it.

1. **Data:** replay Phase 5's recorded demos through SD-Turbo offline (speed doesn't matter) and
   save (game frame → styled frame) pairs.
2. **Train** a small image-to-image network (fast-neural-style / small U-Net class) on those
   pairs. Add a temporal-consistency loss so the student learns *not* to flicker.
3. **Serve** it as `DistilledPipeline`, just another `Pipeline` backend.
4. **Eval** it against live diffusion on fps, latency, flicker, and style/fidelity.

Trade-off: one student per style (or a few), so the agent picks between trained styles instead of
writing free-form prompts. It adds a new skill (training models), so it comes after v0.1, once
Phases 3 and 5 exist.

Read: PyTorch examples, [fast neural style](https://github.com/pytorch/examples/tree/main/fast_neural_style);
Richter et al., [Enhancing Photorealism Enhancement](https://arxiv.org/abs/2105.04619) (GTA V;
watch the videos, read figure 1).

### Offline texture-pack baking

Run diffusion once over Doom's textures and sprites to produce a restyled texture pack. The game
then renders it natively: full fps, no lag, no flicker. It looks less photoreal (geometry and
lighting stay Doom's), and swapping packs likely needs a level reload. The agent could pick
between pre-baked packs. This would be a separate offline tool, not a `Pipeline`.

### Policy distillation (cheap director)

Log the LLM agent's decisions across many runs (scene features → settings chosen), then train a
tiny classifier to imitate them. You get an instant, free, offline director, plus another baseline
between the bandit and the LLM in the eval table.

### Other ideas

- Prompt enemies differently from walls using the labels buffer
- Local LLM comparison (Ollama) in the eval harness
- Hardware comparison on the new laptop

### Hallucination mode (lowest priority)

Add things the player sees but the game doesn't know about: new walls, murals, enemies. The hard
part isn't drawing them; it's keeping them **anchored** as the camera moves. Approaches, in the
order to try them:

1. **Reskin existing objects** (the best first version). Use the labels buffer to restyle each
   enemy or object with its own prompt ("imps → neon jellyfish"). The game still tracks the
   object, so gameplay and visuals stay in sync. Extends "prompt enemies differently" above.
2. **Real placeholders.** Place plain objects in the map (SLADE / ZDoom scripting) that the game
   *does* register (collision, enemies), and let diffusion decide how they look. The engine
   handles anchoring.
3. **World-anchored fakes.** Pick a 3D point, project it to the screen each frame using player
   pose + depth buffer (with occlusion), then either inpaint that patch with a fixed seed or
   paste a once-generated image there and let the style pass blend it in.
4. **Prompt only** (a toy). High strength + "tentacles on the walls". Fun, but objects flicker
   and wander because every frame invents them from scratch.

A possible agent tool: `visual.add_hallucination(prompt, anchor)`, driven by scene info
("health low → walls close in"). This is creative direction, where an LLM should clearly beat
the numeric baselines. Needs Phase 4 (depth conditioning, temporal stability) first.
