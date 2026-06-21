# ax.md — AI & Agentic Tool Usage Reflection

**Team: Chole Bhhature | Bhavya Garg | IIIT Bangalore**
**Problem Statement 09: Occlusion-Aware 3D Scene Reconstruction**

---

## Overview

This document reflects honestly on how AI tools, open-weight models, agentic
workflows, and coding assistants were used throughout the development of
PHANTOM-ECHO REVEAL — including what worked, what failed, and what we would
do differently.

---

## 1. Open-Weight Models Used

> **Runtime AI compliance.** Every model in the evaluated runtime is
> open-weight (table below); the default runtime makes **no proprietary AI API
> call**. The `anthropic` package is not in `requirements.txt`, so no
> Claude/GPT call can execute in a clean install. An *optional, off-by-default*
> Claude planner exists in `src/agent/planner.py` purely to demonstrate the
> planner is policy-agnostic — it is never invoked by `reproduce.sh`, the test
> suite, or any demo mode (see README → "Runtime Compliance"). Claude was used
> as a *development* assistant and critic, reflected honestly in the commit
> co-author trailers and in Section 7 below.

### LLaVA-NeXT-Video (llava-hf/LLaVA-NeXT-Video-7B-hf)
**Role:** Vision-language scene understanding + prompt construction for generation.

LLaVA-NeXT-Video processes the visible RGB frame and generates a structured
natural-language description of the visible scene context. This description
is injected into the VideoScene generation prompt alongside physics bounds
from PHANTOM-LITE.

**What worked:** LLaVA's scene descriptions were remarkably accurate at
identifying semantic classes (chair, sofa, table) from partially occluded
views. It correctly inferred furniture arrangement from shadow patterns
in 7/10 test cases without any fine-tuning.

**What did NOT work:** LLaVA-NeXT-Video's video understanding mode
(multi-frame input) consistently hallucinated room dimensions. In 4/10
tests it claimed rooms were larger than they were, causing VideoScene
to generate geometry outside PHANTOM-LITE bounds. We fixed this by
stripping room-dimension claims from LLaVA output and replacing with
PHANTOM-LITE acoustic measurements.

**Key lesson:** VLMs are excellent at semantics and poor at metric geometry.
Treat their spatial estimates as soft priors, never ground truth.

---

### MobileSAM (dhkim2810/MobileSAM)
**Role:** Instance segmentation — isolates foreground objects from depth frame
to identify occluded region boundaries.

Runs entirely on-device (iPhone Neural Engine equivalent, simulated via
torch.jit on laptop). Latency: ~18ms per frame on M2 MacBook.

**What worked:** MobileSAM's masks were clean and stable across frames when
objects had clear edges (furniture against blank walls). Zero-shot performance
was sufficient — no fine-tuning needed.

**What did NOT work:** MobileSAM merged adjacent objects into single masks
when they had similar textures. A dark sofa against a dark wall was
consistently treated as one region, causing the acoustic bat-sonar to
produce one large sphere constraint instead of two separate ones. We
mitigated this by adding a depth-discontinuity-based boundary detector
as a secondary segmentation pass.

---

### MobileCLIP (apple/MobileCLIP-S2)
**Role:** Semantic classification — assigns WALL/FLOOR/CHAIR/TABLE labels
to each segmented region before routing through the Semantic Affordance Router.

**What worked:** MobileCLIP's zero-shot classification using text prompts
("a photo of a chair", "a photo of a wall") achieved 94.2% accuracy on
standard indoor furniture categories. This met our KPI target without
any fine-tuning.

**What did NOT work:** MobileCLIP confused monitors with paintings (both
flat rectangles on walls) in 3/10 test rooms. The fix was adding acoustic
depth as a disambiguation feature — a 5cm thin flat region (painting)
vs. a 30cm deep region (monitor) are acoustically distinguishable.

---

### VideoScene (stabilityai/stable-video-diffusion variant)
**Role:** Generates 3D Gaussian geometry for GREEN (IMAGINED) occluded regions.

**What worked:** VideoScene generated plausible furniture geometry when given
tight bounding box constraints and semantic class. Chair geometry was
particularly good — leg count, seat height, and backrest were structurally
correct in 8/10 cases.

**What did NOT work:** VideoScene without physics constraints generated
geometry that violated gravity in 60% of cases (floating chairs, tables
embedded in floors). This was the motivation for the PHANTOM-LITE
contradiction engine — physics constraints are now embedded in the
prompt, not applied post-hoc. After constraint injection, physics
violations dropped to 8% of cases.

**Biggest surprise:** VideoScene was significantly better at generating
wall geometry than furniture. Walls are simple flat planes — the model
gets these right almost always. We route wall generation through FAISS
retrieval instead of VideoScene anyway (cheaper, faster, deterministic),
but the observation was unexpected.

---

### SlotLSTM (custom, based on Slot Attention architecture)
**Role:** Structural constraint filter — validates generated furniture
satisfies physical affordance constraints before accepting it.

**What worked:** SlotLSTM reliably rejected impossible furniture configurations
(chair with 3 legs and no backrest, table with surface tilted 20 degrees).
The rejection rate was 23% of VideoScene outputs — meaning 23% of generated
geometry was physically impossible and would have been accepted without this filter.

**What did NOT work:** SlotLSTM was over-aggressive with non-standard furniture.
Bean bags, hammock chairs, and designer furniture with unusual forms were
rejected as "impossible" because they violated the standard affordance
constraints (seat_height: 0.38–0.55m). We added an "unusual furniture"
escape hatch that passes edge cases directly to VideoScene without
SlotLSTM filtering.

---

## 2. Agentic Workflows

### Planning Pipeline
We used Claude (Anthropic) as a planning and reasoning assistant throughout
the design process. The most valuable use was iterative architectural critique:
given a design, Claude identified physical law violations we had missed (e.g.,
the ISM cloud-latency flaw — Flaw 39 — was surfaced by Claude pointing out
that 3ms WiFi jitter = 1.03m depth error at 343 m/s).

The project bible itself (17 versions, 46 flaws fixed) was produced through
a human-AI collaborative loop: propose architecture → Claude critiques →
identify flaw → fix → repeat.

**What worked:** Using AI for architectural reasoning rather than code
generation. Claude was much more useful as a "what could go wrong" engine
than as a code writer.

**What did NOT work:** Asking Claude to generate the full acoustic math
from scratch. The SAS triangulation derivation required human review —
Claude's first attempt used a spherical coordinate system that collapsed
to a degenerate case when all phone positions were coplanar (common in
real walking paths). We caught this in testing and fixed with a baseline
check (Flaw 25 fix: minimum 5cm baseline required for valid triangulation).

---

### Tool Chaining
The pipeline uses the following tool chain automatically:

```
ARKit depth → MobileSAM segment → MobileCLIP classify
    → PHANTOM-LITE contradict → Affordance Router
    → [FAISS retrieve | SlotLSTM filter | VideoScene generate]
    → Bounds validate → Tag & output
```

This runs as a single Python pipeline (`src/main.py`) with no human
intervention required after initial scan. Each tool's output is the
next tool's input — classic linear agentic chain.

**What worked:** The linear chain was robust. Failures at one stage
(e.g., MobileSAM missing a boundary) degraded downstream accuracy
gracefully rather than causing hard failures.

**What did NOT work:** We attempted a more complex agentic loop where
VideoScene output was fed back into LLaVA-NeXT for quality assessment,
which would then re-prompt VideoScene if quality was poor. This
multi-agent loop consistently produced worse results than single-pass
generation — LLaVA's quality assessment was too noisy to reliably
improve generation, and each loop added 800ms of latency. We abandoned
multi-agent feedback loops and kept single-pass linear chains.

**Key lesson:** For time-critical reconstruction, simple linear tool
chains outperform complex agentic feedback loops. Latency of a feedback
loop (2–3 seconds per iteration) is unacceptable in a real-time system.

---

### Memory and Context Handling
The pipeline uses a Redis cache as "working memory" across the processing
of multiple frames from the same room:

- Previously measured acoustic points are cached (key: `room_id:acoustic_points`)
- DDGS Gaussian scene is cached (key: `room_id:gaussian_scene`)
- Contradiction engine results are cached per hypothesis (key: `room_id:hypothesis:region_id`)

Cache TTL: 300 seconds (5 minutes) — matching a typical scan session.

**What worked:** Caching acoustic measurements across multiple frames
significantly improved SAS triangulation accuracy. A single-frame acoustic
measurement gives 3–4 sphere constraints. Accumulated over 30 seconds of
walking, we get 40–60 constraints → triangulation residual drops from 8cm
to 1.2cm.

**What did NOT work:** We initially cached VideoScene outputs too
aggressively. When the user moved the phone and got new depth information
that updated the PHANTOM-LITE bounds for a region, the cached VideoScene
output violated the new bounds. Fix: cache invalidate VideoScene outputs
whenever PHANTOM-LITE bounds change by more than 10cm.

---

### Coding Assistants
Claude was used as a coding assistant for:
- Boilerplate reduction (argparse setup, dataclass definitions)
- Unit test scaffolding
- Debugging the SAS triangulation linear system (rank deficiency edge case)
- WebGL shader code for the viewer

GitHub Copilot was used for:
- scipy.optimize.least_squares call signature
- numpy broadcasting patterns in the contradiction engine

**What worked:** Both tools were excellent at standard library usage
and boilerplate. No manual scipy documentation lookups needed.

**What did NOT work:** Copilot suggested using `np.linalg.solve` instead
of `np.linalg.lstsq` for the SAS linear system. `np.linalg.solve` requires
a square matrix — our system is overdetermined (N-1 equations, 3 unknowns
for N>4 positions). Using `solve` would have silently truncated most
measurements. We caught this in code review.

**Key lesson:** AI coding assistants are overconfident about linear algebra.
Always verify matrix shape assumptions manually.

---

## 3. MCP Servers / External Integrations

No MCP servers were used in the deployed system — all inference runs
locally or on a self-hosted GPU server.

During development, we used:
- **GitHub MCP** (via Claude): automated commit message generation and
  PR description writing. Saved approximately 2 hours across 47 commits.
- **HuggingFace Hub API**: model weight download and versioning.
  All models are pinned to specific commit hashes in requirements.txt
  to ensure reproducibility.

---

## 4. What We Would Do Differently

1. **Start with physics, not models.** We spent the first 5 days trying to
   get VideoScene to generate plausible geometry without constraints. It
   failed. The physics-first approach (Prove → Measure → Imagine) only
   emerged after the 11th version. Starting there would have saved a week.

2. **Implement ISM edge-local from day one.** We lost 3 days debugging
   acoustic measurements before realising the WiFi jitter was the problem,
   not our chirp design. Cloud-based pyroomacoustics was our first
   implementation. The rule should be: anything timing-critical is
   edge-local, always.

3. **Less agentic, more deterministic.** The most reliable parts of the
   system are the deterministic ones: physics laws, ISM filter, SAS
   triangulation. The least reliable are the agentic ones: LLaVA quality
   assessment, multi-agent generation loops. For a real-time system,
   determinism beats flexibility.

4. **FAISS over RAG for structured retrieval.** We initially used a RAG
   pipeline (embed room description → retrieve similar floor plan → generate
   from template) for wall geometry. FAISS with direct geometric queries
   (width/height/depth) was 10x faster and more accurate. Semantic retrieval
   is the wrong tool for metric geometry.

---

## 5. Summary

The most important AI contribution to this project was not code generation
but **architectural reasoning** — using language models as critics to
identify physical impossibilities in proposed designs before implementation.
The 46 flaws fixed across 17 versions were almost all surfaced through
human-AI collaborative critique, not through testing failures.

The least useful AI contribution was **agentic feedback loops** — any
workflow where model output was fed back into another model for quality
assessment degraded results and added unacceptable latency.

The system that shipped is mostly deterministic (physics laws, acoustics,
ISM) with AI used only for the irreducible uncertainty: classifying
occluded object semantics and generating geometry in the final GREEN regions
that physics and acoustics genuinely cannot determine.


## 7. Model Reproducibility — Commit Hashes

All models are pinned to specific commit hashes in `requirements.txt`
and verified by `src/eval/check_model_hashes.py`. Run:

```bash
python -m src.eval.check_model_hashes
```

| Model | HuggingFace repo | Commit hash (first 12) |
|-------|-----------------|------------------------|
| MobileSAM | dhkim2810/MobileSAM | `a9b07f9c0c51` |
| MobileCLIP-S2 | apple/MobileCLIP-S2 | `4e0db7cb1ddb` |
| LLaVA-NeXT-Video-7B | llava-hf/LLaVA-NeXT-Video-7B-hf | `f42d64c890bf` |
| VideoScene (SVD-xt) | stabilityai/stable-video-diffusion-img2vid-xt | `5f8b3e2d1f0c` |

Pinning rationale: open-weight model repos can push new revisions that
change generation quality without changing the model ID. A judge running
our eval 2 months after submission would otherwise get different numbers.
Pinning guarantees exact weight identity.

---

---

## 8. Phase 2 — Agentic Coding for the Real-Time Layer

For Phase 2 we used an agentic coding assistant (Claude, in an
agent harness with shell + file tools) to build the real-time layer
(`src/realtime/`, `src/frontend/`) on top of the frozen v21 modules.

**What worked:**
- **Agent-driven integration testing found a real silent failure.** While
  wiring the engine, the agent ran the v21 pipeline end-to-end and noticed
  `SAS v2: ... → 0 triangulated points` in our own logs — the acoustic
  triangulation (a flagship novelty) had been a silent no-op in v21. The
  fix (BUG-V22-SAS, see technical.md) was developed test-first: the agent
  wrote a reproduction harness with known ground-truth targets, then
  iterated association gates until 20/20 triangulations passed with 0 ghosts.
- **Reusing modules instead of regenerating them.** We constrained the agent
  to import v21 modules unchanged. This kept one code path for eval and demo
  and avoided the classic LLM failure mode of subtly-divergent rewrites.

**What did NOT work:**
- **First SAS fix attempt was wrong.** The agent's initial nearest-neighbour
  association produced ghost points (~1.6m off) that *passed* the residual
  gate — planar-array mirror ambiguity, which neither we nor the agent
  anticipated until plotting the error distances. Lesson: agents iterate
  fast, but geometric degeneracies still need a human-legible hypothesis
  ("why exactly 1.6m?") to converge.
- **Browser-side code cannot be agent-verified headlessly** in our setup;
  the Three.js dashboard needed manual visual passes. We kept the frontend
  dependency-free (single HTML file, one CDN script) to minimise the
  untested surface.
