"""
PHANTOM-ECHO REVEAL — Agentic layer (Prove → Measure → Imagine as a tool-using agent)

This package turns the project's *philosophy* into an actual agentic workflow:
a planner reasons over each genuinely-unknown region and calls the pipeline's
own modules as TOOLS — physics (contradiction engine), acoustics (forward/inverse
DSP + SAS), generation (VideoScene), and next-best-view planning — deciding per
region whether to PROVE, MEASURE, IMAGINE, or hand off to the robot to EXPLORE.

Two planner policies, same tool surface:
  - DeterministicPolicy  — the Prove→Measure→Imagine decision procedure, runs
                           fully offline (no API key, reproducible). Default.
  - LLMPolicy            — optional Claude (claude-opus-4-8) planner that chooses
                           the next tool via forced tool-use. Enabled with
                           PHANTOM_AGENT_LLM=claude (+ ANTHROPIC_API_KEY); falls
                           back to the deterministic policy on any error.

Run:  python -m src.main --mode agent
"""

from src.agent.planner import run_agent, AgentResult  # noqa: F401
