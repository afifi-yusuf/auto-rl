# Auto-RL

**[autoresearch](https://github.com/karpathy/autoresearch), but for RLVR.**

Karpathy's autoresearch lets an agent improve an LLM *pretraining* script overnight: edit the code, train for a few minutes, measure, keep what works, repeat. Auto-RL does the same for **RLVR post-training** (RL with Verifiable Rewards) — and takes it one step further: it doesn't just tune a fixed task, it **invents the task itself**.

You name a domain in plain text — *"be an expert at unit conversions"* — and a [Cursor](https://cursor.com/docs/sdk/python) agent writes the dataset and a programmatic, verifiable reward for it, locks them, then a second agent loop edits a from-scratch GRPO trainer and runs train → evaluate → keep/discard until held-out verified accuracy goes up. As in autoresearch, you don't touch the Python files; you program `program.md` (the agent's standing instructions) and name domains. The agents do the experiments.

Where autoresearch optimizes `val_bpb` on nanochat pretraining, Auto-RL optimizes **held-out verified accuracy** on a domain you invent in a sentence — the same recipe behind real frontier models (e.g. Cursor's Composer, RL-trained in coding environments with verifiable rewards).

## How it works

Two phases, both driven by Cursor Python SDK agents that read and write this repo's own source files:

**Phase 0 — bootstrap the task (`bootstrap.py`).** An agent turns your plain-text domain into a *domain pack*: `domains/<slug>/prepare.py` (problems + a deterministic `verify()`) and `domain_card.json`. It's validated (train/eval disjoint, verifier accepts gold and rejects wrong, baseline has headroom), you approve it, and it's **locked immutable**.

**Phase 1 — the research loop (`orchestrator.py`).** Each iteration an agent reads `program.md` + the experiment ledger and edits **one file**, `train_rl.py` (the GRPO trainer). The orchestrator runs it under a fixed budget, scores held-out accuracy with the locked verifier, and uses git to **commit improvements / revert regressions**.

Runs use a **fixed wall-clock budget** so experiments are comparable, and the metric is **held-out verified accuracy** (higher is better). Because the verifier + eval set are locked before training and the agent only ever edits `train_rl.py`, **the reward can't be gamed** — and it's always programmatic (no LLM-as-judge), so it stays true RLVR.

Both phases talk to agents through a thin [Cursor Python SDK](https://cursor.com/docs/sdk/python) wrapper (`agentlib.py`) — a local agent rooted at the repo, so it can read and edit the project's own files:

```97:107:agentlib.py
@contextmanager
def open_agent(cwd, model: str = DEFAULT_MODEL, api_key: Optional[str] = None) -> Iterator["AgentSession"]:
    """Open a persistent local agent rooted at ``cwd`` for multi-turn use."""
    Agent, CursorAgentError, LocalAgentOptions = _import_sdk()
    key = require_api_key(api_key)
    with Agent.create(
        model=model,
        api_key=key,
        local=LocalAgentOptions(cwd=str(cwd)),
    ) as agent:
        yield AgentSession(agent, CursorAgentError)
```

## Quick start

**Requirements:** Python 3.10–3.12, a Cursor API key (`CURSOR_API_KEY`), and a single NVIDIA GPU for `full`-scale runs (Apple Silicon / MPS works for `smoke`-scale dev).

```bash
# install (uv recommended)
uv venv && uv pip install -e .
# or: python -m venv .venv && . .venv/bin/activate && pip install torch transformers datasets accelerate sympy cursor-sdk

export CURSOR_API_KEY="cursor_..."        # or put it in .env (auto-loaded)

python selftest.py                        # offline sanity check: no GPU/network/API needed

# run one experiment by hand on a built-in domain
AUTO_RL_DOMAIN=gsm8k AUTO_RL_SCALE=smoke python train_rl.py
```

Then go autonomous — invent a domain and let the loop run:

```bash
python bootstrap.py "be an expert at unit conversions" --slug units
AUTO_RL_DOMAIN=units python orchestrator.py --iterations 50
```

A real `units` run, smoke scale, on a Mac:

```
[bootstrap] baseline_acc = 0.750  ->  approved + LOCKED
[orchestrator] baseline eval_acc=0.7500
== iter 1 == "graded near-miss reward within 5%"     -> 0.7500  reverted
== iter 2 == "increase GRPO group size 8 -> 16"      -> 0.8750  KEPT
```

No API key handy? Run `python bootstrap.py "your domain" --print-prompt` to do it in a Cursor chat, then `--validate-only` to validate + approve.

## Project structure

```
harness.py                  — fixed infra + domain-pack contract (do not modify)
train_rl.py                 — GRPO training loop (agent modifies this)
program.md                  — agent instructions (human edits this)
bootstrap.py                — Phase 0: plain-text domain -> locked domain pack
orchestrator.py             — Phase 1: autonomous research loop
agentlib.py                 — Cursor SDK wrapper (lifecycle, streaming, errors)
selftest.py                 — offline self-test (no torch/network/API)
experiments.tsv             — append-only experiment ledger
domains/<slug>/prepare.py   — problems + programmatic verify() (locked after bootstrap)
domains/<slug>/domain_card.json
pyproject.toml              — dependencies
```

Built-in domains: **`gsm8k`** (math), **`units`** (generated unit conversions), **`polymarket`** (forecast resolved prediction markets, YES/NO).

## Design choices

- **Single file to modify.** The agent only touches `train_rl.py` — manageable scope, reviewable diffs.
- **Locked, programmatic reward.** The verifier + eval set are generated, validated, and frozen before training; the agent can't reach them, so the reward can't be hacked. No LLM-as-judge.
- **Fixed time budget.** Every experiment runs for the same wall-clock time, so changes (group size, learning rate, reward shaping, ...) are directly comparable.
- **Two scales, one code path.** `smoke` for laptop/MPS dev, `full` for a single NVIDIA GPU; only sizes and the budget change.

| | `smoke` (Mac dev) | `full` (cloud GPU) |
| --- | --- | --- |
| eval problems | 8 | 200 |
| max new tokens | 160 | 512 |
| group size / prompts-per-step | 4 / 2 | 8 / 8 |
| per-experiment budget | 120 s | 900 s |

## Configuration

Environment variables (all optional except the API key for the agent loops):

| Var | Default | Meaning |
| --- | --- | --- |
| `AUTO_RL_DOMAIN` | `gsm8k` | active domain slug under `domains/` |
| `AUTO_RL_SCALE` | `smoke` | `smoke` or `full` preset |
| `AUTO_RL_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | base model id |
| `AUTO_RL_BUDGET_SEC` | scale preset | per-experiment wall-clock budget |
| `AUTO_RL_OFFLINE` | unset | force offline dataset fallback |
| `CURSOR_API_KEY` | — | Cursor SDK auth (required for the agent loops) |

Device is auto-detected (`cuda` → `mps` → `cpu`).

## Caveats

This is a research scaffold, not a production trainer. The 0.5B default at `smoke` scale (8 eval problems) is built to be cheap to iterate on; results are coarse — use `full` scale on a GPU with more iterations for meaningful gains. The main risk is bootstrap quality: a weak auto-generated dataset or an unlearnable verifier caps everything downstream, which is why the validation, baseline-headroom check, and approval gate exist. Read the domain card before approving.

## License

MIT
