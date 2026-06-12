# Auto-RL

**Autoresearch, but for RLVR post-training.** You name a domain in plain text;
the system bootstraps a *verifiable reward* for it, then an autonomous loop drives
a Cursor agent to repeatedly edit a from-scratch GRPO trainer to make a small base
model an expert at that domain. You wake up to a log of experiments and (hopefully)
a better model.

This is the RL analogue of [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
Where autoresearch optimizes `val_bpb` on nanochat pretraining, Auto-RL optimizes
**held-out verified accuracy** on a domain you choose, using RL with Verifiable
Rewards (RLVR) and group-relative policy optimization (GRPO).

## The idea

```
You: "be an expert at metric/imperial unit conversions"
        │
        ▼
Phase 0  bootstrap.py  ── a Cursor agent writes a domain pack: problems + a
                          PROGRAMMATIC verifier, validated, then you approve it
                          and it is LOCKED immutable.
        │
        ▼
Phase 1  orchestrator.py ── loop: a Cursor agent edits train_rl.py → run GRPO for
                            a fixed budget → eval held-out accuracy → git keep if
                            better, revert if not → repeat.
```

The orchestrator owns the experiment harness; the agent is only the "mutation
function" that edits a single file. The reward (verifier + eval set) is locked and
the agent only ever edits `train_rl.py`, so the reward cannot be gamed.

## Files

| File | Role | Who edits it |
| --- | --- | --- |
| `harness.py` | Fixed infra: scale/device config, model loading, generation, `evaluate()`, the domain-pack contract. | nobody |
| `domains/<slug>/prepare.py` | A domain pack: problems + programmatic `verify()`. Generated in Phase 0, then locked. | bootstrap agent (then frozen) |
| `domains/<slug>/domain_card.json` | Domain metadata (source, samples, baseline). | bootstrap agent |
| `train_rl.py` | From-scratch GRPO training loop. **The one file the research agent edits.** | RL research agent |
| `program.md` | Standing instructions for the research agent. | **you** (the human) |
| `bootstrap.py` | Phase 0: plain-text domain → validated, approved, locked domain pack. | — |
| `orchestrator.py` | Phase 1: the autonomous research loop (Cursor SDK). | — |
| `experiments.tsv` | Append-only ledger of every experiment. | orchestrator |
| `selftest.py` | Offline self-test (no torch/network/API needed). | — |

A built-in **GSM8K** domain ships in `domains/gsm8k/` so the repo is runnable
before you bootstrap anything custom.

## Requirements

- **Python 3.10–3.12** for the full ML stack (PyTorch wheels currently top out
  around 3.12). The offline self-test runs on any 3.10+.
- A single **NVIDIA GPU** (H100/A100/4090) for real `full`-scale runs; **Apple
  Silicon (MPS)** works for `smoke`-scale development.
- A **Cursor API key** for the autonomous agent loop (`CURSOR_API_KEY`).
- [uv](https://docs.astral.sh/uv/) recommended (or plain `pip`).

## Setup

```bash
# with uv (recommended)
uv venv && uv pip install -e .

# or plain pip (use a 3.10-3.12 interpreter)
python -m venv .venv && . .venv/bin/activate
pip install torch transformers datasets accelerate sympy cursor-sdk

export CURSOR_API_KEY="cursor_..."   # from https://cursor.com/dashboard/integrations
```

Verify the wiring with the offline self-test (no heavy deps required):

```bash
python selftest.py
```

## Quick start (built-in GSM8K domain)

`smoke` scale is tiny and runs on a Mac; `full` is for a GPU.

```bash
# Run a single training experiment by hand (sanity check the trainer)
AUTO_RL_DOMAIN=gsm8k AUTO_RL_SCALE=smoke python train_rl.py

# Kick off the autonomous research loop
AUTO_RL_DOMAIN=gsm8k AUTO_RL_SCALE=smoke python orchestrator.py --iterations 20 --allow-unlocked
```

(`--allow-unlocked` lets you run the shipped GSM8K domain without going through the
bootstrap/lock gate; custom domains should be locked first.)

## Bootstrapping a new domain from plain text

```bash
python bootstrap.py "be an expert at metric and imperial unit conversions"
```

This will:
1. Run a Cursor agent that authors `domains/<slug>/prepare.py` + `domain_card.json`,
   choosing the best ground-truth source (a programmatic generator, an existing
   Hugging Face dataset, or hand-synthesized problems) and a **programmatic**
   verifier. If the domain can't be verified programmatically, it stops and tells
   you why (no LLM-judge fallback — this keeps it true RLVR).
2. **Validate** the pack: contract compliance, disjoint train/eval splits, and a
   verifier sanity check (accepts gold answers, rejects wrong ones).
3. Optionally measure the base model's **baseline accuracy** (headroom check).
4. Show you a summary and ask for **approval**, then **lock** the pack immutable.

Then run the loop against it:

```bash
AUTO_RL_DOMAIN=<slug> python orchestrator.py --iterations 50
```

No SDK / API key handy? Generate the prompt and do it manually in a Cursor chat:

```bash
python bootstrap.py "your domain" --print-prompt   # paste into a Cursor chat
python bootstrap.py "your domain" --validate-only   # then validate + approve
```

## Mac dev vs cloud GPU

Everything is driven by two env vars; the code path is identical, only sizes and
the time budget change:

| | `AUTO_RL_SCALE=smoke` (Mac dev) | `AUTO_RL_SCALE=full` (cloud GPU) |
| --- | --- | --- |
| eval problems | 8 | 200 |
| max new tokens | 160 | 512 |
| group size / prompts-per-step | 4 / 2 | 8 / 8 |
| per-experiment budget | 120 s | 900 s |

```bash
# On a rented single NVIDIA GPU:
export CURSOR_API_KEY="cursor_..."
AUTO_RL_SCALE=full AUTO_RL_DOMAIN=gsm8k python orchestrator.py --iterations 100
```

Device is auto-detected: `cuda` → `mps` → `cpu`. Override the base model with
`AUTO_RL_MODEL` (default `Qwen/Qwen2.5-0.5B-Instruct`) and the budget with
`AUTO_RL_BUDGET_SEC`.

## How keep/discard works

The orchestrator commits **only `train_rl.py`** when `eval_acc` improves and reverts
**only `train_rl.py`** otherwise. `harness.py` and `domains/` are never touched by
the loop, so the reward stays honest. Every experiment is appended to
`experiments.tsv` (`id  timestamp  parent_commit  eval_acc  delta  kept  note`).

## Environment variables

| Var | Default | Meaning |
| --- | --- | --- |
| `AUTO_RL_DOMAIN` | `gsm8k` | active domain slug under `domains/` |
| `AUTO_RL_SCALE` | `smoke` | `smoke` or `full` preset |
| `AUTO_RL_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | base model id |
| `AUTO_RL_BUDGET_SEC` | scale preset | per-experiment wall-clock budget |
| `AUTO_RL_METRICS` | `metrics.json` | where `train_rl.py` writes results |
| `AUTO_RL_OFFLINE` | unset | force offline dataset fallback |
| `CURSOR_API_KEY` | — | Cursor SDK auth (required for the agent loops) |

## Caveats

- This is a research scaffold, not a production trainer. The 0.5B default is chosen
  so the loop is cheap to iterate on.
- Bootstrap quality is the main risk: a weak auto-generated dataset or verifier
  caps everything downstream. The validation step + approval gate are the
  mitigations — read the domain card before approving.

## License

MIT
