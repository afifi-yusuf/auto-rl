<!--
  program.md - instructions for the RL research agent.

  This is the file the HUMAN edits to steer the autonomous research (the agent
  edits train_rl.py; you edit this). The orchestrator injects the active domain
  card between the ACTIVE-DOMAIN markers below before each experiment. Treat the
  rest of this file as the "research org's" standing orders and iterate on it to
  make research progress faster.
-->

# Research program: RLVR post-training via GRPO

## Your job

You are running one experiment in an ongoing autonomous research loop. Each
experiment, you make **one focused change to `train_rl.py`** that you believe
will increase held-out **verified accuracy** (`eval_acc`) on the active domain
within a fixed wall-clock training budget. The harness then runs your edited
`train_rl.py`, evaluates it, and either keeps your change (if `eval_acc`
improved) or reverts it.

The metric is `eval_acc`: the fraction of held-out problems the trained model
answers correctly according to the domain's **programmatic verifier**. Higher is
better. It is computed by the harness on a held-out split you cannot see during
training, so you cannot game it.

## Hard rules

1. **Only edit `train_rl.py`.** Do not modify `harness.py`, anything under
   `domains/`, `orchestrator.py`, or this file. Those define the reward and the
   evaluation; changing them is cheating and the loop will discard it anyway.
2. **Keep `train_rl.py` runnable** end-to-end within the time budget. It must
   still load the domain + policy, train, evaluate, and write `metrics.json`
   with an `eval_acc` field.
3. **One concrete hypothesis per experiment.** Change a small number of related
   things so the result is interpretable, not a shotgun of unrelated tweaks.
4. **No reward hacking.** Do not special-case eval problems, hardcode answers,
   or call the verifier to leak labels into training inputs.

## How to work

1. Read the active domain card (below) and skim the latest rows of
   `experiments.tsv` to see what has already been tried and what helped or hurt.
2. Read the current `train_rl.py`. Form a hypothesis (e.g. "raise the learning
   rate", "increase group size for a stronger advantage estimate", "add a small
   KL penalty to stop the policy drifting", "lengthen `MAX_NEW_TOKENS` so
   answers aren't truncated", "add a format/length reward shaping term").
3. Make the edit. State the hypothesis in one sentence as your experiment note.
4. Hand back to the harness to run + score.

## Ideas worth exploring (not exhaustive)

- Learning rate and optimizer (AdamW betas/eps, warmup, schedule, Muon).
- Group size `G` and prompts-per-step (advantage-estimate variance vs throughput
  under the fixed budget).
- Sampling temperature / top-p for rollouts (exploration vs reward signal).
- KL coefficient to a frozen reference (stability vs progress).
- Entropy bonus to avoid premature collapse.
- Reward shaping on top of the binary verifier signal (format adherence, brevity)
  - but the headline metric is always the verifier's accuracy.
- Token budget: are completions getting truncated before the final answer?
- Filtering: skip groups with zero advantage; oversample hard prompts.

## Reading `experiments.tsv`

Tab-separated, append-only ledger of every experiment:
`id  timestamp  parent_commit  eval_acc  delta  kept  note`. `kept=1` rows are on
the main line of progress; `kept=0` rows were reverted. Use it as memory.

<!-- ACTIVE-DOMAIN:BEGIN -->
## Active domain: GSM8K grade-school math (`gsm8k`)

- Goal: make the model an expert at: Be an expert at grade-school math word problems (GSM8K).
- Source: hf_dataset - openai/gsm8k (config 'main'); ships with a small offline fallback set for development without network.
- Answer format: Final answer on its own line as '#### <number>'.
- Verifier: Programmatic numeric exact-match: extract the number after '####' (or the last number in the completion), strip commas, compare to the gold numeric answer with a 1e-6 tolerance. No LLM judgement.
- Baseline accuracy: None
<!-- ACTIVE-DOMAIN:END -->
