"""train_rl.py - the single file the research agent edits.

From-scratch GRPO (Group Relative Policy Optimization) for RLVR post-training.
The agent loads the active domain pack (problems + programmatic verifier) via the
fixed ``harness`` module, samples groups of completions, scores them with the
domain's verifier, turns those rewards into group-normalized advantages, and
takes a policy-gradient step. It trains until the wall-clock budget runs out,
then evaluates held-out verified accuracy and writes ``metrics.json``.

EVERYTHING in the EDITABLE KNOBS block (and indeed this whole file) is fair game
for the agent: optimizer, learning rate, group size, sampling temperature, KL and
entropy coefficients, reward shaping, batch composition, etc. The only hard rules:

  * Do NOT modify harness.py or anything under domains/ (that is the reward; the
    research loop enforces this by only committing changes to this file).
  * Keep writing ``eval_acc`` to the metrics file - it is the optimization target.

Run directly for a single experiment:

    AUTO_RL_DOMAIN=gsm8k AUTO_RL_SCALE=smoke python train_rl.py
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import harness

# ======================= EDITABLE KNOBS (agent tunes these) ================= #
SEED = 0
LEARNING_RATE = 1e-6
GROUP_SIZE = None          # None -> use scale preset (G completions per prompt)
PROMPTS_PER_STEP = None    # None -> use scale preset
SAMPLING_TEMPERATURE = 1.0
MAX_NEW_TOKENS = None      # None -> use scale preset
KL_COEF = 0.0              # >0 loads a frozen reference model for a KL penalty
ENTROPY_COEF = 0.0
MAX_GRAD_NORM = 1.0
ADVANTAGE_EPS = 1e-4       # std floor when normalizing group advantages
# ============================================================================ #

METRICS_PATH = Path(os.environ.get("AUTO_RL_METRICS", "metrics.json"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_knobs() -> dict:
    sc = harness.scale()
    return {
        "group_size": GROUP_SIZE or int(sc["group_size"]),
        "prompts_per_step": PROMPTS_PER_STEP or int(sc["prompts_per_step"]),
        "max_new_tokens": MAX_NEW_TOKENS or int(sc["max_new_tokens"]),
    }


def token_logprobs(model, input_ids, attention_mask, gen_start):
    """Per-token log-probs of the *generated* tokens.

    Returns (logp[B, T-1], gen_mask[B, T-1], entropy[B, T-1]) where gen_mask is 1
    on generated, non-padding positions.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    logp_all = F.log_softmax(logits, dim=-1)
    token_logp = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    entropy = -(logp_all.exp() * logp_all).sum(-1)

    seq_len = input_ids.shape[1]
    pos = torch.arange(seq_len - 1, device=input_ids.device).unsqueeze(0)
    gen_region = pos >= (gen_start - 1)  # targets are shifted by one
    not_pad = attention_mask[:, 1:].bool()
    gen_mask = (gen_region & not_pad).float()
    return token_logp, gen_mask, entropy


def main() -> None:
    set_seed(SEED)
    knobs = resolve_knobs()
    budget = harness.budget_sec()

    domain = harness.load_domain()
    policy = harness.load_policy()
    model, tok, device = policy.model, policy.tokenizer, policy.device
    print(f"[train_rl] domain={domain.NAME!r} device={device} model={policy.model_id}")
    print(f"[train_rl] knobs={knobs} budget={budget}s kl={KL_COEF} lr={LEARNING_RATE}")

    ref_model = None
    if KL_COEF > 0:
        ref_model = harness.load_policy(eval_mode=True).model
        for p in ref_model.parameters():
            p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_problems = domain.get_train_problems()
    if not train_problems:
        raise RuntimeError("domain returned no training problems")

    reward_history: list[float] = []
    step = 0
    start = time.time()
    rng = random.Random(SEED)

    while time.time() - start < budget:
        batch = rng.sample(train_problems, min(knobs["prompts_per_step"], len(train_problems)))
        optimizer.zero_grad(set_to_none=True)
        step_rewards: list[float] = []
        n_groups = 0

        for problem in batch:
            user_text = domain.format_prompt(problem)
            input_text = harness.build_input_text(tok, user_text)
            enc = tok(input_text, return_tensors="pt").to(device)
            prompt_len = enc["input_ids"].shape[1]

            model.eval()
            with torch.no_grad():
                seqs = model.generate(
                    **enc,
                    max_new_tokens=knobs["max_new_tokens"],
                    do_sample=True,
                    temperature=SAMPLING_TEMPERATURE,
                    top_p=1.0,
                    num_return_sequences=knobs["group_size"],
                    pad_token_id=tok.pad_token_id,
                )
            model.train()

            completions = tok.batch_decode(seqs[:, prompt_len:], skip_special_tokens=True)
            rewards = torch.tensor(
                [
                    1.0 if domain.verify(domain.extract_answer(c), problem["reference"]) else 0.0
                    for c in completions
                ],
                device=device,
            )
            step_rewards.extend(rewards.tolist())

            # Group-relative advantages.
            adv = rewards - rewards.mean()
            std = rewards.std()
            if std > ADVANTAGE_EPS:
                adv = adv / (std + ADVANTAGE_EPS)
            if torch.allclose(adv, torch.zeros_like(adv)):
                continue  # whole group same reward -> no learning signal

            attn = (seqs != tok.pad_token_id).long()
            attn[:, :prompt_len] = 1
            logp, gen_mask, entropy = token_logprobs(model, seqs, attn, prompt_len)
            denom = gen_mask.sum(1).clamp(min=1.0)

            pg = -(adv.unsqueeze(1) * logp * gen_mask).sum(1) / denom
            loss = pg.mean()

            if ENTROPY_COEF > 0:
                ent = (entropy * gen_mask).sum(1) / denom
                loss = loss - ENTROPY_COEF * ent.mean()

            if ref_model is not None:
                with torch.no_grad():
                    ref_logp, _, _ = token_logprobs(ref_model, seqs, attn, prompt_len)
                # k3 KL estimator (Schulman): exp(r) - r - 1, r = ref - policy.
                r = (ref_logp - logp).clamp(-20, 20)
                kl = (torch.exp(r) - r - 1.0)
                kl_seq = (kl * gen_mask).sum(1) / denom
                loss = loss + KL_COEF * kl_seq.mean()

            loss.backward()
            n_groups += 1

        if n_groups > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        if step_rewards:
            reward_history.extend(step_rewards)
        step += 1
        recent = reward_history[-64:]
        mean_r = sum(recent) / len(recent) if recent else 0.0
        print(
            f"[train_rl] step={step} groups={n_groups} "
            f"step_reward={sum(step_rewards)/max(len(step_rewards),1):.3f} "
            f"run_reward={mean_r:.3f} elapsed={time.time()-start:.0f}s",
            flush=True,
        )

    # ---- Final held-out evaluation (the experiment's score) ---------------- #
    print("[train_rl] training budget exhausted; running held-out evaluation...")
    policy.model.eval()
    result = harness.evaluate(domain, policy)
    train_reward = sum(reward_history) / len(reward_history) if reward_history else 0.0

    metrics = {
        "eval_acc": result.accuracy,
        "eval_n": result.n,
        "train_reward": train_reward,
        "steps": step,
        "domain": harness.active_domain_slug(),
        "scale": harness.scale_name(),
        "knobs": {
            "learning_rate": LEARNING_RATE,
            "group_size": knobs["group_size"],
            "prompts_per_step": knobs["prompts_per_step"],
            "temperature": SAMPLING_TEMPERATURE,
            "max_new_tokens": knobs["max_new_tokens"],
            "kl_coef": KL_COEF,
            "entropy_coef": ENTROPY_COEF,
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"[train_rl] eval_acc={result.accuracy:.4f} ({result.n_correct}/{result.n})")
    print(f"[train_rl] wrote {METRICS_PATH}")


if __name__ == "__main__":
    main()
