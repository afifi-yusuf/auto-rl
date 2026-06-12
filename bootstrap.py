"""Phase 0: turn a plain-text domain into a verifiable RLVR "domain pack".

Usage:
    python bootstrap.py "be an expert at metric/imperial unit conversions"
    python bootstrap.py "high-school algebra: solve linear equations" --slug algebra

What it does:
  1. Runs a dedicated Cursor SDK agent that authors ``domains/<slug>/prepare.py``
     (implementing the harness domain-pack contract) and ``domain_card.json``,
     choosing the best ground-truth source per domain (HF dataset / programmatic
     generator / synthesis) and a PROGRAMMATIC verifier (no LLM-as-judge).
  2. Validates the generated pack: contract compliance, non-empty + disjoint
     train/eval splits, and verifier sanity (accepts gold answers, rejects wrong
     ones).
  3. Optionally measures the base model's baseline accuracy (headroom check).
  4. Shows you a summary and asks for approval, then locks the pack immutable.

If you don't have the SDK / API key, use ``--print-prompt`` to get a prompt you
can paste into a Cursor chat manually (then re-run with ``--validate-only``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import stat
import sys
from pathlib import Path

import harness
import agentlib

REPO_ROOT = Path(__file__).resolve().parent

# Import statements that would indicate a non-programmatic (LLM / network)
# verifier. Matched as actual imports so dataset ids like "openai/gsm8k" don't
# trip a false positive.
_FORBIDDEN_IMPORTS = re.compile(
    r"(?m)^\s*(?:import|from)\s+(openai|anthropic|cohere|requests|httpx|aiohttp|litellm|urllib)\b"
)


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (s[:40] or "domain").rstrip("_")


def contract_spec() -> str:
    return f"""
The domain pack lives in `domains/{{slug}}/` and consists of two files.

1) `domains/{{slug}}/prepare.py` - a Python MODULE exposing exactly this
   interface (module-level attributes + functions), matching `harness.Domain`:

   - NAME: str                      # short human-readable name
   - DESCRIPTION: str               # one paragraph
   - ANSWER_INSTRUCTIONS: str       # appended to every prompt so the model knows
                                    # how to format its final answer parseably
   - get_train_problems() -> list[dict]   # each: {{"prompt": str, "reference": <any>}}
   - get_eval_problems()  -> list[dict]   # HELD-OUT, disjoint from train
   - format_prompt(problem: dict) -> str  # question + ANSWER_INSTRUCTIONS
   - extract_answer(completion: str) -> str   # pull final answer from raw output
   - verify(model_answer: str, reference) -> bool   # DETERMINISTIC, PROGRAMMATIC

   Import `harness` for helpers (`harness.last_number`, `harness.numbers_equal`,
   `harness.scale()` for eval_limit/train_limit). Respect `harness.scale()`
   limits like the example does. Add `sys.path` insert of the repo root exactly
   as the example does so `import harness` works from any cwd.

2) `domains/{{slug}}/domain_card.json` - metadata with these keys:
   {json.dumps(harness.DOMAIN_CARD_SCHEMA, indent=2)}

   IMPORTANT: each entry in `sample_problems` must include "prompt",
   "reference", and "gold_answer", where `gold_answer` is a correct answer
   string such that `verify(gold_answer, reference)` returns True. The validator
   uses these to sanity-check your verifier.

Study the working reference implementation first: `domains/gsm8k/prepare.py` and
`domains/gsm8k/domain_card.json`.
""".strip()


def build_prompt(domain_text: str, slug: str) -> str:
    return f"""You are bootstrapping a new RLVR (RL with Verifiable Rewards) training
domain for an autonomous research system. The user wants a model that becomes an
expert at:

    "{domain_text}"

Create a self-contained, programmatically-verifiable domain pack with slug
`{slug}`.

REQUIREMENTS:
- Choose the BEST source of ground-truth problems+answers for this domain,
  in this order of preference when feasible:
    (a) a programmatic GENERATOR (e.g. for arithmetic/units/string puzzles) that
        produces unlimited fresh problems with exact answers,
    (b) an existing Hugging Face DATASET that fits well (load lazily; degrade to
        a small baked-in offline fallback set if `datasets` or network is
        unavailable, exactly like the gsm8k example),
    (c) SYNTHESIZED problems with reference answers that you hand-author.
- The verifier MUST be programmatic and deterministic: numeric/symbolic match
  (you may use `sympy`), exact/normalized string match, regex/schema checks, or
  unit-test execution. DO NOT call any LLM or network service inside `verify`.
- If this domain genuinely CANNOT be verified programmatically, DO NOT fake it.
  Instead create `domains/{slug}/FAILED_BOOTSTRAP.md` explaining why, and stop.
- Train and eval splits MUST be disjoint. Make eval a held-out set.
- Make `prepare.py` import cleanly with no heavy work at import time.

{contract_spec()}

Deliverables: write `domains/{slug}/prepare.py` and
`domains/{slug}/domain_card.json` now. After writing, briefly explain your source
choice and how the verifier works."""


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


class ValidationError(Exception):
    pass


def _scan_for_nonprogrammatic(slug: str) -> list[str]:
    src = (harness.domain_dir(slug) / "prepare.py").read_text()
    warnings = []
    for m in set(_FORBIDDEN_IMPORTS.findall(src)):
        warnings.append(f"prepare.py references {m!r} - verifier must be programmatic, not an LLM/network call")
    return warnings


def validate_domain_pack(slug: str) -> dict:
    """Run all structural + verifier checks. Returns a report; raises on hard fail."""
    domain_dir = harness.domain_dir(slug)
    if (domain_dir / "FAILED_BOOTSTRAP.md").exists():
        raise ValidationError(
            f"bootstrap reported failure; see {domain_dir / 'FAILED_BOOTSTRAP.md'}"
        )

    module = harness.load_domain(slug)  # also enforces the contract

    train = module.get_train_problems()
    evals = module.get_eval_problems()
    if not train:
        raise ValidationError("get_train_problems() returned nothing")
    if not evals:
        raise ValidationError("get_eval_problems() returned nothing")

    for name, probs in (("train", train), ("eval", evals)):
        for i, p in enumerate(probs[:50]):
            if not isinstance(p, dict) or "prompt" not in p or "reference" not in p:
                raise ValidationError(
                    f"{name} problem #{i} must be a dict with 'prompt' and 'reference'"
                )

    train_prompts = {p["prompt"] for p in train}
    eval_prompts = {p["prompt"] for p in evals}
    overlap = train_prompts & eval_prompts
    if overlap:
        raise ValidationError(
            f"train/eval splits overlap on {len(overlap)} prompt(s); eval must be held-out"
        )

    card_path = domain_dir / "domain_card.json"
    if not card_path.exists():
        raise ValidationError("domain_card.json missing")
    card = json.loads(card_path.read_text())

    # Verifier sanity using the card's labelled samples.
    samples = card.get("sample_problems") or []
    verifier_checks = []
    for s in samples[:8]:
        ref = s.get("reference")
        gold = s.get("gold_answer")
        if gold is None or ref is None:
            continue
        accepts_gold = bool(module.verify(str(gold), ref))
        rejects_wrong = not bool(module.verify("___definitely_wrong_answer___", ref))
        verifier_checks.append(
            {"gold": gold, "accepts_gold": accepts_gold, "rejects_wrong": rejects_wrong}
        )
    if not verifier_checks:
        raise ValidationError(
            "domain_card.sample_problems lacks gold_answer/reference pairs; "
            "cannot sanity-check the verifier"
        )
    if not all(c["accepts_gold"] for c in verifier_checks):
        raise ValidationError("verifier rejected a known-correct gold answer")
    if not all(c["rejects_wrong"] for c in verifier_checks):
        raise ValidationError("verifier accepted an obviously-wrong answer")

    return {
        "slug": slug,
        "card": card,
        "n_train": len(train),
        "n_eval": len(evals),
        "verifier_checks": verifier_checks,
        "warnings": _scan_for_nonprogrammatic(slug),
        "sample_problems": samples[:3],
    }


def measure_baseline(slug: str, limit: int = 8) -> float | None:
    """Evaluate the base (un-trained) model on the eval split. None if unavailable."""
    try:
        domain = harness.load_domain(slug)
        policy = harness.load_policy(eval_mode=True)
        res = harness.evaluate(domain, policy, limit=limit)
        return res.accuracy
    except Exception as exc:  # missing torch/model/network -> skip gracefully
        print(f"[bootstrap] baseline skipped: {exc}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- #
# Approval gate + lock
# --------------------------------------------------------------------------- #


def print_summary(report: dict, baseline: float | None) -> None:
    card = report["card"]
    bar = "=" * 70
    print(f"\n{bar}\nDOMAIN PACK: {report['slug']}\n{bar}")
    print(f"name            : {card.get('name')}")
    print(f"domain_text     : {card.get('domain_text')}")
    print(f"source          : {card.get('source_type')} - {card.get('source_detail')}")
    print(f"answer_format   : {card.get('answer_format')}")
    print(f"verifier        : {card.get('verifier_description')}")
    print(f"train / eval    : {report['n_train']} / {report['n_eval']} problems")
    print(f"baseline_acc    : {('%.3f' % baseline) if baseline is not None else 'n/a (not measured)'}")
    print("\nVerifier sanity checks:")
    for c in report["verifier_checks"]:
        print(f"  gold={c['gold']!r:<24} accepts_gold={c['accepts_gold']} rejects_wrong={c['rejects_wrong']}")
    print("\nSample problems:")
    for s in report["sample_problems"]:
        print(f"  - {str(s.get('prompt'))[:100]!r} -> gold={s.get('gold_answer')!r}")
    if report["warnings"]:
        print("\nWARNINGS:")
        for w in report["warnings"]:
            print(f"  ! {w}")
    if baseline is not None:
        if baseline <= 0.0:
            print("\nNOTE: baseline accuracy is ~0 - RL may struggle to find any signal.")
        elif baseline >= 0.9:
            print("\nNOTE: baseline accuracy is very high - little headroom for RL to show gains.")
    print(bar)


def lock_domain(slug: str, baseline: float | None) -> None:
    domain_dir = harness.domain_dir(slug)
    card_path = domain_dir / "domain_card.json"
    card = json.loads(card_path.read_text())
    card["baseline_accuracy"] = baseline
    card.setdefault("created_at", dt.datetime.utcnow().isoformat() + "Z")
    card_path.write_text(json.dumps(card, indent=2))

    harness.domain_lock_path(slug).write_text(
        json.dumps({"locked_at": dt.datetime.utcnow().isoformat() + "Z"}, indent=2)
    )
    # Make the pack read-only so accidental edits are obvious.
    for f in ("prepare.py", "domain_card.json"):
        p = domain_dir / f
        if p.exists():
            p.chmod(p.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    print(f"[bootstrap] LOCKED domain {slug!r}. Start research with:")
    print(f"    AUTO_RL_DOMAIN={slug} python orchestrator.py --iterations 50")


def approval_gate(report: dict, baseline: float | None, auto_yes: bool) -> bool:
    print_summary(report, baseline)
    if auto_yes or os.environ.get("AUTO_RL_YES") == "1":
        print("[bootstrap] auto-approving (--yes / AUTO_RL_YES).")
        return True
    try:
        ans = input("\nApprove and LOCK this domain pack for RL? [y/N] ").strip().lower()
    except EOFError:
        ans = ""
    return ans in {"y", "yes"}


# --------------------------------------------------------------------------- #
# Agent-driven generation with validate-and-fix retries
# --------------------------------------------------------------------------- #


def generate_with_agent(domain_text: str, slug: str, model: str, max_fix_attempts: int) -> None:
    prompt = build_prompt(domain_text, slug)
    with agentlib.open_agent(REPO_ROOT, model=model) as session:
        result = session.send(prompt, label=f"bootstrap:{slug}")
        if agentlib.result_failed(result):
            raise RuntimeError(f"bootstrap agent run errored: {getattr(result, 'id', '?')}")

        for attempt in range(1, max_fix_attempts + 1):
            try:
                validate_domain_pack(slug)
                print(f"[bootstrap] validation passed on attempt {attempt}.")
                return
            except (ValidationError, Exception) as exc:
                print(f"[bootstrap] validation failed (attempt {attempt}): {exc}", file=sys.stderr)
                if attempt == max_fix_attempts:
                    raise
                session.send(
                    f"The domain pack failed validation with this error:\n\n{exc}\n\n"
                    "Fix domains/{slug}/prepare.py and/or domain_card.json so it passes. "
                    "Remember: programmatic verifier only, train/eval disjoint, and "
                    "sample_problems must include gold_answer/reference pairs that "
                    "verify() accepts.".replace("{slug}", slug),
                    label=f"bootstrap-fix:{slug}#{attempt}",
                )


def main() -> int:
    ap = argparse.ArgumentParser(description="Bootstrap a verifiable RLVR domain pack.")
    ap.add_argument("domain_text", nargs="?", help="plain-text domain description")
    ap.add_argument("--slug", help="folder name under domains/ (default: derived)")
    ap.add_argument("--model", default=agentlib.DEFAULT_MODEL, help="Cursor agent model")
    ap.add_argument("--max-fix-attempts", type=int, default=3)
    ap.add_argument("--baseline-limit", type=int, default=8)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--yes", action="store_true", help="auto-approve the gate")
    ap.add_argument("--print-prompt", action="store_true",
                    help="print the agent prompt and exit (manual mode)")
    ap.add_argument("--validate-only", action="store_true",
                    help="skip generation; validate + gate an existing pack")
    args = ap.parse_args()

    if not args.domain_text and not args.validate_only:
        ap.error("domain_text is required unless --validate-only")

    slug = args.slug or (slugify(args.domain_text) if args.domain_text else None)
    if not slug:
        ap.error("--slug is required with --validate-only")

    if harness.is_locked(slug):
        print(f"[bootstrap] domain {slug!r} is already locked; nothing to do.")
        return 0

    if args.print_prompt:
        print(build_prompt(args.domain_text, slug))
        return 0

    if not args.validate_only:
        try:
            generate_with_agent(args.domain_text, slug, args.model, args.max_fix_attempts)
        except agentlib.AgentUnavailable as exc:
            print(f"\n[bootstrap] {exc}\n", file=sys.stderr)
            print("Tip: run with --print-prompt to do this manually in a Cursor chat,\n"
                  "then re-run with --validate-only.", file=sys.stderr)
            return 1

    try:
        report = validate_domain_pack(slug)
    except Exception as exc:
        print(f"[bootstrap] validation failed: {exc}", file=sys.stderr)
        return 2

    baseline = None if args.skip_baseline else measure_baseline(slug, args.baseline_limit)

    if approval_gate(report, baseline, args.yes):
        lock_domain(slug, baseline)
        return 0
    print("[bootstrap] not approved; pack left unlocked. Re-run when ready.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
