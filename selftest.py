"""Offline self-test for Auto-RL.

Exercises everything that does NOT require the heavy ML stack (torch/transformers),
network, or a Cursor API key:

  * the domain-pack contract + GSM8K verifier and answer extraction,
  * the harness evaluate() plumbing using a mock generator,
  * bootstrap's validation (contract compliance, disjoint splits, verifier sanity),
  * the orchestrator's experiment runner (with a fake trainer) and its git
    keep/discard + ledger logic (in a throwaway temp repo).

Run:  python selftest.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Force offline + smoke before importing project modules.
os.environ.setdefault("AUTO_RL_OFFLINE", "1")
os.environ.setdefault("AUTO_RL_SCALE", "smoke")
os.environ.setdefault("AUTO_RL_BUDGET_SEC", "5")

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import harness  # noqa: E402

_PASS = 0
_FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# --------------------------------------------------------------------------- #


def test_domain_contract_and_verifier() -> None:
    print("\n[domain contract + verifier]")
    dom = harness.load_domain("gsm8k")
    train = dom.get_train_problems()
    evals = dom.get_eval_problems()
    check("train non-empty", len(train) > 0, str(len(train)))
    check("eval non-empty", len(evals) > 0, str(len(evals)))
    tp = {p["prompt"] for p in train}
    ep = {p["prompt"] for p in evals}
    check("train/eval disjoint", tp.isdisjoint(ep), f"overlap={len(tp & ep)}")

    check("verify accepts gold", dom.verify("72", "72"))
    check("verify rejects wrong", not dom.verify("73", "72"))
    check("verify handles commas", dom.verify("1,000", "1000"))
    check("verify rejects empty", not dom.verify("", "72"))
    check("extract '#### 72'", dom.extract_answer("blah\n#### 72") == "72")
    check("extract last number", dom.extract_answer("the answer is 42 dollars") == "42")
    check("format_prompt includes instructions",
          "####" in dom.format_prompt(evals[0]))


def test_evaluate_plumbing() -> None:
    print("\n[harness.evaluate with mock generator]")
    dom = harness.load_domain("gsm8k")
    evals = dom.get_eval_problems()

    def perfect(prompts):
        # Return the gold answer for each eval problem, in order.
        return [f"reasoning... #### {p['reference']}" for p in evals[: len(prompts)]]

    def useless(prompts):
        return ["I don't know" for _ in prompts]

    r_perfect = harness.evaluate(dom, generate_fn=perfect, limit=len(evals))
    check("perfect generator -> acc 1.0", abs(r_perfect.accuracy - 1.0) < 1e-9,
          f"acc={r_perfect.accuracy}")
    r_bad = harness.evaluate(dom, generate_fn=useless, limit=len(evals))
    check("useless generator -> acc 0.0", r_bad.accuracy == 0.0, f"acc={r_bad.accuracy}")


def test_bootstrap_validation() -> None:
    print("\n[bootstrap validation]")
    import bootstrap
    report = bootstrap.validate_domain_pack("gsm8k")
    check("gsm8k validates", report["n_train"] > 0 and report["n_eval"] > 0)
    check("verifier sanity all-accept-gold",
          all(c["accepts_gold"] for c in report["verifier_checks"]))
    check("verifier sanity all-reject-wrong",
          all(c["rejects_wrong"] for c in report["verifier_checks"]))
    check("no nonprogrammatic warnings", report["warnings"] == [], str(report["warnings"]))
    check("slugify", bootstrap.slugify("Be an EXPERT at unit conversions!") == "be_an_expert_at_unit_conversions")


def test_experiment_runner() -> None:
    print("\n[orchestrator.run_experiment with fake trainer]")
    import orchestrator
    fake = REPO_ROOT / "runs" / "_fake_trainer.py"
    fake.parent.mkdir(exist_ok=True)
    fake.write_text(
        "import json, os\n"
        "open(os.environ['AUTO_RL_METRICS'],'w').write("
        "json.dumps({'eval_acc': 0.5, 'steps': 1, 'train_reward': 0.5}))\n"
        "print('fake trainer ran')\n"
    )
    os.environ["AUTO_RL_TRAIN_CMD"] = f"{sys.executable} {fake}"
    try:
        out = orchestrator.run_experiment("gsm8k", "selftest")
        check("fake experiment ok", out["ok"], str(out))
        check("fake eval_acc==0.5", out["eval_acc"] == 0.5, str(out["eval_acc"]))
    finally:
        os.environ.pop("AUTO_RL_TRAIN_CMD", None)
        fake.unlink(missing_ok=True)

    # Failing trainer (no metrics) -> not ok.
    bad = REPO_ROOT / "runs" / "_bad_trainer.py"
    bad.write_text("import sys; sys.exit(3)\n")
    os.environ["AUTO_RL_TRAIN_CMD"] = f"{sys.executable} {bad}"
    try:
        out = orchestrator.run_experiment("gsm8k", "selftest_bad")
        check("failing experiment not ok", not out["ok"], str(out))
    finally:
        os.environ.pop("AUTO_RL_TRAIN_CMD", None)
        bad.unlink(missing_ok=True)


def test_git_keep_discard() -> None:
    print("\n[orchestrator git keep/discard + ledger in temp repo]")
    import orchestrator
    saved_root, saved_ledger = orchestrator.REPO_ROOT, orchestrator.LEDGER
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run(["git", "init", "-q", "."], cwd=tmp, check=True)
        subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=tmp, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=tmp, check=True)
        (tmp / "train_rl.py").write_text("VERSION = 0\n")
        orchestrator.REPO_ROOT = tmp
        orchestrator.LEDGER = tmp / "experiments.tsv"

        orchestrator.ensure_baseline_commit()
        base_commit = orchestrator.current_commit()

        # Keep: change + commit, then verify it stays after a checkout.
        (tmp / "train_rl.py").write_text("VERSION = 1\n")
        kept_commit = orchestrator.commit_train_file("keep v1")
        check("commit advanced HEAD", kept_commit != base_commit, f"{base_commit}->{kept_commit}")

        # Discard: change then revert -> file restored to committed v1.
        (tmp / "train_rl.py").write_text("VERSION = 2\n")
        orchestrator.revert_train_file()
        restored = (tmp / "train_rl.py").read_text()
        check("revert restores committed version", restored.strip() == "VERSION = 1", restored)

        orchestrator.append_ledger({
            "id": "x", "timestamp": "t", "parent_commit": base_commit,
            "eval_acc": "0.5000", "delta": "+0.5000", "kept": 1, "note": "test",
        })
        tail = orchestrator.ledger_tail()
        check("ledger header present", tail.startswith("id\t"), tail[:40])
        check("ledger row appended", "0.5000" in tail, tail)
    orchestrator.REPO_ROOT, orchestrator.LEDGER = saved_root, saved_ledger


def main() -> int:
    print("Auto-RL offline self-test (no torch / network / API key needed)")
    test_domain_contract_and_verifier()
    test_evaluate_plumbing()
    test_bootstrap_validation()
    test_experiment_runner()
    test_git_keep_discard()
    print(f"\n{'='*50}\nRESULT: {_PASS} passed, {_FAIL} failed\n{'='*50}")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
