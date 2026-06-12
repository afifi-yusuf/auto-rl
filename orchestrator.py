"""Phase 1: the autonomous RLVR research loop.

This is the harness that owns every experiment. The Cursor agent is used purely
as the "mutation function" that proposes one edit to ``train_rl.py`` per
iteration; the orchestrator runs the experiment under a fixed time budget,
evaluates it (via the immutable domain pack), and keeps or reverts the change
with git. Repeat for ``--iterations`` (default: loop a long time, like an
overnight autoresearch run).

    AUTO_RL_DOMAIN=gsm8k AUTO_RL_SCALE=smoke python orchestrator.py --iterations 20

The agent only ever edits ``train_rl.py``: the orchestrator commits *that file
only* on success and reverts *that file only* on failure, so the reward
(``harness.py`` + ``domains/``) cannot be tampered with.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import harness
import agentlib

REPO_ROOT = Path(__file__).resolve().parent
TRAIN_FILE = "train_rl.py"
PROGRAM_FILE = REPO_ROOT / "program.md"
LEDGER = REPO_ROOT / "experiments.tsv"
RUNS_DIR = REPO_ROOT / "runs"
LEDGER_HEADER = "id\ttimestamp\tparent_commit\teval_acc\tdelta\tkept\tnote\n"


# --------------------------------------------------------------------------- #
# git helpers
# --------------------------------------------------------------------------- #


def git(*args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def current_commit() -> str:
    return git("rev-parse", "--short", "HEAD", check=False) or "none"


def ensure_baseline_commit() -> None:
    """Make sure the repo has a clean committed state so reverts work."""
    has_head = git("rev-parse", "--verify", "HEAD", check=False)
    dirty = git("status", "--porcelain", check=False)
    if not has_head or dirty:
        git("add", "-A")
        # Don't fail if there's genuinely nothing to commit.
        proc = subprocess.run(
            ["git", "commit", "-m", "auto-rl: baseline state"],
            cwd=REPO_ROOT, text=True, capture_output=True,
        )
        if proc.returncode != 0 and "nothing to commit" not in (proc.stdout + proc.stderr):
            raise RuntimeError(f"baseline commit failed: {proc.stderr.strip()}")
    print(f"[orchestrator] baseline commit: {current_commit()}")


def commit_train_file(msg: str) -> str:
    git("add", TRAIN_FILE)
    subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, text=True, capture_output=True)
    return current_commit()


def revert_train_file() -> None:
    git("checkout", "--", TRAIN_FILE)


# --------------------------------------------------------------------------- #
# program.md domain injection + ledger
# --------------------------------------------------------------------------- #


def inject_domain_card(slug: str) -> str:
    card_path = harness.domain_dir(slug) / "domain_card.json"
    card = json.loads(card_path.read_text()) if card_path.exists() else {}
    block = (
        f"## Active domain: {card.get('name', slug)} (`{slug}`)\n\n"
        f"- Goal: make the model an expert at: {card.get('domain_text', slug)}\n"
        f"- Source: {card.get('source_type')} - {card.get('source_detail')}\n"
        f"- Answer format: {card.get('answer_format')}\n"
        f"- Verifier: {card.get('verifier_description')}\n"
        f"- Baseline accuracy: {card.get('baseline_accuracy')}\n"
    )
    text = PROGRAM_FILE.read_text()
    begin, end = "<!-- ACTIVE-DOMAIN:BEGIN -->", "<!-- ACTIVE-DOMAIN:END -->"
    if begin in text and end in text:
        pre = text.split(begin)[0]
        post = text.split(end)[1]
        text = f"{pre}{begin}\n{block}{end}{post}"
        PROGRAM_FILE.write_text(text)
    return block


def ledger_tail(n: int = 20) -> str:
    if not LEDGER.exists():
        return "(no experiments yet)"
    lines = LEDGER.read_text().splitlines()
    return "\n".join(lines[:1] + lines[-n:]) if len(lines) > 1 else lines[0]


def append_ledger(row: dict) -> None:
    if not LEDGER.exists():
        LEDGER.write_text(LEDGER_HEADER)
    note = str(row.get("note", "")).replace("\t", " ").replace("\n", " ")[:200]
    line = (
        f"{row['id']}\t{row['timestamp']}\t{row['parent_commit']}\t"
        f"{row['eval_acc']}\t{row['delta']}\t{row['kept']}\t{note}\n"
    )
    with LEDGER.open("a") as f:
        f.write(line)


# --------------------------------------------------------------------------- #
# Running one experiment (train_rl.py under a fixed budget)
# --------------------------------------------------------------------------- #


def run_experiment(slug: str, exp_id: str) -> dict:
    """Run the current train_rl.py as a subprocess. Returns a result dict."""
    RUNS_DIR.mkdir(exist_ok=True)
    metrics_path = RUNS_DIR / f"metrics_{exp_id}.json"
    if metrics_path.exists():
        metrics_path.unlink()
    log_path = RUNS_DIR / f"log_{exp_id}.txt"

    env = dict(os.environ)
    env["AUTO_RL_DOMAIN"] = slug
    env["AUTO_RL_METRICS"] = str(metrics_path)

    budget = harness.budget_sec()
    grace = int(os.environ.get("AUTO_RL_STARTUP_GRACE", "900"))  # startup+compile+eval
    timeout = budget + grace
    train_cmd = os.environ.get("AUTO_RL_TRAIN_CMD", f"{sys.executable} {TRAIN_FILE}")

    start = time.time()
    timed_out = False
    with log_path.open("w") as logf:
        try:
            proc = subprocess.run(
                train_cmd, shell=True, cwd=REPO_ROOT, env=env,
                stdout=logf, stderr=subprocess.STDOUT, timeout=timeout,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -1

    tail = "\n".join(log_path.read_text().splitlines()[-40:]) if log_path.exists() else ""
    metrics = None
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            metrics = None

    ok = (not timed_out) and returncode == 0 and metrics is not None and "eval_acc" in metrics
    return {
        "ok": ok,
        "returncode": returncode,
        "timed_out": timed_out,
        "metrics": metrics,
        "log_tail": tail,
        "elapsed": time.time() - start,
        "eval_acc": (metrics or {}).get("eval_acc"),
    }


# --------------------------------------------------------------------------- #
# Agent prompts
# --------------------------------------------------------------------------- #


def propose_prompt(slug: str, best_acc: float) -> str:
    program = PROGRAM_FILE.read_text()
    return f"""{program}

---
RECENT EXPERIMENT LOG (experiments.tsv):
{ledger_tail()}

Current best eval_acc to beat: {best_acc:.4f}

Now propose and apply ONE concrete change to `{TRAIN_FILE}` that you hypothesize
will increase eval_acc. Edit ONLY `{TRAIN_FILE}`. Keep it runnable within the
time budget. End your message with a single line:
NOTE: <one-sentence description of the change and hypothesis>"""


def repair_prompt(result: dict) -> str:
    reason = "timed out" if result["timed_out"] else f"exited with code {result['returncode']}"
    return f"""Your edit to `{TRAIN_FILE}` failed: the run {reason} and did not produce a
valid metrics.json with eval_acc. Here is the tail of the log:

{result['log_tail']}

Fix `{TRAIN_FILE}` so it runs end-to-end within the budget and writes eval_acc.
Edit ONLY `{TRAIN_FILE}`."""


def extract_note(result_text: str) -> str:
    for line in (result_text or "").splitlines()[::-1]:
        if line.strip().upper().startswith("NOTE:"):
            return line.strip()[5:].strip()
    return "(no note)"


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #


def run_iteration(session, slug: str, best_acc: float, iteration: int, max_repairs: int) -> dict:
    exp_id = f"{int(time.time())}_{iteration}"
    parent = current_commit()

    result_text = ""
    res = session.send(propose_prompt(slug, best_acc), label=f"propose#{iteration}")
    result_text = getattr(res, "result", "") or ""
    if agentlib.result_failed(res):
        print(f"[orchestrator] agent run errored; skipping iteration {iteration}")
        revert_train_file()
        return {"best_acc": best_acc, "kept": False}

    outcome = run_experiment(slug, exp_id)
    repairs = 0
    while not outcome["ok"] and repairs < max_repairs:
        repairs += 1
        print(f"[orchestrator] experiment failed; repair attempt {repairs}/{max_repairs}")
        session.send(repair_prompt(outcome), label=f"repair#{iteration}.{repairs}")
        outcome = run_experiment(slug, exp_id)

    note = extract_note(result_text)
    if not outcome["ok"]:
        print(f"[orchestrator] iteration {iteration}: unrecoverable failure; reverting.")
        revert_train_file()
        append_ledger({
            "id": exp_id, "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds"),
            "parent_commit": parent, "eval_acc": "NA", "delta": "NA", "kept": 0,
            "note": f"FAILED: {note}",
        })
        return {"best_acc": best_acc, "kept": False}

    acc = float(outcome["eval_acc"])
    delta = acc - best_acc
    kept = delta > 0
    if kept:
        commit_train_file(f"auto-rl[{slug}]: eval_acc {acc:.4f} (+{delta:.4f}) - {note}")
        best_acc = acc
    else:
        revert_train_file()

    append_ledger({
        "id": exp_id, "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "parent_commit": parent, "eval_acc": f"{acc:.4f}", "delta": f"{delta:+.4f}",
        "kept": int(kept), "note": note,
    })
    print(f"[orchestrator] iter {iteration}: eval_acc={acc:.4f} delta={delta:+.4f} "
          f"kept={kept} elapsed={outcome['elapsed']:.0f}s")
    return {"best_acc": best_acc, "kept": kept}


def main() -> int:
    ap = argparse.ArgumentParser(description="Autonomous RLVR research loop.")
    ap.add_argument("--domain", default=os.environ.get("AUTO_RL_DOMAIN", "gsm8k"))
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--model", default=agentlib.DEFAULT_MODEL)
    ap.add_argument("--max-repairs", type=int, default=2)
    ap.add_argument("--allow-unlocked", action="store_true",
                    help="proceed even if the domain pack isn't locked (dev only)")
    args = ap.parse_args()

    slug = args.domain
    os.environ["AUTO_RL_DOMAIN"] = slug

    try:
        harness.load_domain(slug)
    except Exception as exc:
        print(f"[orchestrator] cannot load domain {slug!r}: {exc}", file=sys.stderr)
        return 2

    if not harness.is_locked(slug) and not args.allow_unlocked:
        print(f"[orchestrator] domain {slug!r} is not locked. Bootstrap + approve it first:\n"
              f"    python bootstrap.py \"...\" --slug {slug}\n"
              f"or pass --allow-unlocked for development.", file=sys.stderr)
        return 2

    inject_domain_card(slug)
    ensure_baseline_commit()

    # Establish the bar to beat: run the current train_rl.py once.
    print("[orchestrator] running baseline experiment (iteration 0)...")
    baseline = run_experiment(slug, f"{int(time.time())}_baseline")
    best_acc = float(baseline["eval_acc"]) if baseline["ok"] else 0.0
    if baseline["ok"]:
        append_ledger({
            "id": "baseline", "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds"),
            "parent_commit": current_commit(), "eval_acc": f"{best_acc:.4f}",
            "delta": "+0.0000", "kept": 1, "note": "baseline train_rl.py",
        })
        print(f"[orchestrator] baseline eval_acc={best_acc:.4f}")
    else:
        print(f"[orchestrator] baseline run did not produce metrics "
              f"(rc={baseline['returncode']} timed_out={baseline['timed_out']}); "
              f"starting best_acc=0.0\n{baseline['log_tail']}", file=sys.stderr)

    for i in range(1, args.iterations + 1):
        print(f"\n========== iteration {i}/{args.iterations} (best={best_acc:.4f}) ==========")
        try:
            with agentlib.open_agent(REPO_ROOT, model=args.model) as session:
                state = run_iteration(session, slug, best_acc, i, args.max_repairs)
            best_acc = state["best_acc"]
        except agentlib.AgentUnavailable as exc:
            print(f"[orchestrator] {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("\n[orchestrator] interrupted by user.")
            break

    print(f"\n[orchestrator] done. best eval_acc={best_acc:.4f} on domain {slug!r}.")
    print(f"[orchestrator] history: {LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
