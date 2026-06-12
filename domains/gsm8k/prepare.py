"""Built-in example domain pack: GSM8K grade-school math word problems.

This is the reference implementation of the Auto-RL domain-pack contract (see
``harness.Domain``). It is what the bootstrap agent produces for a new domain,
hand-written here so the repo is runnable out of the box.

Verifier: programmatic numeric exact-match (commas stripped, small tolerance).
Source: the ``openai/gsm8k`` dataset when available, with a small baked-in
offline fallback so development / self-tests work with no network or the
``datasets`` package.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Allow importing the repo-level harness regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import harness  # noqa: E402

NAME = "GSM8K grade-school math"
DESCRIPTION = (
    "Multi-step grade-school math word problems with a single integer or decimal "
    "answer. The model must reason step by step and produce a final numeric answer."
)
ANSWER_INSTRUCTIONS = (
    "Solve the problem step by step. Then give the final answer on its own line "
    "in the exact form '#### <number>'."
)

# --------------------------------------------------------------------------- #
# Offline fallback problems (used when the HF dataset is unavailable).
# Each is (question, gold_numeric_answer).
# --------------------------------------------------------------------------- #
_OFFLINE: list[tuple[str, str]] = [
    ("Natalia sold clips to 48 friends in April, and then she sold half as many "
     "clips in May. How many clips did she sell altogether in April and May?", "72"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes "
     "of babysitting. How much did she earn?", "10"),
    ("Betty is saving money for a new wallet which costs $100. Betty has only half "
     "of the money she needs. Her parents decided to give her $15 for that purpose, "
     "and her grandparents twice as much as her parents. How much more money does "
     "Betty need to buy the wallet?", "5"),
    ("James writes a 3-page letter to 2 different friends twice a week. How many "
     "pages does he write a year?", "624"),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many "
     "bolts in total does it take?", "3"),
    ("Mark has a garden with flowers. He planted plants of three different colors "
     "in it. Ten of them are yellow, and there are 80% more of those in purple. "
     "There are only 25% as many green flowers as there are yellow and purple "
     "flowers. How many flowers does Mark have in his garden?", "35"),
    ("Albert is wondering how much pizza he can eat in one day. He buys 2 large "
     "pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has "
     "8 slices. If he eats it all, how many pieces does he eat that day?", "48"),
    ("Ken created a care package to send to his brother. He placed a box on a scale, "
     "and then poured into the box enough jelly beans to bring the weight to 2 "
     "pounds. Then, he added enough brownies to cause the weight to triple. Next, "
     "he added another 2 pounds of jelly beans. And finally, he added enough gummy "
     "worms to double the weight once again. What was the final weight of the box "
     "of goodies, in pounds?", "16"),
    ("Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is "
     "eligible for overtime, which is paid by your hourly wage + 1/2 your hourly "
     "wage. If she works 10 hours every day for 5 days, how much money does she "
     "make?", "990"),
    ("A deep-sea monster rises from the waters once every hundred years to feast on "
     "a ship and sate its hunger. Over three hundred years, it has consumed 847 "
     "people. Ships have been built larger over time, so each new ship has twice as "
     "many people as the last ship. How many people were on the ship the monster ate "
     "in the first hundred years?", "121"),
    ("Tobias is buying a new pair of shoes that costs $95. He has been saving up his "
     "money each month for the past three months. He gets a $5 allowance a month. He "
     "also mows lawns and shovels driveways. He charges $15 to mow a lawn and $7 to "
     "shovel. After buying the shoes, he has $15 in change. If he mows 4 lawns, how "
     "many driveways did he shovel?", "5"),
    ("Randy has 60 mango trees on his farm. He also has 5 less than half as many "
     "coconut trees as mango trees. How many trees does Randy have in all on his "
     "farm?", "85"),
]

_GOLD_RE = re.compile(r"####\s*(-?[\d,]+\.?\d*)")


def _gold_from_solution(solution: str) -> str:
    m = _GOLD_RE.search(solution)
    raw = m.group(1) if m else (harness.last_number(solution) or "")
    return raw.replace(",", "").strip()


def _load_hf(split: str) -> list[harness.Problem] | None:
    if os.environ.get("AUTO_RL_OFFLINE") == "1":
        return None
    try:
        from datasets import load_dataset
    except Exception:
        return None
    try:
        ds = load_dataset("openai/gsm8k", "main", split=split)
    except Exception:
        return None
    problems: list[harness.Problem] = []
    for row in ds:
        problems.append(
            {
                "prompt": row["question"].strip(),
                "reference": _gold_from_solution(row["answer"]),
            }
        )
    return problems


def _offline_problems() -> list[harness.Problem]:
    return [{"prompt": q, "reference": a} for q, a in _OFFLINE]


def get_train_problems() -> list[harness.Problem]:
    problems = _load_hf("train")
    if problems is None:
        # Offline: first 8 of the baked set as "train".
        problems = _offline_problems()[:8]
    limit = harness.scale().get("train_limit")
    if limit is not None:
        problems = problems[:limit]
    return problems


def get_eval_problems() -> list[harness.Problem]:
    problems = _load_hf("test")
    if problems is None:
        # Offline: last 4, disjoint from the offline "train" slice above.
        problems = _offline_problems()[8:]
    limit = harness.scale().get("eval_limit")
    if limit is not None:
        problems = problems[:limit]
    return problems


def format_prompt(problem: harness.Problem) -> str:
    return f"{problem['prompt']}\n\n{ANSWER_INSTRUCTIONS}"


def extract_answer(completion: str) -> str:
    """Prefer the '#### <number>' marker; otherwise fall back to the last number."""
    m = _GOLD_RE.search(completion or "")
    if m:
        return m.group(1).replace(",", "").strip()
    num = harness.last_number(completion or "")
    return num or ""


def verify(model_answer: str, reference) -> bool:
    """Programmatic numeric exact-match (with small float tolerance)."""
    if model_answer is None or model_answer == "":
        return False
    return harness.numbers_equal(str(model_answer), str(reference))
