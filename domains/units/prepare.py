"""Unit conversion domain pack: metric/imperial length, mass, volume, and time.

Verifier: programmatic numeric match with 1.5% relative tolerance and 3-s.f.
agreement. Source: deterministic programmatic generator with exact conversion
factors; produces unlimited disjoint train/eval problems with no network
dependency.
"""

from __future__ import annotations

import math
import random
import re
import sys
from pathlib import Path

# Allow importing the repo-level harness regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import harness  # noqa: E402

NAME = "Unit conversions"
DESCRIPTION = (
    "Convert quantities between metric and imperial units of length, mass, and "
    "volume, and between common units of time. Each problem gives a value in one "
    "unit and asks for the equivalent in a target unit; the answer is a single number."
)
ANSWER_INSTRUCTIONS = (
    "Convert step by step using exact conversion factors. Then give the final "
    "answer on its own line in the exact form '#### <number>'."
)

# Conversion factors to a category base unit (meters, kilograms, liters, seconds).
_LENGTH_TO_METERS: dict[str, tuple[str, float]] = {
    "millimeters": ("mm", 0.001),
    "centimeters": ("cm", 0.01),
    "meters": ("m", 1.0),
    "kilometers": ("km", 1000.0),
    "inches": ("in", 0.0254),
    "feet": ("ft", 0.3048),
    "yards": ("yd", 0.9144),
    "miles": ("mi", 1609.344),
}
_LENGTH_METRIC = ("millimeters", "centimeters", "meters", "kilometers")
_LENGTH_IMPERIAL = ("inches", "feet", "yards", "miles")

_MASS_TO_KG: dict[str, tuple[str, float]] = {
    "milligrams": ("mg", 0.001),
    "grams": ("g", 0.001),
    "kilograms": ("kg", 1.0),
    "ounces": ("oz", 0.028349523125),
    "pounds": ("lb", 0.45359237),
}
_MASS_METRIC = ("milligrams", "grams", "kilograms")
_MASS_IMPERIAL = ("ounces", "pounds")

_VOLUME_TO_LITERS: dict[str, tuple[str, float]] = {
    "milliliters": ("mL", 0.001),
    "liters": ("L", 1.0),
    "fluid ounces": ("fl oz", 0.0295735295625),
    "cups": ("cup", 0.236588236),
    "pints": ("pt", 0.473176473),
    "quarts": ("qt", 0.946352946),
    "gallons": ("gal", 3.785411784),
}
_VOLUME_METRIC = ("milliliters", "liters")
_VOLUME_IMPERIAL = ("fluid ounces", "cups", "pints", "quarts", "gallons")

_TIME_TO_SECONDS: dict[str, tuple[str, float]] = {
    "seconds": ("s", 1.0),
    "minutes": ("min", 60.0),
    "hours": ("hr", 3600.0),
    "days": ("day", 86400.0),
    "weeks": ("wk", 604800.0),
}
_TIME_UNITS = tuple(_TIME_TO_SECONDS.keys())

_CATEGORIES: tuple[tuple[str, dict[str, tuple[str, float]], tuple[str, ...], tuple[str, ...] | None], ...] = (
    ("length", _LENGTH_TO_METERS, _LENGTH_METRIC, _LENGTH_IMPERIAL),
    ("mass", _MASS_TO_KG, _MASS_METRIC, _MASS_IMPERIAL),
    ("volume", _VOLUME_TO_LITERS, _VOLUME_METRIC, _VOLUME_IMPERIAL),
)

# Easy metric-prefix pairs within length, mass, and volume.
_EASY_METRIC_PREFIX: tuple[tuple[str, dict[str, tuple[str, float]], tuple[str, str]], ...] = (
    ("length", _LENGTH_TO_METERS, ("kilometers", "meters")),
    ("length", _LENGTH_TO_METERS, ("meters", "kilometers")),
    ("length", _LENGTH_TO_METERS, ("meters", "centimeters")),
    ("length", _LENGTH_TO_METERS, ("centimeters", "meters")),
    ("length", _LENGTH_TO_METERS, ("millimeters", "centimeters")),
    ("length", _LENGTH_TO_METERS, ("centimeters", "millimeters")),
    ("length", _LENGTH_TO_METERS, ("millimeters", "meters")),
    ("length", _LENGTH_TO_METERS, ("meters", "millimeters")),
    ("mass", _MASS_TO_KG, ("kilograms", "grams")),
    ("mass", _MASS_TO_KG, ("grams", "kilograms")),
    ("mass", _MASS_TO_KG, ("grams", "milligrams")),
    ("mass", _MASS_TO_KG, ("milligrams", "grams")),
    ("mass", _MASS_TO_KG, ("kilograms", "milligrams")),
    ("mass", _MASS_TO_KG, ("milligrams", "kilograms")),
    ("volume", _VOLUME_TO_LITERS, ("liters", "milliliters")),
    ("volume", _VOLUME_TO_LITERS, ("milliliters", "liters")),
)

# Easy time conversions with clean integer ratios.
_EASY_TIME_PAIRS: tuple[tuple[str, str], ...] = (
    ("hours", "seconds"),
    ("seconds", "hours"),
    ("minutes", "seconds"),
    ("seconds", "minutes"),
    ("hours", "minutes"),
    ("minutes", "hours"),
    ("days", "hours"),
    ("hours", "days"),
    ("weeks", "days"),
    ("days", "weeks"),
)

_PROMPT_TEMPLATES = (
    "Convert {value} {from_unit} to {to_unit}.",
    "How many {to_unit} are in {value} {from_unit}?",
    "Express {value} {from_unit} in {to_unit}.",
    "{value} {from_unit} is equal to how many {to_unit}?",
)

_TRAIN_ID_START = 0
_TRAIN_ID_COUNT = 8000
_EVAL_ID_START = 8000
_EVAL_ID_COUNT = 1000

_GOLD_RE = re.compile(r"####\s*(-?[\d,]+\.?\d*)")

_TRAIN_PROMPTS_CACHE: set[str] | None = None


def _format_value(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text or "0"


def _format_answer(value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    rounded = round(value, 8)
    text = f"{rounded:.8f}".rstrip("0").rstrip(".")
    return text or "0"


def _convert(value: float, from_unit: str, to_unit: str, table: dict[str, tuple[str, float]]) -> float:
    base = value * table[from_unit][1]
    return base / table[to_unit][1]


def _make_problem(value: float, from_unit: str, to_unit: str, table: dict[str, tuple[str, float]], rng: random.Random) -> harness.Problem:
    answer = _convert(value, from_unit, to_unit, table)
    prompt = rng.choice(_PROMPT_TEMPLATES).format(
        value=_format_value(value),
        from_unit=from_unit,
        to_unit=to_unit,
    )
    return {"prompt": prompt, "reference": _format_answer(answer)}


def _easy_integer_value(rng: random.Random) -> float:
    return float(rng.randint(1, 200))


def _gen_easy_metric_prefix(rng: random.Random) -> harness.Problem:
    _, table, (from_unit, to_unit) = rng.choice(_EASY_METRIC_PREFIX)
    return _make_problem(_easy_integer_value(rng), from_unit, to_unit, table, rng)


def _gen_easy_time(rng: random.Random) -> harness.Problem:
    from_unit, to_unit = rng.choice(_EASY_TIME_PAIRS)
    return _make_problem(_easy_integer_value(rng), from_unit, to_unit, _TIME_TO_SECONDS, rng)


def _pick_units(
    rng: random.Random,
    metric_units: tuple[str, ...],
    imperial_units: tuple[str, ...] | None,
    all_units: tuple[str, ...],
) -> tuple[str, str]:
    if imperial_units is None:
        from_unit, to_unit = rng.sample(list(all_units), 2)
        return from_unit, to_unit
    if rng.random() < 0.5:
        return rng.choice(metric_units), rng.choice(imperial_units)
    return rng.choice(imperial_units), rng.choice(metric_units)


def _gen_general(rng: random.Random) -> harness.Problem:
    category, table, metric_units, imperial_units = rng.choice(_CATEGORIES)
    all_units = tuple(table.keys())
    from_unit, to_unit = _pick_units(rng, metric_units, imperial_units, all_units)

    if rng.random() < 0.75:
        value = float(rng.randint(1, 500))
    else:
        value = round(rng.uniform(0.25, 250.0), rng.choice([1, 2, 3]))
    return _make_problem(value, from_unit, to_unit, table, rng)


def _gen_time_general(rng: random.Random) -> harness.Problem:
    from_unit, to_unit = rng.sample(list(_TIME_UNITS), 2)
    if rng.random() < 0.75:
        value = float(rng.randint(1, 500))
    else:
        value = round(rng.uniform(0.25, 250.0), rng.choice([1, 2, 3]))
    return _make_problem(value, from_unit, to_unit, _TIME_TO_SECONDS, rng)


def _generate_problem(problem_id: int) -> harness.Problem:
    rng = random.Random(problem_id)
    roll = rng.random()
    if roll < 0.25:
        return _gen_easy_metric_prefix(rng)
    if roll < 0.50:
        return _gen_easy_time(rng)
    if roll < 0.65:
        return _gen_time_general(rng)
    return _gen_general(rng)


def _generate_split(id_start: int, count: int) -> list[harness.Problem]:
    seen_prompts: set[str] = set()
    problems: list[harness.Problem] = []
    problem_id = id_start
    while len(problems) < count:
        problem = _generate_problem(problem_id)
        problem_id += 1
        prompt = problem["prompt"]
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        problems.append(problem)
    return problems


def _train_prompts() -> set[str]:
    global _TRAIN_PROMPTS_CACHE
    if _TRAIN_PROMPTS_CACHE is None:
        _TRAIN_PROMPTS_CACHE = {p["prompt"] for p in _generate_split(_TRAIN_ID_START, _TRAIN_ID_COUNT)}
    return _TRAIN_PROMPTS_CACHE


def get_train_problems() -> list[harness.Problem]:
    problems = _generate_split(_TRAIN_ID_START, _TRAIN_ID_COUNT)
    limit = harness.scale().get("train_limit")
    if limit is not None:
        problems = problems[:limit]
    return problems


def get_eval_problems() -> list[harness.Problem]:
    train_prompts = _train_prompts()
    problems: list[harness.Problem] = []
    seen_prompts: set[str] = set()
    problem_id = _EVAL_ID_START
    while len(problems) < _EVAL_ID_COUNT:
        problem = _generate_problem(problem_id)
        problem_id += 1
        prompt = problem["prompt"]
        if prompt in train_prompts or prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        problems.append(problem)
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


def _parse_number(text: str) -> float | None:
    try:
        return float(str(text).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _round_to_sig_figs(value: float, sig: int = 3) -> float:
    if value == 0:
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    magnitude = abs(value)
    power = sig - 1 - int(math.floor(math.log10(magnitude)))
    return sign * round(magnitude * 10**power) / 10**power


def verify(model_answer: str, reference) -> bool:
    """Programmatic numeric match with relative tolerance and 3-s.f. agreement."""
    if model_answer is None or model_answer == "":
        return False

    ans = _parse_number(str(model_answer))
    ref = _parse_number(str(reference))
    if ans is None or ref is None:
        return False

    if math.isclose(ans, ref, rel_tol=0.015, abs_tol=1e-6):
        return True

    return math.isclose(
        _round_to_sig_figs(ans, 3),
        _round_to_sig_figs(ref, 3),
        rel_tol=0.0,
        abs_tol=1e-9,
    )
