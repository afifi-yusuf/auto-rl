"""Polymarket resolved prediction-market forecasting domain pack.

Verifier: programmatic normalized YES/NO exact-match (prefer a '#### YES' or
'#### NO' marker, else detect yes/no in the completion). Source: resolved binary
Yes/No markets from the Polymarket Gamma API, cached locally on first fetch;
ships with a baked-in offline fallback of well-known resolved markets so
development works with no network.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow importing the repo-level harness regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import harness  # noqa: E402

NAME = "Polymarket resolved forecasts"
DESCRIPTION = (
    "Forecast the binary outcome of resolved Polymarket prediction markets. "
    "Each problem is a real closed market question (with optional description and "
    "close date); the model must predict YES or NO as it would have before "
    "resolution. Ground truth is the actual market resolution."
)
ANSWER_INSTRUCTIONS = (
    "Reason about the question using your knowledge of events, politics, sports, "
    "and markets. Then give your final forecast on its own line in the exact form "
    "'#### YES' or '#### NO'."
)

_DOMAIN_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _DOMAIN_DIR / "resolved_markets.json"
_API_HOST = "gamma-api.polymarket.com"
_EVAL_MODULUS = 5  # ~20% held-out eval split by stable market id

_MARKER_RE = re.compile(r"####\s*(YES|NO)\b", re.IGNORECASE)
_WORD_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

# Baked-in offline fallback: real resolved Polymarket binary markets with
# verified YES/NO outcomes (elections, crypto milestones, sports, tech).
_OFFLINE: list[dict] = [
    {
        "id": "40",
        "prompt": "Will Trump win the 2020 U.S. presidential election?",
        "reference": "NO",
        "end_date": "2020-11-04",
    },
    {
        "id": "76",
        "prompt": "Will $BTC break $20k before 2021?",
        "reference": "YES",
        "end_date": "2021-01-01",
    },
    {
        "id": "75",
        "prompt": "Will the Ethereum 2.0 Genesis Event happen successfully on December 1st, 2020?",
        "reference": "YES",
        "end_date": "2020-12-02",
    },
    {
        "id": "13107",
        "prompt": "Will Joe Biden be inaugurated as President of the USA on January 20th, 2021?",
        "reference": "YES",
        "end_date": "2021-01-20",
    },
    {
        "id": "93",
        "prompt": "Will Donald Trump be inaugurated for his second term as President of the USA on Inauguration Day, January 20th, 2021?",
        "reference": "NO",
        "end_date": "2021-01-20",
    },
    {
        "id": "87",
        "prompt": "Will Donald Trump attend Joe Biden's inauguration ceremony in person?",
        "reference": "NO",
        "end_date": "2021-01-20",
    },
    {
        "id": "19",
        "prompt": "Will Kim Kardashian and Kanye West divorce before Jan 1, 2021?",
        "reference": "NO",
        "end_date": "2021-01-01",
    },
    {
        "id": "12017",
        "prompt": "Will Kim Kardashian or Kanye West file for divorce before March 1, 2021?",
        "reference": "YES",
        "end_date": "2021-03-01",
    },
    {
        "id": "20",
        "prompt": "Will Coinbase begin publicly trading before Jan 1, 2021?",
        "reference": "NO",
        "end_date": "2021-01-01",
    },
    {
        "id": "11987",
        "prompt": "Will Tesla announce a Bitcoin purchase before March 1, 2021?",
        "reference": "YES",
        "end_date": "2021-03-01",
    },
    {
        "id": "8939",
        "prompt": "Will $BTC break $25k before March 1st?",
        "reference": "YES",
        "end_date": "2021-03-01",
    },
    {
        "id": "25367",
        "prompt": "Will $BTC break $50k before April 1st, 2021?",
        "reference": "YES",
        "end_date": "2021-04-01",
    },
    {
        "id": "12015",
        "prompt": "Will President Trump be suspended from Twitter before April 1, 2021?",
        "reference": "YES",
        "end_date": "2021-04-01",
    },
    {
        "id": "29919",
        "prompt": "Will Trump complete his first term?",
        "reference": "YES",
        "end_date": "2021-01-20",
    },
    {
        "id": "31064",
        "prompt": "Will Donald Trump be impeached again before the end of his term?",
        "reference": "YES",
        "end_date": "2021-01-20",
    },
    {
        "id": "35974",
        "prompt": "Will the Senate convict Donald Trump on impeachment by April 29, 2021?",
        "reference": "NO",
        "end_date": "2021-04-29",
    },
    {
        "id": "40936",
        "prompt": "Will Conor McGregor win his UFC 257 match on January 23?",
        "reference": "NO",
        "end_date": "2021-01-24",
    },
    {
        "id": "78435",
        "prompt": "Will there be over 56 points scored in Super Bowl 55?",
        "reference": "NO",
        "end_date": "2021-02-08",
    },
    {
        "id": "140033",
        "prompt": "Will Chadwick Boseman win Best Actor at the 2021 Oscars?",
        "reference": "NO",
        "end_date": "2021-04-26",
    },
    {
        "id": "176506",
        "prompt": "Will the Oversight Board uphold Facebook's decision to indefinitely suspend Donald Trump's account?",
        "reference": "YES",
        "end_date": "2021-05-05",
    },
    {
        "id": "200312",
        "prompt": "Will $DOGE be available to trade on Coinbase by July 1, 2021?",
        "reference": "YES",
        "end_date": "2021-07-01",
    },
    {
        "id": "59",
        "prompt": "Will there be a federal charge filed against Hunter Biden before 2021?",
        "reference": "NO",
        "end_date": "2021-01-01",
    },
    {
        "id": "77",
        "prompt": "Will Donald Trump formally concede the 2020 US Election before December 1st, 2020?",
        "reference": "NO",
        "end_date": "2020-12-01",
    },
    {
        "id": "96",
        "prompt": "Will Trump win any of Pennsylvania, Arizona or Georgia?",
        "reference": "NO",
        "end_date": "2021-01-20",
    },
    {
        "id": "73",
        "prompt": "Will Donald Trump tweet announcing that he won the election before November 5th 2020?",
        "reference": "NO",
        "end_date": "2020-11-05",
    },
    {
        "id": "109",
        "prompt": "Will Trump Pardon Himself in His First Term?",
        "reference": "NO",
        "end_date": "2021-01-20",
    },
    {
        "id": "12576",
        "prompt": "Will Coinbase delist Ripple (XRP) before they begin publicly trading?",
        "reference": "YES",
        "end_date": "2021-04-14",
    },
    {
        "id": "30941",
        "prompt": "Will Joe Biden be President of the USA on March 1, 2021?",
        "reference": "YES",
        "end_date": "2021-03-01",
    },
]

_ALL_PROBLEMS_CACHE: list[harness.Problem] | None = None


def _parse_resolution(market: dict) -> str | None:
    """Return 'YES' or 'NO' for a resolved binary Yes/No market, else None."""
    try:
        outcomes = json.loads(market["outcomes"])
        prices = [float(p) for p in json.loads(market["outcomePrices"])]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if len(outcomes) != 2:
        return None
    if {o.lower() for o in outcomes} != {"yes", "no"}:
        return None
    winner = None
    for outcome, price in zip(outcomes, prices):
        if price > 0.9:
            winner = "YES" if outcome.lower() == "yes" else "NO"
    return winner


def _market_to_problem(market: dict) -> harness.Problem | None:
    reference = _parse_resolution(market)
    if reference is None:
        return None
    question = (market.get("question") or "").strip()
    if not question:
        return None
    description = (market.get("description") or "").strip()
    end_date = market.get("endDateIso") or market.get("endDate") or ""
    if isinstance(end_date, str) and "T" in end_date:
        end_date = end_date.split("T", 1)[0]
    return {
        "id": str(market.get("id", "")),
        "prompt": question,
        "reference": reference,
        "description": description,
        "end_date": end_date,
    }


def _fetch_api_page(offset: int, limit: int = 100) -> list[dict] | None:
    try:
        conn = http.client.HTTPSConnection(_API_HOST, timeout=30)
        conn.request("GET", f"/markets?closed=true&limit={limit}&offset={offset}")
        resp = conn.getresponse()
        if resp.status != 200:
            return None
        data = json.loads(resp.read().decode("utf-8"))
        conn.close()
        return data if isinstance(data, list) else None
    except Exception:
        return None


def _fetch_all_markets() -> list[harness.Problem] | None:
    if os.environ.get("AUTO_RL_OFFLINE") == "1":
        return None
    problems: list[harness.Problem] = []
    seen_ids: set[str] = set()
    offset = 0
    while True:
        batch = _fetch_api_page(offset)
        if not batch:
            break
        for market in batch:
            problem = _market_to_problem(market)
            if problem is None:
                continue
            mid = problem["id"]
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
            problems.append(problem)
        if len(batch) < 100:
            break
        offset += 100
    return problems or None


def _write_cache(problems: list[harness.Problem]) -> None:
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "markets": problems,
    }
    _CACHE_PATH.write_text(json.dumps(payload, indent=2))


def _read_cache() -> list[harness.Problem] | None:
    if not _CACHE_PATH.exists():
        return None
    try:
        payload = json.loads(_CACHE_PATH.read_text())
        markets = payload.get("markets")
        if not isinstance(markets, list) or not markets:
            return None
        return markets
    except Exception:
        return None


def _offline_problems() -> list[harness.Problem]:
    return [dict(p) for p in _OFFLINE]


def _load_all_problems() -> list[harness.Problem]:
    global _ALL_PROBLEMS_CACHE
    if _ALL_PROBLEMS_CACHE is not None:
        return _ALL_PROBLEMS_CACHE

    problems = _read_cache()
    if problems is None:
        problems = _fetch_all_markets()
        if problems is not None:
            _write_cache(problems)
    if problems is None:
        problems = _offline_problems()

    problems = sorted(problems, key=lambda p: int(p.get("id", "0") or "0"))
    deduped: list[harness.Problem] = []
    seen_prompts: set[str] = set()
    for problem in problems:
        prompt = problem["prompt"]
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        deduped.append(problem)
    _ALL_PROBLEMS_CACHE = deduped
    return deduped


def _is_eval(problem: harness.Problem) -> bool:
    try:
        return int(problem.get("id", "0") or "0") % _EVAL_MODULUS == 0
    except ValueError:
        return False


def get_train_problems() -> list[harness.Problem]:
    problems = [p for p in _load_all_problems() if not _is_eval(p)]
    limit = harness.scale().get("train_limit")
    if limit is not None:
        problems = problems[:limit]
    return problems


def get_eval_problems() -> list[harness.Problem]:
    problems = [p for p in _load_all_problems() if _is_eval(p)]
    limit = harness.scale().get("eval_limit")
    if limit is not None:
        problems = problems[:limit]
    return problems


def format_prompt(problem: harness.Problem) -> str:
    parts = [problem["prompt"]]
    end_date = problem.get("end_date")
    if end_date:
        parts.append(f"Market close date: {end_date}")
    description = problem.get("description") or ""
    if description:
        if len(description) > 400:
            description = description[:397] + "..."
        parts.append(f"Description: {description}")
    return "\n\n".join(parts) + f"\n\n{ANSWER_INSTRUCTIONS}"


def _normalize_yes_no(text: str) -> str:
    token = (text or "").strip().upper()
    if token in ("YES", "NO"):
        return token
    return ""


def extract_answer(completion: str) -> str:
    """Prefer '#### YES' / '#### NO'; else detect yes/no in the completion."""
    text = completion or ""
    marker = _MARKER_RE.search(text)
    if marker:
        return marker.group(1).upper()
    for line in reversed(text.splitlines()):
        line = line.strip().rstrip(".")
        norm = _normalize_yes_no(line)
        if norm:
            return norm
    words = _WORD_RE.findall(text)
    if words:
        return words[-1].upper()
    return ""


def verify(model_answer: str, reference) -> bool:
    """Programmatic normalized YES/NO exact string match."""
    if model_answer is None or model_answer == "":
        return False
    answer = _normalize_yes_no(str(model_answer))
    gold = _normalize_yes_no(str(reference))
    return bool(answer) and answer == gold
