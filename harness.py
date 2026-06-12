"""Auto-RL shared harness (immutable infrastructure).

This module is the *fixed* part of the system, analogous to autoresearch's
``prepare.py``. Neither the bootstrap agent nor the RL research agent should edit
this file. It defines:

  * Global configuration: scale presets, device auto-detection, the active model
    id, and the per-experiment time budget.
  * The **domain-pack contract**: every domain lives in ``domains/<slug>/`` and
    exposes a ``prepare.py`` implementing the :class:`Domain` protocol below.
  * Generic model loading / generation utilities so that each domain pack only
    has to implement the *domain-specific* bits (problems + verifier + answer
    formatting), not the boilerplate.
  * ``evaluate(...)`` which computes held-out verified accuracy. The orchestrator
    (not the research agent) runs this, keeping the reward un-gameable.

The point of keeping all of this here is that a generated domain ``prepare.py``
is small and focused on the risky, important part (the verifier and the data),
while the heavy, error-prone plumbing is shared and trusted.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol, runtime_checkable

REPO_ROOT = Path(__file__).resolve().parent
DOMAINS_DIR = REPO_ROOT / "domains"


def load_dotenv(path: Optional[Path] = None) -> None:
    """Minimal .env loader (no dependency). Existing env vars take precedence."""
    env_path = path or (REPO_ROOT / ".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)


# Load .env as soon as the harness is imported so every entry point (train_rl,
# orchestrator, bootstrap) picks up CURSOR_API_KEY and friends.
load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: Default base model. Small enough to smoke-test on a Mac, capable enough to
#: show RLVR gains on a real GPU. Override with ``AUTO_RL_MODEL``.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

#: Scale presets. ``smoke`` keeps everything tiny so the whole loop runs on
#: Apple Silicon for development; ``full`` is for a single NVIDIA GPU.
SCALES: dict[str, dict[str, Any]] = {
    "smoke": {
        "eval_limit": 8,
        "train_limit": 64,
        "max_new_tokens": 160,
        "budget_sec": 120,
        "group_size": 4,
        "prompts_per_step": 2,
    },
    "full": {
        "eval_limit": 200,
        "train_limit": None,  # use everything the domain provides
        "max_new_tokens": 512,
        "budget_sec": 900,
        "group_size": 8,
        "prompts_per_step": 8,
    },
}


def scale_name() -> str:
    name = os.environ.get("AUTO_RL_SCALE", "smoke").strip().lower()
    if name not in SCALES:
        raise ValueError(
            f"AUTO_RL_SCALE={name!r} is not valid; choose one of {sorted(SCALES)}"
        )
    return name


def scale() -> dict[str, Any]:
    """Return the active scale preset (a copy so callers cannot mutate it)."""
    return dict(SCALES[scale_name()])


def model_id() -> str:
    return os.environ.get("AUTO_RL_MODEL", DEFAULT_MODEL_ID)


def budget_sec() -> int:
    """Wall-clock budget for a single experiment (excludes startup/compile)."""
    env = os.environ.get("AUTO_RL_BUDGET_SEC")
    if env:
        return int(env)
    return int(scale()["budget_sec"])


def active_domain_slug() -> str:
    slug = os.environ.get("AUTO_RL_DOMAIN", "gsm8k").strip()
    if not slug:
        raise ValueError("AUTO_RL_DOMAIN is empty")
    return slug


def get_device() -> str:
    """Auto-detect the compute device: cuda > mps > cpu."""
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- #
# Domain-pack contract
# --------------------------------------------------------------------------- #

#: A problem is a plain dict. ``prompt`` is the natural-language task shown to
#: the model; ``reference`` is whatever the domain's ``verify`` needs to grade an
#: answer (a string, number, list of unit tests, etc.). Extra keys are allowed.
Problem = dict


@runtime_checkable
class Domain(Protocol):
    """Interface every ``domains/<slug>/prepare.py`` must satisfy.

    Implementations are ordinary Python modules (not classes); the attributes
    and functions below are looked up at module level. See
    ``domains/gsm8k/prepare.py`` for the reference implementation and
    ``DOMAIN_CARD_SCHEMA`` for the metadata file that accompanies it.
    """

    NAME: str
    DESCRIPTION: str
    # Appended to prompts so the model knows how to format its final answer in a
    # way ``extract_answer`` can parse.
    ANSWER_INSTRUCTIONS: str

    def get_train_problems(self) -> list[Problem]:
        ...

    def get_eval_problems(self) -> list[Problem]:
        ...

    def format_prompt(self, problem: Problem) -> str:
        """Return the user-facing message (question + answer instructions)."""
        ...

    def extract_answer(self, completion: str) -> str:
        """Pull the candidate final answer out of a raw model completion."""
        ...

    def verify(self, model_answer: str, reference: Any) -> bool:
        """Deterministic, programmatic grade. No LLM calls allowed in here."""
        ...


#: JSON schema (informal) for ``domains/<slug>/domain_card.json``. Written by the
#: bootstrap agent, read by the orchestrator and the approval gate.
DOMAIN_CARD_SCHEMA = {
    "slug": "str: folder name under domains/",
    "domain_text": "str: the user's plain-text request",
    "name": "str: short human-readable domain name",
    "description": "str: one paragraph describing the task",
    "source_type": "str: one of 'hf_dataset' | 'generator' | 'synthetic'",
    "source_detail": "str: dataset id, generator description, or synthesis notes",
    "answer_format": "str: how the final answer is expressed",
    "verifier_description": "str: how verify() decides correctness (programmatic)",
    "sample_problems": "list[{prompt, reference, gold_answer}]: a few examples",
    "n_train": "int",
    "n_eval": "int",
    "baseline_accuracy": "float|null: base model accuracy on eval before RL",
    "created_at": "str: ISO timestamp",
}


def domain_dir(slug: str) -> Path:
    return DOMAINS_DIR / slug


def domain_lock_path(slug: str) -> Path:
    return domain_dir(slug) / ".locked"


def is_locked(slug: str) -> bool:
    return domain_lock_path(slug).exists()


def list_domains() -> list[str]:
    if not DOMAINS_DIR.exists():
        return []
    return sorted(
        p.name
        for p in DOMAINS_DIR.iterdir()
        if p.is_dir() and (p / "prepare.py").exists()
    )


def load_domain(slug: Optional[str] = None) -> Any:
    """Import ``domains/<slug>/prepare.py`` and return the module.

    The module is validated against the :class:`Domain` contract.
    """
    slug = slug or active_domain_slug()
    prepare_path = domain_dir(slug) / "prepare.py"
    if not prepare_path.exists():
        raise FileNotFoundError(
            f"Domain {slug!r} not found (expected {prepare_path}). "
            f"Available: {list_domains() or 'none'}. "
            "Bootstrap one with: python bootstrap.py \"<your domain>\""
        )
    mod_name = f"auto_rl_domain_{slug}"
    spec = importlib.util.spec_from_file_location(mod_name, prepare_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    _validate_domain(module, slug)
    return module


def _validate_domain(module: Any, slug: str) -> None:
    required_callables = [
        "get_train_problems",
        "get_eval_problems",
        "format_prompt",
        "extract_answer",
        "verify",
    ]
    required_attrs = ["NAME", "DESCRIPTION", "ANSWER_INSTRUCTIONS"]
    missing = []
    for name in required_callables:
        if not callable(getattr(module, name, None)):
            missing.append(f"{name}()")
    for name in required_attrs:
        if not isinstance(getattr(module, name, None), str):
            missing.append(name)
    if missing:
        raise TypeError(
            f"Domain {slug!r} does not satisfy the contract; missing/invalid: "
            + ", ".join(missing)
        )


# --------------------------------------------------------------------------- #
# Model loading + generation (lazy torch/transformers import)
# --------------------------------------------------------------------------- #


@dataclass
class Policy:
    """A loaded language model + tokenizer, plus its device."""

    model: Any
    tokenizer: Any
    device: str
    model_id: str


def load_policy(model_id_override: Optional[str] = None, *, eval_mode: bool = False) -> Policy:
    """Load the base model + tokenizer onto the auto-detected device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mid = model_id_override or model_id()
    device = get_device()
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16
    # Note: float16 on MPS is numerically unstable for sampling-based generation
    # (produces nan/inf logits after weight updates), so we keep float32 on MPS.

    tokenizer = AutoTokenizer.from_pretrained(mid)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        # transformers >=5 renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype)
    model.to(device)
    if eval_mode:
        model.eval()
    return Policy(model=model, tokenizer=tokenizer, device=device, model_id=mid)


def build_input_text(tokenizer: Any, user_content: str, system: Optional[str] = None) -> str:
    """Render a chat-style prompt using the tokenizer's chat template if any."""
    if getattr(tokenizer, "chat_template", None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    prefix = f"{system}\n\n" if system else ""
    return f"{prefix}{user_content}\n"


# Signature for an injectable generator (used by evaluate); maps prompts to one
# completion each. The default implementation uses the loaded policy.
GenerateFn = Callable[[list[str]], list[str]]


def make_generate_fn(
    policy: Policy,
    *,
    max_new_tokens: Optional[int] = None,
    temperature: float = 0.0,
    system: Optional[str] = None,
) -> GenerateFn:
    """Build a batched greedy/low-temp generation function for evaluation."""
    import torch

    max_new = max_new_tokens or int(scale()["max_new_tokens"])
    tok = policy.tokenizer

    def _generate(user_contents: list[str]) -> list[str]:
        texts = [build_input_text(tok, c, system=system) for c in user_contents]
        enc = tok(texts, return_tensors="pt", padding=True, padding_side="left")
        enc = {k: v.to(policy.device) for k, v in enc.items()}
        do_sample = temperature > 0
        with torch.no_grad():
            out = policy.model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tok.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        return tok.batch_decode(gen, skip_special_tokens=True)

    return _generate


# --------------------------------------------------------------------------- #
# Evaluation (verified accuracy on held-out problems)
# --------------------------------------------------------------------------- #


@dataclass
class EvalResult:
    accuracy: float
    n: int
    n_correct: int
    samples: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "n": self.n,
            "n_correct": self.n_correct,
        }


def evaluate(
    domain: Any,
    policy: Optional[Policy] = None,
    *,
    generate_fn: Optional[GenerateFn] = None,
    problems: Optional[Iterable[Problem]] = None,
    limit: Optional[int] = None,
    keep_samples: int = 3,
) -> EvalResult:
    """Compute held-out verified accuracy for ``domain``.

    Either pass a loaded ``policy`` (real generation) or a ``generate_fn``
    (useful for tests / mock models). Exactly one is used.
    """
    if generate_fn is None:
        if policy is None:
            raise ValueError("evaluate() needs either a policy or a generate_fn")
        generate_fn = make_generate_fn(policy)

    if problems is None:
        problems = domain.get_eval_problems()
    problems = list(problems)
    if limit is None:
        limit = scale()["eval_limit"]
    if limit is not None:
        problems = problems[:limit]

    prompts = [domain.format_prompt(p) for p in problems]
    completions = generate_fn(prompts)
    if len(completions) != len(problems):
        raise RuntimeError(
            f"generate_fn returned {len(completions)} completions for "
            f"{len(problems)} prompts"
        )

    n_correct = 0
    samples: list[dict] = []
    for prob, comp in zip(problems, completions):
        answer = domain.extract_answer(comp)
        ok = bool(domain.verify(answer, prob["reference"]))
        n_correct += int(ok)
        if len(samples) < keep_samples:
            samples.append(
                {
                    "prompt": prob["prompt"][:300],
                    "completion": comp[:300],
                    "extracted": answer,
                    "reference": prob["reference"],
                    "correct": ok,
                }
            )

    n = len(problems)
    acc = n_correct / n if n else 0.0
    return EvalResult(accuracy=acc, n=n, n_correct=n_correct, samples=samples)


# --------------------------------------------------------------------------- #
# Small text helpers shared by domains / verifiers
# --------------------------------------------------------------------------- #

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text: str) -> Optional[str]:
    """Return the last numeric token in ``text`` (commas stripped), or None."""
    matches = _NUM_RE.findall(text or "")
    if not matches:
        return None
    return matches[-1].replace(",", "")


def numbers_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    """Compare two numeric strings with a small tolerance (commas ignored)."""
    def _f(x):
        return float(str(x).replace(",", "").strip())

    try:
        return abs(_f(a) - _f(b)) <= tol
    except (TypeError, ValueError):
        return False
