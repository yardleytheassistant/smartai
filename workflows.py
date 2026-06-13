"""Dynamic workflow primitives — composable orchestration patterns.

These are the model-agnostic building blocks for composing self-correcting
systems out of agents. They take plain callables (which may wrap an open-source
model call, or be pure functions in tests), so the orchestration logic is
verifiable without a live server.

Patterns implemented (the three that earn their place in a self-improving loop):

  fan_out_synthesize  — split work into N independent pieces, run each in its
                        own clean context (in parallel), then synthesize.
  adversarial_verify  — pair a maker with an INDEPENDENT verifier; iterate until
                        the verifier passes or attempts run out.
  loop_until_done     — keep stepping until a stop condition is met (no new
                        findings, no more errors, theory verified).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")
S = TypeVar("S")


def fan_out_synthesize(
    items: Sequence[T],
    worker: Callable[[T], R],
    synthesize: Callable[[list[R]], S],
    *,
    max_workers: int | None = None,
) -> S:
    """Run `worker` on each item (in parallel), then `synthesize` the results.

    Order of results matches the order of `items`. Each worker gets its own
    clean context — ideal when steps shouldn't share state (e.g. grading each
    rule of a Skill against historical examples).
    """
    if not items:
        return synthesize([])
    workers = max_workers or min(8, len(items))
    if workers <= 1 or len(items) == 1:
        results = [worker(it) for it in items]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(worker, items))
    return synthesize(results)


@dataclass
class AdversarialResult:
    met: bool
    attempts: int
    output: R | None = None  # type: ignore[valid-type]
    verdict: object | None = None
    history: list[dict] = field(default_factory=list)


def adversarial_verify(
    make: Callable[[str | None], R],
    verify: Callable[[R], tuple[bool, str]],
    *,
    max_iterations: int = 4,
) -> AdversarialResult:
    """Maker/verifier loop where the verifier is independent of the maker.

    `make(feedback)` produces an artifact (feedback is None on the first try).
    `verify(artifact)` returns (met, feedback). Iterates until met or exhausted.
    """
    feedback: str | None = None
    history: list[dict] = []
    output: R | None = None
    met = False
    attempts = 0
    for attempts in range(1, max_iterations + 1):
        output = make(feedback)
        met, feedback = verify(output)
        history.append({"attempt": attempts, "met": met, "feedback": feedback})
        if met:
            break
    return AdversarialResult(met=met, attempts=attempts, output=output, verdict=feedback, history=history)


def loop_until_done(
    step: Callable[[int, list[R]], R],
    done: Callable[[R, list[R]], bool],
    *,
    max_iterations: int = 10,
) -> list[R]:
    """Repeatedly call `step(i, outputs_so_far)` until `done` is true or capped.

    Returns every step's output in order. Use for "keep going until no new
    findings / no more errors / theory verified" style sweeps.
    """
    outputs: list[R] = []
    for i in range(max_iterations):
        out = step(i, outputs)
        outputs.append(out)
        if done(out, outputs):
            break
    return outputs
