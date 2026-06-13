#!/usr/bin/env python3
"""Live end-to-end demo — watch the system compound across two runs.

This is the first thing that needs a real model server. With Ollama running and
the `novel` model built (./setup.sh), it:

  1. prints durable memory (empty on a fresh workspace),
  2. runs a goal loop (maker -> independent verifier) against a file-based rubric,
  3. prints durable memory again — now carrying the distilled lesson and the
     resume pointer, so a second run consults instead of restarting.

Run:
    source .venv/bin/activate
    python examples/compounding_demo.py

Everything runs locally on open-source models. No cloud, no Claude.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import memory as memory_mod  # noqa: E402
from loop import goal_loop  # noqa: E402
from rubric import Rubric  # noqa: E402

TASK = "Write workspace/fizzbuzz.py and confirm what it would print for 1..15."
RUBRIC = Rubric.from_string(
    "- defines a loop over 1..15\n"
    "- prints Fizz for multiples of 3, Buzz for 5, FizzBuzz for 15\n"
    "- states the expected output"
)


def _print_memory(label: str) -> None:
    mem = memory_mod.load()
    print(f"\n=== durable memory ({label}) ===")
    print("(empty)" if mem.is_empty() else mem.render())


def _on_event(kind: str, data: dict) -> None:
    if kind == "iteration":
        print(f"\n-> iteration {data['n']}/{data['max']}")
    elif kind == "verdict":
        verdict = "MET" if data["met"] else "not met"
        print(f"   verifier: {verdict} (score {data['score']:.2f})")
        if not data["met"] and data["feedback"]:
            print(f"   feedback: {data['feedback']}")


def main() -> None:
    _print_memory("before")
    print(f"\nGoal: {TASK}")
    result = goal_loop(TASK, rubric=RUBRIC, on_event=_on_event)
    print(f"\nResult: {'MET' if result.met else 'UNMET'} in {result.iterations} iteration(s)")
    print("\n--- final answer ---")
    print(result.output)
    _print_memory("after")
    print("\nRun again and the maker will consult the lesson above instead of restarting.")


if __name__ == "__main__":
    main()
