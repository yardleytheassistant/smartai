# Agent operating notes

Reference knowledge the agent can search instead of re-deriving. Add your own
project docs, runbooks, and API references as `.md`/`.txt` files in this folder;
they are chunked and retrieved by `search_knowledge`.

## Model routing
The orchestrator (novel / qwen3.5:122b) plans and delegates. Simple fan-out work
goes to the worker (qwen3.6:35b); hard bounded reasoning to the reasoner
(deepseek-r1:32b); coding tasks to the coder (qwen3-coder:30b); whole-repo or
long-document tasks to the long-context model (llama4:scout). Verification runs
on an independent grader (deepseek-r1:32b) at temperature 0.

## Verification discipline
The agent that produced an artifact never grades it. A verifier sub-agent sees
only the artifact and rubric. Unparseable verdicts are treated as "not met" so
the loop keeps correcting rather than passing on ambiguity.

## Memory contract
Read STATE.md at the start of every session; write to it before stopping.
Verified facts and general rules should be consulted, not re-derived. Open
failures carry a hypothesis to investigate next session.

## Sandbox and safety
File and shell tools are confined to the workspace. The shell tool is off by
default. Sensitive-domain tasks (cyber/bio/chem/distillation) are routed per
SAFETY_POLICY — never handled silently.
