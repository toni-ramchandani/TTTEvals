"""Evaluation toolkits: DeepEval and Ragas, run against the workshop's own traces.

The rest of the workshop writes every evaluator in its own code, so that you
know what each one reads and decides. This package puts two widely used toolkits
beside them, on the same traces and the same human labels:

    DeepEval 4.2.8   G-Eval and DAG judges, RAG metrics, agent metrics, CI
    Ragas 0.4.3      faithfulness, context precision and recall, factual
                     correctness, answer relevancy, tool-call metrics

Three rules hold everywhere in this package:

1. Every model call goes through `helix.llm`, the workshop's own standard-
   library model layer. The toolkits never open a connection of their own,
   so there is no model SDK call and no LiteLLM in the path.
2. Nothing is stored and nothing is uploaded. Telemetry is off, DeepEval
   runs in its read-only file mode, and no toolkit cache is enabled, so a
   run writes no files of its own. `env.prepare` sets this up before either
   toolkit is imported.
3. A toolkit metric is one more evaluator. It earns trust the same way
   yours do: by agreeing with human labels on the failure mode it claims to
   detect.

The toolkits are optional and live in their own virtual environment (see
requirements-toolkits.txt). Nothing outside this package imports them, so
the core workshop runs without them.
"""

from .cases import Case, cases_from_traces

__all__ = ["Case", "cases_from_traces"]
