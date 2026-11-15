"""Whichever toolkits are installed, ready to score workshop cases.

Shared by `metrics`, `day3`, `conversations` and `robustness`, so every
command builds the toolkits the same way: safe defaults first
(`toolkits.env.prepare`), one judge per toolkit through helix.llm, and one
call counter for both. Offline, nothing is built: there is no model to judge
with and nothing stored to show instead.
"""

from __future__ import annotations

from toolkits import env


class Kits:
    """DeepEval and Ragas, each with the workshop judge, counting calls."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.deepeval = self.ragas = None
        self.stats = None
        self._geval: dict[str, object] = {}
        if cfg.offline:
            return
        env.prepare()
        from toolkits.judge import JudgeStats

        self.stats = JudgeStats()
        if env.installed("deepeval"):
            from toolkits import deepeval_metrics as D
            from toolkits.judge import deepeval_judge

            self.D, self.deepeval = D, deepeval_judge(cfg, self.stats)
        if env.installed("ragas"):
            from toolkits import ragas_metrics as R
            from toolkits.judge import ragas_embeddings, ragas_llm

            self.R = R
            self.ragas = R.build(ragas_llm(cfg, self.stats), ragas_embeddings(cfg))

    def d(self, metric_name: str, case):
        """One DeepEval RAG or agent metric on one case, or None if DeepEval is absent."""
        if not self.deepeval:
            return None
        from toolkits.cases import to_deepeval

        builders = {**self.D.rag_metrics(self.deepeval), **self.D.agent_metrics(self.deepeval)}
        return self.D.run(builders[metric_name], to_deepeval(case))

    def d_keep(self, metric_name: str, case):
        """Like `d`, and also returns the metric, so its verdicts can be read."""
        if not self.deepeval:
            return None, None
        from toolkits.cases import to_deepeval

        builders = {**self.D.rag_metrics(self.deepeval), **self.D.agent_metrics(self.deepeval)}
        metric = builders[metric_name]
        return self.D.run(metric, to_deepeval(case)), metric

    def g(self, criterion: str, case):
        """DeepEval's G-Eval on one of `deepeval_metrics.GEVAL`, or None if DeepEval is absent.

        One G-Eval per criterion for the whole run: the judge writes the
        evaluation steps on the first case, and every later case is scored
        with the same steps, as G-Eval intends.
        """
        if not self.deepeval:
            return None
        from toolkits.cases import to_deepeval

        if criterion not in self._geval:
            self._geval[criterion] = self.D.geval_metric(self.deepeval, criterion)
        return self.D.run(self._geval[criterion], to_deepeval(case))

    def geval_steps(self, criterion: str) -> list[str]:
        """The steps the judge wrote for a criterion, once it has scored a case."""
        m = self._geval.get(criterion)
        return list(getattr(m, "evaluation_steps", None) or []) if m is not None else []

    def r(self, metric_name: str, case):
        """One Ragas metric on one case, or None if Ragas is absent."""
        if not self.ragas:
            return None
        return self.R.score(metric_name, self.ragas[metric_name], case)

    def missing(self) -> list[str]:
        return [n for n, v in (("DeepEval", self.deepeval), ("Ragas", self.ragas)) if v is None]


def val(m) -> float | None:
    return None if m is None else m.score


def passes(score: float | None, threshold: float = 0.5) -> bool | None:
    """A toolkit score as pass or fail at a threshold (DeepEval's default is 0.5)."""
    return None if score is None else score >= threshold
