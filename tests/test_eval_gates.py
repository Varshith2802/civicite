"""Regression gate: the extractive baseline must not get worse than the published numbers."""
from civicite.agent.core import CiviCite
from civicite.eval.run import evaluate, load_dataset


def test_extractive_quality_gates(index):
    s = evaluate(CiviCite(index), load_dataset(), "extractive")
    assert s["retrieval_recall@5"] >= 0.95
    assert s["answer_accuracy"] >= 0.80
    assert s["abstain_f1"] >= 0.80
    assert s["faithfulness"] == 1.0
    assert s["tool_date_accuracy"] == 1.0
