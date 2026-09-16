#!/usr/bin/env python3
"""
scripts/eval_intents.py
Standalone CLI for running the full intent classification evaluation.

Usage:
    python scripts/eval_intents.py

Exit codes:
    0 - all thresholds passed
    1 - one or more thresholds failed
    2 - fatal error (missing dataset, etc.)
"""
import asyncio
import os
import sys
import json
import logging

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.core.intent_classifier import classify_intent
from agent.core.eval_intent_classifier import (
    load_eval_dataset,
    get_eval_thresholds,
    evaluate_classifier,
    check_thresholds,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_intents")


def main() -> int:
    print("=" * 60)
    print("Intent Classification Evaluation")
    print("=" * 60)

    dataset = load_eval_dataset()
    if not dataset:
        logger.error("Eval dataset is empty. Check config/intents_eval.yaml")
        return 2

    thresholds = get_eval_thresholds()
    print(f"Loaded {len(dataset)} eval examples")
    print(f"Thresholds: precision>={thresholds.get('min_per_intent_precision')}, "
          f"recall>={thresholds.get('min_per_intent_recall')}, "
          f"F1>={thresholds.get('min_overall_f1')}")
    print()

    async def classify_wrapper(query: str):
        return await classify_intent(query)

    print("Running evaluation...")
    metrics = asyncio.run(evaluate_classifier(classify_wrapper, dataset=dataset))

    print("\nResults:")
    print(f"  Accuracy:   {metrics.get('accuracy', 0):.2%}")
    print(f"  Precision:  {metrics.get('overall_precision', 0):.2%}")
    print(f"  Recall:     {metrics.get('overall_recall', 0):.2%}")
    print(f"  F1:         {metrics.get('overall_f1', 0):.2%}")
    print(f"  Avg latency: {metrics.get('avg_latency_ms', 0):.1f}ms")
    print()

    failures = check_thresholds(metrics, thresholds=thresholds)
    if failures:
        print("FAILED thresholds:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("All thresholds passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
