"""
eval_intent_classifier.py
Evaluation framework for intent classifier.

Computes per-intent precision, recall, and F1.
"""
import asyncio
import os
import time
import logging
import yaml
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("hr_agent")

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EVAL_CONFIG_PATH = os.path.join(_BASE_DIR, "config", "intents_eval.yaml")


def load_eval_dataset(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load evaluation dataset from YAML file."""
    eval_path = path or _EVAL_CONFIG_PATH
    if not os.path.isfile(eval_path):
        logger.warning("Eval dataset not found at %s", eval_path)
        return []
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return []
        examples = data.get("examples", [])
        if not isinstance(examples, list):
            return []
        return [ex for ex in examples if isinstance(ex, dict) and "query" in ex and "expected_intent" in ex]
    except Exception as e:
        logger.warning("Failed to load eval dataset from %s: %s", eval_path, e)
        return []


def get_eval_thresholds(path: Optional[str] = None) -> Dict[str, Any]:
    """Load evaluation thresholds from YAML file."""
    eval_path = path or _EVAL_CONFIG_PATH
    if not os.path.isfile(eval_path):
        return {}
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return {}
        return data.get("thresholds", {})
    except Exception as e:
        logger.warning("Failed to load eval thresholds from %s: %s", eval_path, e)
        return {}


async def evaluate_classifier(
    classify_fn: Callable[[str], Any],
    dataset: Optional[List[Dict[str, Any]]] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run evaluation and return per-intent metrics.

    Args:
        classify_fn: Callable that takes a query string and returns an object
                     with an `intent` attribute (e.g., IntentResult). May be async.
        dataset: Optional pre-loaded dataset list. If None, loads from config.
        path: Optional path to eval dataset YAML.

    Returns:
        Dict with per-intent precision/recall/f1 and overall metrics.
    """
    if dataset is None:
        dataset = load_eval_dataset(path)
    if not dataset:
        return {"error": "empty_dataset"}

    # Per-intent counters
    tp: Dict[str, int] = {}
    fp: Dict[str, int] = {}
    fn: Dict[str, int] = {}

    total = 0
    correct = 0
    latencies: List[float] = []

    for example in dataset:
        query = example.get("query", "")
        expected = example.get("expected_intent", "")
        if not query or not expected:
            continue

        total += 1
        try:
            start = time.perf_counter()
            result = classify_fn(query)
            if asyncio.iscoroutine(result):
                result = await result
            latency = time.perf_counter() - start
            latencies.append(latency)
            predicted = getattr(result, "intent", "general_hr")
        except Exception as e:
            logger.warning("Eval classification failed for '%s': %s", query, e)
            predicted = "general_hr"

        if predicted == expected:
            correct += 1
            tp[expected] = tp.get(expected, 0) + 1
        else:
            fn[expected] = fn.get(expected, 0) + 1
            fp[predicted] = fp.get(predicted, 0) + 1

    # Compute metrics
    all_intents = sorted(set(list(tp.keys()) + list(fp.keys()) + list(fn.keys())))
    per_intent: Dict[str, Dict[str, float]] = {}
    total_prec_num = 0
    total_prec_den = 0
    total_rec_num = 0
    total_rec_den = 0

    for intent in all_intents:
        true_pos = tp.get(intent, 0)
        false_pos = fp.get(intent, 0)
        false_neg = fn.get(intent, 0)

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_intent[intent] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": true_pos + false_neg,
        }
        total_prec_num += true_pos
        total_prec_den += true_pos + false_pos
        total_rec_num += true_pos
        total_rec_den += true_pos + false_neg

    overall_precision = total_prec_num / total_prec_den if total_prec_den > 0 else 0.0
    overall_recall = total_rec_num / total_rec_den if total_rec_den > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "total_examples": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "overall_precision": round(overall_precision, 4),
        "overall_recall": round(overall_recall, 4),
        "overall_f1": round(overall_f1, 4),
        "avg_latency_ms": round(avg_latency * 1000, 2),
        "per_intent": per_intent,
    }


def check_thresholds(metrics: Dict[str, Any], thresholds: Optional[Dict[str, Any]] = None) -> List[str]:
    """Check metrics against thresholds. Returns list of failure messages (empty if pass)."""
    if thresholds is None:
        thresholds = get_eval_thresholds()
    failures = []

    min_precision = thresholds.get("min_per_intent_precision", 0.0)
    min_recall = thresholds.get("min_per_intent_recall", 0.0)
    min_f1 = thresholds.get("min_overall_f1", 0.0)
    max_regression = thresholds.get("max_regression_delta", 1.0)

    for intent, m in metrics.get("per_intent", {}).items():
        if m.get("precision", 1.0) < min_precision:
            failures.append(f"{intent}: precision {m['precision']:.2f} < {min_precision}")
        if m.get("recall", 1.0) < min_recall:
            failures.append(f"{intent}: recall {m['recall']:.2f} < {min_recall}")

    if metrics.get("overall_f1", 1.0) < min_f1:
        failures.append(f"overall F1 {metrics['overall_f1']:.2f} < {min_f1}")

    return failures
