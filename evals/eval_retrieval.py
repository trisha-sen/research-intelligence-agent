"""
Retrieval evaluation: weighted Precision@10 for keyword, semantic,
and hybrid search.
Uses all_topic_prop[topic_id] as ground-truth relevance
(NMF topic weight, 0-1).

Run:
    docker compose up postgres -d
    python3 -m eval.eval_retrieval
"""

import asyncio
import json
import os
import statistics

import mlflow
from dotenv import load_dotenv

from agent_tools import search_abstracts_hybrid
from .eval_queries import EVAL_QUERIES

load_dotenv()

METHODS = {"keyword": 1.0, "semantic": 0.0, "hybrid": 0.5}
TOP_K = 10

_MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlruns/mlflow.db"
    )


async def evaluate():
    scores: dict[tuple[str, str], list[float]] = {
        (et, m): [] for et in EVAL_QUERIES for m in METHODS
    }
    per_topic: dict[str, dict[int, dict[str, float]]] = {et: {} for et in EVAL_QUERIES}

    for eval_type, queries in EVAL_QUERIES.items():
        print(f"---- {eval_type} ----")
        for topic_id, query in sorted(queries.items()):
            per_topic[eval_type][topic_id] = {}
            for method, alpha in METHODS.items():
                papers = await search_abstracts_hybrid(query, limit=TOP_K, alpha=alpha)
                topic_scores = [p["all_topic_prop"][topic_id] for p in papers if p["all_topic_prop"]]
                p10 = statistics.mean(topic_scores) if topic_scores else 0.0
                scores[(eval_type, method)].append(p10)
                per_topic[eval_type][topic_id][method] = round(p10, 4)
                print(f"  topic {topic_id:2d} | {method:<8} | P@10={p10:.4f}")

    return scores, per_topic


def print_table(scores: dict[tuple[str, str], list[float]]) -> None:
    print(f"\n| {'Eval Type':<11} | {'Method':<8} | {'Mean P@10':>9} | {'Min':>5} | {'Max':>5} |")
    print(f"|{'-'*13}|{'-'*10}|{'-'*11}|{'-'*7}|{'-'*7}|")
    for (eval_type, method), s in scores.items():
        print(f"| {eval_type:<11} | {method:<8} | {statistics.mean(s):>9.3f} | {min(s):>5.3f} | {max(s):>5.3f} |")


def log_to_mlflow(scores: dict[tuple[str, str], list[float]], per_topic: dict) -> None:
    mlflow.set_tracking_uri(_MLFLOW_URI)
    mlflow.set_experiment("retrieval_eval")

    with mlflow.start_run():
        for (eval_type, method), s in scores.items():
            mlflow.log_metrics({
                f"p10_mean_{eval_type}_{method}": round(statistics.mean(s), 4),
                f"p10_min_{eval_type}_{method}":  round(min(s), 4),
                f"p10_max_{eval_type}_{method}":  round(max(s), 4),
            })
        mlflow.log_text(json.dumps(per_topic, indent=2), "per_topic_scores.json")


if __name__ == "__main__":
    scores, per_topic = asyncio.run(evaluate())
    print_table(scores)
    log_to_mlflow(scores, per_topic)
    print("\nMLflow run logged.")
