"""Resumable multi-model VLM benchmark on programmatically controlled 3D changes.

The experiment evaluates technical visual discrimination only. It does not use
VLM outputs as evidence of cultural authenticity or as a substitute for experts.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from run_qwen3vl_atomic_pairwise import (
    CASES,
    CRITERIA,
    PROMPT,
    ROOT,
    make_sheet,
    proposition,
)


DEFAULT_MODELS = [
    "qwen3-vl:4b",
    "qwen2.5vl:3b",
    "minicpm-v:latest",
    "llava:7b",
    "moondream:latest",
]
OUT = ROOT / "experiments/exp12_multimodel_vlm_atomic"


def slug(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in value).strip("_")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ollama_chat(
    model: str, sheet: Path, prompt: str, timeout: int
) -> tuple[dict, str, float]:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": "json",
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [base64.b64encode(sheet.read_bytes()).decode("ascii")],
        }],
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 8192,
            "num_predict": 500,
        },
        "keep_alive": "30m",
    }
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        envelope = json.loads(response.read().decode("utf-8"))
    elapsed = round(time.time() - started, 3)
    raw = envelope.get("message", {}).get("content", "")
    result = json.loads(raw)
    return result, raw, elapsed


def validate(result: dict, criterion: str) -> None:
    if result.get("criterion") != criterion:
        raise ValueError("criterion echo mismatch")
    if not isinstance(result.get("criterion_changed"), bool):
        raise ValueError("criterion_changed is not boolean")


def record_path(model: str, case_id: str, criterion: str, order: str) -> Path:
    return OUT / slug(model) / f"{case_id}_{criterion}_{order}.json"


def run_one(
    model: str,
    case_id: str,
    criterion: str,
    order: str,
    timeout: int,
) -> dict:
    intervention, target = CASES[case_id]
    sheet, source_images = make_sheet(case_id, order)
    atomic_proposition = proposition(case_id, criterion, intervention, target)
    prompt = PROMPT.format(
        criterion=criterion,
        definition=CRITERIA[criterion],
        proposition=atomic_proposition,
    )
    errors = []
    for attempt in range(1, 3):
        try:
            repair = (
                "" if attempt == 1
                else "\nReturn all required keys and no text outside JSON."
            )
            result, raw, elapsed = ollama_chat(
                model, sheet, prompt + repair, timeout
            )
            validate(result, criterion)
            expected = criterion == target
            predicted = result["criterion_changed"]
            return {
                "model": model,
                "case_id": case_id,
                "intervention": intervention,
                "target_criterion": target,
                "criterion": criterion,
                "atomic_proposition": atomic_proposition,
                "order": order,
                "expected_changed": expected,
                "predicted_changed": predicted,
                "correct": expected == predicted,
                "schema_attempts": attempt,
                "elapsed_seconds": elapsed,
                "sheet_sha256": sha256(sheet),
                "source_sha256": {
                    path.name: sha256(path) for path in source_images
                },
                "assessment": result,
                "raw_response": raw,
                "errors_before_success": errors,
                "claim_boundary": (
                    "Technical visual discrimination only; "
                    "not cultural validation."
                ),
            }
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            urllib.error.URLError,
        ) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError(f"{model} failed after two attempts: {errors}")


def load_records() -> list[dict]:
    records = []
    for path in OUT.glob("*/*.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            if "predicted_changed" in item:
                records.append(item)
        except json.JSONDecodeError:
            continue
    return records


def metrics(records: list[dict]) -> dict:
    output = {}
    by_model = defaultdict(list)
    for record in records:
        by_model[record["model"]].append(record)
    for model, rows in sorted(by_model.items()):
        tp = sum(r["expected_changed"] and r["predicted_changed"] for r in rows)
        fp = sum(not r["expected_changed"] and r["predicted_changed"] for r in rows)
        tn = sum(not r["expected_changed"] and not r["predicted_changed"] for r in rows)
        fn = sum(r["expected_changed"] and not r["predicted_changed"] for r in rows)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        pairs = defaultdict(list)
        for row in rows:
            pairs[(row["case_id"], row["criterion"])].append(
                row["predicted_changed"]
            )
        complete_pairs = [values for values in pairs.values() if len(values) == 2]
        denominator = math.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
        output[model] = {
            "judgement_count": len(rows),
            "expected_judgement_count": len(CASES) * len(CRITERIA) * 2,
            "completion_rate": round(
                len(rows) / (len(CASES) * len(CRITERIA) * 2), 4
            ),
            "confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
            "accuracy": round((tp + tn) / len(rows), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(tn / (tn + fp), 4) if tn + fp else 0,
            "f1": round(2 * precision * recall / (precision + recall), 4)
            if precision + recall else 0,
            "balanced_accuracy": round(
                ((tp / (tp + fn) if tp + fn else 0)
                 + (tn / (tn + fp) if tn + fp else 0)) / 2,
                4,
            ),
            "matthews_correlation_coefficient": round(
                (tp * tn - fp * fn) / denominator, 4
            ) if denominator else 0,
            "positive_prediction_rate": round(
                (tp + fp) / len(rows), 4
            ),
            "position_consistency_rate": round(
                sum(len(set(values)) == 1 for values in complete_pairs)
                / len(complete_pairs), 4
            ) if complete_pairs else 0,
            "complete_position_pairs": len(complete_pairs),
            "first_pass_schema_validity_rate": round(
                sum(r["schema_attempts"] == 1 for r in rows) / len(rows), 4
            ),
            "mean_elapsed_seconds": round(
                sum(r["elapsed_seconds"] for r in rows) / len(rows), 3
            ),
        }
    return output


def write_aggregate(records: list[dict], requested_models: list[str]) -> None:
    fields = [
        "model", "case_id", "intervention", "target_criterion", "criterion",
        "order", "expected_changed", "predicted_changed", "correct",
        "schema_attempts", "elapsed_seconds",
    ]
    with (OUT / "judgements.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(
            records,
            key=lambda r: (r["model"], r["case_id"], r["criterion"], r["order"]),
        ))
    summary = {
        "experiment": "multi-model atomic technical visual discrimination",
        "requested_models": requested_models,
        "cases": len(CASES),
        "criteria_per_case": len(CRITERIA),
        "position_orders": 2,
        "expected_judgements_per_model": len(CASES) * len(CRITERIA) * 2,
        "metrics_by_model": metrics(records),
        "claim_boundary": (
            "VLM outputs are evaluated against programmatic intervention labels; "
            "they are not cultural validation and do not replace specialists."
        ),
    }
    (OUT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        records = load_records()
        write_aggregate(records, args.models)
        print(json.dumps(metrics(records), indent=2))
        return
    completed = 0
    failures = []
    for model in args.models:
        (OUT / slug(model)).mkdir(exist_ok=True)
        for case_id in CASES:
            for order in ("baseline_left", "variant_left"):
                for criterion in CRITERIA:
                    path = record_path(model, case_id, criterion, order)
                    if path.exists():
                        continue
                    if args.limit and completed >= args.limit:
                        records = load_records()
                        write_aggregate(records, args.models)
                        print(json.dumps(metrics(records), indent=2))
                        return
                    try:
                        record = run_one(
                            model, case_id, criterion, order, args.timeout
                        )
                        path.write_text(
                            json.dumps(record, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        completed += 1
                        print(
                            f"[{model}] {case_id} {criterion} {order}: "
                            f"{record['predicted_changed']} "
                            f"({record['elapsed_seconds']}s)",
                            flush=True,
                        )
                    except RuntimeError as exc:
                        failure = {
                            "model": model, "case_id": case_id,
                            "criterion": criterion, "order": order,
                            "error": str(exc),
                        }
                        failures.append(failure)
                        failure_path = (
                            OUT / slug(model)
                            / f"FAILED_{case_id}_{criterion}_{order}.json"
                        )
                        failure_path.write_text(
                            json.dumps(failure, indent=2), encoding="utf-8"
                        )
                        print(f"ERROR {failure}", flush=True)
    records = load_records()
    write_aggregate(records, args.models)
    (OUT / "failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics(records), indent=2))


if __name__ == "__main__":
    main()
