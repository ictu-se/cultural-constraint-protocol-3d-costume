"""Technical validation extensions for the cultural-constraint simulator.

Produces ablations, multi-seed stability, and injected-failure recovery metrics.
This validates simulator behavior, not cultural correctness.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "run_diverse_authenticity_experiment.py"
OUT = ROOT / "experiments" / "exp03_validation_extensions"
FIG = ROOT / "manuscript" / "figures"

spec = importlib.util.spec_from_file_location("benchmark", SRC)
b = importlib.util.module_from_spec(spec)
assert spec.loader
sys.modules[spec.name] = b
spec.loader.exec_module(b)


def ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get, reverse=True)
    return {name: i + 1 for i, name in enumerate(ordered)}


def spearman(a: dict[str, float], c: dict[str, float]) -> float:
    names = sorted(a)
    n = len(names)
    d2 = sum((a[x] - c[x]) ** 2 for x in names)
    return 1 - 6 * d2 / (n * (n * n - 1))


def simulate(seed: int, scenarios: int, variant: str) -> tuple[dict[str, float], dict[str, float]]:
    rng = random.Random(seed)
    scores: dict[str, list[float]] = defaultdict(list)
    hard: dict[str, list[int]] = defaultdict(list)
    regimes = list(b.REGIMES)
    for i in range(scenarios):
        regime = regimes[i % len(regimes)] if variant != "single_regime" else "balanced"
        severity = rng.uniform(0.10, 0.55)
        noise = rng.uniform(0.55, 1.75)
        threshold = rng.choice([1, 2, 3])
        weights = b.scenario_weights(rng, regime, severity)
        for profile in b.PROFILES:
            if b.profile_status(profile) != "documentary":
                continue
            for method in b.METHODS:
                cs = {}
                for criterion in b.CRITERIA:
                    penalty = 0.0 if variant == "no_interactions" else b.interaction_penalty(profile, method, criterion)
                    if variant == "no_profile_difficulty":
                        neutral = type(profile)(profile.name, profile.family, {x: 0.0 for x in b.CRITERIA}, 0.0, 1.0)
                        penalty = b.interaction_penalty(neutral, method, criterion)
                    value = 3.0 * (method.quality[criterion] - penalty)
                    value += rng.gauss(0, method.variance * noise) * 3.0
                    if variant != "no_failure_events":
                        value -= b.failure_drop(rng, profile, method, criterion, regime)
                    cs[criterion] = b.ordinalize(value)
                audit = 100 * sum(cs[x] * weights[x] for x in b.CRITERIA) / 3
                scores[method.name].append(audit)
                hard[method.name].append(sum(cs[x] < threshold for x in b.CRITERIA))
    return ({k: mean(v) for k, v in scores.items()}, {k: mean(v) for k, v in hard.items()})


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def validation() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    seeds = [20260711 + 97 * i for i in range(10)]
    seed_results = []
    seed_means = {}
    for seed in seeds:
        scores, hard = simulate(seed, 500, "full")
        seed_means[seed] = scores
        for method in scores:
            seed_results.append({"seed": seed, "method": method, "mean_score": round(scores[method], 4), "mean_hard_flags": round(hard[method], 4), "rank": int(ranks(scores)[method])})
    write_rows(OUT / "seed_stability.csv", seed_results)
    correlations = [spearman(ranks(seed_means[x]), ranks(seed_means[y])) for i, x in enumerate(seeds) for y in seeds[i+1:]]

    variants = ["full", "no_profile_difficulty", "single_regime", "no_failure_events", "no_interactions"]
    ablation_rows = []
    base_scores, _ = simulate(20260711, 750, "full")
    base_rank = ranks(base_scores)
    for variant in variants:
        scores, hard = simulate(20260711, 750, variant)
        for method in scores:
            ablation_rows.append({"variant": variant, "method": method, "mean_score": round(scores[method], 4), "score_change_from_full": round(scores[method] - base_scores[method], 4), "mean_hard_flags": round(hard[method], 4), "rank": int(ranks(scores)[method]), "rank_change": int(ranks(scores)[method] - base_rank[method])})
    write_rows(OUT / "ablation.csv", ablation_rows)

    rng = random.Random(20260711)
    counts = {c: dict(tp=0, fp=0, fn=0, tn=0) for c in b.CRITERIA}
    recovery_rows = []
    for criterion in b.CRITERIA:
        for _ in range(12000):
            profile = rng.choice([p for p in b.PROFILES if b.profile_status(p) == "documentary"]); method = rng.choice(b.METHODS); regime = rng.choice(list(b.REGIMES))
            noise = rng.uniform(0.55, 1.75)
            latent = 3.0 * (method.quality[criterion] - b.interaction_penalty(profile, method, criterion))
            baseline = latent + rng.gauss(0, 0.10 * noise)
            injected = rng.random() < 0.35
            observed = latent + rng.gauss(0, 0.10 * noise) - (rng.uniform(0.35, 1.20) if injected else 0.0)
            # Paired-control recovery: detect a criterion-specific decline rather
            # than confusing an already-low baseline with a newly injected event.
            predicted = (baseline - observed) >= 0.30
            key = "tp" if injected and predicted else "fn" if injected else "fp" if predicted else "tn"
            counts[criterion][key] += 1
    for criterion, c in counts.items():
        precision = c["tp"] / max(1, c["tp"] + c["fp"])
        recall = c["tp"] / max(1, c["tp"] + c["fn"])
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        recovery_rows.append({"criterion": criterion, **c, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)})
    write_rows(OUT / "injected_failure_recovery.csv", recovery_rows)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    full = [r for r in ablation_rows if r["variant"] == "full"]
    for v in variants[1:]:
        rows = [r for r in ablation_rows if r["variant"] == v]
        axes[0].plot(range(len(rows)), [r["score_change_from_full"] for r in rows], marker="o", label=v.replace("_", " "))
    axes[0].axhline(0, color="black", lw=.7); axes[0].set_title("Ablation: score change"); axes[0].set_ylabel("Points vs. full"); axes[0].set_xticks(range(8)); axes[0].set_xticklabels([r["method"].split("_")[0] for r in full], rotation=45, ha="right", fontsize=7); axes[0].legend(fontsize=6)
    by_method = defaultdict(list)
    for r in seed_results: by_method[r["method"]].append(r["mean_score"])
    axes[1].bar(range(8), [mean(by_method[m]) for m in by_method], yerr=[pstdev(by_method[m]) for m in by_method], color="#2a9d8f")
    axes[1].set_title("Ten-seed stability"); axes[1].set_ylabel("Mean score ± SD"); axes[1].set_xticks(range(8)); axes[1].set_xticklabels([m.split("_")[0] for m in by_method], rotation=45, ha="right", fontsize=7)
    axes[2].barh(range(7), [r["f1"] for r in recovery_rows], color="#e76f51")
    axes[2].set_yticks(range(7)); axes[2].set_yticklabels([r["criterion"].replace("_", " ") for r in recovery_rows], fontsize=7); axes[2].set_xlim(0, 1); axes[2].set_xlabel("F1"); axes[2].set_title("Injected-failure recovery")
    fig.tight_layout(); fig.savefig(FIG / "technical_validation.pdf", bbox_inches="tight"); fig.savefig(FIG / "technical_validation.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    summary = {"seeds": seeds, "scenarios_per_seed": 500, "profiles_used": 8, "score_scale": "ordinal 0,1,2,3", "mean_pairwise_spearman": round(mean(correlations), 4), "min_pairwise_spearman": round(min(correlations), 4), "macro_precision": round(mean(r["precision"] for r in recovery_rows), 4), "macro_recall": round(mean(r["recall"] for r in recovery_rows), 4), "macro_f1": round(mean(r["f1"] for r in recovery_rows), 4), "note": "Software unit tests and sensitivity analysis only. Fixed ranks reflect researcher-authored capability vectors and are not empirical robustness evidence."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__": validation()
