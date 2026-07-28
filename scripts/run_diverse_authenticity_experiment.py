"""Diverse cultural-authenticity benchmark for Paper 4.

This experiment fixes the over-smoothed behavior of exp01 by introducing:
- more controlled costume-profile variants,
- scenario regimes with different criterion emphases,
- method-specific failure probabilities,
- profile-method interaction penalties,
- per-output stochastic variation.

The experiment remains a rubric simulation, not expert validation.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "exp02_diverse_authenticity_benchmark"

CRITERIA = [
    "garment_components",
    "silhouette_and_proportion",
    "sewing_pattern_plausibility",
    "material_and_drape",
    "motif_and_texture",
    "regional_period_consistency",
    "wearing_context",
]


@dataclass(frozen=True)
class Profile:
    name: str
    family: str
    difficulty: dict[str, float]
    ambiguity: float
    evidence: float


PROFILE_STATUS = {
    "court_inspired_formal": "placeholder",
    "dao_red_headwear_reference": "placeholder",
    "generic_ethnic_minorities_prompt": "rejected",
    "mixed_period_stress_case": "rejected",
}

CULTURAL_CRITERIA = {"regional_period_consistency", "wearing_context"}


@dataclass(frozen=True)
class Method:
    name: str
    quality: dict[str, float]
    variance: float
    failure_rate: float
    metadata_dependence: float


REGIMES = {
    "balanced": {
        "garment_components": 0.16,
        "silhouette_and_proportion": 0.15,
        "sewing_pattern_plausibility": 0.16,
        "material_and_drape": 0.12,
        "motif_and_texture": 0.15,
        "regional_period_consistency": 0.16,
        "wearing_context": 0.10,
    },
    "construction_strict": {
        "garment_components": 0.19,
        "silhouette_and_proportion": 0.18,
        "sewing_pattern_plausibility": 0.24,
        "material_and_drape": 0.10,
        "motif_and_texture": 0.09,
        "regional_period_consistency": 0.11,
        "wearing_context": 0.09,
    },
    "motif_strict": {
        "garment_components": 0.11,
        "silhouette_and_proportion": 0.11,
        "sewing_pattern_plausibility": 0.11,
        "material_and_drape": 0.12,
        "motif_and_texture": 0.28,
        "regional_period_consistency": 0.18,
        "wearing_context": 0.09,
    },
    "heritage_strict": {
        "garment_components": 0.14,
        "silhouette_and_proportion": 0.11,
        "sewing_pattern_plausibility": 0.12,
        "material_and_drape": 0.10,
        "motif_and_texture": 0.17,
        "regional_period_consistency": 0.23,
        "wearing_context": 0.13,
    },
    "visual_lenient": {
        "garment_components": 0.10,
        "silhouette_and_proportion": 0.18,
        "sewing_pattern_plausibility": 0.08,
        "material_and_drape": 0.18,
        "motif_and_texture": 0.20,
        "regional_period_consistency": 0.13,
        "wearing_context": 0.13,
    },
}


PROFILES = [
    Profile(
        "ao_dai_modern_school",
        "ao_dai",
        {
            "garment_components": 0.30,
            "silhouette_and_proportion": 0.35,
            "sewing_pattern_plausibility": 0.48,
            "material_and_drape": 0.34,
            "motif_and_texture": 0.22,
            "regional_period_consistency": 0.25,
            "wearing_context": 0.28,
        },
        ambiguity=0.18,
        evidence=0.78,
    ),
    Profile(
        "ao_dai_modern_wedding",
        "ao_dai",
        {
            "garment_components": 0.42,
            "silhouette_and_proportion": 0.42,
            "sewing_pattern_plausibility": 0.52,
            "material_and_drape": 0.48,
            "motif_and_texture": 0.62,
            "regional_period_consistency": 0.43,
            "wearing_context": 0.58,
        },
        ambiguity=0.25,
        evidence=0.72,
    ),
    Profile(
        "ao_dai_early_20c",
        "ao_dai",
        {
            "garment_components": 0.48,
            "silhouette_and_proportion": 0.56,
            "sewing_pattern_plausibility": 0.60,
            "material_and_drape": 0.50,
            "motif_and_texture": 0.52,
            "regional_period_consistency": 0.66,
            "wearing_context": 0.50,
        },
        ambiguity=0.38,
        evidence=0.56,
    ),
    Profile(
        "ao_ngu_than_ceremonial",
        "ao_ngu_than",
        {
            "garment_components": 0.78,
            "silhouette_and_proportion": 0.72,
            "sewing_pattern_plausibility": 0.82,
            "material_and_drape": 0.58,
            "motif_and_texture": 0.64,
            "regional_period_consistency": 0.76,
            "wearing_context": 0.72,
        },
        ambiguity=0.48,
        evidence=0.50,
    ),
    Profile(
        "ao_tu_than_festival",
        "ao_tu_than",
        {
            "garment_components": 0.68,
            "silhouette_and_proportion": 0.66,
            "sewing_pattern_plausibility": 0.74,
            "material_and_drape": 0.56,
            "motif_and_texture": 0.58,
            "regional_period_consistency": 0.66,
            "wearing_context": 0.70,
        },
        ambiguity=0.42,
        evidence=0.52,
    ),
    Profile(
        "khan_dong_aodai_performance",
        "ao_dai",
        {
            "garment_components": 0.58,
            "silhouette_and_proportion": 0.52,
            "sewing_pattern_plausibility": 0.54,
            "material_and_drape": 0.46,
            "motif_and_texture": 0.70,
            "regional_period_consistency": 0.62,
            "wearing_context": 0.68,
        },
        ambiguity=0.35,
        evidence=0.58,
    ),
    Profile(
        "court_inspired_formal",
        "formal",
        {
            "garment_components": 0.74,
            "silhouette_and_proportion": 0.76,
            "sewing_pattern_plausibility": 0.78,
            "material_and_drape": 0.72,
            "motif_and_texture": 0.86,
            "regional_period_consistency": 0.88,
            "wearing_context": 0.82,
        },
        ambiguity=0.55,
        evidence=0.42,
    ),
    Profile(
        "hmong_festival_reference",
        "ethnic_reference",
        {
            "garment_components": 0.82,
            "silhouette_and_proportion": 0.70,
            "sewing_pattern_plausibility": 0.72,
            "material_and_drape": 0.76,
            "motif_and_texture": 0.94,
            "regional_period_consistency": 0.88,
            "wearing_context": 0.80,
        },
        ambiguity=0.62,
        evidence=0.38,
    ),
    Profile(
        "cham_formal_reference",
        "ethnic_reference",
        {
            "garment_components": 0.70,
            "silhouette_and_proportion": 0.64,
            "sewing_pattern_plausibility": 0.66,
            "material_and_drape": 0.66,
            "motif_and_texture": 0.82,
            "regional_period_consistency": 0.84,
            "wearing_context": 0.74,
        },
        ambiguity=0.56,
        evidence=0.40,
    ),
    Profile(
        "dao_red_headwear_reference",
        "ethnic_reference",
        {
            "garment_components": 0.86,
            "silhouette_and_proportion": 0.66,
            "sewing_pattern_plausibility": 0.68,
            "material_and_drape": 0.70,
            "motif_and_texture": 0.92,
            "regional_period_consistency": 0.90,
            "wearing_context": 0.84,
        },
        ambiguity=0.68,
        evidence=0.36,
    ),
    Profile(
        "generic_ethnic_minorities_prompt",
        "ambiguous_prompt",
        {
            "garment_components": 0.90,
            "silhouette_and_proportion": 0.82,
            "sewing_pattern_plausibility": 0.80,
            "material_and_drape": 0.78,
            "motif_and_texture": 0.98,
            "regional_period_consistency": 1.00,
            "wearing_context": 0.90,
        },
        ambiguity=0.85,
        evidence=0.24,
    ),
    Profile(
        "mixed_period_stress_case",
        "stress",
        {
            "garment_components": 0.88,
            "silhouette_and_proportion": 0.88,
            "sewing_pattern_plausibility": 0.76,
            "material_and_drape": 0.82,
            "motif_and_texture": 0.90,
            "regional_period_consistency": 1.00,
            "wearing_context": 0.86,
        },
        ambiguity=0.78,
        evidence=0.28,
    ),
]


METHODS = [
    Method(
        "text_to_3d_generic",
        {
            "garment_components": 0.54,
            "silhouette_and_proportion": 0.60,
            "sewing_pattern_plausibility": 0.24,
            "material_and_drape": 0.56,
            "motif_and_texture": 0.34,
            "regional_period_consistency": 0.22,
            "wearing_context": 0.30,
        },
        variance=0.24,
        failure_rate=0.20,
        metadata_dependence=0.10,
    ),
    Method(
        "image_to_3d_reconstruction",
        {
            "garment_components": 0.62,
            "silhouette_and_proportion": 0.70,
            "sewing_pattern_plausibility": 0.34,
            "material_and_drape": 0.50,
            "motif_and_texture": 0.42,
            "regional_period_consistency": 0.30,
            "wearing_context": 0.34,
        },
        variance=0.20,
        failure_rate=0.15,
        metadata_dependence=0.15,
    ),
    Method(
        "pattern_driven_generation",
        {
            "garment_components": 0.78,
            "silhouette_and_proportion": 0.76,
            "sewing_pattern_plausibility": 0.84,
            "material_and_drape": 0.48,
            "motif_and_texture": 0.34,
            "regional_period_consistency": 0.38,
            "wearing_context": 0.42,
        },
        variance=0.16,
        failure_rate=0.10,
        metadata_dependence=0.25,
    ),
    Method(
        "texture_motif_transfer",
        {
            "garment_components": 0.56,
            "silhouette_and_proportion": 0.58,
            "sewing_pattern_plausibility": 0.48,
            "material_and_drape": 0.66,
            "motif_and_texture": 0.78,
            "regional_period_consistency": 0.42,
            "wearing_context": 0.40,
        },
        variance=0.18,
        failure_rate=0.12,
        metadata_dependence=0.30,
    ),
    Method(
        "metadata_guided_hybrid",
        {
            "garment_components": 0.78,
            "silhouette_and_proportion": 0.76,
            "sewing_pattern_plausibility": 0.72,
            "material_and_drape": 0.68,
            "motif_and_texture": 0.76,
            "regional_period_consistency": 0.78,
            "wearing_context": 0.76,
        },
        variance=0.15,
        failure_rate=0.08,
        metadata_dependence=0.70,
    ),
    Method(
        "panel_metadata_hybrid",
        {
            "garment_components": 0.84,
            "silhouette_and_proportion": 0.82,
            "sewing_pattern_plausibility": 0.86,
            "material_and_drape": 0.70,
            "motif_and_texture": 0.70,
            "regional_period_consistency": 0.74,
            "wearing_context": 0.70,
        },
        variance=0.13,
        failure_rate=0.06,
        metadata_dependence=0.62,
    ),
    Method(
        "retrieval_reference_guided",
        {
            "garment_components": 0.80,
            "silhouette_and_proportion": 0.78,
            "sewing_pattern_plausibility": 0.62,
            "material_and_drape": 0.78,
            "motif_and_texture": 0.84,
            "regional_period_consistency": 0.82,
            "wearing_context": 0.78,
        },
        variance=0.12,
        failure_rate=0.05,
        metadata_dependence=0.82,
    ),
    Method(
        "expert_reference_upper_bound",
        {
            "garment_components": 0.94,
            "silhouette_and_proportion": 0.93,
            "sewing_pattern_plausibility": 0.92,
            "material_and_drape": 0.90,
            "motif_and_texture": 0.91,
            "regional_period_consistency": 0.95,
            "wearing_context": 0.94,
        },
        variance=0.07,
        failure_rate=0.02,
        metadata_dependence=0.95,
    ),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(v, 0.001) for v in weights.values())
    return {k: max(v, 0.001) / total for k, v in weights.items()}


def scenario_weights(rng: random.Random, regime: str, severity: float) -> dict[str, float]:
    base = REGIMES[regime]
    return normalize({c: base[c] * math.exp(rng.gauss(0, severity)) for c in CRITERIA})


def clamp(value: float, low: float = 0.0, high: float = 3.0) -> float:
    return max(low, min(high, value))


def profile_status(profile: Profile) -> str:
    return PROFILE_STATUS.get(profile.name, "documentary")


def assessable_criteria(profile: Profile) -> list[str]:
    status = profile_status(profile)
    if status == "rejected":
        return []
    if status == "placeholder":
        return [c for c in CRITERIA if c not in CULTURAL_CRITERIA]
    return list(CRITERIA)


def ordinalize(latent: float) -> int:
    """Map a clipped latent response to the four declared rubric anchors."""
    value = clamp(latent)
    if value < 0.5:
        return 0
    if value < 1.5:
        return 1
    if value < 2.5:
        return 2
    return 3


def interaction_penalty(profile: Profile, method: Method, criterion: str) -> float:
    capability_gap = 1.0 - method.quality[criterion]
    penalty = profile.difficulty[criterion] * (0.10 + 0.24 * capability_gap)
    penalty += profile.ambiguity * (1.0 - method.metadata_dependence) * 0.20
    penalty += (1.0 - profile.evidence) * (1.0 - method.metadata_dependence) * 0.12
    if criterion == "motif_and_texture" and profile.family in {"ethnic_reference", "ambiguous_prompt", "stress"}:
        penalty += (1.0 - method.quality["motif_and_texture"]) * 0.18
    if criterion == "regional_period_consistency":
        penalty += profile.ambiguity * (1.0 - method.quality[criterion]) * 0.22
    if criterion == "sewing_pattern_plausibility" and method.name in {"text_to_3d_generic", "image_to_3d_reconstruction"}:
        penalty += 0.10 + profile.difficulty[criterion] * 0.10
    return penalty


def failure_drop(rng: random.Random, profile: Profile, method: Method, criterion: str, regime: str) -> float:
    risk = method.failure_rate
    risk += profile.ambiguity * 0.05
    risk += profile.difficulty[criterion] * 0.04
    if regime == "construction_strict" and criterion in {"garment_components", "sewing_pattern_plausibility"}:
        risk += 0.06
    if regime in {"heritage_strict", "motif_strict"} and criterion in {"motif_and_texture", "regional_period_consistency", "wearing_context"}:
        risk += 0.07
    if rng.random() >= risk:
        return 0.0
    return rng.uniform(0.35, 1.20)


def score(profile: Profile, method: Method, criterion: str, regime: str, rng: random.Random, noise_scale: float) -> int:
    signal = method.quality[criterion] - interaction_penalty(profile, method, criterion)
    signal += rng.gauss(0, method.variance * noise_scale)
    value = 3.0 * signal
    value -= failure_drop(rng, profile, method, criterion, regime)
    return ordinalize(value)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(n_scenarios: int = 6000, seed: int = 20260711) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    regimes = list(REGIMES)
    detail_fields = [
        "scenario_id",
        "regime",
        "profile",
        "method",
        "profile_status",
        "record_status",
        "audit_score",
        "assessable_count",
        "na_count",
        "na_reason",
        "hard_flags",
        "weight_severity",
        "noise_scale",
        "hard_threshold",
        *[f"score_{c}" for c in CRITERIA],
        *[f"weight_{c}" for c in CRITERIA],
    ]
    for scenario_idx in range(n_scenarios):
        regime = regimes[scenario_idx % len(regimes)]
        severity = rng.uniform(0.10, 0.55)
        noise_scale = rng.uniform(0.55, 1.75)
        hard_threshold = rng.choice([1, 2, 3])
        weights = scenario_weights(rng, regime, severity)
        scenario_id = f"s{scenario_idx + 1:06d}"
        method_scores: dict[str, list[float]] = {m.name: [] for m in METHODS}
        for profile in PROFILES:
            status = profile_status(profile)
            assessable = assessable_criteria(profile)
            for method in METHODS:
                criterion_scores = {c: score(profile, method, c, regime, rng, noise_scale) for c in assessable}
                if assessable:
                    assessable_weight = sum(weights[c] for c in assessable)
                    record_weights = {c: weights[c] / assessable_weight for c in assessable}
                    weighted = sum(criterion_scores[c] * record_weights[c] for c in assessable)
                    audit = 100.0 * weighted / 3.0
                    hard = sum(1 for v in criterion_scores.values() if v < hard_threshold)
                    record_status = "assessable"
                else:
                    audit = None
                    hard = 0
                    record_status = "rejected_profile"
                detail_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "regime": regime,
                        "profile": profile.name,
                        "method": method.name,
                        "profile_status": status,
                        "record_status": record_status,
                        "audit_score": "" if audit is None else round(audit, 4),
                        "assessable_count": len(assessable),
                        "na_count": len(CRITERIA) - len(assessable),
                        "na_reason": (
                            "profile rejected: under-specified or internally conflicting"
                            if status == "rejected"
                            else "object-level cultural evidence absent"
                            if status == "placeholder"
                            else ""
                        ),
                        "hard_flags": hard,
                        "weight_severity": round(severity, 4),
                        "noise_scale": round(noise_scale, 4),
                        "hard_threshold": round(hard_threshold, 4),
                        **{f"score_{c}": criterion_scores.get(c, "NA") for c in CRITERIA},
                        **{f"weight_{c}": round(weights[c], 5) for c in CRITERIA},
                    }
                )
                if status == "documentary" and audit is not None:
                    method_scores[method.name].append(audit)
        method_level = [(mean(scores), method) for method, scores in method_scores.items() if scores]
        ordered = sorted(method_level, reverse=True)
        summary_rows.append(
            {
                "scenario_id": scenario_id,
                "regime": regime,
                "winner_method": ordered[0][1],
                "winner_score": round(ordered[0][0], 4),
                "winner_margin": round(ordered[0][0] - ordered[1][0], 4),
                "profiles_aggregated": len(method_scores[ordered[0][1]]),
                "weight_severity": round(severity, 4),
                "noise_scale": round(noise_scale, 4),
                "hard_threshold": round(hard_threshold, 4),
            }
        )
    write_csv(OUT / "scenario_detail.csv", detail_fields, detail_rows)
    write_csv(
        OUT / "scenario_summary.csv",
        [
            "scenario_id",
            "regime",
            "winner_method",
            "winner_score",
            "winner_margin",
            "profiles_aggregated",
            "weight_severity",
            "noise_scale",
            "hard_threshold",
        ],
        summary_rows,
    )
    aggregate(detail_rows, summary_rows, seed, n_scenarios)


def aggregate(detail_rows: list[dict], summary_rows: list[dict], seed: int, n_scenarios: int) -> None:
    by_profile_method: dict[tuple[str, str], list[dict]] = {}
    by_regime_method: dict[tuple[str, str], list[dict]] = {}
    for row in detail_rows:
        by_profile_method.setdefault((row["profile"], row["method"]), []).append(row)
        if row["profile_status"] == "documentary":
            by_regime_method.setdefault((row["regime"], row["method"]), []).append(row)

    pm_rows = []
    for (profile, method), rows in sorted(by_profile_method.items()):
        vals = [float(r["audit_score"]) for r in rows if r["audit_score"] != ""]
        hard = [int(r["hard_flags"]) for r in rows]
        if not vals:
            pm_rows.append(
                {
                    "profile": profile,
                    "method": method,
                    "n": len(rows),
                    "mean_audit_score": "NA",
                    "std_audit_score": "NA",
                    "min_audit_score": "NA",
                    "max_audit_score": "NA",
                    "mean_hard_flags": "NA",
                }
            )
            continue
        pm_rows.append(
            {
                "profile": profile,
                "method": method,
                "n": len(rows),
                "mean_audit_score": round(mean(vals), 4),
                "std_audit_score": round(pstdev(vals), 4),
                "min_audit_score": round(min(vals), 4),
                "max_audit_score": round(max(vals), 4),
                "mean_hard_flags": round(mean(hard), 4),
            }
        )
    write_csv(
        OUT / "aggregate_profile_method.csv",
        [
            "profile",
            "method",
            "n",
            "mean_audit_score",
            "std_audit_score",
            "min_audit_score",
            "max_audit_score",
            "mean_hard_flags",
        ],
        pm_rows,
    )

    rm_rows = []
    for (regime, method), rows in sorted(by_regime_method.items()):
        vals = [float(r["audit_score"]) for r in rows]
        rm_rows.append(
            {
                "regime": regime,
                "method": method,
                "n": len(rows),
                    "mean_audit_score": round(mean(vals), 4),
                    "std_audit_score": round(pstdev(vals), 4),
            }
        )
    write_csv(
        OUT / "aggregate_regime_method.csv",
        ["regime", "method", "n", "mean_audit_score", "std_audit_score"],
        rm_rows,
    )

    winners: dict[str, list[float]] = {}
    regime_winners: dict[tuple[str, str], int] = {}
    for row in summary_rows:
        winners.setdefault(row["winner_method"], []).append(float(row["winner_margin"]))
        regime_winners[(row["regime"], row["winner_method"])] = regime_winners.get((row["regime"], row["winner_method"]), 0) + 1
    win_rows = [
        {
            "method": method,
            "winner_count": len(margins),
            "winner_rate": round(len(margins) / len(summary_rows), 6),
            "mean_winner_margin": round(mean(margins), 4),
        }
        for method, margins in sorted(winners.items())
    ]
    write_csv(OUT / "aggregate_winner_stability.csv", ["method", "winner_count", "winner_rate", "mean_winner_margin"], win_rows)

    regime_win_rows = [
        {"regime": regime, "method": method, "winner_count": count}
        for (regime, method), count in sorted(regime_winners.items())
    ]
    write_csv(OUT / "aggregate_regime_winners.csv", ["regime", "method", "winner_count"], regime_win_rows)

    state = {
        "created_at_utc": utc_now(),
        "seed": seed,
        "scenario_count": n_scenarios,
        "profile_count": len(PROFILES),
        "method_count": len(METHODS),
        "record_count": len(detail_rows),
        "assessable_record_count": sum(r["record_status"] == "assessable" for r in detail_rows),
        "rejected_record_count": sum(r["record_status"] == "rejected_profile" for r in detail_rows),
        "winner_aggregation": "mean audit score over eight documentary profiles for each method within each scenario",
        "score_scale": "ordinal 0,1,2,3 with an explicitly separate latent response",
        "regimes": list(REGIMES),
        "note": "Software-fixture simulation with researcher-authored parameters. Not empirical method comparison, construct validation, or expert evidence.",
    }
    (OUT / "summary.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Experiment 02 - Diverse Cultural Authenticity Benchmark\n\n"
        "This experiment replaces the over-smoothed exp01 design with more profile variants, scenario regimes, profile-method interactions, and stochastic failure events.\n\n"
        "It is still a rubric simulation, not expert validation.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run()
