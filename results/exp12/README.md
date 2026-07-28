# Experiment 12: multi-model atomic VLM benchmark

This experiment tests local VLMs against programmatically known changes in four
procedural 3D áo dài proxy variants. Each model receives a reference/candidate
contact sheet and one atomic technical proposition. Five criteria and two image
orders produce 40 expected decisions per model.

This is a technical visual-discrimination benchmark. It does not test cultural
authenticity, does not create cultural ground truth, and does not replace
Vietnamese costume specialists or community representatives.

## Completed model matrices

| Model | Complete | Accuracy | Balanced accuracy | Recall | Specificity | Positive rate | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL 3B | 40/40 | 0.150 | 0.375 | 0.750 | 0.000 | 0.950 | -0.459 |
| MiniCPM-V | 40/40 | 0.200 | 0.500 | 1.000 | 0.000 | 1.000 | 0.000 |
| LLaVA 7B | 40/40 | 0.275 | 0.500 | 0.875 | 0.125 | 0.875 | 0.000 |

Moondream completed 31/40 parseable decisions; all 31 were positive. Nine
outputs failed the declared schema, primarily by echoing the wrong criterion.
Qwen3-VL 4B produced only 7/40 parseable checkpointed decisions under the
tested Ollama JSON runtime, so its apparent partial-set score is not comparable
and must not be interpreted as model accuracy. The earlier Experiment 11
contains a complete 40-decision Qwen3-VL 8B matrix.

## Interpretation

All three complete new matrices exhibit severe positive-response bias. High
recall therefore does not indicate useful discrimination: specificity is
0–0.125 and balanced accuracy is no greater than chance. Image-order
consistency is high, but a consistently positive response is not a valid judge.
Together with Experiment 11's low positive recall, these results falsify the
hypothesis that any tested local VLM can autonomously score the protocol.

The appropriate computational use is limited to experimental triage after
independent calibration. VLM outputs cannot address the reviewer's request for
real cultural validation.

## Reproduction

Requires Ollama with the named local models:

```text
python scripts/run_multimodel_vlm_atomic_benchmark.py
python scripts/run_multimodel_vlm_atomic_benchmark.py --analyze-only
```

Every successful judgement is stored as an individual JSON checkpoint.
`analysis_summary.json` reports class-aware metrics and completion rates;
`judgements.csv` is the flat analysis table.
