# Experiment 11: atomic reference-anchored VLM judge

This experiment evaluates Qwen3-VL-8B-Instruct on four controlled Blender
variants. Each call asks one binary question for one of five visual/technical
criteria and includes a truthful intervention-specific proposition. Baseline
and variant positions are swapped to test positional consistency.

The 40 judgements yield TP=2, FP=2, TN=30, and FN=6: accuracy 0.80, precision
0.50, recall 0.25, specificity 0.9375, F1 0.3333, position consistency 0.80,
and first-pass schema validity 1.00. These outputs test controlled visual-defect
recognition only. They are not specialist ratings or cultural ground truth.

See `analysis_summary.json` for aggregate metrics, `judgements.csv` for
case-level records, the JSON files for prompts and raw responses, and the PNG
files for labeled contact sheets.
