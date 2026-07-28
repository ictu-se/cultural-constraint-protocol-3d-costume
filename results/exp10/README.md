# Qwen3-VL pairwise controlled-defect experiment

Four Blender proxy variants are compared with the baseline using labeled 2-by-2
contact sheets. The known changes target panel length, hem flare, sleeve length,
and body/slit proportion. Three prompt variants yield 12 comparisons.

Key results:

- some structural change detected: 50%;
- expected-criterion localization: 0%;
- prompt-consistent detection by case: 0%.

The VLM sometimes notices a visual difference but does not reliably map that
difference to the criterion targeted by the controlled intervention. This
negative result prevents using the VLM as an autonomous criterion scorer.

Reproduce with:

```powershell
python scripts/run_qwen3vl_pairwise_defects.py
```
