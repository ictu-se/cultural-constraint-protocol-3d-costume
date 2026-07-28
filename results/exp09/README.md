# Qwen3-VL view-ablation experiment

This experiment evaluates seven assets under four evidence conditions:
front-only, front plus side, front-side-back, and four-view. It contains 28
assessments and 196 criterion records.

Key results:

- exact asset-criterion stability across view conditions: 73.47%;
- cultural NA preservation: 100%;
- first-pass JSON schema validity: 92.86%;
- NA rates: 40.82% front-only, 34.69% front-side, 38.78%
  front-side-back, and 36.73% four-view.

The result measures presentation sensitivity of one VLM. It is not human
cultural validation.

Reproduce with:

```powershell
python scripts/run_qwen3vl_view_ablation.py
python scripts/analyze_qwen3vl_view_ablation.py
```
