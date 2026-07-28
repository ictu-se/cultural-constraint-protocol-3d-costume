# Reproducibility archive

This archive reproduces the simulator, software verification, instrumentation
diagnostic, and record-generation demonstrations. It does not
contain expert ratings and does not validate Vietnamese cultural correctness.

The simulator is a software fixture: its method capability vectors and profile
parameters are researcher-authored. Its rankings are not empirical comparisons.

## Contents

- `scripts/run_diverse_authenticity_experiment.py`: main 6,000-scenario run.
- `scripts/run_validation_extensions.py`: ablations, ten-seed stability, and
  paired injected-failure recovery.
- `scripts/run_asset_level_benchmark.py`: parent-disjoint controlled benchmark
  over 20 reusable garment meshes, six intervention families, and nuisance
  controls.
- `scripts/regenerate_manuscript_figures.py`: figure regeneration.
- `scripts/make_graphical_abstract.py`: code-native graphical abstract.
- `config/`: rubric, metadata schema, documentary evidence matrix, and heritage
  metadata mapping, 12-slot profile catalogue, and complete scoring anchors.
- `results/exp02/`: aggregate outputs and run summary.
- `results/exp03/`: ablation, seed-stability, failure-recovery, and summary.
- `results/exp04/`: software instrumentation diagnostic over 20 meshes and 960
  intervention/control records. The category-balanced split and parent-cluster
  bootstrap test record propagation and selected geometry transforms. This is
  not validation of Equation (1), the seven candidate criteria, or visual
  invariance; three channels read engineered intervention variables and the
  camera/background/lighting null controls are not rerendered.
- `results/exp05/`: five Blender 5.1.2 procedural Vietnamese-costume proxy
  assets (OBJ/GLB/BLEND), standardized renders, parameters, and checksums.
- `results/exp06/`: two TripoSR single-image reconstructions (one licensed
  documentary image and one controlled study render), 512-cubed meshes,
  4096-pixel texture atlases, 3200-by-4000 Cycles renders, tool/model
  provenance, and checksums.
- `results/exp07/`: criterion-level author technical assessments and
  asset-level summaries for all seven named-tool assets.
- `results/exp08/`: 21 Qwen3-VL-8B AI-assisted assessments over seven assets
  and three prompt variants, 147 criterion records, prompt-stability analysis,
  model/input hashes, and an explicit non-substitution claim boundary.
- `results/exp09/`: 28 Qwen3-VL view-ablation assessments, 196 criterion
  records, schema-validity tracking, and view-sensitivity summaries.
- `results/exp10/`: 12 pairwise Qwen3-VL controlled-defect comparisons,
  labeled contact sheets, change/localization records, and negative results.
- `results/exp11/`: 40 atomic, reference-anchored Qwen3-VL judgements over
  four controlled variants, five criteria, and two position orders, with a
  confusion matrix, class-aware metrics, position consistency, and prompts.
- `results/exp12/`: multi-model atomic VLM benchmark with three complete
  40-decision matrices, two documented runtime/schema failures, and explicit
  technical-only claim boundaries.
- `SHA256SUMS.txt`: integrity checks for every archived file.
- `RELEASE_MANIFEST.md`: experiment-group and claim-boundary inventory.
- `tests/test_revised_simulator.py`: regression tests for ordinal cut points,
  record states, and method-level winner aggregation.

The archive includes the full `scenario_detail.csv` (576,000 rows); ZIP
compression keeps the release portable. Regenerate all results deterministically
with:

```powershell
python scripts/run_diverse_authenticity_experiment.py
python scripts/run_validation_extensions.py
python scripts/run_asset_level_benchmark.py
blender --background --python scripts/generate_named_tool_aodai_assets.py
python scripts/document_triposr_run.py
blender --background --python scripts/render_triposr_outputs.py
python scripts/assess_named_tool_assets.py
python scripts/run_qwen3vl_audit.py --asset all --variant all
python scripts/analyze_qwen3vl_audit.py
python scripts/run_qwen3vl_view_ablation.py
python scripts/analyze_qwen3vl_view_ablation.py
python scripts/run_qwen3vl_pairwise_defects.py
python scripts/run_qwen3vl_atomic_pairwise.py
python scripts/run_multimodel_vlm_atomic_benchmark.py
python scripts/regenerate_manuscript_figures.py
python scripts/make_graphical_abstract.py
pytest -q
```

The main run uses seed `20260711`. Validation seed lists and sample counts are
recorded in `results/exp03/summary.json`.

## Environment

- Python 3.12
- pandas
- numpy
- matplotlib
- scipy
- scikit-learn

Install the pinned environment snapshot with:

```powershell
python -m pip install -r requirements.txt
```

## Claim boundary

Method families are encoded archetypes, not measured deployed systems. Profile
difficulties, failure rates, and capability values are researcher-defined.
Outputs support technical traceability and robustness analysis only.
The nuisance tests are deterministic representation/render-state proxies rather
than a substitute for independently rendered photometric evaluation.

## Evidence not contained in this repository

The repository contains no ratings by Vietnamese costume specialists, museum
professionals, historians, makers, or community representatives. It does
contain seven Vietnamese-costume test assets from Blender and TripoSR, but
these do not establish cultural validity or comparative deployed-model
performance. Completed human review remains a prerequisite for any cultural
validation claim.
