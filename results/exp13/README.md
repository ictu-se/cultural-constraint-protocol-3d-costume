# Experiment 13: independent exported-mesh detector

This experiment measures the five procedural Blender assets from their exported
GLB files, independently of the generation script and VLM prompts. It records
mesh hashes, vertex/face counts, connected components, watertightness, Euler
number, axis-aligned bounds, surface area, and signed volume.

All four controlled variants cross at least one predeclared 0.5% geometric
change threshold relative to `BL_AODAI_001`. The detector therefore recovers
4/4 known geometry interventions:

- longer panels: +2.61% x extent, +1.63% surface area;
- flared hem: +17.79% x extent, +7.13% surface area;
- short sleeves: -2.39% y extent, -6.65% surface area;
- slimmer body/higher slit: -2.85% x extent, -5.51% surface area.

This result is an instrumentation check, not a cultural assessment. Criterion
names come from the programmatic intervention manifest; the detector itself
does not infer authenticity, region, period, context, or cultural correctness.

Reproduce with:

```text
python scripts/run_mesh_geometry_detector.py
```
