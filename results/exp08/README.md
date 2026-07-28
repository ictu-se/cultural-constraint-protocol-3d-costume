# Qwen3-VL AI-assisted audit

This experiment uses the official `Qwen/Qwen3-VL-8B-Instruct-GGUF` model
(Apache-2.0), quantized as Q4_K_M with a Q8_0 vision projector. It runs through
llama.cpp b10103 with 99 GPU layers on the NVIDIA RTX 5080 Laptop GPU.

The model is an automated visual evidence gate, not a Vietnamese costume
specialist or community representative. It may describe visible geometry,
identify missing evidence, and produce structured criterion records. It cannot
create cultural authority, community endorsement, object provenance, or human
inter-rater validity. Regional-period and wearing-context fields therefore
remain NA when documentary evidence and qualified assessment are absent.

## Reproduction

```powershell
python scripts/run_qwen3vl_audit.py --asset all --variant all
python scripts/analyze_qwen3vl_audit.py
```

The full run uses temperature 0, three render views for each of seven assets,
three semantically equivalent prompt variants with varied view order, explicit
anti-inference rules, JSON validation, input hashes, model hashes, and a fixed
claim boundary. It produces 21 assessments and 147 criterion records.
`analysis_summary.json` reports 65.31% exact prompt stability, 100% preservation
of NA for regional-period and wearing-context criteria, median absolute
deviation 0, and mean absolute deviation 0.235 versus the author technical
scores. This is not agreement with an expert panel.

## Local model

- Model: `D:/3d modeling/tools/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf`
- Vision projector: `D:/3d modeling/tools/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf`
- Model SHA-256:
  `67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2`
- Projector SHA-256:
  `c6ba85508d82f42590e6eb77d5340369ab6fecf107a7561d809523d8aa5f3bfd`

## Measured runtime

`llama-bench` identified the Vulkan backend and the RTX 5080 Laptop GPU. With
all 99 layers requested on GPU, the local benchmark measured approximately
3349.60 prompt tokens/s for pp256 and 96.84 generated tokens/s for tg64. The
individual audits completed in approximately 11--20 seconds.
