# Business Entity Resolution — reproducible pipeline

Neural-first entity resolution for the Amazon ML Challenge 2026: a from-scratch character n-gram
"fingerprint" bi-encoder generates candidates on the GPU, an XGBoost judge scores candidate pairs,
and a decision layer (one S1 per SX record + per-entity expected-F0.5 cut) produces the matches.
Only the provided data is used; no external API, database or lookup.

## Environment

- Windows or Linux, Python 3.12, an NVIDIA GPU with CUDA (tested: GTX 1650 4 GB), ~16 GB RAM, ~20 GB disk.
- `pip install -r requirements.txt` (torch CUDA wheel: `--index-url https://download.pytorch.org/whl/cu126`).

## Data layout

Set paths with environment variables (defaults in `src/config.py`):

| Variable | Meaning | Default |
|---|---|---|
| `ER_DATA_DIR` | folder containing `train/` and `test/` TSVs from `student_resource/dataset` | `C:\amazon_ml\dataset` |
| `ER_WORK_DIR` | scratch space for intermediate files | `C:\amazon_ml\work` |
| `ER_OUT_DIR` | where the two submission TSVs are written | `<repo>/output` |

## Run end to end

From this folder (`code/business_entity_resolution`):

```bash
python -m pytest tests -q                                   # unit tests
python -m src.run_pipeline prepare                          # normalise all records, E/J/V split, GT rows
python -m src.run_pipeline train-encoder                    # contrastive bi-encoder on split E (GPU)
python -m src.run_pipeline encode --split train             # fingerprints for train records
python -m src.run_pipeline encode --split test              # fingerprints for test records
python -m src.run_pipeline candidates --split train         # GPU kNN for J+V S1s + recall report
python -m src.run_pipeline candidates --split test          # GPU kNN for all test S1s
python -m src.run_pipeline tokens --split train             # token/number/legal arrays (bounded memory)
python -m src.run_pipeline tokens --split test
python -m src.run_pipeline features --split train           # pair features for J and V (optional --floor F)
python -m src.run_pipeline train-judge                      # XGBoost on J (GPU)
python -m src.run_pipeline validate                         # macro F0.5 on V, chooses the decision rule
python -m src.run_pipeline predict-test                     # writes output/*.tsv and runs the validator
```

`python -m src.run_pipeline rethreshold --rule threshold --thr 0.6` re-writes the submission from saved
test probabilities with a different decision rule (no feature recomputation).

## Source map

| File | Role |
|---|---|
| `src/normalize.py` | Brahmic transliteration, name/address normalisation, legal forms, state codes |
| `src/featurize.py` | numba hashing of words + char 3/4-grams into EmbeddingBag ids |
| `src/encoder.py` | fingerprint bi-encoder (training with in-batch InfoNCE, encoding) |
| `src/candidates.py` | GPU brute-force cosine kNN per country, recall report |
| `src/features.py` | token arrays + 44 pair features (cosines, token sets, numbers, fuzzy scores, flags) |
| `src/judge.py` | XGBoost (CUDA) classifier |
| `src/decide.py` | assignment (each SX to one S1), expected-F0.5 cut, threshold rule |
| `src/evaluate.py` | organizer macro F0.5 |
| `src/run_pipeline.py` | CLI stages above |
