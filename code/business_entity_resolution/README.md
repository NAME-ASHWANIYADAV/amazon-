# Business Entity Resolution — reproducible pipeline

Neural-first entity resolution for the Amazon ML Challenge 2026: a from-scratch character n-gram
"fingerprint" bi-encoder generates candidates on the GPU, a stage-A XGBoost filter prunes them, a stacked
XGBoost judge (stage 1 + stage 2 on within-list context) scores candidate pairs, a stage-3 recalibrator adds
generator-structure features, and a decision layer (one S1 per SX record + per-entity exact expected-F0.5 cut
+ lookalike-group rules) produces the matches. Only the provided data is used; no external API, database or
lookup, and no country is hard-coded: France-specific behaviour is keyed on "country not present in the
training data".

## Environment

- Windows or Linux, Python 3.12, an NVIDIA GPU with CUDA (tested: GTX 1650 4 GB), ~16 GB RAM, ~25 GB disk.
- `pip install -r requirements.txt` (torch CUDA wheel: `--index-url https://download.pytorch.org/whl/cu126`).
- All libraries are MIT/Apache/BSD licensed; the largest model is the 0.5 GB fingerprint encoder.

## Data layout

Set paths with environment variables (defaults in `src/config.py`):

| Variable | Meaning | Default |
|---|---|---|
| `ER_DATA_DIR` | folder containing `train/` and `test/` TSVs from `student_resource/dataset` | `C:\amazon_ml\dataset` |
| `ER_WORK_DIR` | scratch space for intermediate files | `C:\amazon_ml\work` |
| `ER_OUT_DIR` | where the two submission TSVs are written | `<repo>/output` |

## Run end to end

From this folder (`code/business_entity_resolution`), with `USE_TF=0` in the environment:

```bash
python -m pytest tests -q                                   # unit tests (61)
python -m src.run_pipeline prepare                          # normalise all records, E/J/V split, GT rows
python -m src.run_pipeline train-encoder                    # contrastive bi-encoder on split E (GPU)
python -m src.run_pipeline encode --split train             # fingerprints for train records
python -m src.run_pipeline encode --split test              # fingerprints for test records
python -m src.run_pipeline candidates --split train         # GPU kNN top-40 for J+V S1s + recall report
python -m src.run_pipeline candidates --split train --queries E   # top-20 lists of the E S1s (competitor table)
python -m src.run_pipeline candidates --split test          # GPU kNN top-40 for all test S1s
python -m src.run_pipeline address-pass --split train --k-addr 10   # + top-10 by address cosine
python -m src.run_pipeline address-pass --split test --k-addr 10
python -m src.run_pipeline tokens --split train             # token/number/legal arrays (bounded memory)
python -m src.run_pipeline tokens --split test
python -m src.run_pipeline vocab --split train              # name-token vocabularies (word log-odds transfer)
python -m src.run_pipeline vocab --split test
python -m src.run_pipeline features --split train           # stage-A filter, context features, J/V matrices
python -m src.run_pipeline train-judge                      # stage-1 XGBoost on J (GPU)
python -m src.run_pipeline validate                         # macro F0.5 on V, chooses the decision rule
python -m src.run_pipeline stage2                           # stage-2 judge (within-list stage-1 context)
python -m src.run_pipeline predict-test                     # stage-1 test probabilities + features
python -m src.run_pipeline stage3-features --split J        # generator-structure features (out-of-fold stage-1 view)
python -m src.run_pipeline stage3-features --split V1
python -m src.run_pipeline stage2-plus                      # stage-2 judge with those features (V +0.0013)
python -m src.run_pipeline predict-stage2-plus              # its test probabilities (becomes the pipeline's stage 2)
python -m src.run_pipeline stage3-features --split train    # generator-structure features on V (stage-2 view)
python -m src.run_pipeline stage3-train                     # stage-3 recalibrator (LightGBM on V)
python -m src.run_pipeline stage3-features --split test
python -m src.run_pipeline stage3-predict                   # stage-3 test probabilities (guarded)
python -m src.run_pipeline rethreshold --stage3 --rule expected_f --shift-rule --lookalike --caps --word-boost --style-add --block-add
```

(`predict-stage2` is the plain stage-2 judge without the generator-structure features, kept for comparison.)

The last command writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` (the stage-A survivors
that the judges score) and runs the organiser's validator. `rethreshold` re-decides saved probabilities without
recomputing features; its flags:

| Flag | Effect |
|---|---|
| `--stage2` / `--stage3` | use the stage-2 / stage-3 probabilities |
| `--rule expected_f` | per-S1 exact expected-F0.5 cut (`--rule threshold --thr T` for a global threshold) |
| `--shift-rule` | never match an SX whose house number is the S1's plus a distractor shift >= 3 while copies confirm the S1 number |
| `--lookalike` | drop lookalike fake groups after assignment (`src/lookalike.py`, rules R1/RM/RB/W/RF/RL/RC/RP by default: same-source anchor conflict, mixed-source groups, all-modified groups, fake words, unseen-country +1/+2 changes, legal swaps where copies never swap, composite shift+truncation numbers, 'partners' at +1/+2; `--rules ...,RW` adds the common-word-swap rule for unseen countries) |
| `--caps` | at most 5 S2 and 6 S3 matches per S1 (train truth maximum) |
| `--word-boost` | in unseen countries, raise exact-address copies whose only extra word is a dual-role word (groupe / developpement / france) |
| `--style-add` | in unseen countries, add unmatched same-number copies whose raw address style (number format, region rendering) agrees with the S1's confident same-source copies (`src/style.py`) |
| `--block-add` | in unseen countries, append same-name / acronym copies at the same house number that the kNN blocking missed (strict street match, unique S1) |
| `--country-thr`, `--country-shift` | per-country threshold / logit shift (not used in the final run) |

## Source map

| File | Role |
|---|---|
| `src/normalize.py` | Brahmic transliteration, name/address normalisation, legal forms, state codes, French abbreviations |
| `src/featurize.py` | numba hashing of words + char 3/4-grams into EmbeddingBag ids |
| `src/encoder.py` | fingerprint bi-encoder (training with in-batch InfoNCE, encoding) |
| `src/candidates.py` | GPU brute-force cosine kNN per country (record + address pass), recall report |
| `src/features.py` | token arrays, per-country idf, 45 pair features + competition features, acronym flag |
| `src/context.py` | generator-aware context: house-number alignment vs the distractor shift set, sibling counts, global name counts, word log-odds; stage-2 within-list features |
| `src/judge.py` | XGBoost (CUDA) classifiers (stage A, stage 1, stage 2) |
| `src/stage3.py` | generator-structure features (per-source address base, co-location counts over all S1s, duplicates, name-edit type, digit-level house-number edit) used inside the stage-2+ judge and by the stage-3 mid-band recalibrator (LightGBM on V) |
| `src/decide.py` | assignment (each SX to one S1), exact expected-F0.5 cut, threshold rule |
| `src/lookalike.py` | lookalike-group rules (same-source anchor conflict, mixed-source groups, all-modified groups, fake words, unseen-country rules), word boost |
| `src/style.py` | address-rendering style from the raw text, name-edit categories, street mismatch, style-agreement additions |
| `src/evaluate.py` | organizer macro F0.5 |
| `src/io_utils.py` | TSV reading, streaming submission writer |
| `src/run_pipeline.py` | CLI stages above |
| `src/cross_encoder.py` | character-level cross-encoder experiment (no gain on validation; not used in the final run) |
