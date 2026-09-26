# Business Entity Resolution — Design (Amazon ML Challenge 2026)

Date: 2026-09-26 · Approach: **B + safety net** (neural-first with an XGBoost safety-net judge)

## 1. Goal and success criteria

For every Source-1 (S1) entity, output the Source-2/3 (SX) records of the same business.
Metric: macro F0.5 per S1 entity (singleton: empty prediction = 1.0, any prediction = 0.0).
Success = best possible **private** leaderboard score. Deadline 2026-09-27 23:59 IST,
5 uploads per day (today's 5 expire at midnight).

## 2. Facts from EDA that shape the design

| Fact | Consequence |
|---|---|
| Train S1 2.21M / SX 10.3M (US, India). Test S1 1.73M / SX 9.97M, adds France (259k S1, no labels) | Everything per country, chunked; nothing hard-coded to a country list |
| Every SX id belongs to at most one S1 (0 exceptions in 7.6M pairs) | Hard assignment: each SX goes only to its best S1 |
| Singletons 5.6%, mean 3.46 matches per S1 (max 11) | Recall matters too; per-entity cut, not a flat threshold |
| 26% of train SX are unmatched; test has 5.7 SX per S1 vs 4.67 in train | Expect more distractors in test; stress-test validation |
| 99.8% of true pairs share a name token, the house number, or ≥2 address words | ≥99.5% shortlist recall is achievable |
| Lookalike negatives sharing a rare name token share the house number only 0.1% (US) / 3% (India) of the time; hardest ones sit on the same street with a shifted number and an extra word ("South", "Holdings") | Address/number features are the main precision weapon |
| 18% of India SX names are in Indic scripts (Devanagari, Bengali, Odia, Telugu, Gujarati, Kannada, Gurmukhi, Tamil, Malayalam) | One shared Brahmic transliteration table |
| Some true pairs have unrelated names (domain names, new trade names) | Address-only evidence must be able to win |
| polars 1.41 crashed natively on list-heavy ops (`map_elements`/`explode`) | Use numpy / pure Python for those steps |

Machine: 15 GB RAM (~5–9 GB free), GTX 1650 4 GB (torch 2.12 cu126, XGBoost CUDA), 6c/12t CPU.
Scripts set `USE_TF=0` (an old TensorFlow breaks `transformers` import).

## 3. Architecture and data flow

```
raw TSV ─► normalize ─► featurize (hashed char n-grams) ─► encoder (GPU) ─► embeddings
        ─► candidates: GPU kNN top-K per S1  ∪  exact-key safety net  ─► candidate_pairs.tsv
        ─► pair features ─► judge v1 (XGBoost GPU) [+ v2 tiny cross-encoder, stacked]
        ─► decide: SX→best S1, per-S1 expected-F0.5 cut ─► matching_results.tsv
```

## 4. Components (each a module in `code/business_entity_resolution/src/`)

**normalize.py** — pure functions, no I/O.
- `transliterate(s)`: Brahmic scripts → Latin through one table keyed on the offset inside
  each script's Unicode block (virama removes the inherent vowel, anusvara → n).
- `norm_name(s)` → `name_norm` (lowercase, Latin accents stripped, brackets/quotes/junk
  prefixes like `>>`, `<<`, `--` removed, `&`→`and`), `name_core` (legal forms, honorifics
  Mr/Dr/Shri/Smt, "the", phone numbers removed), `legal` (canonical legal tokens:
  inc, llc, ltd, pvt, corp, co, plc, pc, pllc, llp, lp, sarl, sas, sasu, eurl, sa, sci, snc, ei),
  flags `was_indic`, `is_domain` (domain stripped of TLD), `alt_names` (split on
  `formerly:` / `dba` / `aka`).
- `norm_addr(s)` → `addr_norm` (null/n/a/`<null>` removed, `#`/`no.`/`n°`/`nº` removed,
  leading zeros stripped, street-type and French abbreviations canonicalised:
  st/str→street, rd→road, ave/av→avenue, blvd/bd→boulevard, dr→drive, ln→lane, ct→court,
  r/r.→rue, ch→chemin, all→allee, imp→impasse, pl→place, rte→route, crs→cours),
  `numbers` (ordered numeric tokens), `addr_empty`.
- Unit-tested on real examples from EDA (transliteration, numbers, abbreviations).

**featurize.py** — numba hashing of char 3/4-grams (word-boundary padded) + word
unigrams into 2^21 buckets; name → [0, 2^20), address → [2^20, 2^21). Returns CSR-style
(indices, offsets) ready for `EmbeddingBag`. Operates on chunks (no full-corpus matrices in RAM).

**encoder.py** — "fingerprint" bi-encoder, trained from scratch.
- EmbeddingBag(2^21 × 64, sum, sparse grads) with a learned per-bucket weight
  (softplus gate) → name tower MLP(64→128→64) and address tower MLP(64→128→64),
  each L2-normalised; record = L2norm(concat(name, addr)).
- Loss: symmetric InfoNCE (τ≈0.05, batch ≈8k positive pairs) on the record embedding
  plus auxiliary InfoNCE on each tower, so name-cosine and address-cosine are meaningful.
- Trained on split E only. v1: in-batch negatives, 2 epochs. v2 (overnight): mined hard
  negatives (same-name S1 in other cities, lookalike SX).
- `encode()` writes fp16 embeddings per split/country to disk (memmap).

**candidates.py**
- GPU brute-force cosine per country: S1 query chunks × SX blocks (fp32 matmul, ~3 G pairs/s),
  running top-K per S1 (K chosen on V from the recall curve, default 40) and, on full passes,
  running top-5 S1 per SX (reverse ranks).
- Safety-net keys: (primary house number + shared rare street word) and (rare core-name token
  + city token), capped per S1; kept only if they add recall on V.
- Writes the final candidate list (exactly what the judge scores) as `candidate_pairs.tsv`.

**features.py** — vectorised pair features (rapidfuzz `cpdist`, numpy):
- neural: cos_record, cos_name, cos_addr, rank of SX in S1's list, gap to best, list size;
- name: token Jaccard/containment, token_set / partial ratio on `name_core`, legal-form
  agree/differ/missing, first-token match, count/IDF of extra tokens, learned "suspicious extra
  word" score (from J: how often a token is an extra word in non-matches vs matches),
  was_indic, is_domain (+ concatenated-name similarity), best score over `alt_names`;
- address: primary number equal, |Δ| and relative Δ of primary numbers, digit edit distance,
  any-number overlap, unit-number match, street-word overlap, city match, empty flags;
- source (S2/S3).

**judge.py** — XGBoost (`device=cuda`, hist, binary:logistic, early stopping on a slice of J).
v1 trained on J's candidate pairs. Final model = v1 features + v2 cross-encoder score +
stage-2 features (stacking).

**cross_encoder.py** (phase 2, overnight) — tiny word-level cross-encoder from scratch:
each word embedded from its hashed char n-grams (+ segment/position + log numeric value),
2-layer Transformer (d=128, 4 heads, ≤48 tokens) over both records, CLS → probability.
Trained on split-E candidate pairs (from the overnight full-train kNN) so its scores on J/V/test
are not leaked; applied only to grey-zone pairs (v1 p ∈ [0.02, 0.98]).

**Stage-2 group features** (phase 2, tomorrow) — from stage-1 probabilities over the full
candidate table: is this S1 the SX's best S1, prob gap to the SX's 2nd-best S1, agreement of the
S1's other high-prob records with this record's house number / core name.

**decide.py**
1. Assignment: keep each SX only for its highest-probability S1.
2. Per-S1 cut: sort remaining candidates by p; choose k ∈ {0..n} maximising expected F0.5
   (Monte-Carlo over independent Bernoulli(p), k = 0 wins when P(no match) is high).
   A global threshold is computed too; the better one on V is used.
3. Optional per-country offset (used for France, set by leaderboard probes).

**evaluate.py** — exact organizer metric (macro F0.5, singleton rule), plus shortlist recall,
precision, recall, reported per country.

**run_pipeline.py** — one command from TSVs to `output/matching_results.tsv` and
`output/candidate_pairs.tsv`; runs the official validator at the end.

## 5. Splits and validation

- Train S1 entities split deterministically by hash of `entity_id`: **E 75%** (encoder),
  **J 15%** (judge), **V 10%** (validation). All SX stay in every pool.
- Tonight's V evaluation scores V's candidates against the full SX pool; SX owned by non-V
  S1s cannot be claimed by their real owner here, so V is slightly pessimistic (safe for F0.5).
  Tomorrow's full-train pass removes this bias.
- Stress test: drop 19% of V S1s (keep their SX) to mimic the test's higher SX/S1 ratio.
  Thresholds must hold up on both views.

## 6. Timeline and submissions

| When | Work |
|---|---|
| 17:45–18:30 | normalize + transliteration + tests, splits, featurizer |
| 18:30–19:30 | encoder v1 training, shortlist recall on V |
| 19:30–20:30 | encode all records, GPU kNN (background) while writing features |
| 20:30–21:45 | features, XGBoost v1, decide, V score |
| 21:45–23:00 | test inference, validator → **Submission #1** |
| 23:00–23:45 | **Submission #2**: threshold variant (learn LB direction) |
| overnight | full-train kNN, encoder v2 (hard negatives), cross-encoder v2 |
| tomorrow | stage-2 + stacking (#3–#4), France probes (#5–#6), finals, package by 23:30 |

Final pick = best on **both** V and public LB (private LB decides). Keep 1–2 slots spare.
Every submission is tagged in git (`sub-1`, `sub-2`, …) with its V score and LB score.

## 7. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Encoder recall too low | TF-IDF shortlist with `sparse_dot_topn` |
| 4 GB VRAM | smaller chunks, fp16 storage |
| RAM | per-country chunks, memmaps on disk |
| polars native crash | numpy / pure Python for list ops |
| Cross-encoder doesn't help | drop it; v1 stays complete |
| Time overrun | v1 alone is a full submission |

## 8. Compliance

Only the provided data. Transliteration table, abbreviation and legal-form lists are
hand-written language rules (documented). All models trained from scratch by us (MIT) plus
XGBoost (Apache-2.0); far below 8B parameters. No external API or lookup anywhere.

## 9. Deliverables

`output/` (matching_results.tsv, candidate_pairs.tsv), `code/business_entity_resolution/`
(`src/`, README.md, pinned requirements.txt), filled `Documentation_template.md`.
