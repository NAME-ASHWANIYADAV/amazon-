# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** GreenBytes
**Team Members:** Lakshay Bansal, Ashwani Yadav, Arpita LNU
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

A neural-first entity-resolution pipeline built only from the provided data. A from-scratch character n-gram
"fingerprint" bi-encoder, trained contrastively on the labelled pairs, maps every record to name and address
vectors; GPU brute-force cosine search per country produces candidates (record pass + address-only pass); a
cheap stage-A filter prunes them to 8.7 pairs per entity (`candidate_pairs.tsv`); a stacked XGBoost judge
(stage 1, then stage 2 on within-list context and generator-structure features) scores each pair from 101
features; a fine-tuned multilingual MiniLM cross-encoder re-scores the pairs nearest the decision boundary; a
LightGBM stage-3 recalibrator adds co-location and digit-edit features; and a decision layer enforces "each
Source-2/3 record belongs to at most one Source-1 entity", picks per entity the number of matches that
maximises the exact expected F0.5, and removes lookalike fake groups with rules derived from the generator's
regularities. Validation macro F0.5 (10% held-out entities): 0.9898. Public leaderboard: 0.981 (best of 7
uploads; rank ≈830). Every leaderboard score was predicted to ±0.001 from validation before uploading.

---

## 2. Methodology

### 2.1 Problem Analysis

Key findings from EDA (train: 2.21M S1, 10.3M S2+S3; test: 1.73M S1, 9.97M S2+S3):

- **One owner per record:** every S2/S3 id belongs to at most one S1 entity (0 exceptions in 7.64M labelled pairs). Used as a hard assignment constraint.
- **Matches per entity:** singletons are 5.6% of S1; the mean is 3.46 matches per S1 (max 5 from S2, 6 from S3). Recall matters as much as precision.
- **Distractors:** 26% of train S2/S3 records match no S1 (1.2 per S1); test has ≈2.3 per S1. Their house numbers are shifted UP by {1,2,3,4,5,7,9,11,13,21} and their names carry descriptor words (group, holdings, india, participations, france) or changed legal forms. True copies get ±1/2 number typos (sign-symmetric). In test, fakes come in groups of 1–3 copies at the same shifted number, which a judge trained on single train fakes reads as agreement.
- **Name noise:** accents, typos and scrambled letters, domain names, "formerly:"/DBA trade names, made-up brand words from a global pool, phone numbers, honorifics, acronyms, legal forms added/dropped/changed, dropped spaces; 18% of India S2/S3 names are phonetic transliterations in nine Brahmic scripts.
- **Address noise:** abbreviations, typos, reordered components, null tokens, zero padding, state codes vs names, 3–5% empty addresses. Empty-address copies cause 81% of validation false negatives: 86% of them have ≥2 same-name S1 entities in the country, an irreducible tie.
- **France:** test only (15% of test S1), unseen in training. French abbreviations (R., AV, BD, N°) and legal forms; names built from generic words (club, amicale, comité, école) that swap between copies; 15.8% of S1s share an exact address with another S1.

### 2.2 Solution Strategy

**Approach Type:** Neural bi-encoder blocking + stacked gradient-boosted pair classifier + cross-encoder blend + generator-aware rules + constrained decision layer.
**Core Innovation:** treating the synthetic generator as the object to model: features and rules encode its regularities (distractor shift set, per-source address base, sibling agreement, dual-role words), a simulated competitor table gives every SX its real competition, and an exact expected-F0.5 cut under the one-owner constraint turns calibrated probabilities into matches. No country is hard-coded: unseen-country behaviour is keyed on "country absent from training".

---

## 3. Candidate Generation (Blocking)

- **Normalisation** (hand-written rules, no lookups): one Brahmic transliteration table for nine scripts; accent folding; legal-form extraction (US/India/France forms); honorific, phone and domain handling; street, French and Indian abbreviation expansion; state-code canonicalisation; number extraction.
- **Fingerprint encoder:** every word and padded char 3/4-gram hashed into 2^20 buckets → EmbeddingBag with learned bucket weights → one MLP tower per field (name, address) → two 64-d unit vectors. Symmetric InfoNCE with in-batch negatives, 8 epochs, batch 8192, trained on 75% of train entities (split E). In-batch top-1 accuracy 99.9%.
- **Search:** exact cosine top-40 per S1 within the same country (brute force on the GPU, ~2.5 G pairs/s) plus a top-10 address-cosine pass that recovers trade-name copies. Empty-address records are rescaled so their score is not capped at 1/√2.
- **Stage-A filter:** a small XGBoost on list/competition features keeps 99.9% of true pairs while pruning 79M test pairs to 15.1M (8.7 per S1). `candidate_pairs.tsv` holds these survivors, the pairs the judges score.
- **Recall ceiling (validation):** 99.6% of true pairs are among the scored candidates (98.3% without the address pass). On test, 1.8% (US) / 3.2% (India) / 8.8% (France) of S2/S3 records reach no list; the French excess is mostly generic-word swaps and spaces-dropped names that the US/India-trained stage A prunes.

---

## 4. Matching Model

**Features (71 per pair + 9 within-list + 21 generator-structure):**
- **Neural / list:** record, name and address cosines; rank; gap to best; z-score in list; list size.
- **Name:** token overlap (plain, containment, IDF-weighted per country), token-set/sort/partial ratios, space-free ratio, Jaro-Winkler, best score over DBA alternatives, legal-form agreement code, acronym flag, transliteration and domain flags.
- **Address:** token overlap and ratios; house numbers (primary/any equality, differences, unmatched numbers); empty-address flag.
- **Competition:** for the same SX, how this S1 compares with every other S1 whose list holds it (computed against a fixed competitor table so train sees the competition test has).
- **Generator context:** aligned house-number offset and whether it is in the distractor shift set / negative / truncated / composite; sibling counts at the same number; empty-address duplicates; global name multiplicity per country; descriptor-word log-odds (extra/missing words) transferred to test tokens through the vocabulary with a French→English equivalence table; stage-A probability.
- **Within-list (stage 2):** statistics of the out-of-fold stage-1 probabilities (confident copies per source, rank, mass of competitors). **Generator structure (stage 2+/3):** per-source address base (extra numbers/tokens not shared by any confident same-source copy), co-located S1 counts at both addresses, exact duplicates in the list, name-edit type (typo / concatenation / novel word / replacement), digit-level number edits.

**Models:** XGBoost (hist, CUDA) for stages A/1/2+, trained on the candidate pairs of 15% of train entities (split J, never used for the encoder) with early stopping on a held-out slice; `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0, 118M parameters) fine-tuned on the J grey zone, applied to the 0.92M test pairs nearest the decision boundary and blended logistically (V +0.0010: it reads raw text and knows French); LightGBM stage-3 recalibrator on the generator-structure features, trained on the validation split with 5-fold grouped CV (+0.0005 V), guarded on test (keeps the stage-2 probability at distractor house-number shifts and for the unseen country).

**Decision:**
1. Assignment: each S2/S3 record is kept only for its highest-probability S1.
2. Lookalike rules (measured on V, cost < 0.0002; +0.009 on the public leaderboard): reject a +1/+2 copy from the same source as a confident copy at the S1 number (and its number group); reject shifted groups mixing S2 and S3, or whose members only add a legal form/descriptor; reject fake descriptor words at shifted numbers; in the unseen country reject +1/+2 copies with a changed word/legal form; add exact-address copies whose names differ only by spacing/permutation/acronym (style-fingerprint checked).
3. Per-entity cut: top-k by exact expected F0.5 (Poisson-binomial with a Poisson tail), k = 0 when "no match" is the better bet; per-source caps 5/6.

---

## 5. Results & Error Analysis

- **Validation macro F0.5 (V, 221k entities):** stage 1 0.9862 → stage 2 0.9870 → stage 2+ 0.9883 → + cross-encoder blend 0.9893 → + stage-3 recalibrator 0.9898. Pair precision 0.998, recall 0.970. The judge is calibrated in every probability bin; the expected-F cut is optimal for it (any threshold change loses on V).
- **Public leaderboard:** 0.963 (v1) → 0.971 (shift rule) → 0.975 (stage 2 + lookalike rules) → 0.977 (stage 3) → 0.978 (stage 2+, France additions) → **0.981 (final: cross-encoder blend + stage 3)**, rank ≈830 of the participating teams. The validation-to-leaderboard gap stayed at ≈0.009 in every upload.
- **Where the remaining loss sits (validation, labelled):** 62% empty-address copies whose name is shared by several S1s (no information left), 15% calibrated 0.5–0.77 pairs that F0.5 correctly leaves out, 8% made-up brand-word records at the exact address (true rate 0.65, no separating signal in token frequency, vocabulary, legal form or address verbatim-ness), 5% house-number digit edits. Four cross-fitted hunts on these buckets gained ≤ +0.00008 each.
- **Where the test-specific loss sits (label-free audit):** submission integrity, id/file-order leaks, country labels, per-country matches per S1 (3.38–3.40 vs 3.46 truth), empty-entity rate (5.5–5.7% vs 5.6%), house-number sign symmetry and probability-band densities were all checked. The test differs from validation only in the US/India mid band (judge p 0.5–0.99 holds ≈3x the validation density, which the cross-encoder calls "no" 79% of the time vs 24% on V) — grouped fakes that shift the judge's context features — and in the unseen French candidate hole above. The lookalike rules remove most of the band excess; the rest, and France, are the 0.009 gap.
- **What did not help:** character-level cross-encoder (+0.00001), transitive rare-token links (0), count priors and joint assignment (< 0.0002), hyper-parameter sweeps (−0.001), simulated missing S1s (test has none: source-balance z-test), self-training on test, per-band label-shift correction of the US/India probabilities (removes mostly true pairs after the rules), a judge retrained on test pairs pseudo-labelled by the rules (neutral: it re-learns the rules).

---

## 6. Conclusion

The synthetic generator has strong regularities; modelling them (shift set, per-source base, sibling agreement, dual-role words) mattered more than model capacity, and label-free audits of the test set predicted every leaderboard score. The two losses the pipeline could not close are intrinsic ties between identically named entities and the train→test shift of the judge's context features under grouped distractors plus the unseen country — a judge that reads raw multilingual text for all candidates (a larger cross-encoder on adequate GPU time) is the natural next step.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`: `src/` (normalize, featurize, encoder, candidates, features, context, judge,
stage3, decide, lookalike, style, unseen, cross_encoder_pt, evaluate, io_utils, run_pipeline), `tests/` (73 pytest
tests), `README.md` (exact stage-by-stage commands and flags), `requirements.txt`. Entry point:
`python -m src.run_pipeline <stage>`.

### B. Additional Results

Label-free checks used to choose between submissions without spending uploads: per-country predictions per S1
against the validation level, the sign-symmetry of ±1/2 house-number offsets (true typos are symmetric, fakes
are +only), probability-band densities and cross-encoder verdict mixes per country against validation, and an
address-style fingerprint (number format, region rendering) that agrees 99% within an entity and 3% between
entities.
