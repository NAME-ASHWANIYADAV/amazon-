# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Team Name]
**Team Members:** [Members]
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

A neural-first entity-resolution pipeline built only from the provided data. A from-scratch character n-gram
"fingerprint" bi-encoder, trained contrastively on the 7.6M labelled pairs, maps every record to name and
address vectors; GPU brute-force cosine search per country produces candidates; an XGBoost judge scores each
candidate pair from ~44 similarity features; and a decision layer enforces "each Source-2/3 record belongs to
at most one Source-1 entity" and picks, per entity, the number of matches that maximises expected F0.5.

---

## 2. Methodology

### 2.1 Problem Analysis

Key findings from EDA (train: 2.21M S1, 10.3M S2+S3; test: 1.73M S1, 9.97M S2+S3):

- **One owner per record:** every S2/S3 id belongs to at most one S1 entity (0 exceptions in 7.64M labelled pairs). We use this as a hard assignment constraint.
- **Matches per entity:** singletons are 5.6% of S1. The mean is 3.46 matches per S1 (1.8 from S2, 1.9 from S3), max 11. Recall matters as much as precision.
- **Distractors:** 26% of S2/S3 records match no S1. The test set has 5.7 S2+S3 records per S1 against 4.67 in train, so it holds more distractors.
- **Signal coverage:** 99.8% of true pairs share a name token, the house number, or at least two address words.
- **Lookalike negatives:** non-matches that share a rare name token share the house number only 0.1% (US) / 3% (India) of the time. The hardest ones sit on the same street with a shifted number and one extra descriptor word ("Sidynis Holdings Group **South** LLC, **1420** Prairie Creek Trl" vs "Sidynis Holdings Group LLC, 1417 Prairie Creek Trail").
- **Name noise:** accents, brackets, typos and scrambled letters, domain names, "formerly:"/DBA trade names, phone numbers, honorifics, legal forms that are added, dropped or moved, and duplicated or dropped tokens.
- **Indic scripts:** 18% of India S2/S3 names are written in Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada or Malayalam. They are phonetic transliterations of the English name.
- **Address noise:** abbreviations, typos, reordered components, null/N/A tokens, zero padding, state codes vs full names vs native script, and 3-5% empty addresses.
- **France:** appears only in test (15% of test S1), with French abbreviations (R., AV, BD, N°) and legal forms (SARL, SAS, EURL, SCI). Names are built from generic words, so the address carries most of the evidence.

### 2.2 Solution Strategy

**Approach Type:** Neural bi-encoder blocking + gradient-boosted pair classifier + constrained decision layer.
**Core Innovation:** a from-scratch, script-agnostic hashed n-gram bi-encoder with separate name/address
towers (typo-, order- and transliteration-robust, millions of records per second on a 4 GB GPU), combined with
an exact expected-F0.5 per-entity cut under the one-owner constraint.

---

## 3. Candidate Generation (Blocking)

- **Normalisation** (hand-written language rules, no lookups):
  - One Brahmic transliteration table covering all nine Indian scripts. They share a Unicode layout, and the table handles inherent-vowel and schwa rules.
  - Accent folding, legal-form extraction, honorific and phone-number removal, domain and "formerly/DBA" handling.
  - Street, French and Indian abbreviation expansion, state-code canonicalisation, number extraction.
- **Fingerprint encoder:**
  - Every word and every padded char 3/4-gram is hashed into 2^21 buckets, then passed through an EmbeddingBag with a learned per-bucket weight.
  - A small MLP tower per field (name, address) produces two 64-d unit vectors. The record vector is their normalised concatenation.
  - Training uses symmetric InfoNCE with in-batch negatives: 8 epochs, batch 8192, one random S2/S3 record per S1 per epoch, on 75% of train entities.
  - In-batch top-1 accuracy reached 99.9%.
- **Search:**
  - Exact cosine top-40 per S1 within the same country, brute force on the GPU in chunks (~2.5 G pairs/s).
  - Empty-address records are rescaled so that their score is not capped at 1/√2.
- **Candidate pairs:** 69.3M on test (1,732,544 × 40).
- **Recall ceiling (validation):** 98.3% of true pairs are within the top 40. Most of the misses have an unrelated trade name with the same address. [update after Phase 2 address-only search]

---

## 4. Matching Model

**Features used (44):**
- **Neural:** record, name and address cosines; rank in the S1 list; gap to the best candidate; z-score within the list; list size.
- **Name:**
  - token intersection, Jaccard, containment both ways, IDF-weighted Jaccard and overlap
  - token-set, token-sort, partial and plain ratios, space-free ratio, Jaro-Winkler
  - best score over formerly/DBA alternatives
  - legal-form agreement code
  - transliterated-script flag, domain-name flag
- **Address:**
  - word intersection, Jaccard, IDF-weighted Jaccard, token-set and plain ratios
  - house numbers: primary equal, any equal, minimum and relative difference, primary-to-primary difference, unmatched numbers, number counts
  - empty-address flag
- **Other:** source (S2 vs S3).

**Model type:** XGBoost (hist, CUDA), trained on the candidate pairs of 15% of train entities. None of these entities were used for the encoder. Early stopping uses a held-out slice of those entities.
**Threshold selection method:**
1. Assignment: each S2/S3 record is kept only for its highest-probability S1.
2. Per-entity cut: the top-k is chosen by exact expected F0.5, computed with a Poisson-binomial model plus a Poisson tail for lower-ranked candidates. k = 0 when "no match" is the better bet.
3. This rule is compared with global thresholds on a 10% validation split and on a stress view with 19% of entities removed.

---

## 5. Results & Error Analysis

- **F0.5 Score (macro, validation):** [fill]
- **Common false positives:** [fill after error analysis]
- **Common false negatives:** mainly pairs where the S2/S3 name is an unrelated trade name and only the address links the records (outside the top-40 shortlist). [fill]

---

## 6. Conclusion

[fill]

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`: `src/` (normalize, featurize, encoder, candidates, features, judge, decide,
evaluate, io_utils, run_pipeline), `tests/` (pytest), `README.md` (exact stage-by-stage commands), `requirements.txt`.
Entry point: `python -m src.run_pipeline <stage>`. Stages: prepare, train-encoder, encode, candidates, tokens,
features, train-judge, validate, predict-test.

### B. Additional Results

[fill]
