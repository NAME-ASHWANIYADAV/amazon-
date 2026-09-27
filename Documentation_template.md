# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** GreenBytes
**Team Members:** Lakshay Bansal, Ashwani Yadav, Arpita LNU
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

A neural-first entity-resolution pipeline built only from the provided data. A from-scratch character n-gram
"fingerprint" bi-encoder, trained contrastively on the labelled pairs, maps every record to name and address
vectors; GPU brute-force cosine search per country produces candidates (record pass + address-only pass); a
cheap stage-A filter prunes them to ~9 pairs per entity; a stacked XGBoost judge (stage 1, then stage 2 on
within-list context) scores each pair from 71 features, including generator-aware ones (house-number shift
alignment, sibling agreement, word log-odds, cross-entity competition); a stage-3 recalibrator adds
generator-structure features (per-source address base, co-location, duplicates, digit-level number edits); and a decision layer enforces
"each Source-2/3 record belongs to at most one Source-1 entity", picks per entity the number of matches that
maximises exact expected F0.5, and removes lookalike fake groups with rules derived from the generator's
regularities. Validation macro F0.5 (10% held-out entities): 0.9888. Public leaderboard: 0.978 (see §5).

---

## 2. Methodology

### 2.1 Problem Analysis

Key findings from EDA (train: 2.21M S1, 10.3M S2+S3; test: 1.73M S1, 9.97M S2+S3):

- **One owner per record:** every S2/S3 id belongs to at most one S1 entity (0 exceptions in 7.64M labelled pairs). Used as a hard assignment constraint.
- **Matches per entity:** singletons are 5.6% of S1; the mean is 3.46 matches per S1 in both train countries (max 5 from S2, 6 from S3, 11 in total). Recall matters as much as precision.
- **Distractors:** 26% of train S2/S3 records match no S1 (1.2 per S1); test has ~2.3 per S1. Their house numbers are shifted UP by {1,2,3,4,5,7,9,11,13,21} and their names carry descriptor words (group, holdings, india, participations, france) or changed legal forms. True copies get ±1/2 number typos (sign-symmetric). In test, fakes come in groups of 1–3 copies at the same shifted number, which a judge trained on single train fakes reads as agreement.
- **Name noise:** accents, typos and scrambled letters, domain names, "formerly:"/DBA trade names, made-up brand words from a global pool, phone numbers, honorifics, acronyms (France: 13x more), legal forms added/dropped/changed, dropped spaces.
- **Indic scripts:** 18% of India S2/S3 names are phonetic transliterations in nine Brahmic scripts.
- **Address noise:** abbreviations, typos, reordered components, null tokens, zero padding, state codes vs names, 3–5% empty addresses. Empty-address copies cause 66% of validation false negatives: their name alone is shared by several S1s, an irreducible tie.
- **France:** test only (15% of test S1). French abbreviations (R., AV, BD, Q, N°) and legal forms; names built from generic words (club, amicale, comité, école) that swap between copies; 15.8% of S1s share an exact address with another S1.

### 2.2 Solution Strategy

**Approach Type:** Neural bi-encoder blocking + stacked gradient-boosted pair classifier + generator-aware rules + constrained decision layer.
**Core Innovation:** treating the synthetic generator as the object to model: features and rules encode its regularities (distractor shift set, per-source address base, sibling agreement, dual-role words), a simulated competitor table gives every SX its real competition, and an exact expected-F0.5 cut under the one-owner constraint turns calibrated probabilities into matches. No country is hard-coded: unseen-country behaviour is keyed on "country absent from training".

---

## 3. Candidate Generation (Blocking)

- **Normalisation** (hand-written rules, no lookups): one Brahmic transliteration table for nine scripts; accent folding; legal-form extraction (US/India/France forms); honorific, phone and domain handling; street, French and Indian abbreviation expansion; state-code canonicalisation; number extraction.
- **Fingerprint encoder:** every word and padded char 3/4-gram hashed into 2^20 buckets → EmbeddingBag with learned bucket weights → one MLP tower per field (name, address) → two 64-d unit vectors, concatenated. Symmetric InfoNCE with in-batch negatives, 8 epochs, batch 8192, trained on 75% of train entities (split E). In-batch top-1 accuracy 99.9%.
- **Search:** exact cosine top-40 per S1 within the same country (brute force on the GPU, ~2.5 G pairs/s) plus a top-10 address-cosine pass that recovers trade-name copies. Empty-address records are rescaled so their score is not capped at 1/√2.
- **Stage-A filter:** a small XGBoost on list/competition features keeps 99.9% of true pairs while pruning 79M test pairs to 15.1M (8.7 per S1). `candidate_pairs.tsv` holds these survivors, the pairs the judges score.
- **Recall ceiling (validation):** 99.6% of true pairs are among the scored candidates (98.3% without the address pass).

---

## 4. Matching Model

**Features (71 per pair):**
- **Neural / list:** record, name and address cosines; rank; gap to best; z-score in list; list size.
- **Name:** token overlap (plain, containment, IDF-weighted per country), token-set/sort/partial/plain ratios, space-free ratio, Jaro-Winkler, best score over DBA alternatives, legal-form agreement code, acronym flag, transliteration and domain flags.
- **Address:** token overlap and ratios; house numbers (primary/any equality, differences, unmatched numbers); empty-address flag.
- **Competition:** for the same SX, how this S1 compares with every other S1 whose list holds it (computed against a fixed competitor table so train sees the competition test has).
- **Generator context:** aligned house-number offset and whether it is in the distractor shift set / negative / truncated / composite; sibling counts at the same number; empty-address duplicates; global name multiplicity (size-normalised per country); descriptor-word log-odds (extra/missing words), transferred to test tokens through the vocabulary with a French→English equivalence table; stage-A probability.
- **Stage 2:** within-list statistics of the out-of-fold stage-1 probabilities (confident copies per source, rank, mass of competitors).
- **Stage 3:** per-source address base (extra numbers/address tokens not shared by any confident same-source copy), co-located S1 counts at both addresses, exact duplicates in the list, name-edit type (typo / concatenation / novel word / replacement).

**Model type:** XGBoost (hist, CUDA) for stages A/1/2+, trained on the candidate pairs of 15% of train entities (split J, never used for the encoder) with early stopping on a held-out slice; a pretrained multilingual MiniLM cross-encoder (Apache-2.0, 118M parameters) fine-tuned on the J grey zone re-scores the pairs the judge is unsure about, nearest to the decision boundary first within a time budget (the GTX 1650 scores ~100 pairs/s), and is blended in logistically (V +0.0010 with every grey-zone pair scored, 90% of it from the 1.07M pairs with |logit| ≤ 3; the largest single gain: it reads the raw text and knows French words); LightGBM stage-3 recalibrator on generator-structure features trained on the 10% validation split (V) with 5-fold grouped CV (+0.0005 V).

**Decision:**
1. Assignment: each S2/S3 record is kept only for its highest-probability S1.
2. Lookalike rules (measured on V, cost < 0.0002): reject a +1/+2 copy from the same source as a confident copy at the S1 number (and its whole number group); reject shifted groups mixing S2 and S3, or whose members all only add a legal form/descriptor; reject fake descriptor words at shifted numbers; in unseen countries reject +1/+2 copies with a changed word/legal form and common-word swaps at the same number.
3. Per-entity cut: top-k by exact expected F0.5 (Poisson-binomial with a Poisson tail), k = 0 when "no match" is the better bet; per-source caps 5/6.

---

## 5. Results & Error Analysis

- **Validation macro F0.5 (V, 221k entities):** stage 1 0.9862 → stage 2 0.9870 → stage 2+ (generator-structure features inside the judge) 0.9883 → + cross-encoder blend 0.9893 → + stage-3 recalibrator ≈0.9898. Pair precision 0.998, recall 0.970.
- **Public leaderboard:** 0.963 (v1) → 0.971 (shift rule) → 0.975 (stage 2 + lookalike rules) → 0.977 (stage 3, France fixes) → 0.978 (stage 2+, raw name counts, France additions) → final upload: stage 2+ with the cross-encoder blend on the 0.92M test pairs nearest the decision boundary and the stage-3 recalibrator (V 0.9898; 5,870,885 matched pairs, 3.40/3.38/3.40 per S1 in US/India/France).
- **Common false positives (test):** lookalike fake groups at shifted house numbers with a legal-form change ('little diner inc / co / ltd | 4311' for 'little diner | 4310'); brand-only copies of a co-located entity. Test has ~2x the train distractor rate and fakes come in groups, which is the main validation-to-leaderboard gap; the rules and stage 3 recover about half of it.
- **Common false negatives:** empty-address copies whose name is shared by several S1s (76% of them irreducible ties on the available data); French acronym and abbreviation copies.
- **Label-free test audit before the final upload:** submission integrity (all 1,732,544 S1 rows, no invalid or duplicate ids), no id/file-order leak (rank correlation 0.0001), per-country matches per S1 3.38–3.40 vs 3.46 truth, house-number sign symmetry, probability-band densities and cross-encoder verdict mixes per country. Test differs from validation only in the US/India mid band (judge p 0.5–0.99 holds ~3x the validation density of candidates, which the cross-encoder calls 'no' 79% of the time vs 24% on V); the lookalike rules remove that excess, and a further per-band label-shift correction was built (`rethreshold --band-shift-file`) but not used: the pairs it would remove have a validation-like cross-encoder verdict mix, i.e. they are mostly true copies.
- **What did not help (measured on V):** a character-level cross-encoder blend (+0.00001), transitive rare-token links (0), count priors and joint assignment (< 0.0002), hyper-parameter sweeps (−0.001), training with simulated missing S1s (test has none: source-balance z-test). Four adversarial cross-fitted hunts on the remaining loss buckets (empty-address ties, name-replaced records at the exact address, the 0.45–0.80 band, entities with no prediction) found ≤ +0.00008 each: the judge is calibrated in every bin and these buckets are generator coin-flips.

---

## 6. Conclusion

The synthetic generator has strong regularities; modelling them (shift set, per-source base, sibling agreement, dual-role words) mattered more than model capacity. The largest remaining losses are intrinsic ties between identically named entities and the unseen-country shift for France.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`: `src/` (normalize, featurize, encoder, candidates, features, context,
judge, stage3, decide, lookalike, evaluate, io_utils, run_pipeline, cross_encoder), `tests/` (61 pytest tests),
`README.md` (exact stage-by-stage commands and flags), `requirements.txt`. Entry point:
`python -m src.run_pipeline <stage>`.

### B. Additional Results

Label-free checks used to choose between submissions without spending uploads: per-country predictions per S1
against the validation level (3.36), the sign-symmetry of ±1/2 house-number offsets (true typos are symmetric,
fakes are +only), p-bin mass per country, and an address-style fingerprint (number format, region rendering)
that agrees 99% within an entity and 3% between entities.
