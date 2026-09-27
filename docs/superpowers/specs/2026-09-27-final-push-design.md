# Final push: three parallel tracks to close the LB gap (2026-09-27)

## Goal
LB 0.97507 (U1) -> as close to 0.9914 as possible in the final submissions. Three tracks run in parallel; each is
measured offline; the winners are combined into one final decision/model.

## Where the gap is (estimates)
- France (15% of test S1): ~0.905 by arithmetic (not measured) -> up to ~0.012 LB.
- US + India (85%): ~0.986 at V level vs ~0.992 for top teams -> ~0.005 LB.

## Track A - France precision/recall (label-free)
- A1 France FP hunt: pattern frequencies of France predictions vs V truth rates (offset 0 lookalikes, no-number
  lookalikes, descriptor words, legal changes, source mix); slot tests; eyeball samples against generator rules.
  Output: rules keyed on "country unseen in training".
- A2 Unseen-country judge: train on US-J only, validate on V-India (a labelled proxy for a new country); ablate
  feature groups (word log-odds, list shape, global counts, competition, idf); route unseen countries to the best
  variant.
- A3 France acronym recall: acronym SX at the exact address with a unique initials-matching S1.

## Track B - Base judge (labelled, V)
- New signals from the residual hunt (transitive rare-token links for empty-address copies, count prior in the
  decision, FP residue rules).
- Final model trained on J+V with fixed rounds (learning curve: 50%->100% of J = +0.00074 V).

## Track C - Char-level cross-encoder (GPU, no download)
- Small transformer over the raw pair text "S1 name | S1 addr || SX name | SX addr", trained from scratch on J
  pairs; scores the grey zone; blended with the XGBoost p (weights fit on one half of V, measured on the other).
- A pretrained multilingual model would need a download (asked separately with file/source/size).

## Evaluation protocol (every candidate)
1. Plain V macro F0.5 overall and per country (labels).
2. Test label-free checks: predictions per S1 per country vs V, +1/+2 vs -1/-2 mirror, p-bin mass vs V.
3. Combine only components with a measured V gain or a label-free test gain and no V loss > 0.0003.

## Combination and uploads
- Final = best model (J+V retrain, stage 2, optional cross-encoder blend) + decision rules (shift, lookalike,
  France rules, caps). 3 uploads: improved candidate, second candidate, final (last upload = intended final).
