# Final submission files — team GreenBytes (Amazon ML Challenge 2026, Business Entity Resolution)

| File | Size | What to do with it |
|---|---|---|
| `FINAL_matching_results.tsv` | 93.6 MB (98,098,489 bytes), MD5 `8879e8368436d21a7205dcaf450ef59d` | **The leaderboard submission.** Upload this file as-is on the Unstop submission page. 1,732,544 rows (one per Source-1 entity), header `source1_entity_id<TAB>matched_entity_ids`, 5,870,885 matched ids, organisers' validator: PASS. |
| `GreenBytes_submission.zip` | 129 MB | The final package: `output/matching_results.tsv` (identical to the file above), `output/candidate_pairs.tsv` (15.13M stage-A candidate pairs), `code/business_entity_resolution/` (src, tests, README, requirements) and `Documentation.md`. Upload where the organisers ask for the code/documentation package. |

Both files are stored with Git LFS: use the **Download** button on the file page (or `git lfs pull` after cloning). Do not open/re-save the TSV in Excel or a text editor before uploading — upload the downloaded file unchanged.

Expected public-leaderboard macro F0.5: 0.979–0.981 (validation 0.9898). Write-up: `../Documentation_template.md`; code: `../code/business_entity_resolution/README.md`.

## Probe uploads (27 Sep evening, after the organisers granted 2 extra submissions)

| File | Change vs FINAL | Purpose |
|---|---|---|
| `PROBE_A_france_strict_matching_results.tsv` | France only: no style/blocking additions, France logit shift −0.8 (3.40 → 3.25 matches per France entity; US/India identical) | Measures France precision on the leaderboard: score up ⇒ France was over-predicting; score down ⇒ re-upload `FINAL_matching_results.tsv`. |
