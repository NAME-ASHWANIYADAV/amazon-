import os

import numpy as np

from src.io_utils import write_submission


def test_write_submission_dedupes_repeated_candidates(tmp_path):
    s1_ids = ["S1-1", "S1-2", "S1-3"]
    sx_ids = np.array(["S2-a", "S2-b", "S3-c"], dtype=object)
    # S1-1 sees S2-a twice (once matched, once not) and S2-b once; S1-2 sees S3-c; S1-3 has no candidates
    s1r = np.array([0, 0, 0, 1], dtype=np.int64)
    sxr = np.array([0, 1, 0, 2], dtype=np.int64)
    mask = np.array([False, False, True, True])
    write_submission(str(tmp_path), s1_ids, sx_ids, s1r, sxr, mask)
    cand = open(os.path.join(tmp_path, "candidate_pairs.tsv"), encoding="utf-8").read().splitlines()
    match = open(os.path.join(tmp_path, "matching_results.tsv"), encoding="utf-8").read().splitlines()
    assert cand == ["source1_entity_id\tcandidate_entity_ids", "S1-1\tS2-a,S2-b", "S1-2\tS3-c", "S1-3\t"]
    assert match == ["source1_entity_id\tmatched_entity_ids", "S1-1\tS2-a", "S1-2\tS3-c", "S1-3\t"]
