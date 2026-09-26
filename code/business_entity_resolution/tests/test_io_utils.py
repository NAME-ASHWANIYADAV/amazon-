import numpy as np

from src.io_utils import write_submission


def test_write_submission_rows_order_and_subset(tmp_path):
    s1_ids = ["S1-a", "S1-b", "S1-c"]
    sx_ids = np.array(["S2-1", "S3-2", "S2-3"], dtype=object)
    s1r = np.array([2, 0, 0, 2])
    sxr = np.array([2, 0, 1, 1])
    mask = np.array([True, False, True, False])
    write_submission(str(tmp_path), s1_ids, sx_ids, s1r, sxr, mask)
    cand = (tmp_path / "candidate_pairs.tsv").read_text(encoding="utf-8").splitlines()
    match = (tmp_path / "matching_results.tsv").read_text(encoding="utf-8").splitlines()
    assert cand == ["source1_entity_id\tcandidate_entity_ids", "S1-a\tS2-1,S3-2", "S1-b\t", "S1-c\tS2-3,S3-2"]
    assert match == ["source1_entity_id\tmatched_entity_ids", "S1-a\tS3-2", "S1-b\t", "S1-c\tS2-3"]
