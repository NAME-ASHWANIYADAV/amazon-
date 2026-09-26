from src.config import HASH_BUCKETS
from src.featurize import hash_features


def test_counts_and_range():
    idx, off = hash_features(["ab cd", "", "x"], 0)
    # "ab": 1 word + 3-grams(<ab, ab>) 2 + 4-grams(<ab>) 1 = 4 ; two tokens = 8 ; "" = 0 ; "x" = 1 + 1 = 2
    assert off.tolist() == [0, 8, 8]
    assert len(idx) == 10 and idx.min() >= 0 and idx.max() < HASH_BUCKETS


def test_field_offset_shifts_ids():
    a, _ = hash_features(["road"], 0)
    b, _ = hash_features(["road"], HASH_BUCKETS)
    assert ((b - a) == HASH_BUCKETS).all()


def test_typo_shares_ngrams():
    a, _ = hash_features(["wesley"], 0)
    b, _ = hash_features(["wesly"], 0)
    assert len(set(a.tolist()) & set(b.tolist())) >= 4


def test_deterministic():
    assert hash_features(["lucas and lee"], 0)[0].tolist() == hash_features(["lucas and lee"], 0)[0].tolist()
