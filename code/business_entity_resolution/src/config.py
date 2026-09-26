"""Shared paths and constants for the entity-resolution pipeline."""
import os

os.environ.setdefault("USE_TF", "0")  # an old TensorFlow install breaks `transformers` imports
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.environ.get("ER_DATA_DIR", r"C:\amazon_ml\dataset")
WORK_DIR = os.environ.get("ER_WORK_DIR", r"C:\amazon_ml\work")
OUT_DIR = os.environ.get("ER_OUT_DIR", os.path.join(ROOT, "output"))

HASH_BUCKETS = 1 << 20  # per field: name ids in [0, 2^20), address ids in [2^20, 2^21)
EMB_DIM = 64            # per tower; record vector = 2 * EMB_DIM
TOP_K = 40              # neural neighbours per S1 before pruning
SPLIT_E, SPLIT_J = 75, 90  # crc32(entity_id) % 100: < 75 -> E, < 90 -> J, else V


def work(*parts):
    """Path under WORK_DIR, creating the parent directory."""
    path = os.path.join(WORK_DIR, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
