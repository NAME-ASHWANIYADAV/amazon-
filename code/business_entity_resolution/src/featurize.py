"""Hash strings into bags of word + char 3/4-gram ids for nn.EmbeddingBag (numba, no Python loops)."""
import numba
import numpy as np

from .config import HASH_BUCKETS

_FNV_OFFSET = np.uint64(14695981039346656037)
_FNV_PRIME = np.uint64(1099511628211)


def _to_codepoints(strings):
    lens = np.fromiter((len(s) for s in strings), dtype=np.int64, count=len(strings))
    offsets = np.zeros(len(strings) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    buf = np.frombuffer("".join(strings).encode("utf-32-le"), dtype=np.uint32)
    return buf, offsets


@numba.njit(cache=True)
def _fnv(buf, a, b, pad_l, pad_r, salt):
    h = _FNV_OFFSET ^ np.uint64(salt)
    if pad_l:
        h = (h ^ np.uint64(60)) * _FNV_PRIME
    for i in range(a, b):
        h = (h ^ np.uint64(buf[i])) * _FNV_PRIME
    if pad_r:
        h = (h ^ np.uint64(62)) * _FNV_PRIME
    return h


@numba.njit(cache=True)
def _count(buf, offsets):
    n = len(offsets) - 1
    counts = np.zeros(n, dtype=np.int64)
    for r in range(n):
        i, e, c = offsets[r], offsets[r + 1], 0
        while i < e:
            while i < e and buf[i] == 32:
                i += 1
            if i >= e:
                break
            j = i
            while j < e and buf[j] != 32:
                j += 1
            L = j - i
            c += 1 + L  # word id + L padded 3-grams
            if L >= 2:
                c += L - 1  # padded 4-grams
            i = j
        counts[r] = c
    return counts


@numba.njit(cache=True)
def _fill(buf, offsets, out_offsets, out, mask, field_offset):
    n = len(offsets) - 1
    for r in range(n):
        i, e, k = offsets[r], offsets[r + 1], out_offsets[r]
        while i < e:
            while i < e and buf[i] == 32:
                i += 1
            if i >= e:
                break
            j = i
            while j < e and buf[j] != 32:
                j += 1
            L = j - i
            out[k] = field_offset + np.int64(_fnv(buf, i, j, False, False, 1) & mask)
            k += 1
            P = L + 2
            for ng in (3, 4):
                for p in range(0, P - ng + 1):
                    a = i + max(p - 1, 0)
                    b = i + min(p + ng - 1, L)
                    out[k] = field_offset + np.int64(_fnv(buf, a, b, p == 0, p + ng == P, ng) & mask)
                    k += 1
            i = j


def hash_features(strings, field_offset):
    """-> (indices, offsets) for nn.EmbeddingBag; ids in [field_offset, field_offset + HASH_BUCKETS)."""
    buf, offsets = _to_codepoints(strings)
    counts = _count(buf, offsets)
    out_offsets = np.zeros(len(strings) + 1, dtype=np.int64)
    np.cumsum(counts, out=out_offsets[1:])
    out = np.empty(out_offsets[-1], dtype=np.int64)
    _fill(buf, offsets, out_offsets, out, np.uint64(HASH_BUCKETS - 1), np.int64(field_offset))
    return out, out_offsets[:-1]
