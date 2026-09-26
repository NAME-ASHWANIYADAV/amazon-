# Entity Resolution Phase 1 (tonight's submission) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a validated `output/matching_results.tsv` + `output/candidate_pairs.tsv` for the test set tonight, using the neural fingerprint encoder for candidates and an XGBoost v1 judge.

**Architecture:** Normalise all records (incl. Brahmic transliteration) → hashed char n-gram bags → contrastive bi-encoder trained on split E → GPU brute-force kNN per country → ~42 pair features → XGBoost (CUDA) trained on split J → assignment + per-S1 expected-F0.5 cut tuned on split V → submission files checked by the official validator. Spec: `docs/superpowers/specs/2026-09-26-entity-resolution-design.md`. Phase 2 (full-train kNN, encoder v2, cross-encoder, stage-2, France probes) gets its own plan after Phase 1 numbers exist.

**Tech Stack:** Python 3.12, polars (I/O only), numpy, numba, torch 2.12 cu126, xgboost 3.4 (CUDA), rapidfuzz 3.14, pytest.

## Global Constraints

- Output TSVs are tab-separated with headers exactly `source1_entity_id\tmatched_entity_ids` and `source1_entity_id\tcandidate_entity_ids`; one row per test S1; empty list = no match; only S2-/S3- ids; no duplicates; matches ⊆ candidates.
- Only the provided data. No external API, database, geocoder or lookup anywhere.
- All models trained from scratch by us (MIT) or XGBoost (Apache-2.0); far below 8B params.
- `country` is an open set: loop over the countries present in the data; never filter to {US, India}.
- `USE_TF=0` is set before anything imports `transformers` (done in `config.py`).
- No polars `map_elements` / `explode` on multi-million-row frames (native crashes seen); use numpy / pure Python.
- Machine: 15 GB RAM (5–9 GB free), GTX 1650 4 GB VRAM. Stream/chunk everything large; big arrays go to memmaps under `WORK_DIR`.
- Paths: `DATA_DIR=C:\amazon_ml\dataset`, `WORK_DIR=C:\amazon_ml\work`, `OUT_DIR=<repo>\output` (env overrides `ER_DATA_DIR`, `ER_WORK_DIR`, `ER_OUT_DIR`).
- All commands run from `code/business_entity_resolution/` (the package root): tests with `python -m pytest tests -q`, stages with `python -m src.run_pipeline <stage>`.

## File Structure

```
code/business_entity_resolution/
  src/__init__.py        empty
  src/config.py          paths, constants, work() helper
  src/io_utils.py        TSV reading, GT pairs, submission writers
  src/normalize.py       transliteration + name/address normalisation (pure)
  src/featurize.py       numba hashing of char n-grams + words
  src/encoder.py         FingerprintModel, train_encoder, encode
  src/candidates.py      GPU kNN with name/address cosines, recall report
  src/features.py        pair features (numba set/number kernels + rapidfuzz)
  src/evaluate.py        organizer macro F0.5
  src/decide.py          assignment + expected-F0.5 cut + threshold rule
  src/judge.py           XGBoost train/predict
  src/run_pipeline.py    CLI stages: prepare, train-encoder, encode, candidates, features, train-judge, validate, predict-test
  tests/test_normalize.py, tests/test_featurize.py, tests/test_evaluate_decide.py
  requirements.txt
```

---

### Task 1: Package scaffold, config and I/O

**Files:**
- Create: `code/business_entity_resolution/src/__init__.py`, `src/config.py`, `src/io_utils.py`, `requirements.txt`

**Interfaces:**
- Produces: `config.DATA_DIR, WORK_DIR, OUT_DIR, HASH_BUCKETS=1<<20, EMB_DIM=64, TOP_K=40, work(*parts)->str`; `io_utils.read_tsv(path)->pl.DataFrame`, `read_source(split, src)->pl.DataFrame`, `read_gt_pairs()->pl.DataFrame[s1, sx]`, `write_submission(out_dir, s1_ids, sx_ids, s1r, sxr, mask)` (streams both TSVs; `s1r/sxr` are row indices, `mask` marks predicted matches).

- [ ] **Step 1: Write the files**

`src/__init__.py`: empty file.

`src/config.py`:
```python
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
```

`src/io_utils.py`:
```python
"""Reading the challenge TSVs and writing submission files."""
import os

import polars as pl

from .config import DATA_DIR


def read_tsv(path):
    """Challenge files contain stray quote characters: disable quoting and read every column as text."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       has_header=True, missing_utf8_is_empty_string=True)


def read_source(split, src):
    return read_tsv(os.path.join(DATA_DIR, split, f"{split}_source{src}.tsv"))


def read_gt_pairs():
    """Ground truth as a long table of (s1, sx) entity ids."""
    gt = read_tsv(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"))
    return (gt.rename({"source1_entity_id": "s1", "matched_entity_ids": "sx"})
            .filter(pl.col("sx") != "")
            .with_columns(pl.col("sx").str.split(","))
            .explode("sx"))


def write_submission(out_dir, s1_ids, sx_ids, s1r, sxr, mask):
    """Stream candidate_pairs.tsv and matching_results.tsv (validator format, one row per S1).

    s1_ids: list of all S1 entity ids (row order); sx_ids: numpy object array of SX ids;
    s1r/sxr: candidate pairs as row indices; mask: True where the pair is a predicted match."""
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    order = np.argsort(s1r, kind="stable")
    s1r, names, mask = s1r[order], sx_ids[sxr[order]], mask[order]
    bounds = np.searchsorted(s1r, np.arange(len(s1_ids) + 1))
    with open(os.path.join(out_dir, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="\n") as fc, \
            open(os.path.join(out_dir, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as fm:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for i, s1 in enumerate(s1_ids):
            lo, hi = bounds[i], bounds[i + 1]
            fc.write(f"{s1}\t{','.join(names[lo:hi])}\n")
            fm.write(f"{s1}\t{','.join(names[lo:hi][mask[lo:hi]])}\n")
```
(The GT explode is a 2.2M-row frame; it already ran fine during EDA.)

`requirements.txt`:
```
numpy==2.2.1
polars==1.41.2
pyarrow==24.0.0
numba==0.60.0
torch==2.12.0+cu126
xgboost==3.4.1
rapidfuzz==3.14.6
pytest
```

- [ ] **Step 2: Smoke-check imports**

Run: `python -c "from src import config, io_utils; print(config.WORK_DIR, len(io_utils.read_gt_pairs()))"`
Expected: `C:\amazon_ml\work 7638365`

- [ ] **Step 3: Commit**

```bash
git add code/business_entity_resolution
git commit -m "feat: pipeline scaffold, config and TSV I/O"
```

---

### Task 2: Normalisation (transliteration, names, addresses)

**Files:**
- Create: `code/business_entity_resolution/src/normalize.py`
- Test: `code/business_entity_resolution/tests/test_normalize.py`

**Interfaces:**
- Produces: `transliterate(text)->str`; `norm_name(raw)->(name_norm, name_core, legal, alt_core, was_indic, is_domain)` (strings; `legal` = space-joined sorted canonical legal tokens; `alt_core` = `|`-joined cores when the name contained formerly/dba/aka, else ""); `norm_addr(raw, country)->(addr_norm, numbers, addr_empty)` (`numbers` = space-joined digit runs, leading zeros stripped, ordinals excluded); `normalize_rows(rows)->list[tuple]` where rows are `(name, addr, country)` and each output tuple is `(name_norm, name_core, legal, alt_core, was_indic, is_domain, addr_norm, numbers, addr_empty)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_normalize.py`:
```python
from src.normalize import norm_addr, norm_name, transliterate


def test_transliterate_hindi_sentence():
    assert transliterate("राम मार्केटिंग प्राइवेट लिमिटेड") == "ram marketing praivet limited"


def test_transliterate_gujarati_medial_schwa():
    assert transliterate("એક્સપોર્ટ્સ") == "eksports"


def test_transliterate_keeps_final_a_after_r_cluster():
    assert transliterate("महाराष्ट्र") == "maharashtra"


def test_transliterate_passthrough_latin():
    assert transliterate("Lucas & Lee") == "Lucas & Lee"


def test_name_accents_brackets_legal():
    nn, core, legal, alt, indic, dom = norm_name("Lesure Sterling (Ráin) Corp")
    assert (core, legal, alt, indic, dom) == ("lesure sterling rain", "corp", "", False, False)


def test_name_dotted_legal_and_honorific():
    _, core, legal, *_ = norm_name("Mr Boyd Newtekone L.L.C.")
    assert (core, legal) == ("boyd newtekone", "llc")


def test_name_domain_only():
    _, core, _, _, _, dom = norm_name(">> classicpeakjones.com")
    assert core == "classicpeakjones" and dom


def test_name_domain_suffix_dropped():
    assert norm_name("Desert Initiative  V | www.desertini.com")[1] == "desert initiative v"


def test_name_formerly_split():
    _, core, legal, alt, *_ = norm_name("Xylozetalum formerly: Corporate Freight Laboratories LLC")
    assert alt == "xylozetalum|corporate freight laboratories" and legal == "llc"


def test_name_phone_removed():
    _, core, legal, *_ = norm_name("Moradabad Tie Private Limited - 9986449917")
    assert (core, legal) == ("moradabad tie", "ltd pvt")


def test_name_indic_flag():
    assert norm_name("हरि एक्सपोर्ट्स")[4] is True


def test_addr_us_street_cdp_state_code():
    assert norm_addr("598 UTICA ST, BUFFALO CDP, NY", "US") == ("598 utica street buffalo ny", "598", False)


def test_addr_us_state_name_and_zero_pad():
    a, nums, _ = norm_addr("Marlborough, Massachusetts, 00307 Hemenway Street", "US")
    assert (a, nums) == ("marlborough ma 307 hemenway street", "307")


def test_addr_nulls_hashes_pobox():
    a, nums, _ = norm_addr("##4231 62, NULL, 1 Ivanhoe Ave, PO Box 6009, Cincinnati, Ohio", "US")
    assert (a, nums) == ("4231 62 1 ivanhoe avenue cincinnati oh", "4231 62 1")


def test_addr_india_code_to_name():
    assert norm_addr("A##301 Shapath Hexa, NULL, Ahmedabad, GJ", "India")[0] == "a 301 shapath hexa ahmedabad gujarat"


def test_addr_france_numero_and_abbrev():
    assert norm_addr("N°30 AV DE BELLEVUE, LA TESTE-DE-BUCH", "France") == (
        "30 avenue de bellevue la teste de buch", "30", False)


def test_addr_saint_vs_street():
    assert norm_addr("Nº 60 AVENUE HECTOR BERLIOZ, ST.-NAZAIRE", "France")[0] == "60 avenue hector berlioz saint nazaire"


def test_addr_ordinal_not_a_number():
    assert norm_addr("150. Beaton Drive, 8th Street", "US")[1] == "150"


def test_addr_empty():
    assert norm_addr("", "US") == ("", "", True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_normalize.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.normalize'`

- [ ] **Step 3: Write the implementation**

`src/normalize.py`:
```python
"""Deterministic text normalisation for business names and addresses (pure functions, no I/O).

All rules are hand-written language knowledge: a Brahmic transliteration table, street/legal
abbreviations and state codes. Nothing is looked up externally.
"""
import re
import unicodedata

# ------------------------------------------------------------------ Brahmic transliteration
# Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada and Malayalam share one
# layout: the code point at offset k inside each 128-wide Unicode block plays the same phonetic role.
_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
         0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
         0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "f", 0x2C: "b",
         0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh",
         0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "gh",
         0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_VOWELS = {0x04: "a", 0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
           0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o",
           0x14: "au", 0x50: "om", 0x60: "ri", 0x61: "li"}
_MATRAS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e",
           0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x62: "li",
           0x63: "li"}
_NASAL = {0x01: "n", 0x02: "n", 0x03: "h"}
# Script-specific code points outside the shared layout: kind C = consonant, F = final consonant.
_SPECIAL = {0x09CE: ("F", "t"), 0x09F0: ("C", "r"), 0x09F1: ("C", "w"), 0x0A70: ("N", "n"),
            0x0B71: ("C", "w"), 0x0D7A: ("F", "n"), 0x0D7B: ("F", "n"), 0x0D7C: ("F", "r"),
            0x0D7D: ("F", "l"), 0x0D7E: ("F", "l"), 0x0D7F: ("F", "k")}
_NUKTA_MAP = {"j": "z", "f": "f", "d": "r", "dh": "rh", "k": "q", "p": "f"}
_KEEP_FINAL_A = frozenset("ryv")  # "राष्ट्र" keeps its final a; "लिमिटेड" does not


def _classify(cp):
    """(kind, latin) for a Brahmic code point, or None for anything else."""
    if cp in _SPECIAL:
        return _SPECIAL[cp]
    if not 0x0900 <= cp < 0x0D80:
        return None
    off = (cp - 0x0900) & 0x7F
    if off in _CONS:
        return "C", _CONS[off]
    if off in _MATRAS:
        return "M", _MATRAS[off]
    if off in _VOWELS:
        return "V", _VOWELS[off]
    if off in _NASAL:
        return "N", _NASAL[off]
    if off == 0x4D:
        return "X", ""  # virama
    if off == 0x3C:
        return "K", ""  # nukta
    if 0x66 <= off <= 0x6F:
        return "D", str(off - 0x66)
    if off in (0x64, 0x65):
        return "S", " "  # danda
    return "Z", ""


def transliterate(text):
    """Brahmic characters -> Latin letters (English-loanword friendly); other characters pass through."""
    if not any(0x0900 <= ord(c) < 0x0D80 for c in text):
        return text
    cls = [_classify(ord(c)) for c in text]
    n = len(text)
    out = []
    pending = False       # last consonant still carries an undecided inherent 'a'
    pend_initial = False  # that consonant starts its word
    last = ""             # latin of that consonant
    cluster = False       # that consonant followed a virama
    after_virama = False
    word_start = True
    for i, ch in enumerate(text):
        c = cls[i]
        if c is None:
            if ch in "\u200c\u200d":
                continue
            if pending and (ch.isalnum() or (cluster and last in _KEEP_FINAL_A)):
                out.append("a")
            pending = after_virama = False
            out.append(ch)
            word_start = not ch.isalnum()
            continue
        kind, lat = c
        if kind == "K":
            if pending and last in _NUKTA_MAP:
                out[-1] = last = _NUKTA_MAP[last]
            continue
        if kind == "X":
            pending, after_virama = False, True
            continue
        if kind == "M":
            pending = after_virama = False
            out.append(lat)
            continue
        if pending:
            nxt = cls[i + 1] if i + 1 < n else None
            medial = kind == "C" and not pend_initial and nxt is not None and nxt[0] == "M"
            if kind != "V" and not medial:
                out.append("a")
            pending = False
        if kind == "C":
            out.append(lat)
            pending, pend_initial, last, cluster = True, word_start, lat, after_virama
            word_start = False
        elif kind == "S":
            out.append(" ")
            word_start = True
        elif lat:
            out.append(lat)
            word_start = False
        after_virama = False
    if pending and cluster and last in _KEEP_FINAL_A:
        out.append("a")
    return "".join(out)


# ------------------------------------------------------------------ shared helpers
_LATIN_EXTRA = str.maketrans({"ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "đ": "d", "ł": "l",
                              "þ": "th", "ı": "i", "°": " ", "º": "o", "ª": "a"})
_DOTTED = re.compile(r"\b(?:[a-z]\.){2,}")  # l.l.c. -> llc, s.a.r.l. -> sarl
_NONALNUM = re.compile(r"[^a-z0-9]+")


def _fold(s):
    s = unicodedata.normalize("NFKD", s.translate(_LATIN_EXTRA))
    return "".join(c for c in s if not unicodedata.combining(c))


def _basic(s):
    s = _fold(transliterate(unicodedata.normalize("NFC", s))).lower()
    return _DOTTED.sub(lambda m: m.group(0).replace(".", ""), s)


def _clean(s):
    return _NONALNUM.sub(" ", s.replace("&", " and ")).strip()


# ------------------------------------------------------------------ names
_LEGAL = {"inc": "inc", "incorporated": "inc", "llc": "llc", "ltd": "ltd", "limited": "ltd",
          "pvt": "pvt", "private": "pvt", "corp": "corp", "corporation": "corp", "co": "co",
          "company": "co", "plc": "plc", "pc": "pc", "pllc": "pllc", "llp": "llp", "lp": "lp",
          "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci",
          "snc": "snc", "ei": "ei", "public": "public", "gmbh": "gmbh", "ag": "ag",
          "aktiengesellschaft": "ag", "5as": "sas"}
_DROP_NAME = frozenset("the and of de du des la le les et d l mr mrs ms dr smt".split())
_ALT_SPLIT = re.compile(r"\b(?:formerly(?:\s+known\s+as)?|fka|dba|aka|trading\s+as)\b\s*:?"
                        r"|\bf/k/a\b|\bd/b/a\b|\ba/k/a\b")
_DOMAIN = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]*)\."
                     r"(?:com|net|org|co\.in|in|co|io|biz|info|us|fr)\b")


def norm_name(raw):
    """-> (name_norm, name_core, legal, alt_core, was_indic, is_domain)."""
    was_indic = any(0x0900 <= ord(c) < 0x0D80 for c in raw)
    s = _basic(raw)
    head = s.split("|")[0]
    if head.strip():
        s = head
    is_domain = False
    m = _DOMAIN.search(s)
    if m:
        if _clean(s[:m.start()] + " " + s[m.end():]):
            s = s[:m.start()] + " " + s[m.end():]   # a real name with a website appended
        else:
            s, is_domain = m.group(1).replace("-", ""), True
    parts = [p for p in (_clean(x) for x in _ALT_SPLIT.split(s)) if p]
    legal, cores = set(), []
    for p in parts:
        keep = []
        for t in p.split():
            if t in _LEGAL:
                legal.add(_LEGAL[t])
            elif t not in _DROP_NAME and not (len(t) >= 5 and t.isdigit()):
                keep.append(t)
        cores.append(" ".join(keep))
    name_norm = " ".join(parts)
    name_core = " ".join(c for c in cores if c) or name_norm
    alt_core = "|".join(c for c in cores if c) if len(parts) > 1 else ""
    return name_norm, name_core, " ".join(sorted(legal)), alt_core, was_indic, is_domain


# ------------------------------------------------------------------ addresses
_NUMSIGN = re.compile(r"\bn\s*[°º]\s*")
_NULLS = re.compile(r"<\s*null\s*>|\bnull\b|\bn/a\b|\bnone\b")
_POBOX = re.compile(r"\b(?:p\s*o\s*box|pmb|post\s+box)\s*#?\s*\d+")
_HALF = re.compile(r"\b1/2\b")
_ORDINAL = re.compile(r"\d+(?:st|nd|rd|th)")
_DIGITS = re.compile(r"\d+")
_ABBR = {"rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "bd": "boulevard",
         "boul": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "cir": "circle",
         "trl": "trail", "pkwy": "parkway", "hwy": "highway", "pl": "place", "sq": "square",
         "trce": "trace", "xing": "crossing", "cres": "crescent", "mt": "mount", "ft": "fort",
         "apt": "unit", "appt": "unit", "apartment": "unit", "suite": "unit", "rm": "unit",
         "room": "unit", "nr": "near", "opp": "opposite", "bldg": "building", "flr": "floor",
         "fl": "floor", "extn": "extension", "ext": "extension", "blk": "block", "sec": "sector",
         "mkt": "market", "dist": "district", "vill": "village", "ch": "chemin", "all": "allee",
         "imp": "impasse", "rte": "route", "crs": "cours", "snte": "sente", "fbg": "faubourg"}
_DIRS = {"n": "north", "s": "south", "e": "east", "w": "west"}
_DROP_ADDR = frozenset(["no", "nos", "door", "hno", "cdp"])
_US = {"alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
       "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
       "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
       "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
       "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
       "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
       "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
       "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
       "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
       "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va",
       "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
       "puerto rico": "pr"}
_IN = {"ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar",
       "cg": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana", "hp": "himachal pradesh",
       "jh": "jharkhand", "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh",
       "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland",
       "od": "odisha", "or": "odisha", "orissa": "odisha", "pb": "punjab", "rj": "rajasthan",
       "sk": "sikkim", "tn": "tamil nadu", "tg": "telangana", "ts": "telangana", "tr": "tripura",
       "up": "uttar pradesh", "uk": "uttarakhand", "ut": "uttarakhand", "wb": "west bengal",
       "dl": "delhi", "jk": "jammu and kashmir", "ch": "chandigarh", "py": "puducherry",
       "pondicherry": "puducherry"}
_FR = {"nord": "hauts de france", "pas de calais": "hauts de france", "gironde": "nouvelle aquitaine",
       "loire atlantique": "pays de la loire"}
# component (whole) -> canonical state/region token; values also map to themselves
_STATES = {"US": {**_US, **{v: v for v in _US.values()}},
           "India": {**_IN, **{v: v for v in _IN.values()}},
           "France": {**_FR, **{v: v for v in _FR.values()}}}


def _expand(toks):
    out = []
    for i, t in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if t in _DROP_ADDR or (t == "h" and nxt == "no"):
            continue
        if t.isdigit():
            out.append(t.lstrip("0") or "0")
        elif t in ("st", "ste"):
            if i == 0 and nxt.isalpha():
                out.append("saint" if t == "st" else "sainte")
            else:
                out.append("street" if t == "st" else "unit")
        elif t == "r" and nxt.isalpha():
            out.append("rue")
        elif t in _DIRS and nxt.isalpha():
            out.append(_DIRS[t])
        else:
            out.append(_ABBR.get(t, t))
    return out


def norm_addr(raw, country=""):
    """-> (addr_norm, numbers, addr_empty)."""
    s = _basic(_NUMSIGN.sub(" ", raw.lower()))
    s = _HALF.sub(" ", _POBOX.sub(" ", _NULLS.sub(" ", s)))
    states = _STATES.get(country, {})
    comps = []
    for comp in s.split(","):
        c = _clean(comp)
        if not c:
            continue
        if c in states:
            comps.append(states[c])
            continue
        c = " ".join(_expand(c.split()))
        if c:
            comps.append(states.get(c, c))
    addr = " ".join(comps)
    nums = [d.lstrip("0") or "0" for t in addr.split() if not _ORDINAL.fullmatch(t)
            for d in _DIGITS.findall(t)]
    return addr, " ".join(nums), addr == ""


def normalize_rows(rows):
    """Worker entry point: [(name, addr, country)] -> list of 9-tuples (see module interfaces)."""
    out = []
    for name, addr, country in rows:
        out.append(norm_name(name) + norm_addr(addr, country))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_normalize.py -q`
Expected: `19 passed`. If a transliteration expectation fails, print the actual output and fix the table/rule, not the test, unless the actual output is equally good for matching (then update the expectation and note why).

- [ ] **Step 5: Commit**

```bash
git add src/normalize.py tests/test_normalize.py
git commit -m "feat: transliteration and name/address normalisation"
```

---

### Task 3: `prepare` stage (parallel normalisation + splits + GT rows)

**Files:**
- Create: `code/business_entity_resolution/src/run_pipeline.py` (stages are added here task by task)

**Interfaces:**
- Consumes: `io_utils.read_source`, `io_utils.read_gt_pairs`, `normalize.normalize_rows`, `config.work`.
- Produces (files): `WORK/prep/{split}_s1.parquet` and `WORK/prep/{split}_sx.parquet` with columns `entity_id, country, name_norm, name_core, legal, alt_core, was_indic, is_domain, addr_norm, numbers, addr_empty` (+ `src` Int8 in sx: 2 or 3; + `part` Utf8 E/J/V in train_s1). Row number = the integer id used everywhere else. `WORK/prep/train_gt_rows.parquet` with `s1_row, sx_row` (Int32).
- Produces (functions in run_pipeline.py): `load_prep(split, which, columns=None)->pl.DataFrame`, `part_of(entity_id)->str`.

- [ ] **Step 1: Write the stage**

`src/run_pipeline.py`:
```python
"""CLI for every pipeline stage: python -m src.run_pipeline <stage> [options]."""
import argparse
import os
import time
import zlib
from multiprocessing import Pool

import numpy as np
import polars as pl

from . import config
from .config import work
from .io_utils import read_gt_pairs, read_source
from .normalize import normalize_rows

PREP_COLS = ["name_norm", "name_core", "legal", "alt_core", "was_indic", "is_domain",
             "addr_norm", "numbers", "addr_empty"]
CHUNK = 100_000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def part_of(entity_id):
    h = zlib.crc32(entity_id.encode()) % 100
    return "E" if h < config.SPLIT_E else ("J" if h < config.SPLIT_J else "V")


def load_prep(split, which, columns=None):
    return pl.read_parquet(work("prep", f"{split}_{which}.parquet"), columns=columns)


def _normalize_frame(df, pool):
    rows = list(zip(df["business_name"].to_list(), df["business_address"].to_list(),
                    df["country"].to_list()))
    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    parts = []
    for res in pool.imap(normalize_rows, chunks):
        parts.append(pl.DataFrame(res, schema=[(c, pl.Boolean if c in ("was_indic", "is_domain", "addr_empty")
                                                else pl.Utf8) for c in PREP_COLS], orient="row"))
    return pl.concat([df.select("entity_id", "country"), pl.concat(parts)], how="horizontal")


def stage_prepare(args):
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        for split in ("train", "test"):
            s1 = _normalize_frame(read_source(split, 1), pool)
            if split == "train":
                s1 = s1.with_columns(pl.Series("part", [part_of(e) for e in s1["entity_id"].to_list()]))
            s1.write_parquet(work("prep", f"{split}_s1.parquet"))
            log(split, "s1", s1.shape)
            sx = pl.concat([_normalize_frame(read_source(split, s), pool)
                            .with_columns(pl.lit(s, dtype=pl.Int8).alias("src")) for s in (2, 3)])
            sx.write_parquet(work("prep", f"{split}_sx.parquet"))
            log(split, "sx", sx.shape)
    s1 = load_prep("train", "s1", ["entity_id"]).with_row_index("s1_row")
    sx = load_prep("train", "sx", ["entity_id"]).with_row_index("sx_row")
    gt = (read_gt_pairs().join(s1, left_on="s1", right_on="entity_id")
          .join(sx, left_on="sx", right_on="entity_id")
          .select(pl.col("s1_row").cast(pl.Int32), pl.col("sx_row").cast(pl.Int32)))
    gt.write_parquet(work("prep", "train_gt_rows.parquet"))
    log("gt rows", gt.shape)


STAGES = {"prepare": stage_prepare}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--floor", type=float, default=None)
    args = ap.parse_args()
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the stage**

Run: `python -m src.run_pipeline prepare`
Expected (≈5–10 min): logs `train s1 (2206821, 13)`, `train sx (10320219, 12)`, `test s1 (1732544, 11)`, `test sx (9969589, 12)`, `gt rows (7638365, 2)`.

- [ ] **Step 3: Spot-check normalisation on real rows**

Run:
```bash
python -c "import polars as pl; d=pl.read_parquet(r'C:\amazon_ml\work\prep\train_sx.parquet'); print(d.filter(pl.col('was_indic')).select('name_norm','addr_norm').head(8)); print(d.filter(pl.col('is_domain')).select('name_core').head(4)); print(d['part'] if 'part' in d.columns else '')"
```
Expected: Indic names appear as readable Latin (e.g. "hari eksports praivet limited"); domain names appear without TLD. Fix `normalize.py` (with a new test) if something is clearly broken.

- [ ] **Step 4: Commit**

```bash
git add src/run_pipeline.py
git commit -m "feat: prepare stage with parallel normalisation, E/J/V split and GT rows"
```

---

### Task 4: Hashed n-gram featurizer

**Files:**
- Create: `code/business_entity_resolution/src/featurize.py`
- Test: `code/business_entity_resolution/tests/test_featurize.py`

**Interfaces:**
- Produces: `hash_features(strings: list[str], field_offset: int) -> (indices int64[nnz], offsets int64[n])` — per string: one hashed word id per token plus hashed char 3- and 4-grams of the token padded with `<` `>`; ids lie in `[field_offset, field_offset + HASH_BUCKETS)`; empty strings give empty bags.

- [ ] **Step 1: Write the failing tests**

`tests/test_featurize.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_featurize.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.featurize'`

- [ ] **Step 3: Write the implementation**

`src/featurize.py`:
```python
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
            c += 1
            if L + 2 >= 3:
                c += L
            if L + 2 >= 4:
                c += L - 1
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
```
(`_count`: a token of length L has L 3-grams when padded (P-3+1 = L) and L-1 4-grams, plus one word id.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_featurize.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/featurize.py tests/test_featurize.py
git commit -m "feat: numba char n-gram hashing featurizer"
```

---

### Task 5: Fingerprint encoder (train on E, encode everything)

**Files:**
- Create: `code/business_entity_resolution/src/encoder.py`
- Modify: `src/run_pipeline.py` (add stages `train-encoder`, `encode`)

**Interfaces:**
- Consumes: `hash_features`, `load_prep`, `WORK/prep/train_gt_rows.parquet`.
- Produces: `FingerprintModel`; `train_encoder(s1_names, s1_addrs, sx_names, sx_addrs, pair_s1, pair_sx, epochs=8, batch=8192, tau=0.05, log=print) -> model` (names/addrs are `pl.Series`; pair arrays index them); `encode_to(model, names: pl.Series, addrs: pl.Series, path) -> np.memmap float16 (n, 2*EMB_DIM)` (rows = L2-normalised concat(name_vec, addr_vec), addr half zero when the address is empty). Files: `WORK/models/encoder.pt`, `WORK/emb/{split}_{s1|sx}.npy`.

- [ ] **Step 1: Write the implementation**

`src/encoder.py`:
```python
"""Neural fingerprint bi-encoder: hashed n-gram bags -> unit name and address vectors."""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EMB_DIM, HASH_BUCKETS
from .featurize import hash_features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FingerprintModel(nn.Module):
    """Shared hashed-n-gram table with a learned per-bucket weight, one small MLP tower per field."""

    def __init__(self, dim=EMB_DIM, buckets=2 * HASH_BUCKETS):
        super().__init__()
        self.emb = nn.EmbeddingBag(buckets, dim, mode="sum", sparse=True)
        self.gate = nn.Embedding(buckets, 1, sparse=True)
        nn.init.normal_(self.emb.weight, std=0.05)
        nn.init.constant_(self.gate.weight, math.log(math.e - 1))  # softplus(gate) starts at 1
        self.name_mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.addr_mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def _field(self, idx, off, bag, mlp):
        w = F.softplus(self.gate(idx)).squeeze(-1)
        v = self.emb(idx, off, per_sample_weights=w)
        wsum = torch.zeros(v.shape[0], device=v.device).index_add_(0, bag, w)
        return F.normalize(mlp(v / wsum.clamp_min(1e-6).unsqueeze(-1)), dim=-1)

    def forward(self, name_bags, addr_bags, addr_mask):
        n = self._field(*name_bags, self.name_mlp)
        a = self._field(*addr_bags, self.addr_mlp) * addr_mask.unsqueeze(-1)
        return n, a, F.normalize(torch.cat([n, a], dim=-1), dim=-1)


def _bags(strings, field_offset):
    idx, off = hash_features(strings, field_offset)
    lengths = np.diff(np.append(off, len(idx)))
    bag = np.repeat(np.arange(len(off)), lengths)
    return (torch.from_numpy(idx).to(DEVICE), torch.from_numpy(off).to(DEVICE),
            torch.from_numpy(bag).to(DEVICE))


def _forward(model, names, addrs):
    mask = torch.tensor([1.0 if a else 0.0 for a in addrs], device=DEVICE)
    return model(_bags(names, 0), _bags(addrs, HASH_BUCKETS), mask)


def _info_nce(q, k, tau):
    logits = q @ k.t() / tau
    y = torch.arange(q.shape[0], device=q.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.t(), y))


def train_encoder(s1_names, s1_addrs, sx_names, sx_addrs, pair_s1, pair_sx, epochs=8, batch=8192,
                  tau=0.05, seed=0, log=print):
    """Contrastive training. Each epoch uses one random SX per S1, so in-batch negatives never share an S1."""
    torch.manual_seed(seed)
    model = FingerprintModel().to(DEVICE)
    opt_s = torch.optim.SparseAdam([model.emb.weight, model.gate.weight], lr=0.01)
    opt_d = torch.optim.Adam([p for n, p in model.named_parameters() if not n.startswith(("emb.", "gate."))],
                             lr=2e-3)
    order = np.argsort(pair_s1, kind="stable")
    ps1, psx = pair_s1[order], pair_sx[order]
    starts = np.flatnonzero(np.r_[True, ps1[1:] != ps1[:-1]])
    counts = np.diff(np.r_[starts, len(ps1)])
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        pick = starts + (rng.random(len(starts)) * counts).astype(np.int64)
        pick = pick[rng.permutation(len(pick))]
        t0, tot, acc, nb = time.time(), 0.0, 0.0, 0
        model.train()
        for b in range(0, len(pick) - batch + 1, batch):
            sel = pick[b:b + batch]
            i1, ix = ps1[sel], psx[sel]
            n1, a1, r1 = _forward(model, s1_names.gather(i1).to_list(), s1_addrs.gather(i1).to_list())
            nx, ax, rx = _forward(model, sx_names.gather(ix).to_list(), sx_addrs.gather(ix).to_list())
            loss = _info_nce(r1, rx, tau) + 0.5 * _info_nce(n1, nx, tau)
            both = (a1.abs().sum(-1) > 0) & (ax.abs().sum(-1) > 0)
            if int(both.sum()) > 1:
                loss = loss + 0.5 * _info_nce(a1[both], ax[both], tau)
            opt_s.zero_grad()
            opt_d.zero_grad()
            loss.backward()
            opt_s.step()
            opt_d.step()
            with torch.no_grad():
                acc += ((r1 @ rx.t()).argmax(1) == torch.arange(len(sel), device=DEVICE)).float().mean().item()
            tot, nb = tot + loss.item(), nb + 1
        log(f"epoch {ep}: loss {tot / nb:.4f}  in-batch top1 {acc / nb:.4f}  {time.time() - t0:.0f}s")
    return model


@torch.no_grad()
def encode_to(model, names, addrs, path, batch=65536):
    """Write float16 record vectors for every row to a .npy memmap at `path`."""
    model.eval()
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(names), 2 * EMB_DIM))
    for b in range(0, len(names), batch):
        _, _, r = _forward(model, names.slice(b, batch).to_list(), addrs.slice(b, batch).to_list())
        out[b:b + batch] = r.half().cpu().numpy()
    out.flush()
    return out
```

- [ ] **Step 2: Add the stages to `run_pipeline.py`**

Add below `stage_prepare`:
```python
def stage_train_encoder(args):
    import torch

    from .encoder import train_encoder
    s1 = load_prep("train", "s1", ["name_norm", "addr_norm", "part"])
    sx = load_prep("train", "sx", ["name_norm", "addr_norm"])
    gt = pl.read_parquet(work("prep", "train_gt_rows.parquet"))
    is_e = (s1["part"] == "E").to_numpy()
    gt = gt.filter(pl.Series(is_e[gt["s1_row"].to_numpy()]))
    log("encoder pairs (E)", gt.height)
    model = train_encoder(s1["name_norm"], s1["addr_norm"], sx["name_norm"], sx["addr_norm"],
                          gt["s1_row"].to_numpy().astype(np.int64), gt["sx_row"].to_numpy().astype(np.int64),
                          log=log)
    torch.save(model.state_dict(), work("models", "encoder.pt"))


def load_encoder():
    import torch

    from .encoder import DEVICE, FingerprintModel
    model = FingerprintModel().to(DEVICE)
    model.load_state_dict(torch.load(work("models", "encoder.pt"), map_location=DEVICE, weights_only=True))
    return model


def stage_encode(args):
    from .encoder import encode_to
    model = load_encoder()
    for which in ("s1", "sx"):
        df = load_prep(args.split, which, ["name_norm", "addr_norm"])
        encode_to(model, df["name_norm"], df["addr_norm"], work("emb", f"{args.split}_{which}.npy"))
        log("encoded", args.split, which, df.height)
```
and extend the registry:
```python
STAGES = {"prepare": stage_prepare, "train-encoder": stage_train_encoder, "encode": stage_encode}
```

- [ ] **Step 3: Train**

Run: `python -m src.run_pipeline train-encoder`
Expected: 8 epoch lines; loss falls epoch over epoch; in-batch top1 ≥ 0.95 by the last epoch; total ≲ 15 min. If top1 < 0.9, raise epochs to 12 and retrain before moving on.

- [ ] **Step 4: Encode train and test** (GPU; run in background while doing Task 7)

Run: `python -m src.run_pipeline encode --split train` then `python -m src.run_pipeline encode --split test`
Expected: `encoded train s1 2206821`, `encoded train sx 10320219`, `encoded test s1 1732544`, `encoded test sx 9969589`; four `.npy` files in `C:\amazon_ml\work\emb` (≈5.9 GB total).

- [ ] **Step 5: Commit**

```bash
git add src/encoder.py src/run_pipeline.py
git commit -m "feat: fingerprint bi-encoder training and encoding stages"
```

---

### Task 6: GPU kNN candidates + recall report

**Files:**
- Create: `code/business_entity_resolution/src/candidates.py`
- Modify: `src/run_pipeline.py` (add stage `candidates`)

**Interfaces:**
- Consumes: embeddings `WORK/emb/{split}_{s1|sx}.npy`, prep `country`/`part` columns, GT rows.
- Produces: `knn(q, d, k) -> (idx int64 (nq,k), cos, cos_name, cos_addr float32 (nq,k))` (cos_addr = -1 when either address is empty); file `WORK/cand/{split}.parquet` with columns `s1 (Int32 row), sx (Int32 row), cos, cos_name, cos_addr (Float32), rank (Int16)`; queries are J+V S1 rows for train, all S1 rows for test. `recall_report(cand, gt_rows, s1_rows)` prints recall at K∈{5,10,20,30,40} and at cos floors.

- [ ] **Step 1: Write the implementation**

`src/candidates.py`:
```python
"""GPU brute-force cosine kNN between S1 and SX fingerprint vectors (per country)."""
import numpy as np
import polars as pl
import torch

from .config import EMB_DIM

DEVICE = "cuda"


def _half_cos(q, d):
    """Cosine between matching halves; -1 where either half is all-zero (empty address)."""
    num = (q.unsqueeze(1) * d).sum(-1)
    den = q.norm(dim=-1, keepdim=True) * d.norm(dim=-1)
    return torch.where(den > 1e-6, num / den.clamp_min(1e-6), torch.full_like(num, -1.0))


@torch.no_grad()
def knn(q_emb, d_emb, k, q_chunk=1024, d_block=300_000):
    D = torch.from_numpy(np.ascontiguousarray(d_emb)).to(DEVICE)  # float16 on GPU
    k = min(k, len(D))
    nq = len(q_emb)
    out = {n: np.empty((nq, k), dtype=np.float32) for n in ("cos", "cos_name", "cos_addr")}
    out_idx = np.empty((nq, k), dtype=np.int64)
    for s in range(0, nq, q_chunk):
        Q = torch.from_numpy(np.asarray(q_emb[s:s + q_chunk], dtype=np.float32)).to(DEVICE)
        best_v = torch.full((len(Q), k), -2.0, device=DEVICE)
        best_i = torch.zeros((len(Q), k), dtype=torch.int64, device=DEVICE)
        for b in range(0, len(D), d_block):
            sims = Q @ D[b:b + d_block].float().T
            v, i = sims.topk(min(k, sims.shape[1]), dim=1)
            v, i = torch.cat([best_v, v], 1), torch.cat([best_i, i + b], 1)
            best_v, pos = v.topk(k, dim=1)
            best_i = i.gather(1, pos)
        Dk = D[best_i].float()
        out["cos"][s:s + len(Q)] = best_v.cpu().numpy()
        out["cos_name"][s:s + len(Q)] = _half_cos(Q[:, :EMB_DIM], Dk[..., :EMB_DIM]).cpu().numpy()
        out["cos_addr"][s:s + len(Q)] = _half_cos(Q[:, EMB_DIM:], Dk[..., EMB_DIM:]).cpu().numpy()
        out_idx[s:s + len(Q)] = best_i.cpu().numpy()
    del D
    torch.cuda.empty_cache()
    return out_idx, out["cos"], out["cos_name"], out["cos_addr"]


def build_candidates(s1_country, sx_country, q_rows, s1_emb, sx_emb, k, log=print):
    """kNN per country for the S1 rows in q_rows. Returns a polars frame sorted by (s1, rank)."""
    frames = []
    for c in sorted(set(s1_country[q_rows].tolist())):
        qr = q_rows[s1_country[q_rows] == c]
        dr = np.flatnonzero(sx_country == c)
        log(f"knn {c}: {len(qr)} queries x {len(dr)} records")
        idx, cos, cn, ca = knn(s1_emb[qr], sx_emb[dr], k)
        kk = idx.shape[1]
        frames.append(pl.DataFrame({
            "s1": np.repeat(qr, kk).astype(np.int32), "sx": dr[idx.ravel()].astype(np.int32),
            "cos": cos.ravel(), "cos_name": cn.ravel(), "cos_addr": ca.ravel(),
            "rank": np.tile(np.arange(kk, dtype=np.int16), len(qr))}))
    return pl.concat(frames).sort("s1", "rank")


def recall_report(cand, gt_rows, s1_rows, log=print):
    """Share of true (s1, sx) pairs of `s1_rows` found in `cand`, by K and by cosine floor."""
    truth = gt_rows.filter(pl.col("s1_row").is_in(pl.Series(np.asarray(s1_rows, dtype=np.int32)).implode()))
    hit = truth.join(cand.select(pl.col("s1").alias("s1_row"), pl.col("sx").alias("sx_row"), "rank", "cos"),
                     on=["s1_row", "sx_row"], how="left")
    n = truth.height
    for K in (5, 10, 20, 30, 40):
        log(f"  recall@{K}: {hit.filter(pl.col('rank') < K).height / n:.4f}")
    for f in (0.2, 0.3, 0.4, 0.5):
        kept = cand.filter(pl.col("cos") >= f)
        log(f"  floor {f}: recall {hit.filter(pl.col('cos') >= f).height / n:.4f}  pairs/S1 "
            f"{kept.height / max(1, cand['s1'].n_unique()):.1f}")
```

- [ ] **Step 2: Add the stage**

```python
def stage_candidates(args):
    from .candidates import build_candidates, recall_report
    s1 = load_prep(args.split, "s1", ["country"] + (["part"] if args.split == "train" else []))
    sx = load_prep(args.split, "sx", ["country"])
    s1_emb = np.load(work("emb", f"{args.split}_s1.npy"), mmap_mode="r")
    sx_emb = np.load(work("emb", f"{args.split}_sx.npy"), mmap_mode="r")
    if args.split == "train":
        q_rows = np.flatnonzero((s1["part"] != "E").to_numpy())
    else:
        q_rows = np.arange(s1.height)
    cand = build_candidates(s1["country"].to_numpy(), sx["country"].to_numpy(), q_rows, s1_emb, sx_emb,
                            config.TOP_K, log=log)
    cand.write_parquet(work("cand", f"{args.split}.parquet"))
    log("candidates", cand.shape)
    if args.split == "train":
        gt = pl.read_parquet(work("prep", "train_gt_rows.parquet"))
        for p in ("J", "V"):
            log(f"recall on {p}:")
            recall_report(cand, gt, np.flatnonzero((s1["part"] == p).to_numpy()), log=log)
```
Registry: add `"candidates": stage_candidates`.

- [ ] **Step 3: Run on train (J+V queries)**

Run: `python -m src.run_pipeline candidates --split train`
Expected (≈15–20 min): per-country knn logs, `candidates (22xxxxxx, 6)`, and recall lines. **Gate:** recall@40 on V ≥ 0.99. Record the smallest floor that keeps recall within 0.002 of recall@40 — that floor is passed to the next stages as `--floor`. If recall@40 < 0.99: retrain the encoder with epochs=12 (Task 5 Step 3) before continuing.

- [ ] **Step 4: Run on test in the background**

Run: `python -m src.run_pipeline candidates --split test`
Expected (≈35 min): `candidates (69301760, 6)` (= 1,732,544 × 40).

- [ ] **Step 5: Commit**

```bash
git add src/candidates.py src/run_pipeline.py
git commit -m "feat: GPU kNN candidate generation with recall report"
```

---

### Task 7: Evaluation and decision rules

**Files:**
- Create: `code/business_entity_resolution/src/evaluate.py`, `src/decide.py`
- Test: `code/business_entity_resolution/tests/test_evaluate_decide.py`

**Interfaces:**
- Produces: `evaluate.f05(pred:set, truth:set)->float`; `evaluate.macro_f05(pred_by_s1:dict, truth_by_s1:dict, s1_ids)->float`; `decide.assign_best(sx, p)->bool mask`; `decide.expected_f05_cut(s1, p, max_k=12, n_samples=256, seed=0)->bool mask`; `decide.threshold_cut(p, t)->bool mask`.

- [ ] **Step 1: Write the failing tests**

`tests/test_evaluate_decide.py`:
```python
import numpy as np
import pytest

from src.decide import assign_best, expected_f05_cut, threshold_cut
from src.evaluate import f05, macro_f05


def test_f05_organizer_example():
    assert f05({"a", "b", "c"}, {"a", "c"}) == pytest.approx(0.7142857, abs=1e-6)


def test_f05_singleton_rules():
    assert f05(set(), set()) == 1.0
    assert f05({"x"}, set()) == 0.0
    assert f05(set(), {"x"}) == 0.0


def test_macro_f05_averages_all_s1():
    assert macro_f05({"s1": {"a"}}, {"s1": {"a"}, "s2": set()}, ["s1", "s2"]) == 1.0


def test_assign_best_keeps_top_s1_per_sx():
    sx = np.array([1, 1, 2])
    p = np.array([0.3, 0.9, 0.5])
    assert assign_best(sx, p).tolist() == [False, True, True]


def test_expected_f05_cut_picks_confident_and_skips_singleton():
    s1 = np.array([0, 0, 0, 1])
    p = np.array([0.95, 0.9, 0.05, 0.02])
    assert expected_f05_cut(s1, p).tolist() == [True, True, False, False]


def test_threshold_cut():
    assert threshold_cut(np.array([0.2, 0.6]), 0.5).tolist() == [False, True]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_evaluate_decide.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

`src/evaluate.py`:
```python
"""The organizer metric: macro-averaged F0.5 over Source-1 entities (singletons included)."""
import numpy as np


def f05(pred, truth):
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_by_s1, truth_by_s1, s1_ids):
    return float(np.mean([f05(pred_by_s1.get(s, set()), truth_by_s1.get(s, set())) for s in s1_ids]))
```

`src/decide.py`:
```python
"""Turn pair probabilities into per-S1 match lists."""
import numpy as np


def assign_best(sx, p):
    """Every SX belongs to at most one S1: keep only its highest-probability pair."""
    order = np.lexsort((-p, sx))
    first = np.r_[True, sx[order][1:] != sx[order][:-1]]
    keep = np.zeros(len(p), dtype=bool)
    keep[order[first]] = True
    return keep


def threshold_cut(p, t):
    return p >= t


def expected_f05_cut(s1, p, max_k=12, n_samples=256, seed=0, chunk=20_000):
    """Per S1, predict the top-k pairs (by p) where k maximises expected F0.5 under independent
    Bernoulli(p) labels; k = 0 is chosen when 'no match' is the better bet."""
    order = np.lexsort((-p, s1))
    s_sorted, p_sorted = s1[order], p[order]
    starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
    counts = np.diff(np.r_[starts, len(s_sorted)])
    rank = np.arange(len(s_sorted)) - np.repeat(starts, counts)
    G = len(starts)
    P = np.zeros((G, max_k), dtype=np.float32)
    m = rank < max_k
    P[np.repeat(np.arange(G), counts)[m], rank[m]] = p_sorted[m]
    rng = np.random.default_rng(seed)
    best_k = np.zeros(G, dtype=np.int64)
    ks = np.arange(1, max_k + 1, dtype=np.float32)
    for g in range(0, G, chunk):
        Pc = P[g:g + chunk]
        Z = rng.random((len(Pc), n_samples, max_k), dtype=np.float32) < Pc[:, None, :]
        T = Z.sum(-1, dtype=np.float32)
        TP = np.cumsum(Z, -1, dtype=np.float32)
        F = (1.25 * TP / (0.25 * T[..., None] + ks)).mean(1)
        F0 = (T == 0).mean(1, dtype=np.float32)
        best_k[g:g + chunk] = np.argmax(np.concatenate([F0[:, None], F], 1), 1)
    pick_sorted = rank < np.repeat(best_k, counts)
    mask = np.zeros(len(p), dtype=bool)
    mask[order] = pick_sorted
    return mask
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evaluate_decide.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py src/decide.py tests/test_evaluate_decide.py
git commit -m "feat: organizer F0.5 metric, assignment and expected-F0.5 cut"
```

---

### Task 8: Pair features

**Files:**
- Create: `code/business_entity_resolution/src/features.py`
- Modify: `src/run_pipeline.py` (add stage `features`)

**Interfaces:**
- Consumes: prep frames, candidate frame (`s1, sx, cos, cos_name, cos_addr, rank`).
- Produces: `FEATURES: list[str]` (42 names, order = column order); `class PairFeaturizer(s1: pl.DataFrame, sx: pl.DataFrame)` with `.compute(cand_chunk: pl.DataFrame) -> np.ndarray float32 (n, 42)`; stage `features --split train [--floor f]` writes `WORK/feat/train_{J|V}.npy` (features), `WORK/feat/train_{J|V}_pairs.parquet` (`s1, sx, label`); J rows ordered so that the last 10% of J S1s (by row % 10 == 0) come last (eval slice for early stopping).

- [ ] **Step 1: Write the implementation**

`src/features.py`:
```python
"""Pair features for the judge: embedding cosines, token-set overlaps, fuzzy scores, numbers, flags."""
from array import array

import numba
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

FEATURES = [
    "cos", "cos_name", "cos_addr", "rank", "gap_best", "z_in_list", "list_size",
    "name_inter", "name_len1", "name_lenx", "name_jacc", "name_cont1", "name_contx", "name_idf_jacc",
    "name_idf_inter", "tset", "tsort", "partial", "ratio", "nospace", "jw", "alt_best", "legal_code",
    "addr_inter", "addr_len1", "addr_lenx", "addr_jacc", "addr_idf_jacc", "addr_tset", "addr_ratio",
    "num_primary_eq", "num_any_eq", "num_log_mindiff", "num_min_rel", "num_n1", "num_nx",
    "num_x_unmatched", "num_x_primary_in_1", "was_indic", "is_domain", "addr_empty_x", "is_s3"]


def _token_csr(strings, vocab, skip_digits=False):
    """Space-separated tokens -> (indptr, sorted unique int32 ids); vocab grows in place."""
    indptr = np.zeros(len(strings) + 1, dtype=np.int64)
    ids = array("i")
    for r, s in enumerate(strings):
        if s:
            ids.extend(sorted({vocab.setdefault(t, len(vocab)) for t in s.split()
                               if not (skip_digits and t.isdigit())}))
        indptr[r + 1] = len(ids)
    return indptr, np.frombuffer(ids, dtype=np.int32).copy()


def add_list_features(cand):
    """Per-S1 list context (needs the whole candidate list, so run before chunking).
    `cand` must be grouped by s1 (it is sorted by s1, rank)."""
    s1r, cos = cand["s1"].to_numpy(), cand["cos"].to_numpy().astype(np.float32)
    starts = np.flatnonzero(np.r_[True, s1r[1:] != s1r[:-1]])
    counts = np.diff(np.r_[starts, len(s1r)])
    mx = np.maximum.reduceat(cos, starts)
    mu = np.add.reduceat(cos, starts) / counts
    sd = np.sqrt(np.maximum(np.add.reduceat(cos * cos, starts) / counts - mu * mu, 0))
    rep = lambda v: np.repeat(v, counts).astype(np.float32)
    return cand.with_columns(pl.Series("gap_best", rep(mx) - cos),
                             pl.Series("z_in_list", (cos - rep(mu)) / (rep(sd) + 1e-3)),
                             pl.Series("list_size", rep(counts)))


def _number_csr(strings):
    indptr = np.zeros(len(strings) + 1, dtype=np.int64)
    vals = array("q")
    for r, s in enumerate(strings):
        if s:
            vals.extend(int(t[:15]) for t in s.split())
        indptr[r + 1] = len(vals)
    return indptr, np.frombuffer(vals, dtype=np.int64).copy()


@numba.njit(parallel=True, cache=True)
def _set_feats(a_ptr, a_ids, b_ptr, b_ids, idf, pa, pb, out):
    """out: inter, |A|, |B|, idf(A∩B), idf(A∪B)."""
    for q in numba.prange(len(pa)):
        i, j = pa[q], pb[q]
        x, xe, y, ye = a_ptr[i], a_ptr[i + 1], b_ptr[j], b_ptr[j + 1]
        inter, wi, wu = 0, 0.0, 0.0
        while x < xe and y < ye:
            if a_ids[x] == b_ids[y]:
                inter += 1
                wi += idf[a_ids[x]]
                wu += idf[a_ids[x]]
                x += 1
                y += 1
            elif a_ids[x] < b_ids[y]:
                wu += idf[a_ids[x]]
                x += 1
            else:
                wu += idf[b_ids[y]]
                y += 1
        while x < xe:
            wu += idf[a_ids[x]]
            x += 1
        while y < ye:
            wu += idf[b_ids[y]]
            y += 1
        out[q, 0] = inter
        out[q, 1] = a_ptr[i + 1] - a_ptr[i]
        out[q, 2] = b_ptr[j + 1] - b_ptr[j]
        out[q, 3] = wi
        out[q, 4] = wu


@numba.njit(parallel=True, cache=True)
def _num_feats(a_ptr, a_val, b_ptr, b_val, pa, pb, out):
    """out: primary_eq, any_eq, log1p(min|diff|), min rel diff, n_a, n_b, b_unmatched, b_primary_in_a.
    -1 marks 'not applicable' (a side without numbers)."""
    for q in numba.prange(len(pa)):
        i, j = pa[q], pb[q]
        x0, x1, y0, y1 = a_ptr[i], a_ptr[i + 1], b_ptr[j], b_ptr[j + 1]
        out[q, 4] = x1 - x0
        out[q, 5] = y1 - y0
        if x1 == x0 or y1 == y0:
            for c in (0, 1, 2, 3, 7):
                out[q, c] = -1.0
            out[q, 6] = y1 - y0
            continue
        prim = a_val[x0]
        peq, anyeq, unmatched, best, brel = 0, 0, 0, 1e18, 1e18
        for y in range(y0, y1):
            v = b_val[y]
            if v == prim:
                peq = 1
            found = 0
            for x in range(x0, x1):
                u = a_val[x]
                if u == v:
                    found = 1
                d = abs(u - v)
                if d < best:
                    best = d
                r = d / max(u, v, 1)
                if r < brel:
                    brel = r
            if found:
                anyeq = 1
            else:
                unmatched += 1
        bprim = 0
        for x in range(x0, x1):
            if a_val[x] == b_val[y0]:
                bprim = 1
        out[q, 0] = peq
        out[q, 1] = anyeq
        out[q, 2] = np.log1p(best)
        out[q, 3] = brel
        out[q, 6] = unmatched
        out[q, 7] = bprim


def _legal_matrix(uniq):
    M = np.zeros((len(uniq), len(uniq)), dtype=np.float32)
    sets = [set(u.split()) for u in uniq]
    for a, sa in enumerate(sets):
        for b, sb in enumerate(sets):
            M[a, b] = (0 if not sa and not sb else 1 if sa == sb else 2 if not sa or not sb
                       else 3 if sa & sb else 4)
    return M


def _cp(scorer, a, b):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / 100.0


class PairFeaturizer:
    """Holds per-record token structures for one split; computes features for candidate chunks."""

    def __init__(self, s1, sx):
        self.s1, self.sx = s1, sx
        vocab_n, vocab_a = {}, {}
        self.n1 = _token_csr(s1["name_core"].to_list(), vocab_n)
        self.nx = _token_csr(sx["name_core"].to_list(), vocab_n)
        self.a1 = _token_csr(s1["addr_norm"].to_list(), vocab_a, skip_digits=True)
        self.ax = _token_csr(sx["addr_norm"].to_list(), vocab_a, skip_digits=True)
        self.num1 = _number_csr(s1["numbers"].to_list())
        self.numx = _number_csr(sx["numbers"].to_list())
        self.idf_n = self._idf(self.nx, len(vocab_n), sx.height)
        self.idf_a = self._idf(self.ax, len(vocab_a), sx.height)
        uniq = sorted(set(s1["legal"].unique().to_list()) | set(sx["legal"].unique().to_list()))
        code = {u: i for i, u in enumerate(uniq)}
        self.legal_M = _legal_matrix(uniq)
        self.legal1 = np.array([code[u] for u in s1["legal"].to_list()], dtype=np.int32)
        self.legalx = np.array([code[u] for u in sx["legal"].to_list()], dtype=np.int32)
        self.flags_x = np.stack([sx["was_indic"].to_numpy(), sx["is_domain"].to_numpy(),
                                 sx["addr_empty"].to_numpy(), (sx["src"] == 3).to_numpy()], 1).astype(np.float32)

    @staticmethod
    def _idf(csr, n_vocab, n_docs):
        df = np.bincount(csr[1], minlength=n_vocab).astype(np.float32)
        return np.log((n_docs + 1) / (df + 1)).astype(np.float32) + 1.0

    def compute(self, cand):
        s1r, sxr = cand["s1"].to_numpy().astype(np.int64), cand["sx"].to_numpy().astype(np.int64)
        n = len(s1r)
        X = np.zeros((n, len(FEATURES)), dtype=np.float32)
        col = {f: i for i, f in enumerate(FEATURES)}
        cos = cand["cos"].to_numpy()
        X[:, col["cos"]] = cos
        X[:, col["cos_name"]] = cand["cos_name"].to_numpy()
        X[:, col["cos_addr"]] = cand["cos_addr"].to_numpy()
        for f in ("rank", "gap_best", "z_in_list", "list_size"):  # list features: see add_list_features
            X[:, col[f]] = cand[f].to_numpy()

        o = np.zeros((n, 5), dtype=np.float32)
        _set_feats(self.n1[0], self.n1[1], self.nx[0], self.nx[1], self.idf_n, s1r, sxr, o)
        X[:, col["name_inter"]], X[:, col["name_len1"]], X[:, col["name_lenx"]] = o[:, 0], o[:, 1], o[:, 2]
        union = np.maximum(o[:, 1] + o[:, 2] - o[:, 0], 1)
        X[:, col["name_jacc"]] = o[:, 0] / union
        X[:, col["name_cont1"]] = o[:, 0] / np.maximum(o[:, 1], 1)
        X[:, col["name_contx"]] = o[:, 0] / np.maximum(o[:, 2], 1)
        X[:, col["name_idf_jacc"]] = o[:, 3] / np.maximum(o[:, 4], 1e-6)
        X[:, col["name_idf_inter"]] = o[:, 3]
        _set_feats(self.a1[0], self.a1[1], self.ax[0], self.ax[1], self.idf_a, s1r, sxr, o)
        X[:, col["addr_inter"]], X[:, col["addr_len1"]], X[:, col["addr_lenx"]] = o[:, 0], o[:, 1], o[:, 2]
        X[:, col["addr_jacc"]] = o[:, 0] / np.maximum(o[:, 1] + o[:, 2] - o[:, 0], 1)
        X[:, col["addr_idf_jacc"]] = o[:, 3] / np.maximum(o[:, 4], 1e-6)

        on = np.zeros((n, 8), dtype=np.float32)
        _num_feats(self.num1[0], self.num1[1], self.numx[0], self.numx[1], s1r, sxr, on)
        for c, f in enumerate(["num_primary_eq", "num_any_eq", "num_log_mindiff", "num_min_rel", "num_n1",
                               "num_nx", "num_x_unmatched", "num_x_primary_in_1"]):
            X[:, col[f]] = on[:, c]

        c1 = self.s1["name_core"].gather(s1r).to_list()
        cx = self.sx["name_core"].gather(sxr).to_list()
        X[:, col["tset"]] = _cp(fuzz.token_set_ratio, c1, cx)
        X[:, col["tsort"]] = _cp(fuzz.token_sort_ratio, c1, cx)
        X[:, col["partial"]] = _cp(fuzz.partial_ratio, c1, cx)
        X[:, col["ratio"]] = _cp(fuzz.ratio, c1, cx)
        X[:, col["nospace"]] = _cp(fuzz.ratio, [s.replace(" ", "") for s in c1], [s.replace(" ", "") for s in cx])
        X[:, col["jw"]] = process.cpdist(c1, cx, scorer=JaroWinkler.normalized_similarity, workers=-1,
                                         dtype=np.float32)
        alt = X[:, col["tset"]].copy()
        altx = self.sx["alt_core"].gather(sxr).to_list()
        for q in np.flatnonzero(np.fromiter((bool(a) for a in altx), dtype=bool, count=n)):
            alt[q] = max(fuzz.token_set_ratio(c1[q], part) for part in altx[q].split("|")) / 100.0
        X[:, col["alt_best"]] = alt
        X[:, col["legal_code"]] = self.legal_M[self.legal1[s1r], self.legalx[sxr]]
        a1 = self.s1["addr_norm"].gather(s1r).to_list()
        ax = self.sx["addr_norm"].gather(sxr).to_list()
        X[:, col["addr_tset"]] = _cp(fuzz.token_set_ratio, a1, ax)
        X[:, col["addr_ratio"]] = _cp(fuzz.ratio, a1, ax)
        X[:, col["was_indic"]:col["is_s3"] + 1] = self.flags_x[sxr]
        return X
```

- [ ] **Step 2: Add the stage**

```python
FEAT_CHUNK = 2_000_000


def _load_split_frames(split):
    cols = ["name_core", "legal", "alt_core", "was_indic", "is_domain", "addr_norm", "numbers", "addr_empty"]
    s1 = load_prep(split, "s1", cols + ["entity_id", "country"] + (["part"] if split == "train" else []))
    sx = load_prep(split, "sx", cols + ["entity_id", "src"])
    return s1, sx


def _pruned_candidates(split, floor):
    """Candidates with cos >= floor, plus per-S1 list features computed on the pruned lists."""
    from .features import add_list_features
    cand = pl.read_parquet(work("cand", f"{split}.parquet"))
    if floor is not None:
        cand = cand.filter(pl.col("cos") >= floor)
    return add_list_features(cand)


def _saved_floor():
    import json
    with open(work("models", "floor.json")) as f:
        return json.load(f)["floor"]


def stage_features(args):
    import json

    from .features import FEATURES, PairFeaturizer
    with open(work("models", "floor.json"), "w") as f:
        json.dump({"floor": args.floor}, f)
    s1, sx = _load_split_frames("train")
    cand = _pruned_candidates("train", args.floor)
    part = s1["part"].to_numpy()
    gt = pl.read_parquet(work("prep", "train_gt_rows.parquet")).with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    cand = cand.join(gt.rename({"s1_row": "s1", "sx_row": "sx"}), on=["s1", "sx"], how="left").with_columns(
        pl.col("label").fill_null(0))
    fz = PairFeaturizer(s1, sx)
    log("featurizer ready")
    for p in ("J", "V"):
        c = cand.filter(pl.Series(part[cand["s1"].to_numpy()] == p))
        if p == "J":  # early-stopping slice (S1 row % 10 == 0) goes last
            c = c.with_columns((pl.col("s1") % 10 == 0).alias("ev")).sort("ev", "s1", "rank").drop("ev")
        out = np.lib.format.open_memmap(work("feat", f"train_{p}.npy"), mode="w+", dtype=np.float32,
                                        shape=(c.height, len(FEATURES)))
        for b in range(0, c.height, FEAT_CHUNK):
            out[b:b + FEAT_CHUNK] = fz.compute(c.slice(b, FEAT_CHUNK))
            log(f"  {p} features {min(b + FEAT_CHUNK, c.height)}/{c.height}")
        out.flush()
        c.select("s1", "sx", "label").write_parquet(work("feat", f"train_{p}_pairs.parquet"))
        log(p, "pairs", c.height, "positives", int(c["label"].sum()))
```
Registry: add `"features": stage_features`.

- [ ] **Step 3: Smoke-test on a small slice**

Run:
```bash
python -c "from src.run_pipeline import _load_split_frames, _pruned_candidates; from src.features import PairFeaturizer, FEATURES; import numpy as np; s1,sx=_load_split_frames('train'); c=_pruned_candidates('train', None).head(2000); X=PairFeaturizer(s1,sx).compute(c); print(X.shape, np.isnan(X).sum()); print(dict(zip(FEATURES, X[0].round(3))))"
```
Expected: `(2000, 42) 0` and a readable feature dict for rank-0 pair (cos near 1, name_jacc high).

- [ ] **Step 4: Run the stage**

Run: `python -m src.run_pipeline features --split train --floor <floor from Task 6>`
Expected: J and V feature files; log lines with pair and positive counts (positives ≈ recall × true pairs of the part).

- [ ] **Step 5: Commit**

```bash
git add src/features.py src/run_pipeline.py
git commit -m "feat: pair features (embedding cosines, token sets, numbers, fuzzy scores)"
```

---

### Task 9: XGBoost judge v1 + validation (choose the decision rule)

**Files:**
- Create: `code/business_entity_resolution/src/judge.py`
- Modify: `src/run_pipeline.py` (add stages `train-judge`, `validate`)

**Interfaces:**
- Consumes: `WORK/feat/train_{J,V}.npy` + `_pairs.parquet`, `decide.*`, `evaluate.*`.
- Produces: `judge.train(X, y, n_eval) -> xgb.Booster` (last `n_eval` rows are the early-stopping slice); `judge.predict(booster, X, chunk=2_000_000) -> np.ndarray float32`; files `WORK/models/judge_v1.json`, `WORK/models/decision.json` (`{"rule": "expected_f"|"threshold", "threshold": float, "floor": float|null}`).

- [ ] **Step 1: Write `src/judge.py`**

```python
"""XGBoost (CUDA) pair classifier."""
import numpy as np
import xgboost as xgb

PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cuda",
          "max_depth": 9, "eta": 0.08, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
          "lambda": 1.0, "max_bin": 256}


def train(X, y, n_eval, rounds=3000):
    cut = len(y) - n_eval
    dtr = xgb.QuantileDMatrix(X[:cut], y[:cut])
    dev = xgb.QuantileDMatrix(X[cut:], y[cut:], ref=dtr)
    return xgb.train(PARAMS, dtr, rounds, evals=[(dev, "eval")], early_stopping_rounds=60, verbose_eval=100)


def predict(booster, X, chunk=2_000_000):
    out = np.empty(len(X), dtype=np.float32)
    for b in range(0, len(X), chunk):
        out[b:b + chunk] = booster.inplace_predict(np.asarray(X[b:b + chunk]),
                                                   iteration_range=(0, booster.best_iteration + 1))
    return out
```

- [ ] **Step 2: Add stages**

```python
def stage_train_judge(args):
    from . import judge
    X = np.load(work("feat", "train_J.npy"), mmap_mode="r")
    pairs = pl.read_parquet(work("feat", "train_J_pairs.parquet"))
    y = pairs["label"].to_numpy().astype(np.float32)
    n_eval = int((pairs["s1"] % 10 == 0).sum())
    booster = judge.train(np.asarray(X), y, n_eval)
    booster.save_model(work("models", "judge_v1.json"))
    log("judge best iteration", booster.best_iteration)


def _truth_by_s1(s1_rows, s1_ids, sx_ids):
    gt = pl.read_parquet(work("prep", "train_gt_rows.parquet")).filter(
        pl.col("s1_row").is_in(pl.Series(s1_rows).implode()))
    truth = {s1_ids[r]: set() for r in s1_rows}
    for a, b in zip(gt["s1_row"].to_list(), gt["sx_row"].to_list()):
        truth[s1_ids[a]].add(sx_ids[b])
    return truth


def _score(s1r, sxr, mask, s1_rows, s1_ids, sx_ids, truth):
    pred = {s1_ids[r]: set() for r in s1_rows}
    for a, b in zip(s1r[mask].tolist(), sxr[mask].tolist()):
        pred[s1_ids[a]].add(sx_ids[b])
    from .evaluate import macro_f05
    return macro_f05(pred, truth, [s1_ids[r] for r in s1_rows])


def stage_validate(args):
    import json

    import xgboost as xgb

    from . import judge
    from .decide import assign_best, expected_f05_cut, threshold_cut
    booster = xgb.Booster(model_file=work("models", "judge_v1.json"))
    s1 = load_prep("train", "s1", ["entity_id", "country", "part"])
    sx_ids = load_prep("train", "sx", ["entity_id"])["entity_id"].to_numpy()
    s1_ids = s1["entity_id"].to_numpy()
    pairs = pl.read_parquet(work("feat", "train_V_pairs.parquet"))
    p = judge.predict(booster, np.load(work("feat", "train_V.npy"), mmap_mode="r"))
    s1r, sxr = pairs["s1"].to_numpy(), pairs["sx"].to_numpy()
    rng = np.random.default_rng(0)
    v_rows = np.flatnonzero((s1["part"] == "V").to_numpy())
    views = {"V": v_rows, "V-stress": v_rows[rng.random(len(v_rows)) >= 0.19]}
    results = {}
    for view, rows in views.items():
        truth = _truth_by_s1(rows, s1_ids, sx_ids)
        inview = np.isin(s1r, rows)
        a, b, pp = s1r[inview], sxr[inview], p[inview]
        keep = assign_best(b, pp)
        a, b, pp = a[keep], b[keep], pp[keep]
        res = {f"thr{t:.2f}": _score(a, b, threshold_cut(pp, t), rows, s1_ids, sx_ids, truth)
               for t in np.arange(0.2, 0.91, 0.05)}
        res["expected_f"] = _score(a, b, expected_f05_cut(a, pp), rows, s1_ids, sx_ids, truth)
        for c in sorted(set(s1["country"].to_numpy()[rows].tolist())):
            crow = rows[s1["country"].to_numpy()[rows] == c]
            cset = set(s1_ids[crow].tolist())
            ct = {k: v for k, v in truth.items() if k in cset}
            m = np.isin(a, crow)
            res[f"expected_f[{c}]"] = _score(a[m], b[m], expected_f05_cut(a[m], pp[m]), crow, s1_ids, sx_ids, ct)
        results[view] = res
        best = max((k for k in res if k.startswith("thr")), key=res.get)
        log(view, "best threshold", best, round(res[best], 5), "| expected_f", round(res["expected_f"], 5))
        log(view, {k: round(v, 4) for k, v in res.items()})
    thr_key = max((k for k in results["V"] if k.startswith("thr")),
                  key=lambda k: results["V"][k] + results["V-stress"][k])
    use_ef = (results["V"]["expected_f"] + results["V-stress"]["expected_f"] >=
              results["V"][thr_key] + results["V-stress"][thr_key])
    decision = {"rule": "expected_f" if use_ef else "threshold", "threshold": float(thr_key[3:]),
                "floor": _saved_floor()}
    with open(work("models", "decision.json"), "w") as f:
        json.dump(decision, f)
    log("decision", decision)
```
Registry: add `"train-judge": stage_train_judge, "validate": stage_validate`.

- [ ] **Step 3: Train the judge**

Run: `python -m src.run_pipeline train-judge`
Expected: eval logloss printed every 100 rounds and falling; `judge best iteration` between 200 and 3000; ≲ 10 min on GPU.

- [ ] **Step 4: Validate**

Run: `python -m src.run_pipeline validate` (the floor is read from `WORK/models/floor.json`, written by the features stage)
Expected: two views logged with macro F0.5 per threshold, expected_f and per-country expected_f; `decision {...}` written. Record the V and V-stress scores in the commit message.

- [ ] **Step 5: Commit**

```bash
git add src/judge.py src/run_pipeline.py
git commit -m "feat: XGBoost judge v1 and validation (V=<score>, V-stress=<score>)"
```

---

### Task 10: Test inference → submission files → validator → Submission #1

**Files:**
- Modify: `src/run_pipeline.py` (add stage `predict-test`)

**Interfaces:**
- Consumes: test prep/cand, `judge_v1.json`, `decision.json`, `PairFeaturizer`, `add_list_features` (via `_pruned_candidates`), `decide.*`, `io_utils.write_submission`.
- Produces: `OUT_DIR/matching_results.tsv`, `OUT_DIR/candidate_pairs.tsv`, `WORK/pred/test_p.npy` (probabilities aligned with the pruned candidate frame, for later re-thresholding without recomputing features).

- [ ] **Step 1: Add the stage**

```python
def _decide(s1r, sxr, p, decision):
    from .decide import assign_best, expected_f05_cut, threshold_cut
    keep = assign_best(sxr, p)
    mask = np.zeros(len(p), dtype=bool)
    idx = np.flatnonzero(keep)
    sub = (expected_f05_cut(s1r[idx], p[idx]) if decision["rule"] == "expected_f"
           else threshold_cut(p[idx], decision["threshold"]))
    mask[idx[sub]] = True
    return mask


def stage_predict_test(args):
    import json
    import subprocess
    import sys

    import xgboost as xgb

    from . import judge
    from .features import PairFeaturizer
    from .io_utils import write_submission
    with open(work("models", "decision.json")) as f:
        decision = json.load(f)
    s1, sx = _load_split_frames("test")
    cand = _pruned_candidates("test", decision["floor"])
    booster = xgb.Booster(model_file=work("models", "judge_v1.json"))
    fz = PairFeaturizer(s1, sx)
    p = np.empty(cand.height, dtype=np.float32)
    for b in range(0, cand.height, FEAT_CHUNK):
        p[b:b + FEAT_CHUNK] = judge.predict(booster, fz.compute(cand.slice(b, FEAT_CHUNK)))
        log(f"  test scored {min(b + FEAT_CHUNK, cand.height)}/{cand.height}")
    np.save(work("pred", "test_p.npy"), p)
    s1r, sxr = cand["s1"].to_numpy(), cand["sx"].to_numpy()
    mask = _decide(s1r, sxr, p, decision)
    write_submission(config.OUT_DIR, s1["entity_id"].to_list(), sx["entity_id"].to_numpy(), s1r, sxr, mask)
    log("matched pairs", int(mask.sum()), "S1 with >=1 match", len(np.unique(s1r[mask])), "of", s1.height)
    validator = os.path.join(config.ROOT, "student_resource", "utils", "validate_submission.py")
    subprocess.run([sys.executable, validator, "--matching", os.path.join(config.OUT_DIR, "matching_results.tsv"),
                    "--candidate", os.path.join(config.OUT_DIR, "candidate_pairs.tsv"),
                    "--test-dir", os.path.join(config.DATA_DIR, "test")], check=True)
```
Registry: add `"predict-test": stage_predict_test`.

- [ ] **Step 2: Run**

Run: `python -m src.run_pipeline predict-test`
Expected: scoring progress logs, matched-pair summary (share of S1 with ≥1 match should be roughly 0.85–0.95), then the validator prints `PASS — no blocking issues found. Safe to submit.` with 1,732,544 rows in both files.

- [ ] **Step 3: Sanity-check a few predictions by eye**

Run:
```bash
python -c "import polars as pl; m=pl.read_csv(r'..\..\output\matching_results.tsv', separator='\t', quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True); t=pl.read_parquet(r'C:\amazon_ml\work\prep\test_s1.parquet', columns=['entity_id','country','name_norm','addr_norm']); print(m.join(t, left_on='source1_entity_id', right_on='entity_id').group_by('country').agg((pl.col('matched_entity_ids')!='').mean().alias('share_with_match'), pl.len()))"
```
Expected: every country (US, India, France) present with a plausible share (not 0, not 1).

- [ ] **Step 4: Commit and tag**

```bash
git add src/run_pipeline.py
git commit -m "feat: test inference stage writing validated submission files"
git tag sub-1
```
Hand `output/matching_results.tsv` to the user for upload; record the public LB score in the tag message later.

---

### Task 11: Submission #2 (threshold probe, no recomputation)

**Files:**
- Modify: `src/run_pipeline.py` (add stage `rethreshold`)

**Interfaces:**
- Consumes: `WORK/pred/test_p.npy`, pruned test candidates, decision floor.
- Produces: new `OUT_DIR/matching_results.tsv` using `--rule threshold --thr <t>`; candidate file unchanged.

- [ ] **Step 1: Add the stage**

```python
def stage_rethreshold(args):
    import json
    with open(work("models", "decision.json")) as f:
        decision = json.load(f)
    decision.update({"rule": args.rule, "threshold": args.thr if args.thr is not None else decision["threshold"]})
    s1 = load_prep("test", "s1", ["entity_id"])
    sx_ids = load_prep("test", "sx", ["entity_id"])["entity_id"].to_numpy()
    cand = _pruned_candidates("test", decision["floor"])
    p = np.load(work("pred", "test_p.npy"))
    s1r, sxr = cand["s1"].to_numpy(), cand["sx"].to_numpy()
    mask = _decide(s1r, sxr, p, decision)
    from .io_utils import write_submission
    write_submission(config.OUT_DIR, s1["entity_id"].to_list(), sx_ids, s1r, sxr, mask)
    log("rethreshold", decision, "matched pairs", int(mask.sum()))
```
In `main()` add `ap.add_argument("--rule", default="threshold", choices=["threshold", "expected_f"])` and `ap.add_argument("--thr", type=float, default=None)`; registry: add `"rethreshold": stage_rethreshold`.

- [ ] **Step 2: Produce the variant**

If submission #1 used `expected_f`, run `python -m src.run_pipeline rethreshold --rule threshold` (best V threshold); otherwise run `python -m src.run_pipeline rethreshold --rule expected_f`. Then run the validator command from Task 10 Step 1 by hand.
Expected: validator `PASS`.

- [ ] **Step 3: Commit and tag**

```bash
git add src/run_pipeline.py
git commit -m "feat: re-threshold stage for leaderboard probes"
git tag sub-2
```

---

## Self-review notes

- Spec coverage (Phase 1): normalize ✓ (T2), featurize ✓ (T4), encoder v1 ✓ (T5), candidates + recall gate ✓ (T6), features ✓ (T8), judge v1 ✓ (T9), decide (assignment, expected-F, threshold) ✓ (T7/T9), evaluate + stress view ✓ (T7/T9), run_pipeline + validator ✓ (T10), submission variants ✓ (T11), splits E/J/V ✓ (T3). Deferred to the Phase 2 plan by design: safety-net exact keys (only if T6 recall gate fails), encoder v2, cross-encoder v2, stage-2 group features, France offsets, README/documentation/package.
- Names used across tasks: `load_prep`, `work`, `hash_features`, `FingerprintModel`, `encode_to`, `build_candidates`, `recall_report`, `PairFeaturizer.compute`, `FEATURES`, `assign_best`, `expected_f05_cut`, `threshold_cut`, `judge.train/predict`, `_decide`, `write_submission`, `add_list_features`, `_pruned_candidates`, `_saved_floor` — consistent.
