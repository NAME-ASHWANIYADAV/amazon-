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
            if ch in "‌‍":
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
    # lowercase before folding: _LATIN_EXTRA keys are lowercase (Œ -> œ -> oe)
    s = _fold(transliterate(unicodedata.normalize("NFC", s)).lower())
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
# whole address component -> canonical state/region token; canonical values map to themselves
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
    """Worker entry point: [(name, addr, country)] -> list of 9-tuples
    (name_norm, name_core, legal, alt_core, was_indic, is_domain, addr_norm, numbers, addr_empty)."""
    return [norm_name(name) + norm_addr(addr, country) for name, addr, country in rows]
