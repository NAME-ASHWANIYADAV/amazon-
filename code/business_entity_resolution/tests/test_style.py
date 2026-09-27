import numpy as np

from src.style import address_style, common_tokens, name_category, street_mismatch, style_additions


def test_address_style_codes():
    assert address_style("N° 112 R. Aristide Briand, Nantes, Loire-Atlantique", "France") == (2 * 4, 2)
    assert address_style("0029 rue Marguerite, Nantes, Pays de la Loire", "France") == (0 * 4 + 2, 1)
    assert address_style("36 - rue Goya, Bordeaux", "France") == (0 * 4 + 1, 0)
    assert address_style("", "France") == (-1, -1)
    assert address_style("12 Main St, Tyler, TX", "US") == (0, 1)          # 'tx' is a canonical state code
    assert address_style("12 Main St, Tyler, Texas", "US") == (0, 2)       # full name maps to the code


def test_name_category():
    assert name_category("amicale du team", "amicale du team") == "EQ"
    assert name_category("amicale du team", "team du amicale") == "PERM"
    assert name_category("amicale du team", "comite du team") == "WSUB1"
    assert name_category("amicale du team", "amicalle du team") == "TYPO1"
    assert name_category("comite du team", "cdt") == "ACRO"
    assert name_category("amicale du team", "amicaleduteam") == "NOSP"
    assert name_category("amicale du team", "amicale du team sportive") == "ADD1"
    assert name_category("amicale du team", "zorbix") == "DISJ"


def test_street_mismatch():
    common = common_tokens(["1 rue de la paix nantes"] * 400 + ["2 avenue victor hugo nantes"] * 400)
    assert street_mismatch("12 rue chevire nantes", "12 rue chevirre nantes", common) == 0   # typo matches
    assert street_mismatch("12 rue chevire nantes", "12 rue goya nantes", common) == 1
    assert street_mismatch("12 rue de la paix", "12 rue de la paix", common) == -1           # only common tokens


def test_style_additions_needs_agreeing_anchor():
    # S1 0: anchors sx 10 (S2, N° format, region) and sx 11 (S3); candidates sx 12 (S2, matching N°) and 13 (S2, '#')
    s1r = np.array([0, 0, 0, 0])
    sxr = np.array([10, 11, 12, 13])
    p = np.array([0.999, 0.999, 0.3, 0.3], dtype=np.float32)
    keep = np.ones(4, bool)
    pred = np.array([True, True, False, False])
    cand = np.array([False, False, True, True])
    nf = np.array([8, 8, 8, 12])            # 8 = 'n° ' prefix, 12 = '#'
    reg = np.array([1, 1, 1, 1])
    src = np.array([2, 3, 2, 2])
    ncat = np.array(["EQ", "EQ", "TYPO1", "TYPO1"], dtype=object)
    smis = np.zeros(4, np.int8)
    got = style_additions(s1r, sxr, p, keep, pred, cand, nf, reg, src, ncat, smis)
    assert got.tolist() == [False, False, True, False]
