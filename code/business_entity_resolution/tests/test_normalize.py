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


def test_name_uppercase_ligature_folded():
    assert norm_name("CŒUR DE FRANCE")[0] == norm_name("Cœur de France")[0] == "coeur de france"


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
