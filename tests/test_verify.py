from civicite.verify.citations import normalize_citation_placement, verify

SRC = {1: "You must report the move to Skatteverket within one week after you have moved.",
       2: "Parents can receive parental benefit for a total of 480 days per child."}


def test_supported_and_moved_citations():
    v = verify("You must report the move within one week. [1]", SRC)
    assert v.label == "verified" and v.faithfulness == 1.0
    assert normalize_citation_placement("rule. [1] Next") == "rule [1]. Next"


def test_wrong_number_is_unsupported():
    v = verify("Parents can receive parental benefit for 400 days per child [2].", SRC)
    assert v.label == "unverified"
    assert "numbers not in source" in v.sentences[0].reason


def test_uncited_and_invalid_citations():
    v = verify("Parents get parental benefit for 480 days per child. You must report a move quickly [7].", SRC)
    assert v.faithfulness == 0.0
    assert v.invalid_citations == [7]
    assert v.citation_coverage == 0.5


def test_calc_sentence_checked_against_tool_result():
    tool = [{"start": "2026-03-10", "deadline": "2026-03-17", "weekday": "Tuesday"}]
    ok = verify("Counting from Tuesday 10 March 2026, the deadline is Tuesday 17 March 2026 [calc].", SRC, tool)
    bad = verify("Counting from Tuesday 10 March 2026, the deadline is Friday 20 March 2026 [calc].", SRC, tool)
    assert ok.label == "verified" and bad.label == "unverified"


def test_non_claims_are_ignored():
    v = verify("I could not find this in the documents I have.", SRC)
    assert v.label == "no-claims"
