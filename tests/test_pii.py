from civicite.guard.pii import classify_identity_number, luhn_ok, redact


def test_luhn():
    assert luhn_ok("8112189876")
    assert not luhn_ok("8112189877")


def test_identity_numbers():
    assert classify_identity_number("811218-9876") == "personnummer"
    assert classify_identity_number("19811218-9876") == "personnummer"
    assert classify_identity_number("8112189877") is None          # bad check digit
    assert classify_identity_number("811278-9873") == "samordningsnummer"  # day 18 + 60
    assert classify_identity_number("811318-9876") is None          # month 13


def test_redact_mixed_text():
    r = redact("I am 811218-9876, mail me at anna.svensson@example.se or call 070-123 45 67. "
               "IBAN SE45 5000 0000 0583 9825 7466, card 4111 1111 1111 1111. Ref 2026-03-10.")
    assert "811218-9876" not in r.text and "[PERSONNUMMER]" in r.text
    assert "[EMAIL]" in r.text and "[PHONE]" in r.text and "[IBAN]" in r.text and "[CARD]" in r.text
    assert "2026-03-10" in r.text  # dates are not personal data
    assert r.found == {"personnummer": 1, "iban": 1, "card": 1, "email": 1, "phone": 1}


def test_no_false_positive_on_plain_numbers():
    r = redact("Parents get 480 days; apply within 90 days; the year is 2026.")
    assert not r.changed
