from civicite.index.search import confidence
from civicite.ingest.loader import Document, chunk_document, html_to_markdown, parse_front_matter


def test_front_matter_and_chunking():
    meta, body = parse_front_matter("---\ntitle: T\nagency: A\n---\n# T\n\n## S1\npara one\n\n## S2\npara two\n")
    assert meta == {"title": "T", "agency": "A"}
    chunks = chunk_document(Document("d", "T", body, "A"))
    assert [c.section for c in chunks] == ["S1", "S2"]
    assert chunks[0].chunk_id == "d#0"


def test_html_to_markdown_drops_navigation():
    title, md = html_to_markdown("<html><head><title>Flytt</title><script>x()</script></head><body><nav>menu</nav>"
                                 "<h1>Anmäl flytt</h1><p>Inom en vecka.</p><footer>f</footer></body></html>")
    assert title == "Flytt" and "# Anmäl flytt" in md and "Inom en vecka." in md
    assert "menu" not in md and "x()" not in md


def test_top_hits(index):
    cases = {
        "How long do I have to report a move?": "skatteverket-moving",
        "How many VAB days per child and year?": "forsakringskassan-temporary-parental-benefit",
        "Hur många dagar föräldrapenning får man?": "forsakringskassan-parental-benefit",
        "Who issues BankID?": "e-identification",
    }
    for q, doc in cases.items():
        assert index.search(q, 3)[0].chunk.doc_id == doc, q


def test_out_of_domain_has_low_confidence(index):
    assert confidence(index.search("What is the capital of Norway?", 5)) < 0.2
    assert confidence(index.search("How do I report a move to Skatteverket?", 5)) > 0.5
