"""Contracts for the deterministic PDF test helper."""

from tests.support.pdf_fixture import build_fixture_pdf, build_text_pdf


def test_build_text_pdf_has_valid_header_pages_and_metadata():
    value = build_text_pdf(
        ("First page", "Second page"),
        title="Fixture Title",
        author="Fixture Author",
        subject="Fixture Subject",
    )

    assert value.startswith(b"%PDF-1.4\n")
    assert value.endswith(b"%%EOF\n")
    assert b"/Count 2" in value
    assert b"First page" in value
    assert b"Second page" in value
    assert b"/Title (Fixture Title)" in value
    assert b"/Author (Fixture Author)" in value
    assert b"/Subject (Fixture Subject)" in value


def test_pdf_helper_is_deterministic():
    assert build_fixture_pdf() == build_fixture_pdf()

