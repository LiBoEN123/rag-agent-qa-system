"""Generate the tiny deterministic text PDF used by Web PDF tests."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "tests" / "fixtures" / "web_pdf_text.pdf"


def _stream(data: bytes) -> bytes:
    """Wrap deterministic PDF drawing commands in a stream object."""
    return (
        f"<< /Length {len(data)} >>\nstream\n".encode()
        + data
        + b"\nendstream"
    )


def _pdf_literal(value: str) -> bytes:
    """Encode the fixture's intentionally ASCII-only PDF literal strings."""
    encoded = value.encode("ascii")
    return (
        encoded.replace(b"\\", b"\\\\")
        .replace(b"(", b"\\(")
        .replace(b")", b"\\)")
    )


def build_text_pdf(
    page_texts: tuple[str, ...],
    *,
    title: str = "",
    author: str = "",
    subject: str = "",
) -> bytes:
    """Build a deterministic Helvetica PDF for local tests."""
    if not page_texts:
        raise ValueError("page_texts must contain at least one page")
    first_page_id = 4
    page_ids = [first_page_id + index * 2 for index in range(len(page_texts))]
    info_id = first_page_id + len(page_texts) * 2
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            f"<< /Type /Pages /Count {len(page_ids)} /Kids ["
            + " ".join(f"{page_id} 0 R" for page_id in page_ids)
            + "] >>"
        ).encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for index, text in enumerate(page_texts):
        page_id = page_ids[index]
        content_id = page_id + 1
        objects.extend(
            [
                (
                    b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    b"/Resources << /Font << /F1 3 0 R >> >> /Contents "
                    + f"{content_id} 0 R >>".encode()
                ),
                _stream(
                    b"BT /F1 18 Tf 72 720 Td ("
                    + _pdf_literal(text)
                    + b") Tj ET"
                ),
            ]
        )
    objects.append(
        b"<< /Title ("
        + _pdf_literal(title)
        + b") /Author ("
        + _pdf_literal(author)
        + b") /Subject ("
        + _pdf_literal(subject)
        + b") >>"
    )

    result = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, value in enumerate(objects, start=1):
        offsets.append(len(result))
        result.extend(f"{object_number} 0 obj\n".encode())
        result.extend(value)
        result.extend(b"\nendobj\n")

    xref_offset = len(result)
    result.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    result.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode())
    result.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R "
            f"/Info {info_id} 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return bytes(result)


def build_fixture_pdf() -> bytes:
    """Build the committed two-page Web PDF fixture."""
    return build_text_pdf(
        ("Governed PDF page one", "Governed PDF page two"),
        title="Governed Web PDF Fixture",
        author="Codex Tests",
        subject="Public PDF extraction",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(build_fixture_pdf())
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
