"""Keep browser session form values out of committed Scholar evidence."""

import re
from pathlib import Path

CAPTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "captures"
INPUT_TAG = re.compile(r"<input\b[^>]*>", re.IGNORECASE | re.DOTALL)
XSRF_NAME = re.compile(r"\bname\s*=\s*(['\"])xsrf\1", re.IGNORECASE)
VALUE = re.compile(r"\bvalue\s*=\s*(['\"])(.*?)\1", re.IGNORECASE | re.DOTALL)


def test_committed_scholar_html_redacts_xsrf_form_values() -> None:
    for path in CAPTURE_DIR.glob("*.html"):
        for tag in INPUT_TAG.findall(path.read_text(encoding="utf-8")):
            if not XSRF_NAME.search(tag):
                continue
            value = VALUE.search(tag)
            assert value is not None and value.group(2) == "[REDACTED]", path
