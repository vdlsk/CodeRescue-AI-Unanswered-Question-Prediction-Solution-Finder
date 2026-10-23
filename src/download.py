"""Download the Chicago Food Inspections dataset.

Regulatory documents must be saved manually into data/regulations/ (PDF, HTML, TXT or MD):
  1. Chicago Municipal Code, Food Establishment chapters (Title 7, Ch. 7-38 / 7-40 / 7-42) from
     https://codelibrary.amlegal.com/codes/chicago/latest/chicago_il/0-0-0-2642808
     (open each chapter, File > Save Page As > HTML, or print to PDF).
  2. Chicago Food Code Rules (CDPH), PDF from chicago.gov (search "Chicago Food Code Rules").
  3. FDA Food Code (the edition Chicago adopted, with Chicago amendments) from fda.gov, PDF.
These sites block scripted downloads or change URLs, which is why this step is manual.

Usage:  python src/download.py
"""
import requests

from config import CSV_NAME, INSPECTIONS_URL, RAW_DIR, REG_DIR


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REG_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / CSV_NAME
    if dest.exists():
        print(f"exists {dest}")
    else:
        print(f"downloading {INSPECTIONS_URL}")
        with requests.get(INSPECTIONS_URL, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    regs = [p.name for p in REG_DIR.iterdir() if p.suffix.lower() in (".pdf", ".html", ".htm", ".txt", ".md")]
    print(f"regulation files found: {regs or 'NONE - see the docstring of this file'}")


if __name__ == "__main__":
    main()
