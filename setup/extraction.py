import os
import re
import pdfplumber

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "..", "PDF")
TEXT_DIR = os.path.join(BASE_DIR, "..", "TEXT")

os.makedirs(TEXT_DIR, exist_ok=True)

BOOKS = [
    ("Boericke_materia_medica.pdf",                                        "Boericke_clean.txt"),
    ("keynotes-and-characteristics-allen.pdf",                             "allen_clean.txt"),
    ("Substance of homeopathy.pdf",                                        "Sankaran_clean.txt"),
    ("Essence of Materia Medica by George Vithoulkas.pdf",                 "vithoulkas_clean.txt"),
    ("lectures-on-homeopathic-materia-medicaby-james-tyler-kent.pdf",      "kent_clean.txt"),
]


def clean_page(text):
    if not text:
        return ""
    text = re.sub(r'\(cid:\d+\)', '', text)           # remove PDF encoding artifacts
    text = re.sub(r'[ \t]+', ' ', text)               # collapse spaces/tabs (not newlines)
    lines = [l.strip() for l in text.split('\n')]
    return '\n'.join(lines)


def is_page_header(line):
    """Skip repeated running headers / page numbers."""
    stripped = line.strip()
    if re.match(r'^\d+$', stripped):                  # lone page number
        return True
    if len(stripped) < 5:
        return True
    return False


def extract_pdf(pdf_path):
    pages_text = []

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"[PDF] {os.path.basename(pdf_path)} -> {total} pages")

        for i, page in enumerate(pdf.pages):
            t = page.extract_text()
            if t:
                pages_text.append(clean_page(t))

            if i % 100 == 0:
                print(f"  page {i}/{total}")

    # join pages with double newline to mark page boundary
    full = '\n\n'.join(pages_text)

    # collapse 3+ blank lines into 2
    full = re.sub(r'\n{3,}', '\n\n', full)

    return full.strip()


def run():
    for pdf_file, txt_file in BOOKS:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        txt_path = os.path.join(TEXT_DIR, txt_file)

        if not os.path.exists(pdf_path):
            print(f"[SKIP] Not found: {pdf_file}")
            continue

        print(f"\n--- Extracting: {pdf_file} ---")
        text = extract_pdf(pdf_path)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        size_kb = len(text) // 1024
        lines = text.count('\n')
        print(f"[OK] {txt_file} | {size_kb} KB | {lines} lines")


if __name__ == "__main__":
    run()
