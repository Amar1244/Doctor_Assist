"""
Structured chunker for homeopathic materia medica books.
Produces JSON chunks ready for ChromaDB ingestion.
"""

import re
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_DIR = os.path.join(BASE_DIR, "..", "TEXT")
OUT_FILE = os.path.join(BASE_DIR, "..", "chunks.json")

MAX_WORDS = 400
OVERLAP_WORDS = 40


# ─────────────────────────────────────
# UTILS
# ─────────────────────────────────────

def word_count(text):
    return len(text.split())


def make_chunk(remedy, source, chunk_type, content):
    content = re.sub(r'\s+', ' ', content).strip()
    return {"remedy": remedy, "source": source, "type": chunk_type, "content": content}


def split_long(content, remedy, source, chunk_type):
    """Split text > MAX_WORDS into overlapping chunks."""
    words = content.split()
    if len(words) <= MAX_WORDS:
        if len(content.strip()) > 80:
            return [make_chunk(remedy, source, chunk_type, content)]
        return []

    chunks = []
    i = 0
    while i < len(words):
        piece = ' '.join(words[i: i + MAX_WORDS])
        if len(piece.strip()) > 80:
            chunks.append(make_chunk(remedy, source, chunk_type, piece))
        i += MAX_WORDS - OVERLAP_WORDS
    return chunks


# ─────────────────────────────────────
# BOERICKE
# Remedies: ALL CAPS line (e.g., "ACONITUM NAPELLUS")
# Sections: "Head.--", "Stomach.--", "Mind.--" etc.
# ─────────────────────────────────────

BOERICKE_SECTION_MAP = {
    "Mind":         "mind",
    "Head":         "head",
    "Eyes":         "physical",
    "Eye":          "physical",
    "Nose":         "physical",
    "Face":         "physical",
    "Mouth":        "physical",
    "Throat":       "physical",
    "Stomach":      "stomach",
    "Abdomen":      "physical",
    "Stool":        "physical",
    "Urinary":      "physical",
    "Male":         "physical",
    "Female":       "female",
    "Respiratory":  "physical",
    "Chest":        "physical",
    "Heart":        "physical",
    "Back":         "physical",
    "Extremities":  "physical",
    "Skin":         "physical",
    "Sleep":        "physical",
    "Fever":        "fever",
    "Modalities":   "modalities",
    "Generals":     "generals",
    "General":      "generals",
}

BOERICKE_SKIP = {
    "PREFACE", "INTRODUCTION", "INDEX", "PART", "CONTENTS", "APPENDIX",
    "MATERIA MEDICA", "TABLE OF CONTENTS", "THERAPEUTIC INDEX",
    "HOMOEOPATHIC MATERIA MEDICA", "HOMEOPATHIC MATERIA MEDICA",
}

BOERICKE_COMMON = {'TO', 'THE', 'OF', 'AND', 'IN', 'FOR', 'BY', 'WITH',
                   'AS', 'FROM', 'AT', 'IS', 'ARE', 'BE', 'AN', 'OR'}


def _boericke_section_type(line):
    for keyword, section_type in BOERICKE_SECTION_MAP.items():
        if re.match(rf'^{keyword}[\.\-—–]', line):
            return section_type
    return None


def chunk_boericke(text):
    lines = text.split('\n')
    chunks = []
    current_remedy = None
    current_type = "general"
    current_lines = []

    def flush():
        if current_remedy and current_lines:
            content = ' '.join(current_lines)
            chunks.extend(split_long(content, current_remedy, "Boericke", current_type))

    for line in lines:
        s = line.strip()
        if not s:
            continue

        s_words = set(s.split())
        if (re.match(r'^[A-Z][A-Z\s\-]+$', s) and
                3 < len(s) < 55 and
                s not in BOERICKE_SKIP and
                not re.search(r'\d', s) and
                not (s_words & BOERICKE_COMMON)):
            flush()
            current_remedy = s.title()
            current_type = "general"
            current_lines = []
            continue

        sec_type = _boericke_section_type(s)
        if sec_type and current_remedy:
            flush()
            current_type = sec_type
            content_start = re.sub(r'^[A-Za-z]+[\.\-—–]+\s*', '', s)
            current_lines = [content_start] if content_start else []
            continue

        if current_remedy:
            current_lines.append(s)

    flush()
    return chunks


# ─────────────────────────────────────
# ALLEN
# ─────────────────────────────────────

def chunk_allen(text):
    lines = text.split('\n')
    chunks = []
    current_remedy = None
    current_lines = []

    SKIP_STARTS = ('The ', 'This ', 'In ', 'It ', 'A ', 'An ')

    def flush():
        if current_remedy and current_lines:
            content = ' '.join(current_lines)
            chunks.extend(split_long(content, current_remedy, "Allen", "keynotes"))

    for line in lines:
        s = line.strip()
        if not s:
            continue

        if (re.match(r'^[A-Z][a-z]+(\s+[A-Za-z]+)*\.$', s) and
                len(s) < 55 and
                not any(s.startswith(x) for x in SKIP_STARTS) and
                not re.search(r'[,;:]', s)):
            flush()
            current_remedy = s.rstrip('.')
            current_lines = []
            continue

        if current_remedy:
            current_lines.append(s)

    flush()
    return chunks


# ─────────────────────────────────────
# KENT
# ─────────────────────────────────────

KENT_SKIP_LINES = {
    "public domain text converted into pdf format by nalanda",
    "nalanda digital library-regional engineering college,calicut,india",
}

KENT_MIND_SECS = {"mind", "mental", "emotion", "delusion", "fear", "anxiety"}

KENT_NOT_REMEDY = {
    "preface", "introduction", "contents", "index", "appendix",
    "lectures", "materia medica", "philosophy",
}

KENT_COMMON = {
    'the', 'of', 'to', 'a', 'an', 'all', 'not', 'they', 'there', 'this',
    'that', 'is', 'are', 'be', 'been', 'has', 'have', 'had', 'in', 'on',
    'at', 'by', 'for', 'with', 'as', 'if', 'or', 'and', 'but', 'so',
    'from', 'into', 'it', 'its', 'we', 'our', 'you', 'he', 'she',
    'who', 'what', 'when', 'where', 'how', 'which', 'than', 'then',
    'do', 'does', 'did', 'will', 'would', 'can', 'could', 'may',
    'might', 'should', 'shall', 'must', 'up', 'down', 'out', 'some',
    'such', 'more', 'most', 'many', 'much', 'no', 'any', 'each',
    'other', 'these', 'those', 'among', 'often', 'only', 'very',
    'just', 'also', 'while', 'after', 'before', 'during', 'always',
    'never', 'every', 'both', 'because', 'therefore', 'however',
}


def chunk_kent(text):
    lines = text.split('\n')
    n = len(lines)
    chunks = []
    current_remedy = None
    current_type = "general"
    current_lines = []

    def flush():
        if current_remedy and current_lines:
            content = ' '.join(current_lines)
            chunks.extend(split_long(content, current_remedy, "Kent", current_type))

    def is_blank(idx):
        return idx >= n or not lines[idx].strip()

    def clean(s):
        s = s.strip()
        if s.lower() in KENT_SKIP_LINES:
            return ""
        if re.match(r'^(Public Domain|Nalanda)', s):
            return ""
        if re.match(r'^\d+$', s):
            return ""
        return s

    i = 0
    while i < n:
        s = clean(lines[i])

        if not s:
            i += 1
            continue

        prev_is_blank = is_blank(i - 1)
        next_is_blank = is_blank(i + 1)

        s_words_lower = set(s.lower().split())
        is_remedy_candidate = (
            prev_is_blank and
            next_is_blank and
            re.match(r'^[A-Z][a-z]+(\s+[a-zA-Z]+)*$', s) and
            len(s) < 40 and
            s.lower() not in KENT_NOT_REMEDY and
            not (s_words_lower & KENT_COMMON) and
            not s.endswith('ing') and
            not s.endswith('ed') and
            not s.endswith('ly')
        )

        if is_remedy_candidate:
            flush()
            current_remedy = s
            current_type = "general"
            current_lines = []
            i += 1
            continue

        sec_match = re.match(r'^([A-Z][a-z]+(?:\s+[a-z]+)?):\s*(.*)', s)
        if sec_match and current_remedy:
            sec_name = sec_match.group(1).lower()
            rest = sec_match.group(2).strip()
            flush()
            if any(k in sec_name for k in KENT_MIND_SECS):
                current_type = "mind"
            elif "general" in sec_name:
                current_type = "generals"
            elif "introduction" in sec_name:
                current_type = "introduction"
            else:
                current_type = "physical"
            current_lines = [rest] if rest else []
            i += 1
            continue

        if current_remedy:
            current_lines.append(s)

        i += 1

    flush()
    return chunks


# ─────────────────────────────────────
# VITHOULKAS
# ─────────────────────────────────────

def chunk_vithoulkas(text):
    lines = text.split('\n')
    chunks = []
    current_remedy = None
    current_lines = []

    abbrev_indices = set()
    for i, line in enumerate(lines):
        s = line.strip()
        if re.match(r'^\([a-z]{2,6}\.?\)$', s):
            abbrev_indices.add(i)
            for j in range(i - 1, max(i - 4, -1), -1):
                prev = lines[j].strip()
                if prev:
                    abbrev_indices.add(j)
                    break

    def flush():
        if current_remedy and current_lines:
            content = ' '.join(current_lines)
            chunks.extend(split_long(content, current_remedy, "Vithoulkas", "essence"))

    remedy_names = {}
    for i, line in enumerate(lines):
        s = line.strip()
        if i + 1 < len(lines):
            next_s = lines[i + 1].strip()
            if not next_s and i + 2 < len(lines):
                next_s = lines[i + 2].strip()
        else:
            next_s = ""

        if (re.match(r'^[A-Z][a-z]+(\s+[a-zA-Z]+)*$', s) and
                len(s) < 45 and
                re.match(r'^\([a-z]{2,6}\.?\)$', next_s)):
            remedy_names[i] = s

    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue

        if re.match(r'^Page \d+ of \d+$', s):
            continue

        if i in remedy_names:
            flush()
            current_remedy = remedy_names[i]
            current_lines = []
            continue

        if re.match(r'^\([a-z]{2,6}\.?\)$', s):
            continue

        if current_remedy:
            current_lines.append(s)

    flush()
    return chunks


# ─────────────────────────────────────
# SANKARAN
# ─────────────────────────────────────

def chunk_sankaran(text):
    paragraphs = re.split(r'\n\n+', text)
    chunks = []
    buffer = []
    buffer_wc = 0

    def flush_buffer():
        if buffer:
            content = ' '.join(buffer)
            content = re.sub(r'\s+', ' ', content).strip()
            if len(content) > 120:
                chunks.append(make_chunk("general", "Sankaran", "concept", content))

    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 60:
            continue
        if re.match(r'^\d{1,3}$', para):
            continue

        wc = word_count(para)
        if buffer_wc + wc > MAX_WORDS:
            flush_buffer()
            buffer = [para]
            buffer_wc = wc
        else:
            buffer.append(para)
            buffer_wc += wc

    flush_buffer()
    return chunks


# ─────────────────────────────────────
# MAIN
# ─────────────────────────────────────

def main():
    tasks = [
        ("Boericke_clean.txt",   chunk_boericke,   "Boericke"),
        ("allen_clean.txt",      chunk_allen,       "Allen"),
        ("kent_clean.txt",       chunk_kent,        "Kent"),
        ("vithoulkas_clean.txt", chunk_vithoulkas,  "Vithoulkas"),
        ("Sankaran_clean.txt",   chunk_sankaran,    "Sankaran"),
    ]

    all_chunks = []

    for filename, chunker_fn, label in tasks:
        path = os.path.join(TEXT_DIR, filename)
        if not os.path.exists(path):
            print(f"[SKIP] {filename} not found")
            continue

        text = open(path, encoding="utf-8").read()
        chunks = chunker_fn(text)
        all_chunks.extend(chunks)

        remedies = set(c["remedy"] for c in chunks)
        types = {}
        for c in chunks:
            types[c["type"]] = types.get(c["type"], 0) + 1

        print(f"[{label}] {len(chunks)} chunks | {len(remedies)} remedies | types: {dict(sorted(types.items()))}")

    seen = set()
    deduped = []
    for c in all_chunks:
        key = (c['remedy'], c['source'], c['content'][:100])
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    print(f"[DEDUP] {len(all_chunks)} -> {len(deduped)} chunks after deduplication")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Total: {len(deduped)} chunks -> {OUT_FILE}")


if __name__ == "__main__":
    main()
