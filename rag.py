"""
RAG module — exposes the 3 functions imported by main.py:
  - rag_available()
  - search_remedies(query, n=6)
  - validate_pqrs_line(line, threshold=0.38)

Improvements over v1:
  - Source weighting   : down-weights Boericke (large) vs Allen/Vithoulkas (precise)
  - Type weighting     : mind/keynotes ranked higher than physical/general
  - Remedy aggregation : chunks grouped by remedy, not returned raw
  - Multi-source boost : remedies confirmed by multiple books score higher
  - Metadata filtering : optional type filter (e.g. mind-only search)
"""

import os
import re
import requests
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_DIR     = r"E:\chroma_db"   # chroma_db lives on E disk; code runs from D disk
COLLECTION = "materia_medica"

_db    = None
_model = None

# ─────────────────────────────────────
# WEIGHTS
# ─────────────────────────────────────

# Down-weight large sources to prevent domination
SOURCE_WEIGHTS = {
    "Allen":      1.00,   # small, precise keynotes — full weight
    "Vithoulkas": 1.00,   # small, essence portraits — full weight
    "Kent":       0.90,   # authoritative classical — near full
    "Boericke":   0.70,   # largest collection — down-weighted
    "Sankaran":   0.60,   # concept-level, less remedy-specific
}

# Reward clinically important section types
TYPE_WEIGHTS = {
    "mind":         1.20,
    "keynotes":     1.15,
    "essence":      1.15,
    "modalities":   1.10,
    "generals":     1.10,
    "fever":        1.00,
    "physical":     1.00,
    "stomach":      1.00,
    "head":         1.00,
    "female":       1.00,
    "general":      0.90,
    "concept":      0.70,
    "introduction": 0.60,
}

# Bonus for remedies confirmed by multiple books
MULTI_SOURCE_BONUS = {1: 0.00, 2: 0.15, 3: 0.25, 4: 0.30}

# Confidence thresholds — below these, LLM expansion is triggered
# Pattern boost multiplies scores up to ×1.8, so threshold is scaled accordingly
SEARCH_LOW_CONFIDENCE  = 2.7    # top remedy score below this → expand (1.5 × 1.8)
VALIDATE_LOW_CONFIDENCE = 0.30  # validate score between this and threshold → expand
SIGNAL_WEAK_THRESHOLD  = 0.60   # per-signal best score below this = vocabulary mismatch

# Signals containing these words are modality signals — carry extra diagnostic weight
_MODALITY_SIGNAL_MARKERS = {"better", "worse", "amelior", "aggravat"}

# ─────────────────────────────────────
# CONTRADICTION DETECTION
# ─────────────────────────────────────

# Each axis maps to keywords that identify it in a symptom line.
# A contradiction is: signal with "better/amelior" on axis X + signal with "worse/aggravat" on axis X.
MODALITY_AXES = {
    "cold":     {"cold", "cool", "chilly", "cooling", "ice"},
    "heat":     {"heat", "hot", "warm", "warmth", "fire"},
    "motion":   {"motion", "moving", "exercise", "exertion", "walking", "movement"},
    "rest":     {"rest", "resting", "lying", "repose", "sitting", "stillness"},
    "pressure": {"pressure", "pressing", "tight", "binding"},
    "wet":      {"wet", "damp", "moisture", "humid", "rain"},
    "eating":   {"eating", "food", "meal", "after eating"},
    "air":      {"open air", "fresh air", "draft", "draught", "windy"},
}

# Per-remedy dampening based on how many axis contradictions a remedy spans
# (spans = its chunks show BOTH better AND worse direction for the same axis)
AXIS_SPAN_DAMPEN   = {0: 0.92, 1: 0.68, 2: 0.50}   # 0-spans → light, 2+-spans → heavy
# Flat dampening for special (non-axis) contradictions, e.g. thirst/thirstless
SPECIAL_CON_DAMPEN = {0: 1.00, 1: 0.88, 2: 0.78, 3: 0.68}

# ─────────────────────────────────────
# STOPWORDS (never score these keywords)
# ─────────────────────────────────────

_STOPWORDS = {
    'the','and','for','with','from','that','this','they','have','been',
    'which','when','more','also','into','than','some','such','very',
    'great','will','not','but','are','was','his','her','their','our',
    'its','can','may','all','any','one','two','has','had','does','did',
    'what','who','how','where','there','here','then','even','over',
    'after','before','about','through','during','while','would','could',
    'should','might','each','both','other','these','those','most','much',
}


# ─────────────────────────────────────
# CONDITIONAL LLM QUERY EXPANSION
# Only fires when initial search confidence is low
# ─────────────────────────────────────

def _expand_query(query: str) -> str:
    """
    Rephrases patient-language symptoms into classical homeopathy terminology.
    Called ONLY when dense+BM25 returns low-confidence results.
    Returns original query if API unavailable or call fails.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return query

    prompt = (
        "You are a classical homeopathy expert.\n"
        "Rephrase these patient symptoms into classical homeopathic materia medica language.\n"
        "Use standard medical and repertory terminology.\n"
        "Examples: 'red face' -> 'flushed face, congestion'\n"
        "          'light sensitivity' -> 'photophobia, aversion to light'\n"
        "          'stomach pain' -> 'gastralgia, epigastric pain'\n"
        "Output only the rephrased symptoms. One concept per line. No explanations.\n\n"
        f"Symptoms:\n{query}"
    )

    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={
                "model":       "google/gemma-3-27b-it",
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens":  300,
            },
            timeout=15
        )
        if r.status_code == 200:
            expanded = r.json()["choices"][0]["message"]["content"].strip()
            if expanded and len(expanded) > 10:
                return expanded
    except Exception:
        pass   # silent fallback — never crash the RAG pipeline

    return query


# ─────────────────────────────────────
# LAZY LOAD
# ─────────────────────────────────────

def _load():
    global _db, _model
    if _db is not None:
        return True
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        client = chromadb.PersistentClient(path=DB_DIR)
        _db    = client.get_collection(COLLECTION)
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        return True
    except Exception as e:
        print(f"[rag] Could not load DB: {e}")
        return False


def rag_available() -> bool:
    if not _load():
        return False
    try:
        return _db.count() > 0
    except Exception:
        return False


# ─────────────────────────────────────
# DENSE SEARCH
# ─────────────────────────────────────

def _dense_search(query: str, n: int = 60, filter_type: str = None):
    """Dense vector search with optional metadata type filter."""
    emb   = _model.encode(query).tolist()
    where = {"type": {"$eq": filter_type}} if filter_type else None
    kwargs = dict(query_embeddings=[emb], n_results=min(n, _db.count()))
    if where:
        kwargs["where"] = where
    res   = _db.query(**kwargs)
    docs  = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    return list(zip(docs, metas, dists))


# ─────────────────────────────────────
# KEYWORD SCORE
# ─────────────────────────────────────

def _keyword_score(text: str, query_words: list) -> float:
    text_lower = text.lower()
    score = 0.0
    for w in query_words:
        if len(w) < 4 or w in _STOPWORDS:
            continue
        count = text_lower.count(w)
        if count:
            score += 1 + 0.5 * (count - 1)   # diminishing returns
    return score


# ─────────────────────────────────────
# CHUNK SCORE (hybrid + weights)
# ─────────────────────────────────────

def _chunk_score(dist: float, doc: str, meta: dict, query_words: list) -> float:
    """
    Final chunk score = (dense_sim × 0.5 + keyword × 0.25)
                        × source_weight × type_weight
    """
    sim        = max(0.0, 1.0 - dist / 2.0)
    kw         = min(_keyword_score(doc, query_words), 4.0)   # cap to prevent dominating sim
    hybrid     = sim * 0.5 + kw * 0.25
    src_w      = SOURCE_WEIGHTS.get(meta.get("source", ""), 0.80)
    type_w     = TYPE_WEIGHTS.get(meta.get("type", ""), 1.00)
    return hybrid * src_w * type_w


# ─────────────────────────────────────
# PATTERN COVERAGE
# ─────────────────────────────────────

_NORM_SUFFIXES = tuple(sorted(
    ('ness', 'tion', 'ance', 'ence', 'ment', 'less', 'ful',
     'ing',  'ive',  'ous',  'ity',  'ent',  'ion',  'ial',
     'ous',  'ed',   'er',   'al',   'ly'),
    key=len, reverse=True,   # longest suffix stripped first
))

def _stem_word(w: str) -> str:
    """Strip one common suffix so morphological variants map to the same root.
    E.g.: indifferent→indiffer, indifference→indiffer, restlessness→restless"""
    for suf in _NORM_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)]
    return w


def _per_signal_scores(signal_lines: list, remedy_chunks: list) -> list:
    """
    For each signal line, returns the best keyword-coverage depth across all chunks
    of this remedy — the fraction of the signal's key words found in the best chunk.
    Returns a list of floats in [0.0, 1.0], one per signal.
    Used for both keynote bonus and vocabulary-mismatch detection.
    """
    scores = []
    for signal in signal_lines:
        words = [w for w in re.findall(r'\b\w{4,}\b', signal.lower())
                 if w not in _STOPWORDS]
        if not words:
            scores.append(0.0)
            continue
        words_norm = [_stem_word(w) for w in words]
        best = 0.0
        for _, doc, _ in remedy_chunks:
            doc_words = {_stem_word(w)
                         for w in re.findall(r'\b\w{4,}\b', doc.lower())
                         if w not in _STOPWORDS}
            frac = sum(1 for w in words_norm if w in doc_words) / len(words)
            if frac > best:
                best = frac
        scores.append(best)
    return scores


# ─────────────────────────────────────
# REMEDY AGGREGATION
# ─────────────────────────────────────

def _aggregate_by_remedy(hits: list, query_words: list, query: str = "") -> list:
    """
    Groups chunks by remedy.
    Per remedy:
      - score = sum of top-2 chunk scores per source + multi-source bonus
      - best_chunks = up to 2 best chunks (different sources preferred)
    Returns list of (remedy_score, remedy, sources_set, best_chunks) sorted desc.
    """
    remedy_data = defaultdict(lambda: {
        "score":       0.0,
        "sources":     set(),
        "chunks":      [],        # (chunk_score, doc, meta)
        "src_scores":  defaultdict(float),
    })

    for doc, meta, dist in hits:
        remedy = meta.get("remedy", "unknown")
        source = meta.get("source", "unknown")
        score  = _chunk_score(dist, doc, meta, query_words)

        d = remedy_data[remedy]
        d["sources"].add(source)
        d["chunks"].append((score, doc, meta))

        # Keep only top-2 scores per source (prevent one source flooding)
        src_chunks = [(s, d2, m) for s, d2, m in d["chunks"] if m.get("source") == source]
        src_chunks.sort(key=lambda x: x[0], reverse=True)
        d["src_scores"][source] = sum(s for s, _, _ in src_chunks[:2])

    signal_lines = [s.strip() for s in query.split('\n') if s.strip()] if query else []
    n_signals    = len(signal_lines)

    # Modality signals (lines with better/worse direction) carry extra diagnostic weight
    mod_weights  = [
        1.5 if any(m in sig.lower() for m in _MODALITY_SIGNAL_MARKERS) else 1.0
        for sig in signal_lines
    ]
    total_mod_w  = sum(mod_weights) or 1.0

    # Pattern boost scales down for short queries (≤3 signals are unreliable for breadth scoring)
    coverage_weight = min(0.8, 0.8 * n_signals / 4) if n_signals else 0.0

    # Build ranked list
    ranked = []
    for remedy, d in remedy_data.items():
        n_sources   = len(d["sources"])
        bonus       = MULTI_SOURCE_BONUS.get(min(n_sources, 4), 0.30)
        total_score = sum(d["src_scores"].values()) + bonus

        # Pattern coverage boost — modality signals weighted 1.5×, short queries scaled down
        per_sig  = _per_signal_scores(signal_lines, d["chunks"])
        coverage = (sum(s * w for s, w in zip(per_sig, mod_weights)) / total_mod_w
                    if signal_lines else 0.0)
        total_score = total_score * (1.0 + coverage * coverage_weight)

        # Keynote bonus — fires when one signal is genuinely dominant AND strong in absolute terms.
        # Dual condition prevents noisy low-quality matches from triggering:
        #   ≥ 0.60 absolute  → signal actually matched well
        #   ≥ 1.6× average   → signal is clearly the standout, not just least-weak
        if len(per_sig) > 1:
            max_idx = per_sig.index(max(per_sig))
            max_s   = per_sig[max_idx]
            avg_s   = sum(per_sig) / len(per_sig)
            if avg_s > 0.0 and max_s >= 0.60 and max_s >= avg_s * 1.6:
                # Fix 1: raised coeff for proportional boost (was 0.9/0.7)
                coeff = 1.3 if (mod_weights and mod_weights[max_idx] > 1.0) else 1.2
                total_score *= (1.0 + (max_s - avg_s) * coeff)
            # Fix 2: pathognomonic override — strong AND dominant keynote beats multi-source noise
            # Guard: ≥0.75 absolute AND ≥1.5× average prevents accidental spikes
            if len(per_sig) >= 3 and avg_s > 0.0 and max_s >= 0.75 and max_s >= avg_s * 1.5:
                total_score += 0.8

        # Pick best 2 chunks preferring different sources
        d["chunks"].sort(key=lambda x: x[0], reverse=True)
        best_chunks = []
        used_sources = set()
        for score, doc, meta in d["chunks"]:
            src = meta.get("source", "")
            if src not in used_sources:
                best_chunks.append((score, doc, meta))
                used_sources.add(src)
            if len(best_chunks) >= 2:
                break

        ranked.append((total_score, remedy, d["sources"], best_chunks))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


# ─────────────────────────────────────
# CONTRADICTION DETECTION
# ─────────────────────────────────────

def _detect_contradictions(signal_lines: list) -> list:
    """
    Returns a list of contradiction dicts, each with keys:
      axis      — the modality axis that conflicts (e.g. "cold")
      better    — the signal line containing "better ... cold"
      worse     — the signal line containing "worse  ... cold"
    Also detects the thirst/thirstless pair specially.
    """
    # Direction prefixes — intentionally partial so "amelior" matches "ameliorated",
    # "aggravat" matches "aggravated", "worsen" matches "worsening", etc.
    BETTER_WORDS = {"better", "amelior", "relieved", "relief", "improve"}
    WORSE_WORDS  = {"worse",  "aggravat", "worsen",  "increase"}

    def _has_dir(line, prefixes):
        """Prefix match for direction words — \bword (no trailing boundary)."""
        low = line.lower()
        return any(re.search(r'\b' + re.escape(p), low) for p in prefixes)

    def _has(line, words):
        """Full word-boundary match for axis keywords — prevents 'rest' ↔ 'restlessness'."""
        low = line.lower()
        return any(
            bool(re.search(r'\b' + re.escape(w) + r'\b', low)) if ' ' not in w
            else w in low          # multi-word phrases: substring is fine
            for w in words
        )

    # Classify each line: {axis -> {"better": line | None, "worse": line | None}}
    axis_map = {ax: {"better": None, "worse": None} for ax in MODALITY_AXES}

    for line in signal_lines:
        is_better = _has_dir(line, BETTER_WORDS)
        is_worse  = _has_dir(line, WORSE_WORDS)
        if not (is_better or is_worse):
            continue
        if is_better and is_worse:
            continue   # compound line — ambiguous direction, skip to avoid false positive
        for axis, keywords in MODALITY_AXES.items():
            if _has(line, keywords):
                if is_better and axis_map[axis]["better"] is None:
                    axis_map[axis]["better"] = line
                if is_worse and axis_map[axis]["worse"] is None:
                    axis_map[axis]["worse"] = line

    conflicts = []
    for axis, sides in axis_map.items():
        if sides["better"] and sides["worse"]:
            conflicts.append({
                "axis":   axis,
                "better": sides["better"],
                "worse":  sides["worse"],
            })

    # Thirst special case
    thirst_lines    = [l for l in signal_lines if "thirst"    in l.lower()
                       and "thirstless" not in l.lower()]
    thirstless_lines = [l for l in signal_lines if "thirstless" in l.lower()]
    if thirst_lines and thirstless_lines:
        conflicts.append({
            "axis":   "thirst",
            "better": thirst_lines[0],
            "worse":  thirstless_lines[0],
        })

    return conflicts


def _remedy_spans_conflict(axis_keywords: set, best_chunks: list) -> bool:
    """
    Returns True if this remedy's best chunks show BOTH better-direction AND
    worse-direction language in the context of the given axis keywords.
    Remedies that span a contradiction are genuinely ambiguous for that axis
    and deserve heavier dampening than remedies unrelated to it.
    """
    _BETTER = {"better", "amelior", "relief", "improve"}
    _WORSE  = {"worse",  "aggravat", "worsen"}

    better_found = worse_found = False
    for _, doc, _ in best_chunks:
        low = doc.lower()
        has_axis = any(
            bool(re.search(r'\b' + re.escape(kw) + r'\b', low)) if ' ' not in kw
            else kw in low
            for kw in axis_keywords
        )
        if not has_axis:
            continue
        if any(re.search(r'\b' + re.escape(p), low) for p in _BETTER):
            better_found = True
        if any(re.search(r'\b' + re.escape(p), low) for p in _WORSE):
            worse_found = True
    return better_found and worse_found


# ─────────────────────────────────────
# SEARCH REMEDIES  (main API)
# ─────────────────────────────────────

def search_remedies(query: str, n: int = 6, filter_type: str = None) -> str:
    """
    Search books for chunks matching the query.
    Returns remedy-grouped, source-weighted, multi-source-boosted excerpts
    as a formatted string for injection into the LLM prompt.
    """
    if not _load():
        return "[RAG unavailable — DB not loaded]"

    try:
        query_words = re.findall(r'\b\w{3,}\b', query.lower())
        hits        = _dense_search(query, n=80, filter_type=filter_type)

        if not hits:
            return "[RAG: no relevant book excerpts found]"

        signal_lines = [s.strip() for s in query.split('\n') if s.strip()]
        ranked = _aggregate_by_remedy(hits, query_words, query)

        if not ranked:
            return "[RAG: no relevant book excerpts found]"

        # ── Vocabulary mismatch detection ──────────────────────
        # Counts signals where even the best chunk match is weak —
        # indicates patient-language terms not found in materia medica
        weak_count = 0
        for sig in signal_lines:
            sig_words = re.findall(r'\b\w{3,}\b', sig.lower())
            best = max((_chunk_score(d, doc, meta, sig_words)
                        for doc, meta, d in hits), default=0.0)
            if best < SIGNAL_WEAK_THRESHOLD:
                weak_count += 1
        vocab_mismatch = weak_count > len(signal_lines) / 2
        # ──────────────────────────────────────────────────────

        # ── Top-remedy coverage-gap check ──────────────────────
        # Even if aggregate vocab_mismatch doesn't fire, the #1 remedy
        # may still miss most signals (patient language ≠ materia medica).
        # Guards: ≥3 signals, ≥2 weak, >40% of signals weak.
        top_coverage_gap = False
        if ranked and len(signal_lines) >= 3:
            _, _, _, top_chunks = ranked[0]
            top_per_sig = _per_signal_scores(signal_lines, top_chunks)
            top_weak    = sum(1 for s in top_per_sig if s < 0.25)
            top_coverage_gap = top_weak >= 2 and top_weak > len(signal_lines) * 0.4
        # ──────────────────────────────────────────────────────

        # ── Conditional LLM expansion ──────────────────────────
        # Fires on low aggregate confidence OR vocabulary mismatch
        # OR when top remedy has coverage gaps across most signals
        if ranked[0][0] < SEARCH_LOW_CONFIDENCE or vocab_mismatch or top_coverage_gap:
            expanded = _expand_query(query)
            if expanded != query:
                exp_words  = re.findall(r'\b\w{3,}\b', expanded.lower())
                exp_hits   = _dense_search(expanded, n=80, filter_type=filter_type)
                exp_ranked = _aggregate_by_remedy(exp_hits, exp_words, expanded)
                # Merge: keep expanded result if it found a better top remedy
                if exp_ranked and exp_ranked[0][0] > ranked[0][0]:
                    ranked = exp_ranked
        # ──────────────────────────────────────────────────────

        # ── Contradiction detection + targeted dampening ──────────
        conflicts = _detect_contradictions(signal_lines)
        n_conflicts = len(conflicts)

        if conflicts:
            # Axis conflicts: per-remedy dampening based on spanning
            axis_conflicts    = [c for c in conflicts if c["axis"] in MODALITY_AXES]
            special_conflicts = [c for c in conflicts if c["axis"] not in MODALITY_AXES]
            special_dampen    = SPECIAL_CON_DAMPEN.get(min(len(special_conflicts), 3), 0.68)

            dampened = []
            for total_score, remedy, sources, best_chunks in ranked:
                n_spans     = sum(1 for c in axis_conflicts
                                  if _remedy_spans_conflict(
                                      MODALITY_AXES[c["axis"]], best_chunks))
                span_dampen = AXIS_SPAN_DAMPEN.get(min(n_spans, 2), 0.50)
                final       = total_score * span_dampen * special_dampen
                dampened.append((final, remedy, sources, best_chunks))
            ranked = sorted(dampened, key=lambda x: x[0], reverse=True)
        # ──────────────────────────────────────────────────────

        lines        = []
        seen_content = []   # for deduplication

        def _is_duplicate(text: str) -> bool:
            """True if text shares >60% words with any already-seen excerpt."""
            words = set(text.lower().split())
            for seen in seen_content:
                overlap = len(words & seen) / max(len(words | seen), 1)
                if overlap > 0.60:
                    return True
            seen_content.append(words)
            return False

        if n_conflicts:
            warn_lines = ["⚠️  MIXED MODALITIES DETECTED — results dampened, please review case:"]
            for c in conflicts:
                warn_lines.append(f"    {c['axis']:10s}: \"{c['better'].strip()}\"  vs  \"{c['worse'].strip()}\"")
            conf_label = {1: "LOW", 2: "VERY LOW", 3: "UNRELIABLE"}.get(n_conflicts, "UNRELIABLE")
            warn_lines.append(f"    Case confidence: {conf_label}")
            lines.append("\n".join(warn_lines))

        for total_score, remedy, sources, best_chunks in ranked[:n]:
            src_label = ", ".join(sorted(sources))
            header    = f"[{remedy}] — sources: {src_label} | score: {round(total_score, 2)}"
            excerpts  = []
            for _, doc, meta in best_chunks:
                excerpt = doc[:250].strip()
                if _is_duplicate(excerpt):
                    continue
                src  = meta.get("source", "")
                sec  = meta.get("type", "")
                tag  = f"  [{src}" + (f"/{sec}" if sec not in ("general","keynotes","essence","concept","") else "") + "]"
                excerpts.append(f"{tag} {excerpt}")
            if excerpts:
                lines.append(header + "\n" + "\n".join(excerpts))

        return "\n\n".join(lines) if lines else "[RAG: no relevant book excerpts found]"

    except Exception as e:
        return f"[RAG error: {e}]"


# ─────────────────────────────────────
# VALIDATE A SINGLE PQRS LINE
# ─────────────────────────────────────

def validate_pqrs_line(line: str, threshold: float = 0.38) -> dict:
    """
    Search DB for a single PQRS symptom line.
    Returns: {"confirmed": bool, "score": float, "remedy": str, "source": str}
    """
    empty = {"confirmed": False, "score": 0.0, "remedy": "", "source": ""}

    if not line or not line.strip() or not _load():
        return empty

    try:
        query_words = re.findall(r'\b\w{3,}\b', line.lower())
        hits        = _dense_search(line, n=20)
        if not hits:
            return empty

        # Score and sort
        scored = []
        for doc, meta, dist in hits:
            s = _chunk_score(dist, doc, meta, query_words)
            scored.append((s, meta))
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_meta = scored[0]

        # ── Conditional expansion ─────────────────────────────
        # Score in "weak zone" (between low-confidence floor and threshold)
        # → try rephrasing before giving up
        if VALIDATE_LOW_CONFIDENCE <= best_score < threshold:
            expanded = _expand_query(line)
            if expanded != line:
                exp_words = re.findall(r'\b\w{3,}\b', expanded.lower())
                exp_hits  = _dense_search(expanded, n=20)
                exp_scored = [
                    (_chunk_score(dist, doc, meta, exp_words), meta)
                    for doc, meta, dist in exp_hits
                ]
                exp_scored.sort(key=lambda x: x[0], reverse=True)
                if exp_scored and exp_scored[0][0] > best_score:
                    best_score, best_meta = exp_scored[0]
        # ─────────────────────────────────────────────────────

        if best_score < threshold:
            return empty

        return {
            "confirmed": True,
            "score":     round(best_score, 3),
            "remedy":    best_meta.get("remedy", ""),
            "source":    best_meta.get("source", ""),
        }

    except Exception as e:
        print(f"[rag.validate_pqrs_line] error: {e}")
        return empty
