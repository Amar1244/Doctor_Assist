"""
PQRS Pipeline — pure Python functions, no Streamlit UI.
Steps: preprocess → extract → score → generate → validate → advanced_validate
Import and call these functions directly from any script or FastAPI endpoint.
"""

import os
import json
import re
import requests
from dotenv import load_dotenv

load_dotenv()
_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ─────────────────────────────────────
# CASE TYPE PROFILES
# ─────────────────────────────────────
CASE_PROFILES = {
    "Acute": {
        "weights":   {"causation": 5, "modality": 3, "time": 3, "mental": 1, "concomitant": 2, "periodicity": 1, "peculiar": 3},
        "min_score": 3,
        "priority":  "causation (what triggered), onset, modalities (what relieves/aggravates)",
        "focus":     "Focus on what started the complaint, what triggers it, what relieves it. Reject chronic constitutional symptoms.",
    },
    "Chronic": {
        "weights":   {"peculiar": 4, "causation": 3, "modality": 3, "mental": 3, "periodicity": 3, "concomitant": 2, "time": 2},
        "min_score": 4,
        "priority":  "peculiarity, periodicity, generals (thermal, food, sleep), mental state",
        "focus":     "Focus on characteristic, recurring, and unusual symptoms. Reject acute symptoms without a chronic pattern.",
    },
    "Mental / Emotional": {
        "weights":   {"mental": 5, "causation": 5, "peculiar": 3, "modality": 2, "concomitant": 2, "time": 2, "periodicity": 1},
        "min_score": 4,
        "priority":  "mental/emotional expressions, causation of mental state, peculiar behavior",
        "focus":     "Only include strongly expressed, characteristic mental states with causation or peculiarity. Reject vague emotions.",
    },
    "Mixed (Acute + Chronic)": {
        "weights":   {"causation": 4, "mental": 3, "peculiar": 3, "modality": 2, "time": 2, "concomitant": 2, "periodicity": 2},
        "min_score": 4,
        "priority":  "causation, modalities, mental/emotional states, peculiar symptoms",
        "focus":     "Balance acute triggers with chronic constitutional symptoms. Include both if characteristic.",
    },
}

QUALIFYING_TAGS = {"causation", "modality", "peculiar", "mental"}


# ─────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────
def _call_llm(prompt: str) -> str | None:
    if not _API_KEY:
        return None
    payload = {
        "model":       "google/gemma-3-27b-it",
        "messages":    [{"role": "user", "content": "/no_think\n\n" + prompt}],
        "temperature": 0.2,
        "max_tokens":  6000,
        "reasoning":   {"effort": "none"},
        "provider":    {"order": ["Fireworks", "Together", "Novita", "Nebius"], "allow_fallbacks": True},
    }
    for _ in range(3):
        try:
            res = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {_API_KEY}", "Content-Type": "application/json"},
                json=payload,
                timeout=120,
            )
            if res.status_code == 401:
                return None
            if res.status_code in (429, 502, 503, 504):
                continue
            choices = res.json().get("choices", [])
            if not choices:
                return None
            msg = choices[0].get("message", {})
            return (msg.get("content") or msg.get("reasoning") or "").strip() or None
        except Exception:
            continue
    return None


# ─────────────────────────────────────
# JSON HELPER
# ─────────────────────────────────────
def _extract_json(text: str):
    if not text:
        return None
    text = re.sub(r"```(?:json)?\s*", "", text).strip().replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    s, e = text.find("{"), text.rfind("}") + 1
    if s != -1 and e > s:
        try:
            return json.loads(text[s:e])
        except Exception:
            pass
    s, e = text.find("["), text.rfind("]") + 1
    if s != -1 and e > s:
        try:
            return json.loads(text[s:e])
        except Exception:
            pass
    return None


# ─────────────────────────────────────
# STEP 1 — PREPROCESS + DETECT CC
# ─────────────────────────────────────
def preprocess_case(case_text: str) -> dict:
    """
    Cleans case text and auto-detects the chief complaint via LLM.
    Returns: {"chief_complaint": str, "cleaned": str, "error": str|None}
    """
    prompt = f"""You are a medical case structuring assistant.

Tasks:
1. Identify the chief complaint — the PRIMARY reason the patient came in (2-5 words max).
2. Clean and preserve ALL information from the case text.
3. Organize into sections: Mental, Physical, Modalities, Generals, History.

Return STRICT JSON:

{{
  "chief_complaint": "short phrase — primary complaint only (e.g. hair fall, knee pain)",
  "cleaned": "full cleaned structured case text — preserve every detail"
}}

Rules:
- Do NOT add new information
- Do NOT interpret or summarize
- Keep original meaning intact

Case:
{case_text}
"""
    content = _call_llm(prompt)
    if not content:
        return {"chief_complaint": "", "cleaned": case_text, "error": "LLM call failed"}
    parsed = _extract_json(content)
    if not parsed:
        return {"chief_complaint": "", "cleaned": case_text, "error": "Could not parse response"}
    return {
        "chief_complaint": parsed.get("chief_complaint", "").strip().lower(),
        "cleaned":         parsed.get("cleaned", case_text),
        "error":           None,
    }


# ─────────────────────────────────────
# STEP 2 — EXTRACT + TAG SYMPTOMS
# ─────────────────────────────────────
def extract_symptoms(cleaned: str, chief_complaint: str = "") -> dict:
    """
    LLM extracts every symptom from the cleaned case and assigns tags.
    Returns: {"symptoms": [...], "error": str|None}
    """
    cc = chief_complaint.strip().lower()
    cc_rule = f"""
Chief Complaint: "{cc}"
- Do NOT extract "{cc}" as a standalone symptom.
- You MAY extract at most ONE qualifying symptom directly about "{cc}" only if it has a
  unique modality (e.g. "{cc} worse after washing") or a clear Never Well Since cause.
- PRIORITIZE everything beyond the chief complaint: mental states, generals,
  concomitants, peculiar expressions, causation of the whole case.
""" if cc else ""

    prompt = f"""
You are a senior homeopathy expert.

Your task: Extract and tag every symptom from the case below.
{cc_rule}
For EACH symptom:
- Write a clear, atomic symptom in standard homeopathic language
- Assign ALL applicable tags from: causation, modality, mental, concomitant, periodicity, time, peculiar
- A symptom may have multiple tags

What to extract (cover ALL categories):
1. Causation       — what triggered or started the complaint (e.g. "never well since miscarriage")
2. Modalities      — anything better or worse from external/internal factors
3. Mental/Emotional — exact expressions used by the patient, characteristic mental states
4. Generals        — thermal reaction, thirst, appetite, craving, aversion, sleep, dreams, sweat, energy
5. Concomitants    — symptoms that always appear together
6. Time/Periodicity — specific time of day, periodic recurrence
7. Peculiar        — strange, rare, or unexpected symptoms

IMPORTANT:
- Capture patient's EXACT words for mental symptoms
- Do NOT merge multiple symptoms into one
- Do NOT invent symptoms not present in the case
- Do NOT miss generals (thermal, thirst, appetite, cravings, aversions, sleep, dreams)

Return STRICT JSON:

{{
  "symptoms": [
    {{
      "symptom": "clear homeopathic symptom description",
      "tags": ["tag1", "tag2"]
    }}
  ]
}}

IMPORTANT: Return ONLY valid JSON.

Full Case:
{cleaned}
"""
    content = _call_llm(prompt)
    if not content:
        return {"symptoms": [], "error": "LLM call failed"}
    parsed = _extract_json(content)
    if not parsed:
        return {"symptoms": [], "error": "Could not parse response"}
    return {"symptoms": parsed.get("symptoms", []), "error": None}


# ─────────────────────────────────────
# STEP 3 — SCORE SYMPTOMS  (pure Python, no LLM)
# ─────────────────────────────────────
def score_symptoms(symptoms: list, case_type: str = "Mixed (Acute + Chronic)", chief_complaint: str = "") -> tuple:
    """
    Scores symptoms using case-type weights. No LLM call.
    Returns: (scored: list, skipped: list)
    """
    profile   = CASE_PROFILES.get(case_type, CASE_PROFILES["Mixed (Acute + Chronic)"])
    weights   = profile["weights"]
    min_score = profile["min_score"]
    cc_lower  = chief_complaint.lower().strip()
    cc_words  = set(cc_lower.split()) if cc_lower else set()
    cc_quota  = 1

    def is_cc_variant(sym: str) -> bool:
        s = sym.lower().strip()
        if s == cc_lower:
            return True
        if cc_lower and s.startswith(cc_lower) and (len(s) == len(cc_lower) or s[len(cc_lower)] == " "):
            return True
        first = set(s.split()[:4])
        return bool(cc_words and len(cc_words & first) >= max(1, len(cc_words) - 1))

    _TAG_NORM = {
        "causal": "causation", "cause": "causation", "causality": "causation",
        "modalities": "modality", "modulation": "modality", "aggravation": "modality", "amelioration": "modality",
        "mentals": "mental", "emotional": "mental", "emotion": "mental",
        "concomitant symptom": "concomitant", "concomitants": "concomitant", "accompanying": "concomitant",
        "periodic": "periodicity", "recurrence": "periodicity", "recurring": "periodicity",
        "timing": "time", "time of day": "time",
        "peculiarity": "peculiar", "rare": "peculiar", "strange": "peculiar", "unusual": "peculiar",
    }

    scored, skipped = [], []
    for item in symptoms:
        tags    = [_TAG_NORM.get(t.lower().strip(), t.lower().strip()) for t in item.get("tags", [])]
        symptom = item.get("symptom", "").strip()
        sc      = sum(weights.get(t, 0) for t in tags)

        if cc_lower and is_cc_variant(symptom):
            if not set(tags) & QUALIFYING_TAGS:
                skipped.append({**item, "score": sc, "skip_reason": "CC variant — no qualifying tag"})
                continue
            if cc_quota <= 0:
                skipped.append({**item, "score": sc, "skip_reason": "CC variant — quota used (max 1 allowed)"})
                continue
            cc_quota -= 1

        if sc < min_score:
            skipped.append({**item, "score": sc, "skip_reason": f"Score {sc} below threshold {min_score}"})
            continue

        if not set(tags) & QUALIFYING_TAGS:
            skipped.append({**item, "score": sc, "skip_reason": "No qualifying PQRS tag"})
            continue

        scored.append({"symptom": symptom, "tags": tags, "score": sc})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored, skipped


# ─────────────────────────────────────
# STEP 4 — GENERATE FINAL PQRS
# ─────────────────────────────────────
def generate_pqrs(scored: list, cleaned: str, case_type: str = "Mixed (Acute + Chronic)", chief_complaint: str = "") -> dict:
    """
    LLM selects best 3-6 PQRS from scored candidates and converts to Kent/Synthesis rubric language.
    Returns: {"pqrs": [...], "error": str|None}
    """
    profile = CASE_PROFILES.get(case_type, CASE_PROFILES["Mixed (Acute + Chronic)"])
    cc      = chief_complaint.strip()
    cc_rule = (
        f'\nChief Complaint: "{cc}"\n'
        f'- Do NOT select "{cc}" as a PQRS symptom.\n'
        f'- At most ONE symptom may qualify "{cc}" with a unique modality or Never Well Since cause.\n'
        f'- Prioritize mental symptoms, generals, and peculiar expressions over chief complaint variants.\n'
    ) if cc else ""

    prompt = f"""
You are a senior homeopathy doctor selecting the final PQRS symptoms for repertorization.

Case Type: {case_type}
Priority: {profile['priority']}
{profile['focus']}
{cc_rule}
You have the FULL case text below for reference, plus a pre-scored list of candidates.
Use the full case to verify and understand each candidate — do not rely only on the candidate text.

SELECTION PRIORITY ORDER — follow this hierarchy strictly:
  1. Constitutional causation    — "Never Well Since" or "Ailments from" covering the WHOLE case
                                   ALWAYS select this first if present. It outranks every other symptom.
  2. Peculiar patient expression — exact words that are strange, rare, or unexpected for the condition
  3. Combination modality        — two or more modalities on the same symptom
  4. Concomitant symptoms        — two symptoms that always appear together
  5. Fixed time generals         — specific recurring time pattern (e.g. waking 3 AM, pain every Monday)
  6. Strong mental WITH causation — characteristic emotional state plus its triggering cause
  7. Single local modality       — one aggravation/amelioration on CC (last resort, max 1 slot)

Your task:
- Select the BEST 3 to 6 PQRS symptoms following the priority order above
- Each must be CHARACTERISTIC — unique to THIS patient, not applicable to any patient
- Transform each into proper homeopathic RUBRIC language (Kent/Synthesis repertory style)

CLINICAL TRANSFORMATION RULES — write in rubric format:
- Causation         → "GENERALS - Ailments from, [cause]; e.g. Ailments from, miscarriage; Ailments from, grief"
- Mental symptoms   → "MIND - [rubric]; e.g. MIND - Misunderstood, feels; MIND - Weeping, talking about complaints, when"
- Generals          → "GENERALS - [rubric]; e.g. GENERALS - Food - sweets, desires; GENERALS - Cold - aggravation"
- Modalities        → embed into symptom: "worse cold, better warmth" or "aggravated washing"
- Peculiar          → keep exact patient language + rubric; e.g. "MIND - Delusion, unloved, feels"
- Body part symptom → "[BODY PART] - [rubric] - [modality if any]"

Examples of CORRECT clinical transformation:
  Case has "never well since miscarriage" covering whole case
  → ALWAYS select first: "GENERALS - Ailments from, miscarriage"

  Raw: "patient cries easily when talking about the miscarriage"
  → "MIND - Weeping, talking about complaints, when; causation: miscarriage"

  Raw: "feels nobody understands her"
  → "MIND - Misunderstood, feels"

  Raw: "wakes 3 AM cannot fall asleep again"
  → "SLEEP - Waking, midnight, after, 3 AM"

Each selected symptom MUST have at least ONE of:
  1. Clear causation (what triggered the whole illness or this symptom)
  2. Clear modality (what makes it better or worse)
  3. Peculiarity (strange, rare, or unexpected)
  4. Strong, characteristic mental/emotional expression WITH its cause or context

RUBRIC FORMAT RULES — strictly one rubric per item:
- Each JSON item must contain EXACTLY ONE rubric (one "SECTION - Description" string)
- NEVER join two rubrics with a semicolon into a single item — e.g. DO NOT write:
    "SKIN - Eruptions, aggravation, sun; SKIN - Eruptions, aggravation, water"
  Instead create TWO separate items, one for sun and one for water
- If a symptom has two modalities, embed both in the same rubric description:
    "SKIN - Eruptions, aggravation, sun and water" (one rubric, two modalities listed)

GROUNDING RULE — only use what the patient actually said:
- Every rubric you generate MUST be traceable to a specific phrase or sentence in the FULL CASE TEXT above
- Do NOT add symptoms based on what "usually" goes with a complaint — that is your knowledge, not this patient's case
- If the case does not mention weather, do NOT add a weather modality
- If the case does not mention sleep, do NOT add a sleep rubric
- If the case does not mention thirst, do NOT add thirst
- Do NOT infer symptoms from the remedy you suspect — reason from case to remedy, never from remedy back to case
- When in doubt whether a symptom is in the case: skip it

REJECT — do NOT include:
- Chief complaint alone without qualifier or modality
- Bare emotions without causation or peculiarity — "grief", "despair", "anxiety" alone are NOT valid PQRS
- Local CC modality when a constitutional causation exists and has not yet been selected
- Vague or common symptoms applicable to any patient with the same complaint
- Duplicate variants of the same symptom
- Symptoms without any individualizing feature

FULL CASE (for reference):
{cleaned}

SCORED CANDIDATES:
{json.dumps(scored, indent=2)}

Return STRICT JSON:

{{
  "pqrs": [
    {{
      "symptom": "patient language or raw description",
      "rubric":  "SECTION - Rubric in Kent/Synthesis format",
      "reason":  "why this is PQRS for THIS patient",
      "score":   0
    }}
  ]
}}

IMPORTANT: Return ONLY valid JSON. No explanation outside JSON.
"""
    content = _call_llm(prompt)
    if not content:
        return {"pqrs": [], "error": "LLM call failed"}
    parsed = _extract_json(content)
    if not parsed:
        return {"pqrs": [], "error": "Could not parse response"}
    pqrs_items = parsed.get("pqrs", [])
    scored_lookup = [(s["symptom"].lower(), s["score"]) for s in scored]
    for item in pqrs_items:
        if item.get("score", 0) == 0:
            sym_words = set(item.get("symptom", "").lower().split())
            best_score, best_overlap = 0, 0
            for cand_text, cand_score in scored_lookup:
                overlap = len(sym_words & set(cand_text.split()))
                if overlap > best_overlap:
                    best_overlap, best_score = overlap, cand_score
            if best_overlap > 0:
                item["score"] = best_score
    return {"pqrs": pqrs_items, "error": None}


# ─────────────────────────────────────
# STEP 5 — CLINICAL VALIDATION
# ─────────────────────────────────────
def validate_pqrs(pqrs: list) -> dict:
    """
    LLM clinical review — finds errors and returns corrected PQRS list.
    Returns: {"errors": [...], "corrected_pqrs": [...], "error": str|None}
    """
    prompt = f"""
You are a senior homeopathy clinician performing a strict PQRS quality review.

Review the following PQRS symptom list and identify clinical errors.

INSTRUCTIONS:
1. Point out which symptoms are NOT valid PQRS and explain why.
2. Identify issues such as:
   - Inclusion of life situations instead of symptoms
   - Vague or common mental states (e.g. "stressed", "anxious", "sad")
   - Over-specific events (e.g. dates, one-time incidents) instead of patterns
   - Duplicate or redundant symptoms
   - Chief complaint included without a qualifying modality or causation
   - Social behaviors that are not extreme or consistent enough to be rubrics

3. For each incorrect symptom, explain how to correct it — convert to clinical rubric form if possible.
4. Provide a corrected final PQRS list.

IMPORTANT RULES:
- PQRS must be CHARACTERISTIC, REPRODUCIBLE, and CLINICALLY USEFUL for repertorization
- Every symptom must have at least ONE of: modality, causation, peculiarity, or strong mental state
- DO NOT include: chief complaints alone, vague emotions, life events as symptoms, duplicates
- Corrected symptoms must be in Kent/Synthesis rubric format (e.g. MIND - Grief, ailments from)

PQRS TO REVIEW:
{json.dumps(pqrs, indent=2)}

Return STRICT JSON:

{{
  "errors": [
    {{
      "symptom": "original symptom text",
      "issue":   "what is wrong with it",
      "fix":     "corrected clinical rubric or REMOVE"
    }}
  ],
  "corrected_pqrs": [
    {{
      "symptom": "patient language description",
      "rubric":  "SECTION - Rubric in Kent/Synthesis format",
      "reason":  "why this is valid PQRS"
    }}
  ]
}}

IMPORTANT: Return ONLY valid JSON.
"""
    content = _call_llm(prompt)
    if not content:
        return {"errors": [], "corrected_pqrs": pqrs, "error": "LLM call failed"}
    parsed = _extract_json(content)
    if not parsed:
        return {"errors": [], "corrected_pqrs": pqrs, "error": "Could not parse response"}
    return {
        "errors":         parsed.get("errors", []),
        "corrected_pqrs": parsed.get("corrected_pqrs", pqrs),
        "error":          None,
    }


# ─────────────────────────────────────
# STEP 6 — ADVANCED REASONING REVIEW
# ─────────────────────────────────────
def advanced_validate(pqrs: list, cleaned: str) -> dict:
    """
    Deep reasoning review — checks for over-interpretation, missing stronger PQRS,
    wrong prioritization, non-clinical phrasing, and upgrade errors.
    Returns: {"errors": [...], "missing_pqrs": [...], "final_pqrs": [...], "error": str|None}
    """
    prompt = f"""
You are a senior homeopathy clinician performing a deep reasoning-level PQRS review.

Your task: identify subtle clinical mistakes in the PQRS selection below.

PRIORITY ORDER (strictly enforce this):
  1. Combination modalities  — two or more modalities on the same symptom (highest value)
  2. Single clear modality   — one definite aggravation or amelioration
  3. Concomitants            — symptoms always appearing together
  4. Periodicity             — fixed time pattern, regular recurrence
  5. Mental symptoms         — only if strongly expressed, with causation or peculiarity

FIVE ERROR TYPES TO CHECK:

1. OVER-INTERPRETATION
   - Do NOT label behavior as pathology unless clearly extreme and consistent
   - Example: "bullying sibling occasionally" ≠ MIND - Cruelty
   - Example: "sometimes irritable" ≠ MIND - Irritability (needs frequency + context)

2. MISSING STRONGER PQRS
   - If the case has clear modalities or concomitants that were ignored in favor of mental symptoms, flag it
   - Modalities and concomitants always outrank mental symptoms

3. WRONG PRIORITIZATION
   - If a mental symptom was selected but a modality/concomitant was available, that is a priority error
   - List what should have been selected instead

4. NON-CLINICAL PHRASING
   - Symptoms must be short, observable, and mappable to a repertory rubric
   - Reject explanatory or narrative language (e.g., "patient tends to feel overwhelmed when...")
   - Correct form: "MIND - Overwhelmed, feeling of" or "MIND - Anxiety, anticipation, from"

5. UPGRADE ERROR
   - Do NOT upgrade patient language to a stronger rubric than what is present
   - "I feel alone sometimes" ≠ MIND - Delusion, alone, he is
   - "I worry a lot" ≠ MIND - Anxiety, hypochondriacal
   - Stay at the level of what the patient actually expressed

FULL CASE (for cross-reference):
{cleaned}

PQRS TO REVIEW:
{json.dumps(pqrs, indent=2)}

Return STRICT JSON:

{{
  "errors": [
    {{
      "symptom":    "original symptom",
      "error_type": "Over-interpretation | Missing stronger PQRS | Wrong prioritization | Non-clinical phrasing | Upgrade error",
      "explanation":"why this is wrong",
      "corrected":  "corrected rubric or REMOVE"
    }}
  ],
  "missing_pqrs": [
    {{
      "symptom": "symptom from case that should have been included",
      "rubric":  "SECTION - Rubric",
      "reason":  "why this ranks higher than what was selected"
    }}
  ],
  "final_pqrs": [
    {{
      "symptom": "patient language",
      "rubric":  "SECTION - Rubric in Kent/Synthesis format",
      "reason":  "why this is valid PQRS"
    }}
  ]
}}

IMPORTANT: Return ONLY valid JSON.
"""
    content = _call_llm(prompt)
    if not content:
        return {"errors": [], "missing_pqrs": [], "final_pqrs": pqrs, "error": "LLM call failed"}
    parsed = _extract_json(content)
    if not parsed:
        return {"errors": [], "missing_pqrs": [], "final_pqrs": pqrs, "error": "Could not parse response"}
    return {
        "errors":       parsed.get("errors", []),
        "missing_pqrs": parsed.get("missing_pqrs", []),
        "final_pqrs":   parsed.get("final_pqrs", pqrs),
        "error":        None,
    }


# ─────────────────────────────────────
# FULL PIPELINE  (convenience wrapper)
# ─────────────────────────────────────
def run_pipeline(case_text: str, case_type: str = "Mixed (Acute + Chronic)") -> dict:
    """
    Runs all 6 steps end-to-end. Useful for scripting or testing outside the UI.
    Returns a single dict with every stage's output.
    """
    step1 = preprocess_case(case_text)
    if step1["error"]:
        return {"error": step1["error"]}

    cleaned = step1["cleaned"]
    cc      = step1["chief_complaint"]

    step2 = extract_symptoms(cleaned, cc)
    if step2["error"]:
        return {"error": step2["error"]}

    scored, skipped = score_symptoms(step2["symptoms"], case_type, cc)
    if not scored:
        return {"error": "No symptoms passed scoring — case may be too sparse or case type mismatch"}

    step4 = generate_pqrs(scored, cleaned, case_type, cc)
    if step4["error"]:
        return {"error": step4["error"]}

    step5 = validate_pqrs(step4["pqrs"])
    step6 = advanced_validate(step5["corrected_pqrs"], cleaned)

    return {
        "chief_complaint":   cc,
        "cleaned":           cleaned,
        "symptoms":          step2["symptoms"],
        "scored":            scored,
        "skipped":           skipped,
        "pqrs_raw":          step4["pqrs"],
        "pqrs_validated":    step5["corrected_pqrs"],
        "pqrs_final":        step6["final_pqrs"],
        "validation_errors": step5["errors"],
        "reasoning_errors":  step6["errors"],
        "missing_pqrs":      step6["missing_pqrs"],
        "step5_error":       step5.get("error"),
        "step6_error":       step6.get("error"),
    }
