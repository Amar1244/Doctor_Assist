import os
import re
import json
import requests
import streamlit as st
from dotenv import load_dotenv

# ---------------- INIT ----------------
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

st.set_page_config(layout="wide")
st.title("🧠 HomeoAssist - PQRS Pipeline")

if st.button("🔄 Clear Case / Start Over", key="reset"):
    st.session_state.data = {}
    st.rerun()

# ---------------- CASE TYPE PROFILES ----------------
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

# ---------------- LLM CALL ----------------
def call_llm(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "HomeoAssist"
    }
    data = {
        "model": "google/gemma-3-27b-it",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    try:
        res = requests.post(url, headers=headers, json=data, timeout=60)
        if res.status_code == 401:
            return {"error": "API key invalid or missing (401)"}
        if res.status_code == 429:
            return {"error": "Rate limit exceeded (429) — wait and retry"}
        if res.status_code in (502, 503, 504):
            return {"error": f"Server error {res.status_code}"}
        return res.json()
    except Exception as e:
        return {"error": str(e)}

# ---------------- HELPERS ----------------
def extract_json(text):
    if not text:
        return None
    text = re.sub(r"```(?:json)?\s*", "", text).strip().replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except Exception:
            pass
    start, end = text.find("["), text.rfind("]") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except Exception:
            pass
    return None

def get_content(res, step_name):
    if "error" in res:
        st.error(f"[{step_name}] API error: {res['error']}")
        return None
    choices = res.get("choices", [])
    if not choices:
        st.error(f"[{step_name}] No choices returned:")
        st.json(res)
        return None
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        st.error(f"[{step_name}] Empty LLM response.")
        return None
    return content

def parse_or_show(content, step_name):
    parsed = extract_json(content)
    if parsed is None:
        st.error(f"[{step_name}] Could not parse JSON. Raw output:")
        st.code(content, language="text")
    return parsed

def score_symptom(tags, weights):
    return sum(weights.get(t, 0) for t in tags)

# ---------------- SESSION STATE ----------------
if "data" not in st.session_state:
    st.session_state.data = {}

# ---------------- CASE TYPE SELECTOR ----------------
st.markdown("---")
case_type = st.radio(
    "Select Case Type",
    options=list(CASE_PROFILES.keys()),
    horizontal=True,
    index=3,
)
profile = CASE_PROFILES[case_type]
st.session_state.data["case_type"] = case_type

c1, c2, c3 = st.columns(3)
w = profile["weights"]
top_tags = sorted(w, key=lambda k: w[k], reverse=True)[:3]
c1.caption(f"**Priority:** {profile['priority']}")
c2.caption(f"**Min score:** {profile['min_score']}")
c3.caption(f"**Top tags:** {', '.join(top_tags)}")
st.markdown("---")

if st.session_state.data.get("chief_complaint"):
    st.info(f"**Chief Complaint (auto-detected):** {st.session_state.data['chief_complaint']}")

# ---------------- INPUT ----------------
case_text = st.text_area("Paste Full Case", height=280)

# ================================================================
# STEP 1 — PREPROCESS + DETECT CHIEF COMPLAINT
# ================================================================
if st.button("1️⃣  Preprocess & Detect CC"):
    if not case_text.strip():
        st.warning("Paste a case first.")
        st.stop()

    prompt = f"""
You are a medical case structuring assistant.

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
    res = call_llm(prompt)
    content = get_content(res, "Step 1")
    if content:
        parsed = parse_or_show(content, "Step 1")
        if parsed:
            st.session_state.data["cleaned"] = parsed
            st.session_state.data["chief_complaint"] = parsed.get("chief_complaint", "").strip().lower()
            st.success(f"Chief Complaint detected: **{st.session_state.data['chief_complaint']}**")
            st.write(parsed)

# ================================================================
# STEP 2 — EXTRACT + TAG  (single LLM call, full case context)
# ================================================================
if st.button("2️⃣  Extract & Tag Symptoms"):
    cleaned_data = st.session_state.data.get("cleaned", {})
    cleaned = cleaned_data.get("cleaned", "") if isinstance(cleaned_data, dict) else ""
    if not cleaned:
        st.warning("Run Step 1 first.")
        st.stop()

    cc = st.session_state.data.get("chief_complaint", "")
    cc_rule = f"""
Chief Complaint: "{cc}"
- Do NOT extract "{cc}" as a standalone symptom.
- You MAY extract at most ONE qualifying symptom directly about "{cc}" only if it has a
  unique modality (e.g. "{cc} worse after washing") or a clear Never Well Since cause
  (e.g. "never well since miscarriage → {cc}").
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
1. Causation       — what triggered or started the complaint (e.g., "never well since miscarriage")
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
    res = call_llm(prompt)
    content = get_content(res, "Step 2")
    if content:
        parsed = parse_or_show(content, "Step 2")
        if parsed:
            st.session_state.data["tagged_symptoms"] = parsed
            symptoms = parsed.get("symptoms", [])
            st.success(f"Extracted {len(symptoms)} symptoms.")
            st.write(parsed)

# ================================================================
# STEP 3 — SCORE  (pure Python, no LLM)
# ================================================================
if st.button("3️⃣  Score Symptoms (Python)"):
    tagged_data = st.session_state.data.get("tagged_symptoms", {})
    symptoms = tagged_data.get("symptoms", []) if isinstance(tagged_data, dict) else []
    if not symptoms:
        st.warning("Run Step 2 first.")
        st.stop()

    active_profile = CASE_PROFILES[st.session_state.data.get("case_type", "Mixed (Acute + Chronic)")]
    weights   = active_profile["weights"]
    min_score = active_profile["min_score"]

    _TAG_NORM = {
        "causal": "causation", "cause": "causation", "causality": "causation",
        "modalities": "modality", "modulation": "modality", "aggravation": "modality", "amelioration": "modality",
        "mentals": "mental", "emotional": "mental", "emotion": "mental",
        "concomitant symptom": "concomitant", "concomitants": "concomitant", "accompanying": "concomitant",
        "periodic": "periodicity", "recurrence": "periodicity", "recurring": "periodicity",
        "timing": "time", "time of day": "time",
        "peculiarity": "peculiar", "rare": "peculiar", "strange": "peculiar", "unusual": "peculiar",
    }

    # Warn if key tags are completely missing
    all_tags = {_TAG_NORM.get(t.lower().strip(), t.lower().strip()) for item in symptoms for t in item.get("tags", [])}
    if "causation" not in all_tags:
        st.warning("⚠️ No causation found. Case may be incomplete or causation not mentioned.")
    if "modality" not in all_tags:
        st.warning("⚠️ No modality found. PQRS quality will be lower.")

    scored = []
    skipped = []
    cc = st.session_state.data.get("chief_complaint", "")
    cc_lower = cc.lower().strip() if cc else ""
    cc_words = set(cc_lower.split()) if cc_lower else set()
    cc_variant_allowed = 1  # at most ONE CC-variant symptom passes

    def is_cc_variant(sym: str) -> bool:
        """True if symptom is primarily about the chief complaint."""
        s = sym.lower().strip()
        if s == cc_lower:
            return True
        # starts with CC phrase (e.g. "hair fall after abortion")
        if cc_lower and s.startswith(cc_lower) and (len(s) == len(cc_lower) or s[len(cc_lower)] == " "):
            return True
        # majority of first 4 words are CC words (catches reordered phrasing)
        first_words = set(s.split()[:4])
        if cc_words and len(cc_words & first_words) >= max(1, len(cc_words) - 1):
            return True
        return False

    for item in symptoms:
        tags    = [_TAG_NORM.get(t.lower().strip(), t.lower().strip()) for t in item.get("tags", [])]
        symptom = item.get("symptom", "").strip()
        sc      = score_symptom(tags, weights)

        # CC-variant filter — block unless it has a qualifying tag AND quota not used
        if cc_lower and is_cc_variant(symptom):
            if not set(tags) & QUALIFYING_TAGS:
                skipped.append({**item, "score": sc, "skip_reason": "CC variant — no qualifying tag"})
                continue
            if cc_variant_allowed <= 0:
                skipped.append({**item, "score": sc, "skip_reason": "CC variant — quota used (max 1 allowed)"})
                continue
            cc_variant_allowed -= 1  # consume the one allowed slot

        # Skip if score below threshold
        if sc < min_score:
            skipped.append({**item, "score": sc, "skip_reason": f"Score {sc} below threshold {min_score}"})
            continue

        # Skip if no qualifying tag at all
        if not set(tags) & QUALIFYING_TAGS:
            skipped.append({**item, "score": sc, "skip_reason": "No qualifying PQRS tag"})
            continue

        scored.append({"symptom": symptom, "tags": tags, "score": sc})

    scored.sort(key=lambda x: x["score"], reverse=True)

    st.info(
        f"Case type: **{case_type}** | "
        f"Min score: **{min_score}** | "
        f"**{len(scored)}** passed, **{len(skipped)}** skipped"
    )
    st.session_state.data["scored"] = scored

    st.subheader("✅ Scored Candidates")
    st.write(scored)
    if skipped:
        with st.expander(f"Skipped ({len(skipped)})"):
            st.write(skipped)

# ================================================================
# STEP 4 — FINAL PQRS  (LLM with full case context)
# ================================================================
if st.button("4️⃣  Generate Final PQRS"):
    scored = st.session_state.data.get("scored", [])
    if not scored:
        st.warning("Run Step 3 first.")
        st.stop()

    active_case    = st.session_state.data.get("case_type", "Mixed (Acute + Chronic)")
    active_profile = CASE_PROFILES[active_case]
    cc             = st.session_state.data.get("chief_complaint", "")
    cleaned_data   = st.session_state.data.get("cleaned", {})
    cleaned        = cleaned_data.get("cleaned", "") if isinstance(cleaned_data, dict) else ""

    cc_rule = (
        f'\nChief Complaint: "{cc}"\n'
        f'- Do NOT select "{cc}" as a PQRS symptom.\n'
        f'- At most ONE symptom may qualify "{cc}" with a unique modality or Never Well Since cause.\n'
        f'- Prioritize mental symptoms, generals, and peculiar expressions over chief complaint variants.\n'
    ) if cc else ""

    prompt = f"""
You are a senior homeopathy doctor selecting the final PQRS symptoms for repertorization.

Case Type: {active_case}
Priority: {active_profile['priority']}
{active_profile['focus']}
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
    res = call_llm(prompt)
    content = get_content(res, "Step 4")
    if content:
        parsed = parse_or_show(content, "Step 4")
        if parsed:
            final = parsed.get("pqrs", [])
            scored_lookup = [(s["symptom"].lower(), s["score"]) for s in scored]
            for item in final:
                if item.get("score", 0) == 0:
                    sym_words = set(item.get("symptom", "").lower().split())
                    best_score, best_overlap = 0, 0
                    for cand_text, cand_score in scored_lookup:
                        overlap = len(sym_words & set(cand_text.split()))
                        if overlap > best_overlap:
                            best_overlap, best_score = overlap, cand_score
                    if best_overlap > 0:
                        item["score"] = best_score
            st.session_state.data["final_pqrs"] = final

            if not final:
                st.warning("No PQRS symptoms returned. Check scored candidates.")
                st.stop()

            st.subheader(f"🎯 Final PQRS Symptoms  [{active_case}]")
            if cc:
                st.caption(f"Chief complaint excluded: **{cc}**")
            st.markdown("---")
            for i, item in enumerate(final, 1):
                st.markdown(f"**{i}. {item.get('symptom', '')}**")
                if item.get("rubric"):
                    st.markdown(f"&nbsp;&nbsp;&nbsp;📖 `{item['rubric']}`")
                st.caption(f"Score: {item.get('score', 0)} | {item.get('reason', '')}")
            st.markdown("---")
            st.json(parsed)

# ================================================================
# STEP 5 — PQRS CLINICAL VALIDATOR  (LLM review of Step 4 output)
# ================================================================
if st.button("5️⃣  Validate PQRS (Clinical Review)"):
    final_pqrs = st.session_state.data.get("final_pqrs", [])
    if not final_pqrs:
        st.warning("Run Step 4 first.")
        st.stop()

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
{json.dumps(final_pqrs, indent=2)}

Return STRICT JSON:

{{
  "errors": [
    {{
      "symptom": "original symptom text",
      "issue":   "what is wrong with it",
      "fix":     "corrected clinical rubric or 'REMOVE'"
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
    res = call_llm(prompt)
    content = get_content(res, "Step 5")
    if content:
        parsed = parse_or_show(content, "Step 5")
        if parsed:
            errors = parsed.get("errors", [])
            corrected = parsed.get("corrected_pqrs", [])
            st.session_state.data["validated_pqrs"] = corrected

            st.subheader("🔍 Clinical Validation Report")
            st.markdown("---")

            if errors:
                st.error(f"**{len(errors)} issue(s) found:**")
                for e in errors:
                    with st.expander(f"❌ {e.get('symptom', '')}"):
                        st.markdown(f"**Issue:** {e.get('issue', '')}")
                        fix = e.get("fix", "")
                        if fix and fix.upper() != "REMOVE":
                            st.markdown(f"**Corrected rubric:** `{fix}`")
                        else:
                            st.markdown("**Action:** REMOVE from PQRS")
            else:
                st.success("No clinical errors found — PQRS is valid.")

            st.markdown("---")
            st.subheader("✅ Corrected Final PQRS")
            for i, item in enumerate(corrected, 1):
                st.markdown(f"**{i}. {item.get('symptom', '')}**")
                if item.get("rubric"):
                    st.markdown(f"&nbsp;&nbsp;&nbsp;📖 `{item['rubric']}`")
                st.caption(item.get("reason", ""))
            st.markdown("---")
            st.json(parsed)

# ================================================================
# STEP 6 — ADVANCED CLINICAL VALIDATOR  (reasoning-level review)
# ================================================================

if st.button("6️⃣  Advanced Validate (Reasoning Review)"):
    # Prefer Step 5 corrected output; fall back to Step 4
    pqrs_to_review = st.session_state.data.get("validated_pqrs") or st.session_state.data.get("final_pqrs", [])
    if not pqrs_to_review:
        st.warning("Run Step 4 (and optionally Step 5) first.")
        st.stop()

    cleaned_data = st.session_state.data.get("cleaned", {})
    cleaned      = cleaned_data.get("cleaned", "") if isinstance(cleaned_data, dict) else ""

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
{json.dumps(pqrs_to_review, indent=2)}

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
    res = call_llm(prompt)
    content = get_content(res, "Step 6")
    if content:
        parsed = parse_or_show(content, "Step 6")
        if parsed:
            errors      = parsed.get("errors", [])
            missing     = parsed.get("missing_pqrs", [])
            final       = parsed.get("final_pqrs", [])
            st.session_state.data["final_validated_pqrs"] = final

            st.subheader("🧠 Advanced Clinical Review")
            st.markdown("---")

            if errors:
                st.error(f"**{len(errors)} reasoning error(s) found:**")
                for e in errors:
                    with st.expander(f"❌ [{e.get('error_type','?')}]  {e.get('symptom','')}"):
                        st.markdown(f"**Why wrong:** {e.get('explanation','')}")
                        fix = e.get("corrected", "")
                        if fix and fix.upper() != "REMOVE":
                            st.markdown(f"**Corrected:** `{fix}`")
                        else:
                            st.markdown("**Action:** REMOVE")
            else:
                st.success("No reasoning errors found.")

            if missing:
                st.warning(f"**{len(missing)} stronger symptom(s) were missed:**")
                for m in missing:
                    with st.expander(f"⚠️ Missing: {m.get('symptom','')}"):
                        st.markdown(f"📖 `{m.get('rubric','')}`")
                        st.caption(m.get("reason", ""))

            st.markdown("---")
            st.subheader("✅ Final Validated PQRS")
            for i, item in enumerate(final, 1):
                st.markdown(f"**{i}. {item.get('symptom', '')}**")
                if item.get("rubric"):
                    st.markdown(f"&nbsp;&nbsp;&nbsp;📖 `{item['rubric']}`")
                st.caption(item.get("reason", ""))
            st.markdown("---")
            st.json(parsed)
