"""
AI Powered Clinical Support — FastAPI + Embedded UI
Run: uvicorn main:app --reload
  or: python main.py
"""

import os
import re
import base64
import tempfile
import requests
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from rag import search_remedies, rag_available, validate_pqrs_line
    HAS_RAG = True
except ImportError:
    HAS_RAG = False
    def search_remedies(q, n=6): return ""
    def rag_available(): return False
    def validate_pqrs_line(line, threshold=0.38): return {"confirmed": False, "score": 0.0, "remedy": "", "source": ""}

from pqrs import (
    extract_symptoms, score_symptoms, generate_pqrs,
    validate_pqrs, advanced_validate, CASE_PROFILES,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(APP_DIR, ".env"))

# Ollama local server — set OLLAMA_BASE_URL in .env or default to localhost
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "gemma3:12b")

app = FastAPI(title="DR - AI Powered Clinical Support")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class StripApiMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.scope.get('path', '')
        if path.startswith('/api/'):
            request.scope['path'] = path[4:]
            request.scope['raw_path'] = request.scope['path'].encode()
        return await call_next(request)

app.add_middleware(StripApiMiddleware)

REPORT_TXT_PATH = os.path.join(tempfile.gettempdir(), "doctor_assistant_report.txt")
REPORT_HTML_PATH = os.path.join(tempfile.gettempdir(), "doctor_assistant_report.html")
report_cache: dict = {"txt": REPORT_TXT_PATH, "html": REPORT_HTML_PATH}


# ============================================================
# PYDANTIC MODELS
# ============================================================
class CaseRequest(BaseModel):
    case: str

class StepRequest(BaseModel):
    case_data: str
    step12: str = ""
    raw_case: str = ""

class Step4Request(BaseModel):
    case_data: str
    step1: str
    step12: str
    step15: str
    step2: str
    step3: str
    step3_filtered: str

class PqrsExtractRequest(BaseModel):
    case_data: str
    case_type: str = "Mixed (Acute + Chronic)"
    chief_complaint: str = ""

class PqrsGenerateRequest(BaseModel):
    scored: list
    cleaned: str
    case_type: str = "Mixed (Acute + Chronic)"
    chief_complaint: str = ""

class PqrsValidateRequest(BaseModel):
    pqrs: list

class PqrsAdvancedRequest(BaseModel):
    pqrs: list
    cleaned: str

class PqrsRagRequest(BaseModel):
    pqrs: list

class ReportBuildRequest(BaseModel):
    case_input: str = ""
    step1: str
    step12: str = ""
    step15: str = ""
    step2: str = ""
    pqrs_text: str = ""
    confirmations_text: str = ""
    remedies_text: str = ""


# ============================================================
# LLM
# ============================================================
async def call_llm(prompt: str) -> str:
    data = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": (
                "You are a senior homeopathy doctor.\n"
                "STRICT RULES:\n- Only structured output\n- No reasoning\n"
                "- No paragraphs\n- Only bullet format\n- Start directly from required section"
            )},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 6000,
    }
    last_error = ""
    async with httpx.AsyncClient(timeout=180) as client:
        for _ in range(3):
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=data,
            )
            if resp.status_code in (502, 503, 504):
                last_error = f"[ERROR] vLLM call failed ({resp.status_code}): {resp.text}"
                continue
            if resp.status_code != 200:
                return f"[ERROR] vLLM call failed ({resp.status_code}): {resp.text}"
            body = resp.json()
            if "choices" not in body:
                return f"[ERROR] Unexpected vLLM response: {body}"
            break
        else:
            return last_error
    content = (body["choices"][0]["message"].get("content") or "").strip()
    if not content:
        return "[ERROR] No response generated"
    for marker in ["Patient Details:", "Chief Complaints:"]:
        if marker in content:
            content = marker + content.split(marker)[1]
            break
    content = content.replace("**", "").replace("*", "")
    return "\n".join(line.strip() for line in content.split("\n") if line.strip())


# ============================================================
# VISION / PDF EXTRACTION
# ============================================================
async def extract_image_text(image_path: str) -> str:
    if not image_path:
        return ""
    try:
        ext  = os.path.splitext(image_path)[1].lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": (
                    "This is a homeopathic patient case note. "
                    "Transcribe ALL the text exactly as written — "
                    "preserve every word, number, and detail. Output plain text only."
                )}
            ]}],
            "temperature": 0.1,
            "max_tokens": 4000
        }
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
            )
        if resp.status_code != 200:
            return f"[ERROR] Vision API failed: {resp.text[:200]}"
        content = (resp.json()["choices"][0]["message"].get("content") or "").strip()
        return content
    except Exception as e:
        return f"[ERROR] Image extraction failed: {str(e)}"


async def extract_pdf_text(pdf_path: str) -> str:
    if not HAS_PDF:
        return "[ERROR] pdfplumber not installed"
    try:
        parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    parts.append(text.strip())
                else:
                    tmp = os.path.join(tempfile.gettempdir(), f"pdf_page_{i}.png")
                    try:
                        page.to_image(resolution=200).save(tmp)
                        v = await extract_image_text(tmp)
                        if v and not v.startswith("[ERROR]"):
                            parts.append(v)
                    finally:
                        try: os.remove(tmp)
                        except: pass
        extracted = "\n".join(parts).strip()
        return extracted if extracted else "[ERROR] No readable text found"
    except Exception as e:
        return f"[ERROR] Could not read PDF: {str(e)}"


# ============================================================
# STEP FUNCTIONS
# ============================================================
async def run_step1(case: str) -> str:
    prompt = f"""
You are a clinical data extraction engine for homeopathic case taking.

Your job is to convert the case into a COMPLETE and EXHAUSTIVE structured case sheet.

CRITICAL RULES:
- Extract EVERY single detail from the case — nothing should be skipped
- DO NOT summarize, shorten or paraphrase
- DO NOT skip any line or sentence
- DO NOT infer or assume anything not written
- If something is not explicitly mentioned → write "Not mentioned"
- Preserve ALL information exactly as given
- Every bullet point must carry real information

IMPORTANT:
- Extract ALL information including:
  - negative statements ("no", "none")
  - indirect mentions
  - partial or weak information
- Do NOT write "not mentioned" if any related info exists

FORMAT RULES:
- Each section must start on a new line with its heading
- Each point must be a bullet starting with "-"
- Sub-points use indented "-"
- No paragraph text
- No merging of sections

OUTPUT FORMAT:

Patient Details:
- Name:
- Age:
- Sex:
- Location/City:
- Occupation/Studies:

Chief Complaints:
- (List every complaint with duration)

History of Present Illness:
- (Full chronological history — when started, how progressed, every event)

Location of Complaints:
- (Exact body parts affected)

Sensation & Appearance:
- (How it looks, feels — describe skin/rash/symptom in detail)

Modalities:
- Worse:
  - (ONLY physical factors — weather, time, temperature, activity, clothing, touch, food, bathing, season, etc.)
  - Do NOT include mental/emotional triggers here — those go under Mental Profile
- Better:
  - (ONLY physical factors — cream, moisture, warmth, rest, time of day, clothing, etc.)
  - Do NOT include mental/emotional relief here — those go under Mental Profile

Associated / Concomitant Symptoms:
- (Symptoms that appear along with chief complaint)

Sleep:
- (Quality, disturbance, position, time of waking, reason)

Thermal Reaction:
- (Hot/Cold/Moderate preference)

General Symptoms:
- Appetite: (hunger, eating habits, food intake, meal frequency)
- Thirst: (water intake, hydration, drinks, fluid consumption, how much water)
- Stool: (bowel movements, constipation, loose, frequency)
- Urine: (frequency, colour, burning, quantity)
- Sweat: (perspiration, sweating, odour, location)
- Skin (general): (texture, dryness, oiliness, colour, sensitivity)

Food Preferences:
- Desires:
- Aversions:

Mental & Emotional Profile:
- Temperament:
- Mood & Mood Changes:
- Stress Response:
- Anger:
- Grief/Sadness:
- Joy/Happiness:
- Fears: (ANY statement about disliking, not tolerating, avoiding, or being disturbed by something — e.g. "does not like to be alone", "afraid of dark", "fears death" — ALL must be listed here)
- Ambitions/Expectations of Self:
- Procrastination/Laziness:
- Mental Stress & Triggers:

SPECIAL RULE FOR FEARS:
- ANY phrase like "does not like to be alone", "cannot tolerate", "avoids", "disturbed by", "afraid of" MUST be captured under Fears — do NOT write "Not mentioned" if such phrases appear anywhere in the case

Personality Profile:
- (Detailed character description — organized/disorganized, introvert/extrovert, strong-willed, etc.)

Social & Relationship History:
- Family Relationships:
- Friendships:
- Bullying/Conflicts:

Hobbies & Activities:
- (Sports, interests, what makes them happy or sad)

Past Medical History:
- (All past illnesses, treatments, hospitalizations)

Childhood History:
- (Illnesses, notable events)

Family History:
- (Every family member's illness — maternal and paternal side)

Miasmatic Indicators:
- (Hereditary, chronic, or recurring patterns)

Previous Treatments:
- (All past medicines, therapies, home remedies)

Current Medications:
- (All current medicines)

Investigations / Lab Reports:
- (Tests done, results mentioned)

Triggers Identified:
- (Every trigger — weather, stress, activity, clothing, food)

Negative Findings:
HARD LIMIT: Maximum 5 items and minimum 2. Write exactly the items confirmed absent, then STOP. Do not continue.
- ONLY include things the case EXPLICITLY confirmed as absent or ruled out with a direct statement
- A confirmed negative has an explicit quote like "no specific food trigger" or "no major illness except..."
- STRICTLY FORBIDDEN: Do NOT write "Not mentioned" under any circumstances
- STRICTLY FORBIDDEN: Do NOT list names, personal details, or demographic fields not provided
- STRICTLY FORBIDDEN: Do NOT list anything that was simply not discussed or not asked
- If fewer than 3 real confirmed negatives exist, list only those — leave the section short

Case:
{case}
"""
    return await call_llm(prompt)


async def check_local_params(raw_case: str) -> str:
    """Check complaint-specific (LOCAL) parameters — scans full case for complaint info."""
    prompt = f"""You are a clinical parameter scanner for homeopathic case intake.

TASK: For each parameter below, scan the case and answer ✓ or ✗.

RULE — ✓ if ANY meaningful information exists (direct, indirect, negative, or past):
- Onset / Trigger → ✓ if time of onset OR triggering cause is mentioned in any form
- Duration → ✓ if any time period or duration is mentioned
- Progress → ✓ if any change over time ("worse", "better", "same", "improving", "recurring", "spreading")
- Location & spread → ✓ if any body part or area is described
- Sensation → ✓ if any feeling, quality, or appearance of the complaint is described
- Modalities → ✓ if anything makes it better or worse (weather, time, activity, rest, etc.)
- Treatment taken → ✓ if any medicine, therapy, home remedy, or "no treatment taken" is mentioned
- Associated complaint Present/Absent → ✓ if any other symptom alongside chief complaint is mentioned
- Onset & Duration (associated) → ✓ if time info for any associated complaint is mentioned
- Sensation & Modalities (associated) → ✓ if any sensation or modality for associated complaint is mentioned

✗ ONLY if completely absent — zero connection anywhere in the case.

Output ONLY this structure with ✓ or ✗. No explanations.

CHIEF COMPLAINT:
- Onset / Trigger:
- Duration:
- Progress (increasing/decreasing):
- Location & spread:
- Sensation:
- Modalities (better/worse):
- Treatment taken so far:

ASSOCIATED COMPLAINTS:
- Present/Absent:
- Onset & Duration:
- Sensation & Modalities:
- Treatment taken:

CASE:
{raw_case}
"""
    return await call_llm(prompt)


async def check_global_params(raw_case: str) -> str:
    """Check global parameters — scans full case once for each parameter."""
    prompt = f"""You are a clinical parameter scanner for homeopathic case intake.

TASK: For each parameter below, scan the ENTIRE case and answer ✓ or ✗.

RULE — ✓ if ANY of the following exist for that parameter:
- Direct mention (positive or negative — "no fears" = ✓ Fears & anxieties, we know the answer)
- Indirect mention ("resolved on its own" → ✓ Outcome)
- Past event ("previously had anxiety" → ✓ Fears & anxieties, past tense still counts)
- Confirmed absent ("no past illness" → ✓ Past illnesses, knowing the answer = data present)

✗ ONLY if completely absent — not mentioned in any form, direct, indirect, negative, or past.

SPECIAL RULES:
- Maternal side → ✓ if "mother", "mom", or any maternal relative is mentioned with any health info
- Paternal side → ✓ if "father", "dad", or any paternal relative is mentioned with any health info
- Similar illness in family → ✓ if the same or related disease appears in any family member
- Thermal reaction → ✓ if body temperature preference is mentioned (fan, AC, weather, season, bathing). NOT food/drink temperature alone.
- MENSTRUAL HISTORY: detect sex from case — Male patient → N/A | Female with menstrual info → ✓ | Female with no menstrual info → ✗

Output ONLY this structure with ✓, ✗, or N/A. No explanations.

PAST HISTORY:
- Past illnesses with age:
- Surgeries:
- Outcome (cured/recurring/persisting):

FAMILY HISTORY:
- Maternal side:
- Paternal side:
- Similar illness in family:

MENSTRUAL HISTORY:
- Menstrual History:

PERSONAL HISTORY:
- Appetite:
- Food desires & aversions:
- Thirst (frequency or quantity):
- Sweat (location, odour, stain):
- Bowels (regularity, consistency):
- Urine (frequency, colour, burning):
- Sleep (hours, refreshing/unrefreshing):
- Sleep position:
- Dreams:
- Addictions:
- Thermal reaction (fan, AC, bathing, season):

LIFE SPACE:
- Family setup & relations:
- Work / Study environment:
- Upbringing & life difficulties:

MENTAL GENERALS:
- Major stress events:
- Anxiety / nervousness / confidence:
- Anger triggers & expression:
- Fears & anxieties:
- Sensitivities:
- Emotional reactions:
- Childhood & scholastic performance:

Hobbies described:

CASE:
{raw_case}
"""
    return await call_llm(prompt)


def merge_checklist(local_result: str, global_result: str) -> str:
    """Combine LOCAL (complaint) + GLOBAL results into full checklist."""
    return f"{local_result.strip()}\n\n{global_result.strip()}"


async def evidence_check(step12_output: str, case_data: str) -> str:
    """Re-checks only ✗ parameters against the raw case — all ✓ are preserved untouched."""
    lines = step12_output.split("\n")

    # Collect only the ✗ parameters for re-checking
    uncertain = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("-") and "✗" in stripped:
            param = stripped.lstrip("-").replace("✗", "").strip().rstrip(":")
            if param:
                uncertain.append(param)

    if not uncertain:
        return step12_output  # nothing to re-check

    param_list = "\n".join(f"- {p}" for p in uncertain)

    prompt = f"""You are re-checking parameters that were initially marked ✗ in a clinical checklist.

TASK: For each parameter below, scan the ENTIRE case. Does ANY meaningful info exist?

RULE — ✓ if ANY of the following exist:
- Direct mention (positive or negative — "no fears" = ✓, knowing the answer counts)
- Indirect mention ("resolved on its own" → ✓ Outcome)
- Past event ("previously had anxiety" → ✓ Fears & anxieties)
- Confirmed absent ("no past illness" → ✓ Past illnesses)

✗ ONLY if completely absent in any form — not present, not absent, not past, not indirect.

PARAMETERS TO RE-CHECK:
{param_list}

Reply in this exact format for each parameter:
- Parameter name: ✓ or ✗

CASE:
{case_data}
"""
    recheck_raw = await call_llm(prompt)

    # Parse re-check results into a corrections dict
    corrections: dict[str, str] = {}
    for line in recheck_raw.split("\n"):
        stripped = line.strip().lstrip("-").strip()
        if "✓" in stripped:
            key = stripped.replace("✓", "").strip().rstrip(":").strip().lower()
            corrections[key] = "✓"
        elif "✗" in stripped:
            key = stripped.replace("✗", "").strip().rstrip(":").strip().lower()
            corrections[key] = "✗"

    # Apply corrections — only touch lines that were ✗
    result_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("-") and "✗" in stripped:
            param_lower = stripped.lstrip("-").replace("✗", "").strip().rstrip(":").strip().lower()
            for key, symbol in corrections.items():
                if key in param_lower or param_lower in key:
                    line = line.replace("✗", symbol)
                    break
        result_lines.append(line.rstrip().replace("~", "✗").replace("?", ""))

    return "\n".join(result_lines)


def _extract_missing_params(step12: str) -> list:
    missing = []
    for line in step12.split("\n"):
        line = line.strip()
        if line.startswith("-") and "✗" in line:
            param = line.lstrip("-").replace("✗", "").strip().rstrip(":").strip()
            if param:
                missing.append(param)
    return missing


def _has_minimal_extracted_value(structured_output: str, param: str, raw_case: str = "") -> bool:
    """
    Conservative safety check for Step 1.5.
    If Step 1 already extracted any real value for a parameter, do not report it missing.
    """
    if not structured_output or not param:
        return False

    combined_text = f"{structured_output}\n{raw_case}".lower()
    param_lower = param.lower()
    keyword_map = {
        "urine": ["urine", "urinary"],
        "addictions": ["addictions", "smoking", "alcohol", "tobacco", "substance"],
        "dreams": ["dreams", "dream"],
        "treatment": [
            "treatment", "treated", "medicine", "medication", "previous treatments",
            "current medications", "diet", "bandage", "bandaging", "creatine",
            "supplement", "therapy", "managed", "management", "improves with",
        ],
        "outcome": ["outcome", "cured", "recurring", "persisting", "improved", "healed", "treated", "resolved"],
        "appetite": ["appetite"],
        "thirst": ["thirst"],
        "sweat": ["sweat"],
        "bowels": ["stool", "bowel"],
        "sleep": ["sleep"],
    }

    keys = []
    for marker, aliases in keyword_map.items():
        if marker in param_lower:
            keys.extend(aliases)
    if not keys:
        keys = [re.sub(r"\s*\(.*?\)", "", param_lower).split("/")[0].strip()]

    absent_values = ("not mentioned", "not stated", "unknown", "nil", "n/a", "na", "")
    minimal_patterns = {
        "urine": [
            r"\burine\s*[:\-]\s*(normal|clear|yellow|pale|dark|frequent|burning|no burning)",
            r"\burinary\s+(normal|complaint|frequency|burning)",
        ],
        "addictions": [
            r"\baddictions?\s*[:\-]\s*(none|nil|no|absent|smoking|alcohol|tobacco)",
            r"\b(no|denies)\s+(smoking|alcohol|tobacco|addiction)",
            r"\b(smoking|alcohol|tobacco|cigarette|gutkha)\b",
        ],
        "dreams": [
            r"\bdreams?\s*[:\-]\s*\S+",
            r"\bdreams?\s+(of|about)\b",
            r"\bdream\s+of\b",
            r"\bdreamt\b",
            r"\bnightmare\b",
        ],
        "treatment": [
            r"\b(treatment|treated|medicine|medication|tablet|therapy|bandag(?:e|ing)|diet|creatine|supplement|managed|management)\b",
            r"\b(improves?|better)\s+with\s+(diet|medicine|rest|therapy|bandag(?:e|ing))\b",
        ],
        "outcome": [
            r"\b(treated|healed|improved|resolved|cured|better|recovered)\b",
        ],
    }
    for marker, patterns in minimal_patterns.items():
        if marker in param_lower and any(re.search(pattern, combined_text) for pattern in patterns):
            return True

    for line in structured_output.splitlines():
        cleaned = line.strip().lstrip("-").strip()
        if ":" not in cleaned:
            continue
        label, value = cleaned.split(":", 1)
        label_lower = label.lower()
        value_lower = value.strip().lower()
        if any(key and key in label_lower for key in keys):
            if value_lower not in absent_values and not value_lower.startswith("not mentioned"):
                return True
    for key in keys:
        if key and re.search(rf"\b{re.escape(key)}\b", combined_text):
            nearby_absent = re.search(rf"\b{re.escape(key)}\b\s*[:\-]\s*(not mentioned|not stated|unknown)\b", combined_text)
            if not nearby_absent:
                return True
    return False


_PARAM_SECTIONS = {
    "CHIEF COMPLAINT":       ["Onset / Trigger", "Duration", "Progress (increasing/decreasing)",
                               "Location & spread", "Sensation", "Modalities (better/worse)",
                               "Treatment taken so far"],
    "ASSOCIATED COMPLAINTS": ["Present/Absent", "Onset & Duration",
                               "Sensation & Modalities", "Treatment taken"],
    "PAST HISTORY":          ["Past illnesses with age", "Surgeries",
                               "Outcome (cured/recurring/persisting)"],
    "FAMILY HISTORY":        ["Maternal side", "Paternal side", "Similar illness in family"],
    "MENSTRUAL HISTORY":     ["Menstrual History"],
    "PERSONAL HISTORY":      ["Appetite", "Food desires & aversions", "Thirst (frequency or quantity)",
                               "Sweat (location, odour, stain)", "Bowels (regularity, consistency)",
                               "Urine (frequency, colour, burning)", "Sleep (hours, refreshing/unrefreshing)",
                               "Sleep position", "Dreams", "Addictions",
                               "Thermal reaction (fan, AC, bathing, season)"],
    "LIFE SPACE":            ["Family setup & relations", "Work / Study environment",
                               "Upbringing & life difficulties"],
    "MENTAL GENERALS":       ["Major stress events", "Anxiety / nervousness / confidence",
                               "Anger triggers & expression", "Fears & anxieties",
                               "Sensitivities", "Emotional reactions",
                               "Childhood & scholastic performance"],
    "HOBBIES":               ["Hobbies described"],
}


def _group_missing_by_section(missing: list) -> str:
    """Format a flat list of missing params into heading → sub-heading groups."""
    grouped: dict[str, list] = {}
    unmatched: list = []

    for param in missing:
        found = False
        for section, params in _PARAM_SECTIONS.items():
            for p in params:
                if param.lower() in p.lower() or p.lower() in param.lower():
                    grouped.setdefault(section, []).append(p)
                    found = True
                    break
            if found:
                break
        if not found:
            unmatched.append(param)

    lines = ["Missing Parameters:"]
    for section, params in _PARAM_SECTIONS.items():
        if section in grouped:
            lines.append(f"\n{section}:")
            for p in grouped[section]:
                lines.append(f"  - {p}")
    if unmatched:
        lines.append("\nOTHER:")
        for p in unmatched:
            lines.append(f"  - {p}")
    return "\n".join(lines)


async def run_step15(step12: str = "", structured_output: str = "", raw_case: str = "") -> str:
    missing = _extract_missing_params(step12)
    if not missing:
        return "Missing Parameters:\n- None — all parameters are complete."

    missing = [
        param for param in missing
        if not _has_minimal_extracted_value(structured_output, param, raw_case)
    ]
    if not missing:
        return "Missing Parameters:\n- None — all parameters are complete."

    if not structured_output.strip():
        return _group_missing_by_section(missing)

    candidates = "\n".join(f"- {param}" for param in missing)
    prompt = f"""
You are a clinical data completeness auditor.

CASE:
{raw_case or structured_output}

EXTRACTED DATA:
{structured_output}

CHECKLIST:
{step12}

POSSIBLE MISSING PARAMETERS:
{candidates}

TASK:
Mark a parameter as missing ONLY if there is absolutely NO information present in the extracted data.

IMPORTANT RULES:
- If ANY valid information is present → DO NOT mark as missing
- Minimal information is still valid
- "Normal", "none", "treated", "healed", "improved", one dream, any diet/lifestyle change, bandaging, supplement, medicine, or therapy are valid information
- If a parameter has a value other than "Not mentioned" → DO NOT mark it missing

Examples:
- Urine: "normal" → NOT missing
- Addictions: "none" → NOT missing
- Dreams: even one dream → NOT missing
- Treatment: any past treatment → NOT missing
- Treatment: diet change, bandaging, supplement, or management response → NOT missing
- Outcome: "treated", "healed", "improved" → NOT missing

DO NOT:
- Require detailed structure
- Ignore extracted data
- Overthink

OUTPUT FORMAT — group by section heading, then sub-heading:

Missing Parameters:

SECTION HEADING:
  - Sub-parameter name

(Repeat for each section that has missing parameters)

If none are truly missing, write exactly:
Missing Parameters:
- None — all parameters are complete.
"""
    audited = (await call_llm(prompt)).strip()
    return audited if audited else "Missing Parameters:\n- None — all parameters are complete."


async def run_step2(case_data: str) -> str:
    prompt = f"""
You are an experienced homeopathic physician doing a follow-up interview.

Based on this case:
{case_data}

Ask ONLY the 5 most important questions to fill genuine gaps clinically relevant to remedy selection.

RULES:
- Maximum 5 questions
- Each question must explore an UNKNOWN area not yet covered in the case
- Questions must be OPEN-ENDED — ask what, where, when, how — never "do you have X symptom?"
- Do NOT ask leading questions that contain a specific symptom the patient has not mentioned
- Do NOT fish for remedy keynotes — e.g. do not ask "do you feel burning in the soles of your feet?" or "do you feel better after passing stool?" unless the patient has already mentioned these
- Do NOT suggest symptoms through your questions — this biases the case
- Questions must come from genuine gaps: missing modalities, unexplored sensations, unknown sleep details, unconfirmed fears, unclear causation
- No repetition of what is already known
- Pay attention to patient's sex and age — do NOT ask gender-irrelevant questions
- If patient is male — do NOT ask about menstruation, periods, or female-specific symptoms

FORMAT:

Top 5 Questions to Ask the Patient:
1. ...
2. ...
3. ...
4. ...
5. ...
"""
    return await call_llm(prompt)


async def run_step3(case_data: str) -> str:
    prompt = f"""
You are a senior classical homeopathic physician.

From this case:
{case_data}

═══════════════════════════════════════════════
TASK 1 — PQRS Symptoms (Peculiar, Queer, Rare, Strange)
═══════════════════════════════════════════════

PQRS are the DIFFERENTIATING symptoms — what makes THIS patient
different from every other patient with the same condition.

SELF-CHECK — Before writing each line, answer these 3 questions:
1. Is this something the PATIENT directly experiences?
   (Not a treatment effect, not a medication observation, not a doctor's finding)
2. Is this unexpected or unusual for this condition?
   (Would the majority of patients with this condition have it? If YES → not PQRS)
3. Would an experienced homeopath find this clinically surprising?
Only write the line if ALL 3 answers are YES.

WHAT QUALIFIES AS PQRS:
- Peculiar causation — the specific event or exposure that originally triggered the illness
- Peculiar modality combination — unusual pairing of aggravation AND amelioration together
  (a single common modality alone does not qualify — the specific combination must be unusual)
- Time-specific aggravation — symptom worsens at a specific clock hour
  (e.g. 3 AM, 4 AM, midnight, 11 PM) — "at night" alone is too vague and does NOT qualify
- Action-response PQRS — peculiar relief or worsening from a specific action
  (e.g. relief only after scratching until bleeding, worse immediately after bathing,
   better after eating, worse after first motion but better with continued motion)
  The action-to-response link must be specific and clinically unexpected
- Thermal contradiction — a reaction that contradicts the patient's own general thermal state
  (e.g. a chilly patient who wants cold applications on the affected part)
- Sensation paradox — an unusual quality of sensation unexpected for the condition
  (e.g. burning relieved by heat in an inflammatory condition)

CONCOMITANTS ARE VALID PQRS:
- A concomitant = a symptom that consistently appears TOGETHER WITH the chief complaint
  (e.g. cough + cold EVERY TIME skin flares during weather change — the two systems react in sync)
- Concomitants are NOT common symptoms — they are peculiar linkages between body systems
- When the case shows two complaints triggered by the SAME cause at the SAME time → Grade 2 concomitant

STRICT EXCLUSIONS — do NOT write any of these:
- The main sensation of the condition itself as a STANDALONE line
  (itching in skin disease, pain in joints, breathlessness in asthma — common, not PQRS by themselves)
- Common aggravations without peculiar pairing
  (cold in joint pains alone, damp in skin alone — expected, not PQRS)
- Sleep disturbance directly caused by the main complaint as a standalone line
- Mental generals as standalone lines (temperament, mood, anger, personality)
  — UNLESS they show a peculiar causation link to the physical complaint
- Treatment history of any kind
- Medication effects or consequences
- Family history, social history, background as standalone lines

GRADING — label each PQRS line with its clinical tier before writing it:

[GRADE 1] — highest value, must match for remedy selection:
  • Time-specific aggravation at an exact clock hour (3 AM, 4 AM, midnight, 11 PM)
  • Peculiar causation (the specific event or exposure that triggered the illness)
  • Thermal contradiction (patient's reaction contradicts their own general thermal state)
  • Sensation paradox (unexpected sensation quality for this condition)

[GRADE 2] — strong value, should match:
  • Action-response PQRS (peculiar relief or worsening caused by a specific action)
  • Peculiar modality combination (unusual aggravation + amelioration pairing together)

[GRADE 3] — supporting value, confirms but does not select on its own:
  • Characteristic particular with a clear modality that is still unusual for this patient
  • Simple concomitant link between two systems under the same trigger

OUTPUT RULES:
- Maximum 5 candidates
- Each candidate MUST include three lines: SYMPTOM, CATEGORY, WHY PECULIAR
- Begin the SYMPTOM line with its grade label: [GRADE 1], [GRADE 2], or [GRADE 3]
- CATEGORY must be one of: causation | time-aggravation | modality-combination | action-response | thermal-contradiction | sensation-paradox | concomitant
- WHY PECULIAR must explain in one line why THIS patient's finding is unusual for THIS condition
- If fewer than 3 genuine PQRS exist, write only those — do NOT pad
- If no genuine PQRS exist, write exactly:
  No genuine PQRS identified — case needs more clinical data.

CANDIDATE FORMAT (repeat for each):
[GRADE N] <one clear sentence describing the symptom>
  CATEGORY: <one of the categories above>
  WHY PECULIAR: <one line of clinical reasoning>


═══════════════════════════════════════════════
TASK 2 — Totality of Symptoms
═══════════════════════════════════════════════

Totality = The complete symptom picture that together points to ONE remedy.

Include ALL of the following that are present in the case:
- Physical generals: thermal reaction, food desires/aversions, sleep, thirst, sweat, skin
- Mental generals: temperament, stress response, emotional triggers, fears, mood
- Characteristic particulars: exact location, sensation quality, modalities of the chief complaint
- PQRS symptoms (repeat from above)
- Miasmatic background: hereditary patterns, chronic/recurring nature

DO NOT include: background info, social history, hobbies, medications, treatment history.

FORMAT — use this exact structure:

PQRS (Peculiar, Queer, Rare, Strange):
- [GRADE 1] line 1
- [GRADE 2] line 2
(max 5 lines; label each with [GRADE 1], [GRADE 2], or [GRADE 3])

Totality of Symptoms:
- Physical Generals:
  - [point]
- Mental Generals:
  - [point]
- Characteristic Particulars:
  - [point]
- PQRS:
  - [repeat PQRS lines here with grade labels]
- Miasmatic Background:
  - [point]
"""
    return await call_llm(prompt)


def split_step3(result: str) -> tuple:
    """
    Splits LLM output into (pqrs_block, totality_block).
    Tries the primary marker first, then a fallback.
    Returns (full_result, "") if neither marker is found.
    """
    for marker in ["Totality of Symptoms:", "Totality:"]:
        if marker in result:
            parts = result.split(marker, 1)
            return parts[0].strip(), marker + "\n" + parts[1].strip()
    return result, ""


# ============================================================
# PQRS FILTER ENGINE (LLM-based, dynamic)
# ============================================================

# Safety net only — treatment/medication observations are NEVER valid PQRS.
# All other filtering is done by the context-aware LLM validator below.
TREATMENT_BLOCKLIST = [
    "ayurveda", "ayurvedic", "cyclosporine", "tofacitinib",
    "steroid", "corticosteroid", "immunosuppressant",
    "antihistamine", "cetirizine", "allegra", "nebulizer",
    "allopathic", "prescribed", "responded to", "under treatment",
    "on medication", "weaned", "managed with", "due to medication",
    "side effect", "pharmacological", "drug effect",
    "after starting medication", "after stopping medication",
    "since starting medication",
]


def _safety_net(line: str) -> bool:
    ll = line.lower()
    return any(kw in ll for kw in TREATMENT_BLOCKLIST)


def _strip_grade(line: str) -> str:
    for prefix in ("[GRADE 1] ", "[GRADE 2] ", "[GRADE 3] "):
        if line.startswith(prefix):
            return line[len(prefix):]
    return line


def _grade(line: str) -> int:
    match = re.search(r"\[GRADE\s+([123])\]", line, re.I)
    return int(match.group(1)) if match else 3


_SKIP_LINE = re.compile(r'^(TASK\s+\d|PQRS\s*\(|[\u2500\u2550=\-─═]{4,})', re.I)


def extract_pqrs_candidates(pqrs_block: str) -> list:
    """
    STAGE 1 PARSER — parses the structured PQRS block into candidate dicts.
    Each candidate: {grade, symptom, category, why, line}.
    Handles the "no genuine pqrs" passthrough separately.
    """
    candidates = []
    current = None

    for raw in pqrs_block.split("\n"):
        line = raw.strip().lstrip("-•*").strip()
        if not line:
            continue
        if _SKIP_LINE.match(line):
            continue
        if line.endswith(":") and not re.search(r"\[GRADE\s+[123]\]", line, re.I):
            continue
        if line.lower().startswith("no genuine pqrs"):
            return [{"grade": 0, "symptom": line, "category": "", "why": "", "line": line}]

        if re.match(r"^\[GRADE\s+[123]\]", line, re.I):
            if current:
                candidates.append(current)
            current = {
                "grade": _grade(line),
                "symptom": _strip_grade(line),
                "category": "",
                "why": "",
                "line": line,
            }
        elif current and re.match(r"^category\s*:", line, re.I):
            current["category"] = line.split(":", 1)[1].strip()
        elif current and re.match(r"^why(\s+peculiar)?\s*:", line, re.I):
            current["why"] = line.split(":", 1)[1].strip()

    if current:
        candidates.append(current)

    # Deduplicate by normalized symptom text
    seen = set()
    unique = []
    for c in candidates:
        key = c["symptom"].lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _parse_verdicts(text: str, expected: int) -> list:
    """Parses 'CANDIDATE N: ACCEPT/REJECT\\nREASON: ...' blocks from validator output."""
    verdicts = []
    pattern = re.compile(
        r"CANDIDATE\s+\d+\s*:\s*(ACCEPT|REJECT)[^\n]*\n\s*REASON\s*:\s*([^\n]+(?:\n(?!\s*CANDIDATE\s+\d).*)*)",
        re.I,
    )
    for m in pattern.finditer(text):
        verdicts.append({
            "verdict": m.group(1).strip().upper(),
            "reason": m.group(2).strip(),
        })
    while len(verdicts) < expected:
        verdicts.append({"verdict": "ACCEPT", "reason": "Validator returned no verdict — default accept."})
    return verdicts[:expected]


async def validate_pqrs_with_llm(case_data: str, candidates: list) -> list:
    """
    STAGE 2 — Context-aware LLM judge.
    Given the full case + candidates with reasoning, decides ACCEPT or REJECT per candidate.
    Returns candidates annotated with 'verdict' and 'verdict_reason'.
    """
    if not candidates:
        return []
    # "No genuine PQRS" passthrough — skip validation
    if len(candidates) == 1 and candidates[0]["grade"] == 0:
        return candidates

    block = ""
    for i, c in enumerate(candidates, 1):
        block += f"\nCANDIDATE {i}:\n"
        block += f"  SYMPTOM: [GRADE {c['grade']}] {c['symptom']}\n"
        block += f"  CATEGORY: {c['category'] or 'unspecified'}\n"
        block += f"  WHY PECULIAR: {c['why'] or 'not provided'}\n"

    prompt = f"""
You are a senior classical homeopath auditing PQRS candidates for remedy selection.

FULL CASE (context for judgement):
{case_data}

CANDIDATES EXTRACTED:
{block}

For EACH candidate, decide if it is a genuine PQRS for THIS patient's specific condition.

VALID PQRS (any of these qualifies):
- Specific causation event (grief, shock, exposure that triggered illness onset)
- Exact clock-hour aggravation (3 AM, 4 AM, midnight, 11 PM)
- Thermal contradiction (patient's own thermal reaction contradicted at affected part)
- Sensation paradox (sensation quality unexpected for the condition)
- Peculiar modality combination (unusual aggravation + amelioration pairing)
- Action-response link (specific action → specific relief or worsening)
- Concomitant — two systems consistently triggered TOGETHER by the same cause
  (e.g. skin flare + cough occur together during every weather change — this IS a valid concomitant PQRS, not a common symptom)

REJECT ONLY IF:
- Treatment or medication observation
- Standalone common symptom of the condition with no peculiar pairing or signal
- Background / social / family history not linked to causation
- Vague modality (e.g. "at night", "in evening") without exact time or peculiar combination
- Mental general (mood, anger, temperament) as standalone trait with no causation link to physical complaint
- Sleep disturbance caused directly by the chief complaint

OUTPUT FORMAT — EXACTLY one block per candidate, in order, nothing else:

CANDIDATE 1: ACCEPT
REASON: <one line explaining why>

CANDIDATE 2: REJECT
REASON: <one line explaining why>
"""
    response = await call_llm(prompt)
    verdicts = _parse_verdicts(response, len(candidates))

    for c, v in zip(candidates, verdicts):
        c["verdict"] = v["verdict"]
        c["verdict_reason"] = v["reason"]
    return candidates


def classify_pqrs_candidates(candidates: list) -> tuple:
    """
    STAGE 3 — Applies safety-net + uses LLM verdicts to split into accepted/rejected.
    If no Grade 1 or Grade 2 PQRS remains accepted, forces needs-more-data flow.
    """
    accepted, rejected = [], []

    for c in candidates:
        # "No genuine PQRS" passthrough
        if c["grade"] == 0:
            rejected.append({"line": c["line"], "reason": "The model found no differentiating symptom."})
            continue

        # Hard safety net — medication/treatment is never valid PQRS
        if _safety_net(c["symptom"]):
            rejected.append({"line": c["line"], "reason": "Treatment or medicine response is not PQRS."})
            continue

        verdict = c.get("verdict", "ACCEPT")
        reason = c.get("verdict_reason", "")
        if verdict == "ACCEPT":
            accepted.append(c["line"])
        else:
            rejected.append({"line": c["line"], "reason": reason or "Rejected by validator."})

    has_strong = any(_grade(l) in (1, 2) for l in accepted)
    needs_more_data = []
    if not has_strong:
        accepted = []
        needs_more_data.extend([
            "No Grade 1 or Grade 2 PQRS confirmed — ask for more differentiating details.",
            "Clarify exact time modalities, if any.",
            "Clarify peculiar aggravation/amelioration combinations.",
            "Clarify causation, concomitants, and thermal contradictions.",
        ])

    return accepted, rejected, needs_more_data


def has_accepted_pqrs(filtered_text: str) -> bool:
    """True only when the ACCEPTED PQRS section contains at least one real bullet."""
    accepted = filtered_text.split("REJECTED / NOT USED FOR REMEDY SELECTION:", 1)[0]
    return "- No genuine PQRS found" not in accepted and bool(re.search(r"^-\s+\[GRADE\s+[12]\]", accepted, re.I | re.M))


def tag_with_rag(pqrs_lines: list) -> list:
    """
    Tags each PQRS line with book confirmation if RAG is available.
    Returns lines untagged if RAG is not available.
    """
    if not rag_available():
        return pqrs_lines

    tagged = []
    for line in pqrs_lines:
        if line.lower().startswith("no genuine pqrs"):
            tagged.append(line)
            continue
        symptom = _strip_grade(line.strip())
        v = validate_pqrs_line(symptom)
        if v["confirmed"]:
            tag = f"  ✓ confirmed — {v['source']} (score {v['score']})"
        else:
            tag = "  ✗ not found in Boericke / Allen"
        tagged.append(f"{line.strip()}{tag}")
    return tagged


def format_pqrs_output(accepted_lines: list, rejected_lines: list, needs_more_data: list) -> str:
    parts = ["ACCEPTED PQRS:"]
    if accepted_lines:
        parts.append("\n".join(f"- {line}" for line in accepted_lines))
    else:
        parts.append("- No genuine PQRS found - case needs more clinical data.")

    parts.append("\nREJECTED / NOT USED FOR REMEDY SELECTION:")
    if rejected_lines:
        parts.append("\n".join(
            f"- {item['line']} | Reason: {item['reason']}" for item in rejected_lines
        ))
    else:
        parts.append("- None.")

    if needs_more_data:
        parts.append("\nNEEDS MORE DATA:")
        parts.append("\n".join(f"- {item}" for item in needs_more_data))

    return "\n".join(parts)


# ============================================================
# STEP 1.2 FINAL OVERRIDES
# ============================================================
def audit_step12(step12_output: str, case_data: str) -> str:
    """Deterministic cleanup only — no LLM call. Strips ~ and ? characters."""
    cleaned = []
    for line in step12_output.split("\n"):
        line = line.rstrip().replace("~", "✗")
        if line.strip().startswith("-"):
            line = line.replace("?", "")
        cleaned.append(line)
    return "\n".join(cleaned)


def auto_correct_step12(checklist: str, case_data: str) -> str:
    """
    Minimal deterministic safety layer.
    Keeps semantic LLM decisions intact and only fixes formatting, male menstrual N/A,
    and common hallucinated fields that have no explicit support in the case.
    """
    case_lower = case_data.lower()

    def _set(ln: str, sym: str) -> str:
        idx = ln.rfind(":")
        if idx == -1:
            return ln
        return ln[:idx + 1] + " " + sym

    def _any(patterns: list) -> bool:
        return any(re.search(pattern, case_lower) for pattern in patterns)

    male_pats = [
        r"\bmale\b", r"\bman\b", r"\bboy\b", r"\bhe\b",
        r"\bhis\b", r"\bmr\b", r"[mM]/\d+", r"\d+/[mM]",
    ]
    female_pats = [
        r"\bfemale\b", r"\bwoman\b", r"\bgirl\b", r"\bshe\b",
        r"\bher\b", r"\bmrs\b", r"\bms\b", r"\bmiss\b",
        r"[fF]/\d+", r"\d+/[fF]",
    ]
    dream_pats = [
        r"\bdreams?\s+(of|about)\b", r"\bdreamt\b", r"\bnightmare\b",
        r"\brecurring\s+dream\b", r"\bdream\s+content\b",
        r"\bdreams?\s*[:\-]\s*\S+", r"\bfalling\s+into\b", r"\bfalling\b",
        r"\bpit\b", r"\bchased\b", r"\bflying\b", r"\bdead\b", r"\bsnake\b",
        r"\bno\s+dreams?\b", r"\bno\s+significant\s+dreams?\b",
        r"\bdreams?\s+not\b", r"\bfrequent\s+dreams?\b", r"\bdreams?\b",
    ]
    hobby_pats = [
        r"\bhobb(y|ies)\b", r"\binterest(ed)?\s+in\b", r"\blikes?\s+(reading|music|painting|gardening|cooking|cricket|football|walking|cycling|swimming|photography|chess|yoga|dance|dancing|singing|drawing|travel|writing)\b",
        r"\benjoys?\s+(reading|music|painting|gardening|cooking|cricket|football|walking|cycling|swimming|photography|chess|yoga|dance|dancing|singing|drawing|travel|writing)\b",
    ]
    same_illness_map = [
        (r"\bcough\b", [r"\bcough\b", r"\bchronic\s+cough\b", r"\bbronchitis\b"]),
        (r"\bacidity\b|acid\s+reflux|gerd", [r"\bacidity\b", r"\bgastritis\b", r"\bgerd\b", r"\bacid\s+reflux\b", r"\bhyperacidity\b"]),
        (r"\bback\s+pain\b", [r"\bback\s+pain\b", r"\bback\s+issues?\b"]),
        (r"\bmigraine\b", [r"\bmigraine\b"]),
        (r"\bheadache\b", [r"\bheadache\b", r"\bmigraine\b"]),
        (r"\bdiabetes\b", [r"\bdiabetes\b", r"\bblood\s+sugar\b", r"\bdiabetic\b"]),
        (r"\bhypertension\b|high\s+blood\s+pressure", [r"\bhypertension\b", r"\bhigh\s+blood\s+pressure\b"]),
        (r"\beczema\b|atopic\s+dermatitis", [r"\beczema\b", r"\batopic\b", r"\bdermatitis\b"]),
        (r"\bpsoriasis\b", [r"\bpsoriasis\b"]),
        (r"\brhinitis\b|allergic\s+rhinitis", [r"\brhinitis\b", r"\ballergic\s+rhinitis\b"]),
        (r"\bkidney\s+stone\b|renal\s+calculi", [r"\bkidney\s+stone\b", r"\brenal\s+calculi\b"]),
        (r"\bthyroid\b", [r"\bthyroid\b", r"\bhypothyroid\b", r"\bhyperthyroid\b"]),
        (r"\barthritis\b", [r"\barthritis\b", r"\bjoint\s+pain\b"]),
        (r"\bchest\s+pain\b", [r"\bchest\s+pain\b", r"\bchest\s+discomfort\b", r"\bchest\s+tightness\b"]),
        (r"\bangina\b", [r"\bangina\b", r"\banginal\b"]),
    ]

    is_male = _any(male_pats)
    is_female = _any(female_pats)
    has_dream = _any(dream_pats)
    has_hobby = _any(hobby_pats)

    first_sent = re.split(r"[.\n]", case_lower)[0]
    same_illness_synonyms = None
    for chief_pattern, synonyms in same_illness_map:
        if re.search(chief_pattern, first_sent):
            same_illness_synonyms = synonyms
            break
    family_rel = r"\b(father|mother|sister|brother|grandfather|grandmother|uncle|aunt|parent)\b"
    family_text = " ".join(
        sent for sent in re.split(r"[.\n]", case_lower)
        if re.search(family_rel, sent)
    )
    has_same_family_illness = bool(
        same_illness_synonyms and any(re.search(pattern, family_text) for pattern in same_illness_synonyms)
    )

    cleaned = []
    for raw_line in checklist.split("\n"):
        line = raw_line.rstrip().replace("~", "✗")
        if line.strip().startswith("-"):
            line = line.replace("?", "")

        stripped = line.strip()
        if stripped.startswith("- Menstrual History:") and is_male and not is_female:
            line = _set(line, "N/A")
        elif stripped.startswith("- Dreams:") and "✓" in line and not has_dream:
            line = _set(line, "✗")
        elif stripped.startswith("- Hobbies") and "✓" in line and not has_hobby:
            line = _set(line, "✗")
        elif stripped.startswith("- Similar illness") and same_illness_synonyms:
            if has_same_family_illness and "✗" in line:
                line = _set(line, "✓")
            elif not has_same_family_illness and "✓" in line:
                line = _set(line, "✗")

        cleaned.append(line)

    return "\n".join(cleaned)


# ============================================================
# REPORT GENERATION
# ============================================================
def format_section_html(title: str, content: str) -> str:
    lines = content.strip().split("\n")
    html = f'<div class="section"><h2>{title}</h2>'
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.endswith(":") and not line.startswith("-"):
            html += f'<h3>{line}</h3>'
        elif line.startswith("-"):
            html += f'<p class="bullet">{line}</p>'
        else:
            html += f'<p>{line}</p>'
    return html + '</div>'


def build_reports(case_input, step1, step12, step15, step3, confirmations_text, remedies_text):
    full = f"""DOCTOR ASSISTANT - COMPLETE CASE REPORT
=====================================

CASE INPUT
{case_input}

=====================================

STEP 1: CASE ANALYSIS
{step1}

=====================================

STEP 2: PARAMETER COMPLETENESS CHECKLIST
{step12}

=====================================

STEP 3: MISSING DATA OUTPUT
{step15}

=====================================

STEP 4: PQRS ANALYSIS
{step3}

=====================================

STEP 5: BOOK EVIDENCE
{confirmations_text}

=====================================

STEP 6: REMEDY SEARCH
{remedies_text}
"""
    txt_path = REPORT_TXT_PATH
    with open(txt_path, "w", encoding="utf-8-sig") as f:
        f.write(full.replace("✓", "Y").replace("✗", "N"))

    html_path = REPORT_HTML_PATH
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>DR - AI Powered Clinical Support Report</title>
<style>
  body{{font-family:Georgia,serif;max-width:900px;margin:40px auto;color:#1a1a2e;padding:20px}}
  .header{{background:linear-gradient(135deg,#1e3a8a,#3b82f6);color:white;padding:30px;border-radius:12px;margin-bottom:30px}}
  .header h1{{margin:0;font-size:28px}} .header p{{margin:5px 0 0;opacity:.85;font-size:13px}}
  .section{{background:#f8fafc;border-left:4px solid #3b82f6;padding:20px 25px;margin-bottom:24px;border-radius:0 8px 8px 0;page-break-inside:avoid}}
  .section h2{{color:#1e3a8a;font-size:16px;text-transform:uppercase;margin:0 0 15px}}
  .section h3{{color:#374151;font-size:14px;margin:14px 0 6px}}
  .section p{{margin:4px 0;font-size:13px;color:#374151;line-height:1.6}}
  .footer{{text-align:center;color:#9ca3af;font-size:11px;margin-top:40px;padding-top:20px;border-top:1px solid #e5e7eb}}
  @media print{{.header{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}}}
</style></head><body>
<div class="header"><h1>DR - AI Powered Clinical Support</h1><p>Clinical Decision Support Report</p></div>
{format_section_html("Case Input", case_input)}
{format_section_html("Step 1: Case Analysis", step1)}
{format_section_html("Step 2: Parameter Completeness Checklist", step12)}
{format_section_html("Step 3: Missing Data Output", step15)}
{format_section_html("Step 4: PQRS Analysis", step3)}
{format_section_html("Step 5: Book Evidence", confirmations_text)}
{format_section_html("Step 5: Remedy Search", remedies_text)}
<div class="footer">Generated by DR - AI Powered Clinical Support &mdash; Clinical Decision Support System</div>
</body></html>""")

    return txt_path, html_path


# ============================================================
# API ENDPOINTS
# ============================================================
@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    tmp = os.path.join(tempfile.gettempdir(), f"upload_{file.filename}")
    try:
        with open(tmp, "wb") as f:
            f.write(await file.read())
        return {"text": await extract_pdf_text(tmp)}
    finally:
        try: os.remove(tmp)
        except: pass


@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
    tmp = os.path.join(tempfile.gettempdir(), f"upload_img{ext}")
    try:
        with open(tmp, "wb") as f:
            f.write(await file.read())
        return {"text": await extract_image_text(tmp)}
    finally:
        try: os.remove(tmp)
        except: pass


@app.post("/case/analyse")              # Step 1 — full structured case extraction
async def api_case_analyse(req: CaseRequest):
    return {"result": await run_step1(req.case)}

@app.post("/case/completeness")         # Step 1.2 — parameter completeness checklist (✓ ✗)
async def api_case_completeness(req: StepRequest):
    raw_case = req.raw_case or req.case_data
    local_result  = await check_local_params(raw_case)
    global_result = await check_global_params(raw_case)
    merged    = merge_checklist(local_result, global_result)
    validated = await evidence_check(merged, raw_case)
    combined_case = req.case_data
    if req.raw_case.strip():
        combined_case = f"""RAW CASE:
{req.raw_case}

STRUCTURED EXTRACTION:
{req.case_data}
"""
    audited = audit_step12(validated, combined_case)
    final   = auto_correct_step12(audited, combined_case)
    return {"result": final}

@app.post("/case/missing-data")         # Step 2.5 — extracted-data-driven missing data audit
async def api_case_missing_data(req: StepRequest):
    return {"result": await run_step15(req.step12, req.case_data, req.raw_case)}

@app.post("/case/pqrs")                 # Step 3 — PQRS extraction + totality of symptoms
async def api_case_pqrs(req: StepRequest):
    raw_output           = await run_step3(req.case_data)                       # Stage 1 — extract
    pqrs_block, totality = split_step3(raw_output)
    candidates           = extract_pqrs_candidates(pqrs_block)
    validated            = await validate_pqrs_with_llm(req.case_data, candidates)  # Stage 2 — LLM judge
    accepted, rejected, needs_more_data = classify_pqrs_candidates(validated)
    tagged_lines         = tag_with_rag(accepted)                         # Stage 3 — book evidence
    filtered_text        = format_pqrs_output(tagged_lines, rejected, needs_more_data)
    return {"pqrs_raw": pqrs_block, "filtered": filtered_text, "totality": totality}

@app.post("/case/remedies")             # Step 4 — top 3 remedy suggestions from validated PQRS
async def api_case_remedies(req: Step4Request):
    _, totality_only = split_step3(req.step3)

    if not has_accepted_pqrs(req.step3_filtered):
        result = """No remedy selected.

Reason:
- No accepted Grade 1 or Grade 2 PQRS is available.
- Selecting a remedy from common symptoms would be unreliable.

Next clinical step:
- Ask for exact time modalities, peculiar aggravation/amelioration, causation, concomitants, and thermal contradictions.
- Re-run PQRS after collecting differentiating symptoms."""
        txt_path, html_path = build_reports(
            req.case_data, req.step1, req.step12, req.step15,
            req.step3, req.step3_filtered, result
        )
        report_cache["txt"]  = txt_path
        report_cache["html"] = html_path
        return {"result": result}

    # RAG: search books using accepted PQRS lines only (no headers/rejected lines)
    rag_section = ""
    if rag_available():
        accepted_lines = [
            l.lstrip("- ").strip()
            for l in req.step3_filtered.splitlines()
            if l.strip().startswith("- ") and
               "REJECTED" not in l and
               "NEEDS MORE" not in l and
               "No genuine" not in l
        ]
        query    = "\n".join(accepted_lines) + "\n" + totality_only
        rag_hits = search_remedies(query, n=6)
        if rag_hits and not rag_hits.startswith("[RAG"):
            rag_section = f"""
BOOK REFERENCES — Boericke, Kent, Allen, Vithoulkas, Sankaran:
(Use these as supporting evidence when matching remedies to PQRS)
{rag_hits}
"""

    prompt = f"""
You are a senior classical homeopathic physician.

Based on the PQRS & Totality below, suggest the top 3 remedies.

VALIDATED PQRS:
Use ONLY the ACCEPTED PQRS section below as the PRIMARY basis.
Do NOT use anything under REJECTED / NOT USED FOR REMEDY SELECTION or NEEDS MORE DATA.
{req.step3_filtered}

TOTALITY OF SYMPTOMS (for supporting match only — Physical Generals, Mental Generals, Miasmatic background):
{totality_only}
{rag_section}
FULL CASE (for context only):
{req.case_data}

GRADING PRIORITY — the PQRS lines are labelled [GRADE 1], [GRADE 2], [GRADE 3]:
- [GRADE 1] symptoms MUST match — a remedy that cannot cover a GRADE 1 symptom is disqualified
- [GRADE 2] symptoms SHOULD match — strong supporting evidence
- [GRADE 3] symptoms are supporting confirmation only — do not select on GRADE 3 alone
- If no GRADE 1 symptoms are present, GRADE 2 symptoms become the must-match criteria

RULES FOR REMEDY SELECTION:
- The PRIMARY basis for selecting each remedy must be the VALIDATED PQRS symptoms listed above
- Each remedy must directly match at least one VALIDATED PQRS symptom — if it cannot, do not include it
- Use only accepted Grade 1 or Grade 2 PQRS for remedy selection
- Never select a remedy from rejected symptoms, common symptoms, or needs-more-data prompts
- After matching PQRS, you may support with Physical Generals and Mental Generals from the Totality
- If BOOK REFERENCES are provided above, use them as evidence to strengthen or confirm your match
- Common symptoms of the condition (discharge, redness, itching, pain, swelling) are NOT valid primary match reasons — these are found in all remedies covering this condition
- Do NOT use treatment history, medication effects, or past therapy as PQRS justification
- Do NOT suggest a remedy only because it covers the diagnosis — it must match the VALIDATED PQRS

FORMAT:
1. Remedy Name
   - GRADE match: (which [GRADE 1] / [GRADE 2] PQRS symptom this remedy covers — be specific)
   - Book evidence: (quote from Boericke/Allen if available)
   - Supporting match: (Physical generals / Mental generals that also fit)

2. Remedy Name
   - PQRS match: ...
   - Book evidence: ...
   - Supporting match: ...

3. Remedy Name
   - PQRS match: ...
   - Book evidence: ...
   - Supporting match: ...
"""
    result = await call_llm(prompt)
    txt_path, html_path = build_reports(
        req.case_data, req.step1, req.step12, req.step15,
        req.step3, req.step3_filtered, result
    )
    report_cache["txt"]  = txt_path
    report_cache["html"] = html_path
    return {"result": result}


@app.get("/download/txt")
async def download_txt():
    path = report_cache.get("txt") or REPORT_TXT_PATH
    if not path or not os.path.exists(path):
        return JSONResponse({"error": "Generate Step 4 first"}, status_code=404)
    return FileResponse(path, media_type="text/plain", filename="doctor_assistant_report.txt")

@app.get("/download/html")
async def download_html():
    path = report_cache.get("html") or REPORT_HTML_PATH
    if not path or not os.path.exists(path):
        return JSONResponse({"error": "Generate Step 4 first"}, status_code=404)
    return FileResponse(path, media_type="text/html", filename="doctor_assistant_report.html")


@app.post("/pqrs/extract")
def api_pqrs_extract(req: PqrsExtractRequest):
    """Step 3a — LLM extract+tag symptoms, then Python score."""
    result = extract_symptoms(req.case_data, req.chief_complaint)
    if result["error"]:
        return {"error": result["error"], "symptoms": [], "scored": [], "skipped": []}
    symptoms = result["symptoms"]
    scored, skipped = score_symptoms(symptoms, req.case_type, req.chief_complaint)
    return {"symptoms": symptoms, "scored": scored, "skipped": skipped,
            "case_type": req.case_type, "min_score": CASE_PROFILES.get(req.case_type, {}).get("min_score", 4)}


@app.post("/pqrs/generate")
def api_pqrs_generate(req: PqrsGenerateRequest):
    """Step 3b — LLM selects final PQRS from scored candidates."""
    result = generate_pqrs(req.scored, req.cleaned, req.case_type, req.chief_complaint)
    if result["error"]:
        return {"error": result["error"], "pqrs": []}
    return {"pqrs": result["pqrs"], "cleaned": req.cleaned}


@app.post("/pqrs/validate")
def api_pqrs_validate(req: PqrsValidateRequest):
    """Step 3c — LLM clinical validation of PQRS."""
    result = validate_pqrs(req.pqrs)
    if result["error"]:
        return {"error": result["error"], "errors": [], "corrected_pqrs": req.pqrs}
    return {"errors": result["errors"], "corrected_pqrs": result["corrected_pqrs"]}


@app.post("/pqrs/advanced")
def api_pqrs_advanced(req: PqrsAdvancedRequest):
    """Step 3d — LLM advanced reasoning review."""
    result = advanced_validate(req.pqrs, req.cleaned)
    if result["error"]:
        return {"error": result["error"], "errors": [], "missing_pqrs": [], "final_pqrs": req.pqrs}
    return {"errors": result["errors"], "missing_pqrs": result["missing_pqrs"],
            "final_pqrs": result["final_pqrs"]}


@app.post("/pqrs/rag")
def api_pqrs_rag(req: PqrsRagRequest):
    """Step 4 — RAG book confirmation per line + full remedy search."""
    if not HAS_RAG or not rag_available():
        return {"available": False, "confirmations": [], "remedies": "RAG not available."}
    confirmations = []
    for item in req.pqrs:
        line = item.get("rubric") or item.get("symptom", "")
        if not line:
            continue
        v = validate_pqrs_line(line)
        confirmations.append({
            "line":      line,
            "confirmed": v["confirmed"],
            "remedy":    v.get("remedy", ""),
            "source":    v.get("source", ""),
            "score":     v.get("score", 0.0),
        })
    symptom_lines = [c["line"] for c in confirmations if c["line"]]
    remedies = search_remedies("\n".join(symptom_lines), n=5) if symptom_lines else ""
    return {"available": True, "confirmations": confirmations, "remedies": remedies}


@app.post("/report/build")
def api_report_build(req: ReportBuildRequest):
    """Builds TXT + HTML report from the new PQRS engine flow and caches for download."""
    txt_path, html_path = build_reports(
        req.case_input, req.step1, req.step12, req.step15,
        req.pqrs_text, req.confirmations_text, req.remedies_text
    )
    report_cache["txt"]  = txt_path
    report_cache["html"] = html_path
    return {"ok": True}


# ============================================================
# AUTH / ADMIN PROXY  (forward to auth-backend on port 8000)
# ============================================================
AUTH_BACKEND = "http://localhost:8000"
FRONTEND_DIR = os.path.join(APP_DIR, "dist")

def _proxy(method, url, request_headers, body, params):
    fwd_headers = {k: v for k, v in request_headers.items()
                   if k.lower() not in ('host', 'content-length')}
    resp = requests.request(method=method, url=url, headers=fwd_headers,
                            content=body, params=params, timeout=30)
    try:
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception:
        return JSONResponse(content={"detail": resp.text}, status_code=resp.status_code)

@app.api_route("/auth/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_auth(path: str, request: Request):
    body = await request.body()
    return _proxy(request.method, f"{AUTH_BACKEND}/auth/{path}",
                  request.headers, body, dict(request.query_params))

@app.api_route("/admin/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_admin(path: str, request: Request):
    body = await request.body()
    return _proxy(request.method, f"{AUTH_BACKEND}/admin/{path}",
                  request.headers, body, dict(request.query_params))


# ============================================================
# FRONTEND — serve React build
# ============================================================
if os.path.isdir(os.path.join(FRONTEND_DIR, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")

@app.get("/{full_path:path}", include_in_schema=False)
async def serve_react(full_path: str):
    if os.path.isdir(FRONTEND_DIR):
        candidate = os.path.join(FRONTEND_DIR, full_path)
        if os.path.isfile(candidate):
            return FileResponse(candidate)
        index = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index):
            return FileResponse(index)
    return JSONResponse({"error": "Frontend not built. Run: npm run build in doctor-app/frontend"}, status_code=404)

# LEGACY EMBEDDED UI (kept below, no longer served)
def _old_embedded_html():
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Powered Clinical Support</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body{font-family:'Inter','Segoe UI',sans-serif;}
  .output-box{font-family:'Courier New',monospace;font-size:13px;line-height:1.7;white-space:pre-wrap;background:#f8fafc;border:1.5px solid #bfdbfe;border-radius:10px;padding:16px;min-height:80px;max-height:520px;overflow-y:auto;color:#1e293b;}
  .step-card{background:white;border-radius:16px;box-shadow:0 2px 16px rgba(0,0,0,0.07);border:1px solid #e2e8f0;padding:28px 32px;margin-bottom:24px;}
  .step-badge{background:linear-gradient(135deg,#1e3a8a,#2563eb);color:white;border-radius:20px;padding:4px 16px;font-size:0.72rem;font-weight:700;letter-spacing:.5px;text-transform:uppercase;}
  .step-title{color:#1e3a8a;font-size:1.05rem;font-weight:700;}
  .btn-primary{background:linear-gradient(135deg,#1e3a8a,#2563eb);color:white;border:none;border-radius:10px;padding:10px 28px;font-weight:600;font-size:.95rem;cursor:pointer;box-shadow:0 4px 14px rgba(37,99,235,.3);transition:opacity .2s;display:inline-flex;align-items:center;gap:8px;}
  .btn-primary:hover{opacity:.88;} .btn-primary:disabled{opacity:.5;cursor:not-allowed;}
  .btn-dl{background:white;color:#1e3a8a;border:2px solid #bfdbfe;border-radius:10px;padding:9px 24px;font-weight:600;font-size:.9rem;cursor:pointer;text-decoration:none;display:inline-flex;align-items:center;gap:8px;transition:all .2s;}
  .btn-dl:hover{background:#eff6ff;border-color:#2563eb;}
  .lbl{font-weight:700;color:#1e3a8a;font-size:.78rem;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px;display:block;}
  .spinner{display:none;width:18px;height:18px;border:3px solid rgba(255,255,255,.4);border-top-color:white;border-radius:50%;animation:spin .8s linear infinite;}
  @keyframes spin{to{transform:rotate(360deg)}}
  .hidden{display:none!important;}
  textarea{border-radius:10px;border:1.5px solid #bfdbfe;font-size:.9rem;color:#1e293b;background:white;line-height:1.7;padding:12px 14px;width:100%;resize:vertical;outline:none;font-family:inherit;}
  textarea:focus{border-color:#2563eb;box-shadow:0 0 0 3px rgba(37,99,235,.12);}
  hr.div{border:none;border-top:2px solid #bfdbfe;margin-bottom:20px;}
  @keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
  @keyframes pulseGlow{0%,100%{box-shadow:0 2px 16px rgba(0,0,0,.07)}50%{box-shadow:0 4px 28px rgba(37,99,235,.22),0 0 0 3px rgba(37,99,235,.1)}}
  .step-card.running{border:2px solid #2563eb;animation:pulseGlow 2s ease-in-out infinite;}
  .output-box.fade-in{animation:fadeIn .35s ease;}
  .status-pill{font-size:.74rem;font-weight:700;padding:3px 12px;border-radius:20px;margin-left:4px;}
  .status-pill.running{background:#eff6ff;color:#2563eb;}
  .status-pill.done{background:#f0fdf4;color:#059669;}
  .timing{font-size:.73rem;color:#94a3b8;margin-top:5px;}
  #topBar{position:fixed;top:0;left:0;height:3px;background:linear-gradient(90deg,#2563eb,#38bdf8);width:0%;transition:width .45s ease;z-index:9999;}
</style>
</head>
<body class="bg-gray-50 min-h-screen">
<div id="topBar"></div>

<div style="background:linear-gradient(135deg,#1e3a8a 0%,#2563eb 60%,#3b82f6 100%);color:white;padding:32px 40px;box-shadow:0 4px 24px rgba(37,99,235,.3);">
  <div style="max-width:860px;margin:0 auto;">
    <h1 style="margin:0 0 6px;font-size:2rem;font-weight:800;">&#129504; AI Powered Clinical Support</h1>
    <p style="margin:0;font-size:.78rem;letter-spacing:2.5px;text-transform:uppercase;opacity:.78;">Clinical Decision Support &nbsp;·&nbsp; Step-by-Step Case Analysis</p>
  </div>
</div>

<div style="max-width:860px;margin:32px auto;padding:0 20px;">

  <!-- STEP 1 -->
  <div class="step-card" id="s1">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
      <span class="step-badge">Step 1</span>
      <span class="step-title">Case Analysis — Full Extraction</span>
      <span class="status-pill" id="status1" style="display:none;"></span>
    </div>
    <hr class="div">
    <div style="margin-bottom:16px;">
      <span class="lbl">Patient Case</span>
      <textarea id="caseIn" rows="12" placeholder="Paste the full patient case here..."></textarea>
    </div>
    <button class="btn-primary" id="btn1" onclick="runStep1()">
      <span class="spinner" id="sp1"></span><span>▶ Start Analysis</span>
    </button>
    <div class="hidden" id="w1" style="margin-top:20px;">
      <span class="lbl">Case Analysis Output</span>
      <div class="output-box" id="o1"></div>
    </div>
  </div>

  <!-- STEP 1.2 -->
  <div class="step-card hidden" id="s12">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
      <span class="step-badge">Step 2</span>
      <span class="step-title">Parameter Completeness Checklist</span>
      <span class="status-pill" id="status12" style="display:none;"></span>
    </div>
    <hr class="div">
    <button class="btn-primary" id="btn12" onclick="runStep12()">
      <span class="spinner" id="sp12"></span><span>▶ Check Parameters</span>
    </button>
    <div class="hidden" id="w12" style="margin-top:20px;">
      <span class="lbl">Checklist — ✓ Covered &nbsp;✗ Missing</span>
      <div class="output-box" id="o12"></div>
    </div>
  </div>

  <!-- STEP 2.5 -->
  <div class="step-card hidden" id="s15">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
      <span class="step-badge">Step 3</span>
      <span class="step-title">Missing Data Output</span>
      <span class="status-pill" id="status15" style="display:none;"></span>
    </div>
    <hr class="div">
    <button class="btn-primary" id="btn15" onclick="runStep15()">
      <span class="spinner" id="sp15"></span><span>▶ Check Missing Info</span>
    </button>
    <div class="hidden" id="w15" style="margin-top:20px;">
      <span class="lbl">Missing Data Output</span>
      <div class="output-box" id="o15"></div>
    </div>
  </div>

  <!-- STEP 2 — commented out
  <div class="step-card hidden" id="s2">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
      <span class="step-badge">Step 3</span>
      <span class="step-title">Top 5 Questions to Ask</span>
      <span class="status-pill" id="status2" style="display:none;"></span>
    </div>
    <hr class="div">
    <button class="btn-primary" id="btn2" onclick="runStep2()">
      <span class="spinner" id="sp2"></span><span>▶ Generate Questions</span>
    </button>
    <div class="hidden" id="w2" style="margin-top:20px;">
      <span class="lbl">Questions Output</span>
      <div class="output-box" id="o2"></div>
    </div>
  </div>
  -->

  <!-- STEP 3 — PQRS Analysis (all sub-steps in one click) -->
  <div class="step-card hidden" id="s3">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
      <span class="step-badge">Step 4</span>
      <span class="step-title">PQRS Analysis</span>
      <span class="status-pill" id="status3" style="display:none;"></span>
    </div>
    <hr class="div">
    <div style="display:none;margin-bottom:16px;">
      <span class="lbl">Case Type</span>
      <select id="caseTypeSelect" style="border:1.5px solid #bfdbfe;border-radius:8px;padding:8px 12px;font-size:.9rem;color:#1e293b;background:white;width:100%;outline:none;">
        <option value="Mixed (Acute + Chronic)">Mixed (Acute + Chronic)</option>
        <option value="Acute">Acute</option>
        <option value="Chronic">Chronic</option>
        <option value="Mental / Emotional">Mental / Emotional</option>
      </select>
    </div>
    <div style="margin-bottom:16px;">
      <span class="lbl">Chief Complaint (optional — auto-detected from Step 1)</span>
      <input id="ccInput" type="text" placeholder="e.g. hair fall, knee pain" style="border:1.5px solid #bfdbfe;border-radius:8px;padding:8px 12px;font-size:.9rem;color:#1e293b;background:white;width:100%;outline:none;">
    </div>
    <button class="btn-primary" id="btn3" onclick="runAllPqrs()">
      <span class="spinner" id="sp3"></span><span>▶ Run PQRS Analysis</span>
    </button>
    <!-- Progress tracker -->
    <div class="hidden" id="pqrsProgress" style="margin-top:18px;display:flex;flex-direction:column;gap:6px;">
      <div id="prog3a" style="font-size:.85rem;color:#6b7280;">○ Extract &amp; score symptoms</div>
      <div id="prog3b" style="font-size:.85rem;color:#6b7280;">○ Generate final PQRS</div>
      <div id="prog3c" style="font-size:.85rem;color:#6b7280;">○ Clinical validation</div>
      <div id="prog3d" style="font-size:.85rem;color:#6b7280;">○ Advanced reasoning review</div>
    </div>
    <!-- Sub-step outputs (revealed progressively) -->
    <div class="hidden" id="w3a" style="margin-top:20px;">
      <span class="lbl" style="color:#6b7280;">📋 Scored Candidates</span>
      <div class="output-box" id="o3a"></div>
    </div>
    <div class="hidden" id="w3b" style="margin-top:16px;">
      <span class="lbl" style="color:#059669;">🎯 Final PQRS with Rubrics</span>
      <div class="output-box" id="o3b" style="border-color:#a7f3d0;background:#f0fdf4;"></div>
    </div>
    <div class="hidden" id="w3c" style="margin-top:16px;">
      <span class="lbl" style="color:#dc2626;">🔍 Clinical Errors</span>
      <div class="output-box" id="o3cErr" style="margin-bottom:16px;border-color:#fca5a5;background:#fff5f5;"></div>
      <span class="lbl" style="color:#059669;">✅ Corrected PQRS</span>
      <div class="output-box" id="o3cOk" style="border-color:#a7f3d0;background:#f0fdf4;"></div>
    </div>
    <div class="hidden" id="w3d" style="margin-top:16px;">
      <span class="lbl" style="color:#dc2626;">🧠 Reasoning Errors</span>
      <div class="output-box" id="o3dErr" style="margin-bottom:16px;border-color:#fca5a5;background:#fff5f5;"></div>
      <span class="lbl" style="color:#059669;">✅ Final Validated PQRS</span>
      <div class="output-box" id="o3dOk" style="border-color:#a7f3d0;background:#f0fdf4;"></div>
    </div>
  </div>

  <!-- STEP 4 — Book Evidence + Remedies (RAG) -->
  <div class="step-card hidden" id="s4">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
      <span class="step-badge">Step 5</span>
      <span class="step-title">Book Evidence &amp; Remedy Search</span>
      <span class="status-pill" id="status4" style="display:none;"></span>
    </div>
    <hr class="div">
    <button class="btn-primary" id="btn4" onclick="runStep4()">
      <span class="spinner" id="sp4"></span><span>▶ Search Books &amp; Suggest Remedies</span>
    </button>
    <div class="hidden" id="w4" style="margin-top:20px;">
      <span class="lbl" style="color:#1e3a8a;">📚 Per-line Book Confirmation</span>
      <div class="output-box" id="o4conf" style="margin-bottom:16px;"></div>
      <span class="lbl" style="color:#059669;">💊 Top Remedies from Books</span>
      <div class="output-box" id="o4rem" style="border-color:#a7f3d0;background:#f0fdf4;"></div>
    </div>
  </div>

  <!-- DOWNLOAD -->
  <div class="step-card hidden" id="sDl">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
      <span class="step-badge">Report</span>
      <span class="step-title">Download Full Case Report</span>
    </div>
    <hr class="div">
    <div style="display:flex;gap:12px;flex-wrap:wrap;">
      <button class="btn-dl" onclick="downloadFile('/api/download/txt','doctor_assistant_report.txt')">⬇ Download TXT</button>
      <button class="btn-dl" onclick="downloadFile('/api/download/html','doctor_assistant_report.html')">⬇ Download HTML</button>
    </div>
    <p style="color:#64748b;font-size:.8rem;margin-top:10px;">HTML: open in browser → Ctrl+P → Save as PDF</p>
  </div>

</div>

<script>
const S = { caseData:'', step1:'', step12:'', step15:'', step2:'',
            caseType:'Mixed (Acute + Chronic)', chiefComplaint:'',
            scored:[], pqrs:[], validatedPqrs:[], finalPqrs:[] };

function loading(id, on) {
  document.getElementById('btn'+id).disabled = on;
  document.getElementById('sp'+id).style.display = on ? 'block' : 'none';
}
function show(id) {
  const el = document.getElementById(id);
  el.classList.remove('hidden');
  setTimeout(() => el.scrollIntoView({behavior:'smooth',block:'start'}), 120);
}
function setProgress(pct) {
  const bar = document.getElementById('topBar');
  bar.style.width = pct + '%';
  if (pct >= 100) setTimeout(() => { bar.style.width = '0%'; }, 700);
}
function setPill(id, type, text) {
  const el = document.getElementById('status' + id);
  if (!el) return;
  el.className = 'status-pill ' + type;
  el.textContent = text;
  el.style.display = 'inline-block';
}
function setRunning(cardId, btnId, msg) {
  document.getElementById(cardId).classList.add('running');
  document.getElementById('btn' + btnId).disabled = true;
  document.getElementById('sp' + btnId).style.display = 'block';
  setPill(btnId, 'running', msg);
}
function setDone(cardId, btnId, elapsed) {
  document.getElementById(cardId).classList.remove('running');
  document.getElementById('btn' + btnId).disabled = false;
  document.getElementById('sp' + btnId).style.display = 'none';
  setPill(btnId, 'done', '✅ Done · ' + elapsed + 's');
}
function fadeOutput(id) {
  const el = document.getElementById(id);
  el.classList.remove('fade-in');
  void el.offsetWidth;
  el.classList.add('fade-in');
}
async function api(url, body) {
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  return r.json();
}

function extractCC(text) {
  const lines = text.split('\\n');
  let inCC = false;
  for (const line of lines) {
    const t = line.trim();
    if (t.toLowerCase().startsWith('chief complaint')) { inCC = true; continue; }
    if (inCC && t.startsWith('-')) {
      let cc = t.slice(1).trim();
      const sinceIdx = cc.toLowerCase().indexOf(' since ');
      const forIdx   = cc.toLowerCase().indexOf(' for ');
      const cutIdx   = Math.min(sinceIdx === -1 ? 999 : sinceIdx, forIdx === -1 ? 999 : forIdx);
      if (cutIdx < 999) cc = cc.slice(0, cutIdx).trim();
      return cc;
    }
    if (inCC && t && !t.startsWith('-')) break;
  }
  return '';
}
async function downloadFile(url, filename) {
  const r = await fetch(url);
  if (!r.ok) { alert('Generate Step 4 first to create the report.'); return; }
  const blob = await r.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(a.href);
}
async function runStep1() {
  const txt = document.getElementById('caseIn').value.trim();
  if (!txt) { alert('Please enter a patient case first.'); return; }
  const t0 = Date.now();
  setRunning('s1', '1', '🔍 Extracting full case...');
  setProgress(15);
  const d = await api('/api/case/analyse', {case: txt});
  S.caseData = txt; S.step1 = d.result;
  const o1 = document.getElementById('o1');
  o1.textContent = d.result;
  fadeOutput('o1');
  document.getElementById('w1').classList.remove('hidden');
  const cc = extractCC(d.result);
  if (cc) document.getElementById('ccInput').value = cc;
  setDone('s1', '1', ((Date.now()-t0)/1000).toFixed(1));
  setProgress(100);
  show('s12');
}
async function runStep12() {
  const t0 = Date.now();
  setRunning('s12', '12', '✅ Checking completeness...');
  setProgress(15);
  const d = await api('/api/case/completeness', {case_data: S.step1, raw_case: S.caseData});
  S.step12 = d.result;
  document.getElementById('o12').textContent = d.result;
  fadeOutput('o12');
  document.getElementById('w12').classList.remove('hidden');
  setDone('s12', '12', ((Date.now()-t0)/1000).toFixed(1));
  setProgress(100);
  show('s15');
}
async function runStep15() {
  const t0 = Date.now();
  setRunning('s15', '15', '🔎 Scanning for missing data...');
  setProgress(15);
  const d = await api('/api/case/missing-data', {case_data: S.step1, raw_case: S.caseData, step12: S.step12});
  S.step15 = d.result;
  document.getElementById('o15').textContent = d.result;
  fadeOutput('o15');
  document.getElementById('w15').classList.remove('hidden');
  setDone('s15', '15', ((Date.now()-t0)/1000).toFixed(1));
  setProgress(100);
  show('s3');
}
/* Step 3 commented out
async function runStep2() {
  const t0 = Date.now();
  setRunning('s2', '2', '💬 Generating questions...');
  setProgress(15);
  const d = await api('/api/case/questions', {case_data: S.caseData});
  S.step2 = d.result;
  document.getElementById('o2').textContent = d.result;
  fadeOutput('o2');
  document.getElementById('w2').classList.remove('hidden');
  setDone('s2', '2', ((Date.now()-t0)/1000).toFixed(1));
  setProgress(100);
  show('s3');
}
*/
async function runAllPqrs() {
  const caseTypeEl = document.getElementById('caseTypeSelect');
  S.caseType = caseTypeEl ? caseTypeEl.value : S.caseType;
  S.chiefComplaint = document.getElementById('ccInput').value.trim();
  const t0pqrs = Date.now();
  setRunning('s3', '3', '📊 Running PQRS...');
  setProgress(5);
  document.getElementById('pqrsProgress').classList.remove('hidden');

  // Sub-step 3a — extract & score
  setProgress(20);
  document.getElementById('prog3a').textContent = '⏳ Extracting & scoring symptoms...';
  const d3a = await api('/api/pqrs/extract', {
    case_data: S.step1, case_type: S.caseType, chief_complaint: S.chiefComplaint
  });
  S.scored = d3a.scored || [];
  const lines3a = [];
  if (d3a.error) lines3a.push('ERROR: ' + d3a.error);
  lines3a.push('Case type: ' + S.caseType + ' | Min score: ' + d3a.min_score);
  lines3a.push(S.scored.length + ' passed, ' + (d3a.skipped||[]).length + ' skipped\\n');
  S.scored.forEach((s,i) => lines3a.push((i+1)+'. ['+s.score+'] '+s.symptom+' ('+s.tags.join(', ')+')'));
  document.getElementById('o3a').textContent = lines3a.join('\\n');
  document.getElementById('w3a').classList.remove('hidden');
  document.getElementById('prog3a').textContent = '✅ Symptoms extracted — ' + S.scored.length + ' scored';
  setProgress(40);

  // Sub-step 3b — generate final PQRS
  document.getElementById('prog3b').textContent = '⏳ Generating final PQRS...';
  const d3b = await api('/api/pqrs/generate', {
    scored: S.scored, cleaned: S.step1,
    case_type: S.caseType, chief_complaint: S.chiefComplaint
  });
  S.pqrs = d3b.pqrs || [];
  const lines3b = [];
  if (d3b.error) lines3b.push('ERROR: ' + d3b.error);
  S.pqrs.forEach((p,i) => {
    lines3b.push((i+1)+'. '+p.symptom);
    if (p.rubric) lines3b.push('   📖 '+p.rubric);
    if (p.reason) lines3b.push('   → '+p.reason);
    lines3b.push('');
  });
  document.getElementById('o3b').textContent = lines3b.join('\\n');
  document.getElementById('w3b').classList.remove('hidden');
  document.getElementById('prog3b').textContent = '✅ PQRS generated — ' + S.pqrs.length + ' symptoms';
  setProgress(60);

  // Sub-step 3c — clinical validation
  document.getElementById('prog3c').textContent = '⏳ Running clinical validation...';
  const d3c = await api('/api/pqrs/validate', { pqrs: S.pqrs });
  S.validatedPqrs = d3c.corrected_pqrs || S.pqrs;
  const errLines3c = (d3c.errors||[]).length
    ? (d3c.errors||[]).map(e => '❌ '+e.symptom+'\\n   Issue: '+e.issue+'\\n   Fix: '+e.fix).join('\\n\\n')
    : '✅ No errors found.';
  const okLines3c = S.validatedPqrs.map((p,i) =>
    (i+1)+'. '+p.symptom+(p.rubric?'\\n   📖 '+p.rubric:'')+(p.reason?'\\n   → '+p.reason:'')
  ).join('\\n\\n');
  document.getElementById('o3cErr').textContent = errLines3c;
  document.getElementById('o3cOk').textContent = okLines3c;
  document.getElementById('w3c').classList.remove('hidden');
  document.getElementById('prog3c').textContent = '✅ Validation complete — ' + (d3c.errors||[]).length + ' error(s) found';
  setProgress(80);

  // Sub-step 3d — advanced reasoning review
  document.getElementById('prog3d').textContent = '⏳ Running advanced reasoning review...';
  const d3d = await api('/api/pqrs/advanced', { pqrs: S.validatedPqrs, cleaned: S.step1 });
  S.finalPqrs = d3d.final_pqrs || S.validatedPqrs;
  const errLines3d = (d3d.errors||[]).length
    ? (d3d.errors||[]).map(e => '['+e.error_type+'] '+e.symptom+'\\n   '+e.explanation+'\\n   Fix: '+e.corrected).join('\\n\\n')
    : '✅ No reasoning errors found.';
  const finalLines = S.finalPqrs.map((p,i) =>
    (i+1)+'. '+p.symptom+(p.rubric?'\\n   📖 '+p.rubric:'')+(p.reason?'\\n   → '+p.reason:'')
  ).join('\\n\\n');
  document.getElementById('o3dErr').textContent = errLines3d;
  document.getElementById('o3dOk').textContent = finalLines;
  document.getElementById('w3d').classList.remove('hidden');
  document.getElementById('prog3d').textContent = '✅ Advanced review done — ' + S.finalPqrs.length + ' final PQRS';

  setDone('s3', '3', ((Date.now()-t0pqrs)/1000).toFixed(1));
  setProgress(100);
  show('s4');
}
async function runStep4() {
  const t0 = Date.now();
  setRunning('s4', '4', '📚 Searching book evidence...');
  setProgress(15);
  const d = await api('/api/pqrs/rag', { pqrs: S.finalPqrs.length ? S.finalPqrs : S.validatedPqrs });
  const pqrsList = S.finalPqrs.length ? S.finalPqrs : S.validatedPqrs;
  const pqrsLines = pqrsList.map((p,i) =>
    (i+1)+'. '+p.symptom+(p.rubric ? '\\n   Rubric: '+p.rubric : '')+(p.reason ? '\\n   Reason: '+p.reason : '')
  ).join('\\n\\n');
  let confLines = 'RAG not available.';
  let remediesText = 'RAG not available.';
  if (!d.available) {
    document.getElementById('o4conf').textContent = 'RAG not available — run build_db.py first.';
    document.getElementById('w4').classList.remove('hidden');
  } else {
    confLines = (d.confirmations||[]).map(c =>
      c.confirmed ? '✓ '+c.line+'\\n  → '+c.remedy+' ('+c.source+', score '+c.score+')'
                  : '— '+c.line+'\\n  → not confirmed in books'
    ).join('\\n\\n') || 'No lines to confirm.';
    remediesText = d.remedies || 'No results.';
    document.getElementById('o4conf').textContent = confLines;
    fadeOutput('o4conf');
    document.getElementById('o4rem').textContent = remediesText;
    fadeOutput('o4rem');
    document.getElementById('w4').classList.remove('hidden');
  }
  setProgress(80);
  const rb = await fetch('/api/report/build', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      case_input: S.caseData,
      step1: S.step1, step12: S.step12, step15: S.step15, step2: S.step2,
      pqrs_text: pqrsLines,
      confirmations_text: confLines,
      remedies_text: remediesText
    })
  });
  if (!rb.ok) { alert('Report build failed — check server logs.'); return; }
  setDone('s4', '4', ((Date.now()-t0)/1000).toFixed(1));
  setProgress(100);
  show('sDl');
}
</script>
</body>
</html>"""
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
