# Step12 Completeness Checklist Improvements - Progress Tracker

## Plan Status: IMPLEMENTATION IN PROGRESS 🔄

**File:** HomeoAi/doctor-app/backend/main.py

### 1. [ ] Add evidence_check() function
- Add after audit_step12()
- Fixed softer prompt: \"closest match + minor interpretation\"

### 2. [ ] Update run_step12() prompt  
- Add: \"Do NOT assume missing details. But DO accept clearly implied clinical meaning.\"
- Semantic examples kept

### 3. [ ] Simplify audit_step12() prompt
- \"Only fix clearly wrong ✓. Do NOT downgrade borderline ✓.\"

### 4. [ ] Update api_case_completeness() flow
```
raw = run_step12()
validated = evidence_check(raw, case_data)  # NEW
audited = audit_step12(validated, case_data)
final = auto_correct_step12(audited, case_data)  # Simplified
```

### 5. [ ] Simplify auto_correct_step12()
**KEEP ONLY:**
- Menstrual N/A for males  
- Replace \"~\" → \"✗\"
- Remove hallucinated fields (dreams/hobbies if absent)

**REMOVE:**
- Urine strict 3-part check ❌
- Sweat location+odour ❌
- Sleep hours+quality ❌
- Bowels regularity+consistency ❌

## Testing Steps
1. `cd HomeoAi/doctor-app/backend`
2. `python main.py` (port 8001)
3. Test `/case/completeness` with sample case
4. Verify balanced ✓/✗ (no over-rejection)
5. Check reports generate

## Expected Results
- \"I eat less\" → Appetite ✓
- Indirect thermal → ✓  
- Hallucinated dreams → ✗
- 96-98% accuracy on natural cases

**Next:** Code edits → test → mark complete ✅
