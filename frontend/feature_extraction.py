import re
import json

def extract_diagnosis(text: str) -> str:
    match = re.search(r"diagnosis[:\s]+([\w\s\-]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"diagnosed with ([\w\s\-]+?)(,|\.|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def extract_stage(text: str) -> str:
    match = re.search(r"stage[:\s]+([IV0-9]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else ""

def extract_mutations(text: str) -> list:
    return re.findall(r"(EGFR|STK11|T790M|KRAS|ALK|ROS1|BRAF|MET|RET|HER2|CD20)", text, re.IGNORECASE)

def extract_ecog(text: str) -> int:
    match = re.search(r"ECOG[\s=:-]*([0-4])", text, re.IGNORECASE)
    return int(match.group(1)) if match else -1

def extract_comorbidities(text: str) -> list:
    comorbidities = ["diabetes", "hypertension", "hepatitis", "COPD", "TB", "asthma"]
    found = [c for c in comorbidities if re.search(c, text, re.IGNORECASE)]
    return found

def extract_medications(text: str) -> list:
    meds = ["metformin", "insulin", "aspirin", "statin", "prednisone", "rituximab", "CHOP"]
    found = [m for m in meds if re.search(m, text, re.IGNORECASE)]
    return found

def extract_lab_values(text: str) -> dict:
    labs = {}
    for lab in ["ALT", "AST", "creatinine", "HbA1c", "WBC", "hemoglobin"]:
        match = re.search(rf"{lab}[\s:=]+([0-9.]+)", text, re.IGNORECASE)
        if match:
            value = match.group(1).rstrip('.').strip()
            try:
                labs[lab] = float(value)
            except ValueError:
                pass
    return labs

def extract_features(text: str) -> dict:
    return {
        "diagnosis": extract_diagnosis(text),
        "stage": extract_stage(text),
        "mutations": extract_mutations(text),
        "ecog": extract_ecog(text),
        "comorbidities": extract_comorbidities(text),
        "medications": extract_medications(text),
        "lab_values": extract_lab_values(text),
    } 