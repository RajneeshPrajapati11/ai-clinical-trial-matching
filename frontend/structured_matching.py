def match_trials(features, trials):
    results = []
    for trial in trials:
        explanation = []
        eligible = True
        inc = trial["inclusion"]
        if inc.get("diagnosis") and inc["diagnosis"].lower() not in features["diagnosis"].lower():
            eligible = False
            explanation.append(f"Diagnosis mismatch: needs {inc['diagnosis']}")
        if inc.get("mutation") and inc["mutation"].upper() not in [m.upper() for m in features["mutations"]]:
            eligible = False
            explanation.append(f"Mutation mismatch: needs {inc['mutation']}")
        if inc.get("ecog_max") is not None and features["ecog"] > inc["ecog_max"]:
            eligible = False
            explanation.append(f"ECOG too high: must be ≤ {inc['ecog_max']}")
        exc = trial.get("exclusion", {})
        for comorb in exc.get("comorbidities", []):
            if comorb.lower() in [c.lower() for c in features["comorbidities"]]:
                eligible = False
                explanation.append(f"Excluded due to comorbidity: {comorb}")
        if eligible:
            explanation = [
                f"Diagnosis matches: {inc['diagnosis']}" if inc.get('diagnosis') else None,
                f"Mutation matches: {inc['mutation']}" if inc.get('mutation') else None,
                f"ECOG within limit: {features['ecog']} ≤ {inc['ecog_max']}" if inc.get('ecog_max') is not None else None,
                f"No exclusion comorbidities present"
            ]
            explanation = [e for e in explanation if e]
        results.append({
            "trial_id": trial["id"],
            "trial_name": trial["name"],
            "eligible": eligible,
            "explanation": explanation if explanation else ["Eligible"]
        })
    return results 