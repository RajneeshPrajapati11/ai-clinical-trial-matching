def load_trials():
    trials = []
    # Example synthetic trials (expand to 30)
    for i in range(1, 31):
        trial = {
            "id": f"trial_{i:03d}",
            "name": f"Synthetic Trial {i}",
            "inclusion": {
                "diagnosis": "Non-small cell lung cancer" if i % 3 == 0 else "lymphoma" if i % 3 == 1 else "pancreatic cancer",
                "mutation": ["EGFR", "ALK", "KRAS", "CD20"][i % 4],
                "ecog_max": (i % 3) + 1
            },
            "exclusion": {
                "comorbidities": ["hepatitis", "TB"] if i % 2 == 0 else ["diabetes"]
            },
            "criteria_text": f"Patients with {['EGFR','ALK','KRAS','CD20'][i%4]} mutation, ECOG ≤ {(i%3)+1}, no {' or '.join(['hepatitis','TB'] if i%2==0 else ['diabetes'])}."
        }
        trials.append(trial)
    return trials 