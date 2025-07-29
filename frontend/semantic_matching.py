from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def semantic_match(patient_text, trials, top_n=3):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode patient text
        patient_embedding = model.encode([patient_text])
        
        # Encode all trial criteria texts
        trial_texts = [trial["criteria_text"] for trial in trials]
        trial_embeddings = model.encode(trial_texts)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(patient_embedding, trial_embeddings)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[:top_n]
        
        matches = []
        for idx in top_indices:
            matches.append({
                "trial_id": trials[idx]["id"],
                "trial_name": trials[idx]["name"],
                "similarity": float(similarities[idx])
            })
        
        return matches
    
    except Exception as e:
        # Fallback: return simple text-based matches if embedding fails
        print(f"Semantic matching failed, using fallback: {str(e)}")
        return fallback_semantic_match(patient_text, trials, top_n)

def fallback_semantic_match(patient_text, trials, top_n=3):
    """Fallback method using simple keyword matching"""
    patient_lower = patient_text.lower()
    
    # Simple keyword scoring
    scores = []
    for trial in trials:
        score = 0
        trial_text = trial["criteria_text"].lower()
        
        # Check for common medical terms
        medical_terms = ["cancer", "tumor", "mutation", "stage", "ecog", "diagnosis"]
        for term in medical_terms:
            if term in patient_lower and term in trial_text:
                score += 1
        
        # Check for specific conditions
        conditions = ["lung", "lymphoma", "pancreatic", "breast", "colon"]
        for condition in conditions:
            if condition in patient_lower and condition in trial_text:
                score += 2
        
        scores.append((score, trial))
    
    # Sort by score and return top matches
    scores.sort(reverse=True)
    matches = []
    for i, (score, trial) in enumerate(scores[:top_n]):
        matches.append({
            "trial_id": trial["id"],
            "trial_name": trial["name"],
            "similarity": 1.0 - (i * 0.1)  # Simple similarity score
        })
    
    return matches 