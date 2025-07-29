from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def semantic_match_advanced(patient_text, trials, top_n=3, use_chromadb=True):
    """
    Advanced semantic matching with ChromaDB support when available
    """
    if use_chromadb:
        try:
            return chromadb_semantic_match(patient_text, trials, top_n)
        except Exception as e:
            print(f"ChromaDB failed, falling back to simple approach: {str(e)}")
            return simple_semantic_match(patient_text, trials, top_n)
    else:
        return simple_semantic_match(patient_text, trials, top_n)

def chromadb_semantic_match(patient_text, trials, top_n=3):
    """
    ChromaDB-based semantic matching with advanced features
    """
    try:
        import chromadb
        from chromadb.config import Settings
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        
        # Create or get collection
        collection = chroma_client.get_or_create_collection(
            name="clinical_trials",
            metadata={"description": "Clinical trial criteria embeddings"}
        )
        
        # Clear previous data
        try:
            collection.delete(where={})
        except:
            pass
        
        # Add trial embeddings with rich metadata
        for trial in trials:
            embedding = model.encode(trial["criteria_text"]).tolist()
            collection.add(
                documents=[trial["criteria_text"]],
                embeddings=[embedding],
                ids=[trial["id"]],
                metadatas=[{
                    "name": trial["name"],
                    "diagnosis": trial["inclusion"].get("diagnosis", ""),
                    "mutation": trial["inclusion"].get("mutation", ""),
                    "ecog_max": trial["inclusion"].get("ecog_max", ""),
                    "exclusions": str(trial.get("exclusion", {}).get("comorbidities", []))
                }]
            )
        
        # Query with patient text
        patient_embedding = model.encode(patient_text).tolist()
        results = collection.query(
            query_embeddings=[patient_embedding],
            n_results=top_n,
            include=["metadatas", "distances"]
        )
        
        matches = []
        for i, trial_id in enumerate(results["ids"][0]):
            matches.append({
                "trial_id": trial_id,
                "trial_name": results["metadatas"][0][i]["name"],
                "similarity": results["distances"][0][i],
                "metadata": results["metadatas"][0][i]
            })
        
        return matches
        
    except ImportError:
        raise Exception("ChromaDB not available")
    except Exception as e:
        raise Exception(f"ChromaDB error: {str(e)}")

def simple_semantic_match(patient_text, trials, top_n=3):
    """
    Simple semantic matching using sentence transformers and cosine similarity
    """
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

# For backward compatibility
def semantic_match(patient_text, trials, top_n=3):
    """Main function - uses simple approach by default for deployment compatibility"""
    return simple_semantic_match(patient_text, trials, top_n)