from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

def semantic_match(patient_text, trials, top_n=3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
    # Use get_or_create_collection to avoid error if collection exists
    if hasattr(chroma_client, 'get_or_create_collection'):
        collection = chroma_client.get_or_create_collection(name="trials")
    else:
        try:
            collection = chroma_client.create_collection(name="trials")
        except Exception:
            collection = chroma_client.get_collection(name="trials")
    # Optionally clear previous data to avoid duplicates
    try:
        collection.delete(where={})
    except Exception:
        pass
    for trial in trials:
        embedding = model.encode(trial["criteria_text"]).tolist()
        collection.add(
            documents=[trial["criteria_text"]],
            embeddings=[embedding],
            ids=[trial["id"]],
            metadatas=[{"name": trial["name"]}]
        )
    patient_embedding = model.encode(patient_text).tolist()
    results = collection.query(
        query_embeddings=[patient_embedding],
        n_results=top_n
    )
    matches = []
    for i, trial_id in enumerate(results["ids"][0]):
        matches.append({
            "trial_id": trial_id,
            "trial_name": results["metadatas"][0][i]["name"],
            "similarity": results["distances"][0][i]
        })
    return matches 