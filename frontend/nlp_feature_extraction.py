import spacy
import en_core_sci_sm

# Load the scispaCy model once
nlp = en_core_sci_sm.load()

def extract_medical_entities(text):
    """
    Extracts medical entities from the input text using scispaCy's NER.
    Returns a list of (entity_text, entity_label) tuples.
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Example usage (for testing):
if __name__ == "__main__":
    sample_text = "Patient diagnosed with diffuse large B-cell lymphoma, stage III, with CD20 mutation. Medications include rituximab and CHOP. No history of diabetes."
    print(extract_medical_entities(sample_text)) 