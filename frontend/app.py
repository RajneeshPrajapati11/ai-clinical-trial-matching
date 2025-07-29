import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import json
from feature_extraction import extract_features
from trial_data import load_trials
from structured_matching import match_trials
from semantic_matching import semantic_match

st.title("AI in Clinical Trial Matching")
st.write("Paste or upload a patient report to find matching clinical trials.")

report_text = st.text_area("Paste patient report here:", height=200)
uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])

if uploaded_file is not None:
    report_text = uploaded_file.read().decode("utf-8")
    st.text_area("Report content:", report_text, height=200, key="uploaded")

# --- Extraction and Matching ---
if st.button("Extract Features and Match Trials") and report_text.strip():
    features = extract_features(report_text)
    trials = load_trials()
    matches = match_trials(features, trials)
    match_df = pd.DataFrame(matches)
    sem_matches = semantic_match(report_text, trials)
    sem_df = pd.DataFrame(sem_matches)
    # Store in session state
    st.session_state["features"] = features
    st.session_state["match_df"] = match_df
    st.session_state["sem_df"] = sem_df
    st.session_state["extracted"] = True
else:
    # Only show extracted if session state is set
    if "extracted" in st.session_state and st.session_state["extracted"]:
        features = st.session_state["features"]
        match_df = st.session_state["match_df"]
        sem_df = st.session_state["sem_df"]
    else:
        features = None
        match_df = None
        sem_df = None

# --- Display Results ---
if features is not None and match_df is not None and sem_df is not None:
    st.subheader("Extracted Features")
    st.write("**Diagnosis:**", features["diagnosis"])
    st.write("**Stage:**", features["stage"])
    st.write("**Mutations:**", ", ".join(features["mutations"]) if features["mutations"] else "None")
    st.write("**ECOG:**", features["ecog"] if features["ecog"] != -1 else "Not found")
    st.write("**Comorbidities:**", ", ".join(features["comorbidities"]) if features["comorbidities"] else "None")
    st.write("**Medications:**", ", ".join(features["medications"]) if features["medications"] else "None")
    if features["lab_values"]:
        st.write("**Lab Values:**")
        st.table(pd.DataFrame(list(features["lab_values"].items()), columns=["Lab", "Value"]))
    else:
        st.write("**Lab Values:** None")

    # --- Explainability Dashboard ---
    st.subheader(":mag: Explainability Dashboard")
    st.write("Below you can see for each trial the specific reason(s) for eligibility or exclusion.")
    explain_data = []
    for idx, row in match_df.iterrows():
        trial = row['trial_name']
        eligible = row['eligible']
        explanation = row['explanation']
        if isinstance(explanation, list):
            explanation_str = "; ".join(str(e) for e in explanation)
        else:
            explanation_str = str(explanation)
        explain_data.append({
            "Trial Name": trial,
            "Eligible": eligible,
            "Reason(s)": explanation_str
        })
    explain_df = pd.DataFrame(explain_data)
    st.dataframe(explain_df, use_container_width=True)


    st.subheader("Top Semantic Matches")
    st.dataframe(sem_df[["trial_name", "similarity"]])
    fig2, ax2 = plt.subplots()
    ax2.bar(sem_df["trial_name"], sem_df["similarity"])
    ax2.set_ylabel("Cosine Similarity (lower is better)")
    ax2.set_title("Top Semantic Matches")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig2)

    def create_pdf(features, match_df, sem_df):
        import tempfile
        def safe(s):
            if isinstance(s, str):
                return s.replace("≤", "<=").replace("≥", ">=")
            if isinstance(s, list):
                return [safe(x) for x in s]
            return s
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "AI in Clinical Trial Matching Results", ln=True, align="C")
        pdf.ln(10)
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 10, "Extracted Features:", ln=True)
        # Table for features
        features_table = [
            ["Diagnosis", safe(features['diagnosis'])],
            ["Stage", safe(features['stage'])],
            ["Mutations", ', '.join(safe(features['mutations']))],
            ["ECOG", safe(features['ecog'])],
            ["Comorbidities", ', '.join(safe(features['comorbidities']))],
            ["Medications", ', '.join(safe(features['medications']))],
            ["Lab Values", json.dumps(safe(features['lab_values'])) if features['lab_values'] else "None"]
        ]
        col_width = pdf.w / 3
        for row in features_table:
            pdf.set_font("Arial", style="B", size=11)
            pdf.cell(col_width, 8, str(row[0]), border=1)
            pdf.set_font("Arial", size=11)
            pdf.cell(0, 8, str(row[1]), border=1, ln=True)
        pdf.ln(5)
        # Structured Trial Matches Table
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "Structured Trial Matches:", ln=True)
        pdf.set_font("Arial", style="B", size=10)
        pdf.cell(50, 8, "Trial Name", border=1)
        pdf.cell(20, 8, "Eligible", border=1)
        pdf.cell(0, 8, "Explanation", border=1, ln=True)
        pdf.set_font("Arial", size=10)
        for idx, row in match_df.iterrows():
            y_before = pdf.get_y()
            x = pdf.get_x()
            pdf.cell(50, 8, safe(row['trial_name'])[:30], border=1)
            pdf.cell(20, 8, str(row['eligible']), border=1)
            x_expl = pdf.get_x()
            y_expl = pdf.get_y()
            explanation = ', '.join(safe(row['explanation']))
            explanation_height = pdf.get_string_width(explanation) / (pdf.w - x_expl - 10) * 8 + 8
            pdf.multi_cell(0, 8, explanation, border=1)
            pdf.set_xy(x, max(y_before + explanation_height, pdf.get_y()))
            if pdf.get_y() > pdf.h - 30:
                pdf.add_page()
        pdf.ln(5)
        # Top Semantic Matches Table
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "Top Semantic Matches:", ln=True)
        pdf.set_font("Arial", style="B", size=10)
        pdf.cell(60, 8, "Trial Name", border=1)
        pdf.cell(0, 8, "Similarity", border=1, ln=True)
        pdf.set_font("Arial", size=10)
        for idx, row in sem_df.iterrows():
            pdf.cell(60, 8, safe(row['trial_name'])[:35], border=1)
            pdf.cell(0, 8, f"{row['similarity']:.3f}", border=1, ln=True)
            if pdf.get_y() > pdf.h - 30:
                pdf.add_page()
        pdf.ln(5)
        # Add charts as images, resizing to avoid MemoryError
        from PIL import Image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp1, tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp2:
            eligible_counts = match_df["eligible"].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(eligible_counts, labels=eligible_counts.index.map(str), autopct='%1.1f%%', startangle=90)
            ax1.set_title("Eligibility Distribution (Structured)")
            plt.savefig(tmp1.name, bbox_inches='tight')
            plt.close(fig1)
            fig2, ax2 = plt.subplots()
            ax2.bar(sem_df["trial_name"], sem_df["similarity"])
            ax2.set_ylabel("Cosine Similarity (lower is better)")
            ax2.set_title("Top Semantic Matches")
            plt.xticks(rotation=45, ha='right')
            plt.savefig(tmp2.name, bbox_inches='tight')
            plt.close(fig2)
            # Resize images to max width/height (e.g., 800x600)
            for tmp_img in [tmp1.name, tmp2.name]:
                with Image.open(tmp_img) as im:
                    im = im.convert('RGB')
                    im.thumbnail((800, 600), Image.LANCZOS)
                    im.save(tmp_img, format='PNG')
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, "Visualizations", ln=True)
            pdf.image(tmp1.name, w=pdf.w * 0.8)
            pdf.ln(5)
            pdf.image(tmp2.name, w=pdf.w * 0.8)
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        return pdf_bytes

    # Always show the download button when results are available
    if features is not None and match_df is not None and sem_df is not None:
        pdf_bytes = create_pdf(features, match_df, sem_df)
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name="clinical_trial_matching_results.pdf",
            mime="application/pdf"
        )
