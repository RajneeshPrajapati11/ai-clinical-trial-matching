import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import json
from datetime import datetime
from feature_extraction import extract_features
from trial_data import load_trials
from structured_matching import match_trials
from semantic_matching_advanced import semantic_match_advanced, semantic_match

# Page configuration
st.set_page_config(
    page_title="AI Clinical Trial Matching System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🏥 AI Clinical Trial Matching System</h1>', unsafe_allow_html=True)
st.markdown("### Advanced AI-powered clinical trial matching with dual analysis approach")

# Sidebar for advanced options
with st.sidebar:
    st.header("⚙️ Advanced Settings")
    
    # Semantic matching options
    st.subheader("🔍 Semantic Matching")
    use_advanced_semantic = st.checkbox("Use Advanced Semantic Matching", value=True, 
                                       help="Enable ChromaDB-based semantic matching with rich metadata")
    
    if use_advanced_semantic:
        st.info("Advanced mode: Uses ChromaDB for enhanced semantic search with metadata")
    else:
        st.info("Standard mode: Uses scikit-learn for reliable deployment")
    
    # Matching parameters
    st.subheader("📊 Matching Parameters")
    top_n_matches = st.slider("Number of Top Matches", min_value=1, max_value=10, value=5)
    
    # Feature extraction options
    st.subheader("🔧 Feature Extraction")
    show_extraction_details = st.checkbox("Show Extraction Details", value=False)
    
    # Visualization options
    st.subheader("📈 Visualizations")
    show_charts = st.checkbox("Show Charts", value=True)
    
    # About section
    st.markdown("---")
    st.markdown("### 📋 About")
    st.markdown("""
    **Features:**
    - 🤖 AI-powered semantic matching
    - 📋 Structured eligibility matching
    - 📊 Interactive visualizations
    - 📄 PDF report generation
    - 🔄 Real-time processing
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Patient Report Input")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📝 Paste Text", "📁 Upload File"],
        horizontal=True
    )
    
    report_text = ""
    
    if input_method == "📝 Paste Text":
        report_text = st.text_area(
            "Paste patient medical report here:",
            height=200,
            placeholder="Enter patient medical report with diagnosis, stage, mutations, ECOG score, comorbidities, medications, and lab values..."
        )
    else:
        uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
        if uploaded_file is not None:
            report_text = uploaded_file.read().decode("utf-8")
            st.text_area("Report content:", report_text, height=200, key="uploaded")

with col2:
    st.subheader("📋 Sample Report")
    st.markdown("""
    **Example format:**
    ```
    Patient: John Doe, Age 65
    Diagnosis: Non-small cell lung cancer, Stage IIIB
    ECOG: 1
    Mutations: EGFR positive
    Comorbidities: Hypertension
    Medications: Metformin, Aspirin
    Lab Values: ALT 45, AST 38, Creatinine 1.2
    ```
    """)

# Processing section
if st.button("🚀 Extract Features and Match Trials", type="primary") and report_text.strip():
    with st.spinner("Processing patient report..."):
        try:
            # Extract features
            features = extract_features(report_text)
            trials = load_trials()
            
            # Structured matching
            matches = match_trials(features, trials)
            match_df = pd.DataFrame(matches)
            
            # Semantic matching (advanced or standard)
            if use_advanced_semantic:
                sem_matches = semantic_match_advanced(report_text, trials, top_n_matches)
            else:
                sem_matches = semantic_match(report_text, trials, top_n_matches)
            sem_df = pd.DataFrame(sem_matches)
            
            # Store in session state
            st.session_state["features"] = features
            st.session_state["match_df"] = match_df
            st.session_state["sem_df"] = sem_df
            st.session_state["extracted"] = True
            st.session_state["use_advanced"] = use_advanced_semantic
            
            st.success("✅ Analysis completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.info("💡 Try using the standard semantic matching mode if advanced mode fails.")

# Display results
if "extracted" in st.session_state and st.session_state["extracted"]:
    features = st.session_state["features"]
    match_df = st.session_state["match_df"]
    sem_df = st.session_state["sem_df"]
    use_advanced = st.session_state.get("use_advanced", False)
    
    # Results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Extracted Features", "📋 Structured Matches", "🤖 Semantic Matches", "📊 Analytics"])
    
    with tab1:
        st.subheader("🔍 Extracted Medical Features")
        
        # Feature display with better formatting
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.write("**🏥 Diagnosis:**", features["diagnosis"] if features["diagnosis"] else "Not found")
            st.write("**📊 Stage:**", features["stage"] if features["stage"] else "Not found")
            st.write("**🧬 Mutations:**", ", ".join(features["mutations"]) if features["mutations"] else "None")
            st.write("**📈 ECOG Score:**", features["ecog"] if features["ecog"] != -1 else "Not found")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.write("**💊 Comorbidities:**", ", ".join(features["comorbidities"]) if features["comorbidities"] else "None")
            st.write("**💉 Medications:**", ", ".join(features["medications"]) if features["medications"] else "None")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Lab values table
        if features["lab_values"]:
            st.subheader("🔬 Laboratory Values")
            lab_df = pd.DataFrame(list(features["lab_values"].items()), columns=["Test", "Value"])
            st.dataframe(lab_df, use_container_width=True)
        else:
            st.info("No laboratory values found in the report.")
        
        # Extraction details (if enabled)
        if show_extraction_details:
            with st.expander("🔧 Extraction Details"):
                st.json(features)
    
    with tab2:
        st.subheader("📋 Structured Eligibility Matches")
        
        # Summary statistics
        eligible_count = match_df["eligible"].sum()
        total_count = len(match_df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trials", total_count)
        with col2:
            st.metric("Eligible Trials", eligible_count)
        with col3:
            st.metric("Eligibility Rate", f"{(eligible_count/total_count)*100:.1f}%")
        
        # Results table
        st.dataframe(
            match_df[["trial_name", "eligible", "explanation"]],
            use_container_width=True,
            hide_index=True
        )
        
        # Visualizations
        if show_charts:
            # Add color explanation
            st.markdown("""
            **📊 Color Coding:**
            - 🟢 **Green**: Eligible trials
            - 🔴 **Red**: Ineligible trials
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                eligible_counts = match_df["eligible"].value_counts()
                
                # Better color scheme: Red for ineligible, Green for eligible
                colors = []
                labels = []
                for idx in eligible_counts.index:
                    if idx == True:  # Eligible
                        colors.append('#51cf66')  # Green
                        labels.append('Eligible')
                    else:  # Ineligible
                        colors.append('#ff6b6b')  # Red
                        labels.append('Ineligible')
                
                ax1.pie(eligible_counts, labels=labels, autopct='%1.1f%%', 
                       startangle=90, colors=colors)
                ax1.set_title("Eligibility Distribution (Structured)")
                st.pyplot(fig1)
            
            with col2:
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                top_eligible = match_df[match_df["eligible"] == True].head(10)
                if not top_eligible.empty:
                    ax2.barh(range(len(top_eligible)), [1]*len(top_eligible))
                    ax2.set_yticks(range(len(top_eligible)))
                    ax2.set_yticklabels(top_eligible["trial_name"], fontsize=8)
                    ax2.set_title("Top Eligible Trials")
                    ax2.set_xlabel("Eligibility")
                    st.pyplot(fig2)
    
    with tab3:
        st.subheader("🤖 Semantic Similarity Matches")
        
        if use_advanced:
            st.info("🔬 Using Advanced Semantic Matching with ChromaDB")
        else:
            st.info("📊 Using Standard Semantic Matching with scikit-learn")
        
        # Similarity scores explanation
        st.markdown("""
        **Similarity Scores:**
        - **Lower values** = Better matches (more similar)
        - **Higher values** = Less similar matches
        """)
        
        # Results table
        st.dataframe(
            sem_df[["trial_name", "similarity"]],
            use_container_width=True,
            hide_index=True
        )
        
        # Visualizations
        if show_charts:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            bars = ax3.bar(range(len(sem_df)), sem_df["similarity"])
            ax3.set_ylabel("Similarity Score (lower is better)")
            ax3.set_title("Semantic Similarity Scores")
            ax3.set_xticks(range(len(sem_df)))
            ax3.set_xticklabels(sem_df["trial_name"], rotation=45, ha='right')
            
            # Color bars based on similarity with better color scheme
            for i, bar in enumerate(bars):
                similarity = sem_df.iloc[i]["similarity"]
                if similarity < 0.3:
                    bar.set_color('#51cf66')  # Green for excellent matches
                elif similarity < 0.6:
                    bar.set_color('#ffd43b')  # Yellow for moderate matches
                else:
                    bar.set_color('#ff6b6b')  # Red for poor matches
            
            # Add color legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#51cf66', label='Excellent (< 0.3)'),
                Patch(facecolor='#ffd43b', label='Moderate (0.3-0.6)'),
                Patch(facecolor='#ff6b6b', label='Poor (> 0.6)')
            ]
            ax3.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            st.pyplot(fig3)
    
    with tab4:
        st.subheader("📊 Comprehensive Analytics")
        
        # Comparison of both approaches
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Structured vs Semantic")
            
            # Find overlap between approaches
            structured_eligible = set(match_df[match_df["eligible"]]["trial_name"])
            semantic_top = set(sem_df["trial_name"])
            overlap = structured_eligible.intersection(semantic_top)
            
            st.metric("Structured Eligible", len(structured_eligible))
            st.metric("Semantic Top Matches", len(semantic_top))
            st.metric("Overlap", len(overlap))
            
            if overlap:
                st.success(f"✅ {len(overlap)} trials appear in both approaches!")
        
        with col2:
            st.subheader("🎯 Match Quality")
            
            # Quality metrics
            avg_similarity = sem_df["similarity"].mean()
            best_similarity = sem_df["similarity"].min()
            
            st.metric("Average Similarity", f"{avg_similarity:.3f}")
            st.metric("Best Similarity", f"{best_similarity:.3f}")
            
            if best_similarity < 0.3:
                st.success("🎉 Excellent semantic matches found!")
            elif best_similarity < 0.6:
                st.warning("⚠️ Moderate semantic matches")
            else:
                st.error("❌ Poor semantic matches - consider different input")
    
    # PDF Download Section
    st.markdown("---")
    st.subheader("📄 Download Comprehensive Report")
    
    def create_advanced_pdf(features, match_df, sem_df, use_advanced):
        def safe_text(text):
            """Safely encode text for PDF generation"""
            if text is None:
                return ""
            if isinstance(text, str):
                # Replace problematic characters
                text = text.replace("≤", "<=").replace("≥", ">=")
                text = text.replace("°", " degrees").replace("±", "+/-")
                # Remove or replace other problematic Unicode characters
                text = text.encode('latin1', 'replace').decode('latin1')
                return text[:100]  # Limit length to avoid overflow
            return str(text)
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "AI Clinical Trial Matching System - Advanced Report", ln=True, align="C")
        pdf.ln(10)
        
        # Add timestamp
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(0, 10, f"Analysis Mode: {'Advanced' if use_advanced else 'Standard'}", ln=True)
        pdf.ln(10)
        
        # Features section
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 10, "Extracted Features:", ln=True)
        features_table = [
            ["Diagnosis", safe_text(features['diagnosis'])],
            ["Stage", safe_text(features['stage'])],
            ["Mutations", safe_text(', '.join(features['mutations']))],
            ["ECOG", safe_text(features['ecog'])],
            ["Comorbidities", safe_text(', '.join(features['comorbidities']))],
            ["Medications", safe_text(', '.join(features['medications']))],
            ["Lab Values", safe_text(json.dumps(features['lab_values']) if features['lab_values'] else "None")]
        ]
        
        col_width = pdf.w / 3
        for row in features_table:
            pdf.set_font("Arial", style="B", size=10)
            pdf.cell(col_width, 8, safe_text(row[0]), border=1)
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 8, safe_text(row[1]), border=1, ln=True)
        
        pdf.ln(5)
        
        # Structured matches
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 10, "Structured Trial Matches:", ln=True)
        pdf.set_font("Arial", style="B", size=10)
        pdf.cell(50, 8, "Trial Name", border=1)
        pdf.cell(20, 8, "Eligible", border=1)
        pdf.cell(0, 8, "Explanation", border=1, ln=True)
        pdf.set_font("Arial", size=10)
        
        for idx, row in match_df.iterrows():
            pdf.cell(50, 8, safe_text(row['trial_name'])[:30], border=1)
            pdf.cell(20, 8, str(row['eligible']), border=1)
            explanation = safe_text(', '.join(row['explanation']))
            pdf.multi_cell(0, 8, explanation[:100], border=1)
            if pdf.get_y() > pdf.h - 30:
                pdf.add_page()
        
        pdf.ln(5)
        
        # Semantic matches
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 10, "Semantic Matches:", ln=True)
        pdf.set_font("Arial", style="B", size=10)
        pdf.cell(60, 8, "Trial Name", border=1)
        pdf.cell(0, 8, "Similarity", border=1, ln=True)
        pdf.set_font("Arial", size=10)
        
        for idx, row in sem_df.iterrows():
            pdf.cell(60, 8, safe_text(row['trial_name'])[:35], border=1)
            pdf.cell(0, 8, f"{row['similarity']:.3f}", border=1, ln=True)
            if pdf.get_y() > pdf.h - 30:
                pdf.add_page()
        
        # Add visualizations
        try:
            import tempfile
            from PIL import Image
            
            # Create pie chart for structured matches
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            eligible_counts = match_df["eligible"].value_counts()
            colors = []
            labels = []
            for idx in eligible_counts.index:
                if idx == True:  # Eligible
                    colors.append('#51cf66')  # Green
                    labels.append('Eligible')
                else:  # Ineligible
                    colors.append('#ff6b6b')  # Red
                    labels.append('Ineligible')
            
            ax1.pie(eligible_counts, labels=labels, autopct='%1.1f%%', 
                   startangle=90, colors=colors)
            ax1.set_title("Eligibility Distribution (Structured)")
            
            # Save pie chart
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp1:
                plt.savefig(tmp1.name, bbox_inches='tight', dpi=150)
                plt.close(fig1)
                
                # Add pie chart to PDF
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(0, 10, "Visualizations", ln=True, align="C")
                pdf.ln(5)
                pdf.image(tmp1.name, w=pdf.w * 0.8, x=pdf.w * 0.1)
            
            # Create bar chart for semantic matches
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            bars = ax2.bar(range(len(sem_df)), sem_df["similarity"])
            ax2.set_ylabel("Similarity Score (lower is better)")
            ax2.set_title("Semantic Similarity Scores")
            ax2.set_xticks(range(len(sem_df)))
            ax2.set_xticklabels(sem_df["trial_name"], rotation=45, ha='right', fontsize=8)
            
            # Color bars based on similarity
            for i, bar in enumerate(bars):
                similarity = sem_df.iloc[i]["similarity"]
                if similarity < 0.3:
                    bar.set_color('#51cf66')  # Green for excellent matches
                elif similarity < 0.6:
                    bar.set_color('#ffd43b')  # Yellow for moderate matches
                else:
                    bar.set_color('#ff6b6b')  # Red for poor matches
            
            # Add color legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#51cf66', label='Excellent (< 0.3)'),
                Patch(facecolor='#ffd43b', label='Moderate (0.3-0.6)'),
                Patch(facecolor='#ff6b6b', label='Poor (> 0.6)')
            ]
            ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
            
            plt.tight_layout()
            
            # Save bar chart
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp2:
                plt.savefig(tmp2.name, bbox_inches='tight', dpi=150)
                plt.close(fig2)
                
                # Add bar chart to PDF
                pdf.ln(10)
                pdf.image(tmp2.name, w=pdf.w * 0.8, x=pdf.w * 0.1)
            
            # Add summary statistics
            pdf.ln(10)
            pdf.set_font("Arial", size=11)
            pdf.cell(0, 10, "Summary Statistics:", ln=True)
            pdf.set_font("Arial", size=10)
            
            eligible_count = match_df["eligible"].sum()
            total_count = len(match_df)
            avg_similarity = sem_df["similarity"].mean()
            best_similarity = sem_df["similarity"].min()
            
            pdf.cell(0, 8, f"Total Trials Analyzed: {total_count}", ln=True)
            pdf.cell(0, 8, f"Eligible Trials: {eligible_count} ({(eligible_count/total_count)*100:.1f}%)", ln=True)
            pdf.cell(0, 8, f"Average Semantic Similarity: {avg_similarity:.3f}", ln=True)
            pdf.cell(0, 8, f"Best Semantic Match: {best_similarity:.3f}", ln=True)
            
        except Exception as e:
            # If visualization fails, add a note
            pdf.add_page()
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 10, "Note: Visualizations could not be generated for this report.", ln=True)
            pdf.cell(0, 10, "Please view the web interface for interactive charts.", ln=True)
        
        try:
            pdf_bytes = pdf.output(dest='S').encode('latin1')
        except UnicodeEncodeError:
            # Fallback: create a simpler PDF without problematic characters
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, "AI Clinical Trial Matching Report", ln=True, align="C")
            pdf.ln(10)
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.cell(0, 10, f"Analysis Mode: {'Advanced' if use_advanced else 'Standard'}", ln=True)
            pdf.ln(10)
            pdf.cell(0, 10, "Report generated successfully. View results in the web interface.", ln=True)
            pdf_bytes = pdf.output(dest='S').encode('latin1')
        
        return pdf_bytes
    
    # Download button
    pdf_bytes = create_advanced_pdf(features, match_df, sem_df, use_advanced)
    st.download_button(
        label="📥 Download Advanced PDF Report",
        data=pdf_bytes,
        file_name=f"clinical_trial_matching_advanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏥 AI Clinical Trial Matching System | Advanced Edition</p>
    <p>Powered by Streamlit, Sentence Transformers, and Advanced NLP</p>
</div>
""", unsafe_allow_html=True)
