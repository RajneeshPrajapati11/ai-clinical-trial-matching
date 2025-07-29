# AI Clinical Trial Matching System

## 📋 Project Overview

The AI Clinical Trial Matching System is an intelligent web application designed to automatically match patients with appropriate clinical trials based on their medical reports. The system combines structured rule-based matching with semantic similarity analysis to provide comprehensive trial recommendations.

### 🎯 Key Features

- **Single Patient Processing**: Designed to handle one patient's medical report at a time
- **Dual Matching Approach**: Combines structured eligibility criteria matching with semantic similarity analysis
- **Comprehensive Feature Extraction**: Automatically extracts medical features from unstructured text
- **Interactive Web Interface**: User-friendly Streamlit-based web application
- **PDF Report Generation**: Downloadable comprehensive reports with visualizations
- **Real-time Processing**: Instant analysis and matching results

## 🏗️ System Architecture

### Core Components

```
frontend/
├── app.py                 # Main Streamlit web application
├── feature_extraction.py  # Medical feature extraction engine
├── trial_data.py         # Clinical trial data management
├── structured_matching.py # Rule-based eligibility matching
├── semantic_matching.py  # AI-powered semantic similarity
└── requirements.txt      # Python dependencies
```

### Data Flow

1. **Input**: Patient medical report (text or file upload)
2. **Feature Extraction**: Parse and extract structured medical data
3. **Trial Loading**: Load available clinical trials
4. **Dual Matching**: Apply both structured and semantic matching
5. **Results Display**: Show matches with explanations and visualizations
6. **Report Generation**: Create downloadable PDF reports

## 🔧 Technical Implementation

### 1. Feature Extraction (`feature_extraction.py`)

The system automatically extracts the following medical features from patient reports:

#### Extracted Features:
- **Diagnosis**: Primary disease condition
- **Stage**: Disease staging (e.g., Stage I, II, III, IV)
- **Mutations**: Genetic mutations (EGFR, STK11, T790M, KRAS, ALK, ROS1, BRAF, MET, RET, HER2, CD20)
- **ECOG Score**: Performance status (0-4 scale)
- **Comorbidities**: Existing conditions (diabetes, hypertension, hepatitis, COPD, TB, asthma)
- **Medications**: Current medications (metformin, insulin, aspirin, statin, prednisone, rituximab, CHOP)
- **Lab Values**: Laboratory test results (ALT, AST, creatinine, HbA1c, WBC, hemoglobin)

#### Extraction Methods:
- **Regular Expression Patterns**: Pattern-based text extraction
- **Case-Insensitive Matching**: Robust matching regardless of text case
- **Flexible Format Support**: Handles various text formats and structures

### 2. Trial Data Management (`trial_data.py`)

#### Synthetic Trial Generation:
- **30 Clinical Trials**: Automatically generated for demonstration
- **Diverse Conditions**: Covers lung cancer, lymphoma, and pancreatic cancer
- **Multiple Mutations**: EGFR, ALK, KRAS, CD20 mutations
- **Variable Criteria**: Different ECOG requirements and exclusion criteria

#### Trial Structure:
```python
{
    "id": "trial_001",
    "name": "Synthetic Trial 1",
    "inclusion": {
        "diagnosis": "Non-small cell lung cancer",
        "mutation": "EGFR",
        "ecog_max": 1
    },
    "exclusion": {
        "comorbidities": ["hepatitis", "TB"]
    },
    "criteria_text": "Detailed eligibility criteria text"
}
```

### 3. Structured Matching (`structured_matching.py`)

#### Rule-Based Eligibility Assessment:
- **Inclusion Criteria Matching**: Checks diagnosis, mutations, and ECOG scores
- **Exclusion Criteria Checking**: Identifies disqualifying comorbidities
- **Detailed Explanations**: Provides specific reasons for eligibility/ineligibility

#### Matching Logic:
1. **Diagnosis Match**: Verifies patient diagnosis against trial requirements
2. **Mutation Check**: Confirms presence of required genetic mutations
3. **ECOG Validation**: Ensures performance status meets trial limits
4. **Comorbidity Screening**: Checks for exclusionary conditions

### 4. Semantic Matching (`semantic_matching.py`)

#### AI-Powered Similarity Analysis:
- **Sentence Transformers**: Uses 'all-MiniLM-L6-v2' model for text embedding
- **ChromaDB Integration**: Vector database for efficient similarity search
- **Cosine Similarity**: Measures semantic similarity between patient reports and trial criteria

#### Semantic Analysis Process:
1. **Text Embedding**: Convert patient report and trial criteria to vectors
2. **Vector Storage**: Store trial embeddings in ChromaDB
3. **Similarity Search**: Find most similar trials using cosine distance
4. **Ranking**: Return top matches with similarity scores

### 5. Web Application (`app.py`)

#### Streamlit Interface Features:
- **Text Input**: Large text area for pasting patient reports
- **File Upload**: Support for .txt file uploads
- **Real-time Processing**: Instant feature extraction and matching
- **Interactive Results**: Expandable sections with detailed information
- **Data Visualization**: Pie charts and bar graphs for results
- **PDF Export**: Comprehensive downloadable reports

#### User Interface Components:
- **Input Section**: Text area and file uploader
- **Feature Display**: Structured presentation of extracted medical data
- **Matching Results**: Two separate sections for structured and semantic matches
- **Visualizations**: Charts showing eligibility distribution and similarity scores
- **Download Section**: PDF report generation

## 🚀 Installation and Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step-by-Step Installation

1. **Clone or Download the Project**
   ```bash
   # Navigate to your project directory
   cd Aitrials
   ```

2. **Install Dependencies**
   ```bash
   cd frontend
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   python -c "import streamlit, pandas, matplotlib, fpdf; print('All dependencies installed successfully')"
   ```

### Dependencies

The following packages are required:

```
streamlit==1.46.1      # Web application framework
requests               # HTTP library for API calls
matplotlib             # Data visualization
pandas                 # Data manipulation and analysis
fpdf                   # PDF generation
sentence-transformers  # AI text embedding models
chromadb               # Vector database for similarity search
```

## 🎮 Usage Guide

### Starting the Application

1. **Navigate to the Frontend Directory**
   ```bash
   cd frontend
   ```

2. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```

3. **Access the Application**
   - Open your web browser
   - Navigate to `http://localhost:8501`
   - The app will automatically open in your default browser

### Using the Application

#### Step 1: Input Patient Data
- **Option A**: Paste patient medical report into the text area
- **Option B**: Upload a .txt file containing the patient report

#### Step 2: Process the Report
- Click the "Extract Features and Match Trials" button
- Wait for the system to process the report (usually takes a few seconds)

#### Step 3: Review Results
The application will display:

1. **Extracted Features Section**
   - Diagnosis, stage, mutations
   - ECOG score, comorbidities, medications
   - Lab values in a structured table

2. **Structured Trial Matches**
   - List of trials with eligibility status
   - Detailed explanations for each match
   - Pie chart showing eligibility distribution

3. **Semantic Trial Matches**
   - Top similar trials based on text similarity
   - Similarity scores (lower is better)
   - Bar chart visualization

#### Step 4: Download Report
- Click the "Download PDF" button
- Receive a comprehensive PDF report with all results and visualizations

### Example Patient Report Format

```
Patient: John Doe, Age 65
Diagnosis: Non-small cell lung cancer, Stage IIIB
ECOG: 1
Mutations: EGFR positive
Comorbidities: Hypertension
Medications: Metformin, Aspirin
Lab Values: ALT 45, AST 38, Creatinine 1.2
```

## 📊 Output and Results

### Feature Extraction Results
- **Structured Data**: All extracted features are displayed in organized sections
- **Lab Values Table**: Numerical values presented in a clear table format
- **Missing Data Handling**: Graceful handling of missing or unclear information

### Matching Results

#### Structured Matching
- **Eligibility Status**: Clear indication of eligible/ineligible trials
- **Explanation List**: Detailed reasons for each match decision
- **Visual Summary**: Pie chart showing distribution of eligible vs ineligible trials

#### Semantic Matching
- **Similarity Scores**: Cosine similarity scores (0-1 scale, lower is better)
- **Ranked Results**: Trials ordered by similarity
- **Visual Comparison**: Bar chart showing similarity scores

### PDF Report Contents
1. **Executive Summary**: Overview of extracted features
2. **Detailed Features**: Complete breakdown of medical data
3. **Structured Matches**: Eligibility results with explanations
4. **Semantic Matches**: Similarity-based recommendations
5. **Visualizations**: Charts and graphs from the web interface
6. **Trial Details**: Comprehensive trial information

## 🔍 Technical Details

### AI/ML Components

#### Sentence Transformers
- **Model**: all-MiniLM-L6-v2
- **Purpose**: Convert text to high-dimensional vectors
- **Performance**: Fast inference with good semantic understanding
- **Output**: 384-dimensional embeddings

#### ChromaDB Vector Database
- **Purpose**: Store and query trial embeddings
- **Features**: Efficient similarity search, metadata storage
- **Configuration**: Anonymized telemetry disabled for privacy

### Data Processing Pipeline

1. **Text Preprocessing**: Case normalization, whitespace handling
2. **Feature Extraction**: Regular expression-based parsing
3. **Data Validation**: Type checking and value validation
4. **Vector Generation**: AI model inference for embeddings
5. **Similarity Computation**: Cosine distance calculations
6. **Result Ranking**: Sorting and filtering of matches

### Performance Considerations

- **Processing Speed**: Typically 2-5 seconds for complete analysis
- **Memory Usage**: Efficient vector storage and retrieval
- **Scalability**: Modular design allows for easy expansion
- **Accuracy**: Combines rule-based precision with AI-powered similarity

## 🛠️ Customization and Extension

### Adding New Features
1. **Medical Features**: Extend `feature_extraction.py` with new extraction functions
2. **Trial Criteria**: Modify `trial_data.py` to include additional eligibility factors
3. **Matching Rules**: Update `structured_matching.py` with new matching logic
4. **UI Components**: Enhance `app.py` with additional visualization options

### Integration Possibilities
- **EHR Integration**: Connect to electronic health record systems
- **Real Trial Data**: Replace synthetic data with actual clinical trial databases
- **API Endpoints**: Expose matching functionality via REST API
- **Multi-language Support**: Extend for international clinical trials

## 🔒 Privacy and Security

### Data Handling
- **Local Processing**: All data processing occurs locally
- **No External Storage**: Patient data is not stored or transmitted
- **Session-based**: Data persists only during the current session
- **Secure Downloads**: PDF reports generated locally

### Compliance Considerations
- **HIPAA Awareness**: Designed with healthcare privacy in mind
- **Data Minimization**: Only processes necessary medical information
- **Audit Trail**: Clear logging of processing steps

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

2. **Missing Dependencies**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Import Errors**
   ```bash
   python -c "import feature_extraction, trial_data, structured_matching, semantic_matching"
   ```

4. **Memory Issues**
   - Close other applications
   - Restart the Streamlit server

### Performance Optimization
- **Large Reports**: Consider breaking very long reports into sections
- **Multiple Users**: Each user session is independent
- **Resource Usage**: Monitor system resources during processing

## 📈 Future Enhancements

### Planned Features
- **Multi-language Support**: International clinical trial matching
- **Advanced NLP**: More sophisticated text understanding
- **Real-time Updates**: Live trial database integration
- **Mobile Interface**: Responsive design for mobile devices
- **Batch Processing**: Handle multiple patients simultaneously

### Research Opportunities
- **Improved Embeddings**: Domain-specific medical language models
- **Federated Learning**: Privacy-preserving model training
- **Explainable AI**: Better understanding of matching decisions
- **Clinical Validation**: Real-world accuracy assessment

## 📞 Support and Contact

### Getting Help
- **Documentation**: This README provides comprehensive guidance
- **Code Comments**: Well-documented source code for technical questions
- **Error Messages**: Clear error reporting in the application

### Contributing
- **Code Quality**: Follow existing code style and patterns
- **Testing**: Test new features thoroughly
- **Documentation**: Update documentation for any changes

## 📄 License

This project is designed for educational and research purposes. Please ensure compliance with local healthcare regulations and privacy laws when using in clinical settings.

---

**Note**: This system uses synthetic data for demonstration purposes. For clinical use, replace with real clinical trial databases and ensure proper validation and regulatory compliance.