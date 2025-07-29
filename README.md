# AI Clinical Trial Matching System

## 📋 Project Overview

The AI Clinical Trial Matching System is an intelligent web application designed to automatically match patients with appropriate clinical trials based on their medical reports. The system combines structured rule-based matching with semantic similarity analysis to provide comprehensive trial recommendations.

### 🎯 Key Features

- **Single Patient Processing**: Designed to handle one patient's medical report at a time
- **Dual Matching Approach**: Combines structured eligibility criteria matching with semantic similarity analysis
- **Advanced AI Integration**: ChromaDB-powered semantic matching with rich metadata
- **Comprehensive Feature Extraction**: Automatically extracts medical features from unstructured text
- **Interactive Web Interface**: User-friendly Streamlit-based web application with advanced controls
- **Enhanced Visualizations**: Professional charts with intuitive color coding
- **Comprehensive PDF Reports**: Downloadable reports with visualizations and analytics
- **Real-time Processing**: Instant analysis and matching results
- **Smart Fallback System**: Graceful degradation when advanced features are unavailable

## 🏗️ System Architecture

### Core Components

```
frontend/
├── app.py                        # Main Streamlit web application (Advanced Edition)
├── feature_extraction.py         # Medical feature extraction engine
├── trial_data.py                # Clinical trial data management
├── structured_matching.py       # Rule-based eligibility matching
├── semantic_matching.py         # Standard semantic similarity (deployment-safe)
├── semantic_matching_advanced.py # Advanced ChromaDB-based semantic matching
└── requirements.txt             # Python dependencies
```

### Data Flow

1. **Input**: Patient medical report (text or file upload)
2. **Feature Extraction**: Parse and extract structured medical data
3. **Trial Loading**: Load available clinical trials
4. **Dual Matching**: Apply both structured and semantic matching
5. **Advanced Analytics**: Generate comprehensive insights and visualizations
6. **Results Display**: Show matches with explanations and interactive charts
7. **Report Generation**: Create downloadable PDF reports with visualizations

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

### 4. Semantic Matching (Dual Approach)

#### Standard Semantic Matching (`semantic_matching.py`):
- **Sentence Transformers**: Uses 'all-MiniLM-L6-v2' model for text embedding
- **Scikit-learn Integration**: Cosine similarity for efficient matching
- **Deployment-Safe**: Works reliably on all platforms including Streamlit Cloud

#### Advanced Semantic Matching (`semantic_matching_advanced.py`):
- **ChromaDB Integration**: Vector database for enhanced semantic search
- **Rich Metadata**: Stores additional trial information with embeddings
- **Advanced Features**: Better similarity algorithms and metadata filtering
- **Smart Fallback**: Automatically switches to standard mode if ChromaDB unavailable

#### Semantic Analysis Process:
1. **Text Embedding**: Convert patient report and trial criteria to vectors
2. **Vector Processing**: Store and query embeddings efficiently
3. **Similarity Search**: Find most similar trials using cosine distance
4. **Ranking**: Return top matches with similarity scores

### 5. Advanced Web Application (`app.py`)

#### Enhanced Streamlit Interface Features:
- **Modern Design**: Professional styling with custom CSS
- **Advanced Sidebar**: Configuration options and settings
- **Tabbed Interface**: Organized results in separate sections
- **Interactive Controls**: Adjustable parameters and visualization options
- **Real-time Processing**: Instant feature extraction and matching
- **Professional Visualizations**: Color-coded charts and graphs
- **Comprehensive Analytics**: Advanced insights and metrics
- **Enhanced PDF Export**: Complete reports with visualizations

#### User Interface Components:
- **Advanced Settings Sidebar**: Semantic matching mode, parameters, visualization options
- **Input Section**: Text area and file uploader with sample format
- **Feature Display**: Structured presentation of extracted medical data
- **Results Tabs**: Separate sections for features, structured matches, semantic matches, and analytics
- **Interactive Visualizations**: Charts with color coding and legends
- **Analytics Dashboard**: Comprehensive metrics and insights
- **Download Section**: Advanced PDF report generation

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
   python -c "import streamlit, pandas, matplotlib, fpdf, sentence_transformers, sklearn; print('All dependencies installed successfully')"
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
scikit-learn           # Machine learning and similarity computation
numpy                  # Numerical computing
chromadb               # Vector database for advanced semantic search
Pillow                 # Image processing for PDF visualizations
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

### Using the Advanced Application

#### Step 1: Configure Advanced Settings
- **Semantic Matching Mode**: Choose between Advanced (ChromaDB) or Standard (scikit-learn)
- **Matching Parameters**: Adjust number of top matches (1-10)
- **Feature Extraction**: Toggle detailed extraction information
- **Visualizations**: Enable/disable charts and graphs

#### Step 2: Input Patient Data
- **Option A**: Paste patient medical report into the text area
- **Option B**: Upload a .txt file containing the patient report

#### Step 3: Process the Report
- Click the "🚀 Extract Features and Match Trials" button
- Wait for the system to process the report (usually takes a few seconds)
- View real-time processing status and success messages

#### Step 4: Review Comprehensive Results
The application will display results in organized tabs:

1. **🔍 Extracted Features Tab**
   - Diagnosis, stage, mutations in organized boxes
   - ECOG score, comorbidities, medications
   - Lab values in structured table format
   - Optional detailed extraction information

2. **📋 Structured Matches Tab**
   - Summary metrics (total trials, eligible count, eligibility rate)
   - Detailed results table with explanations
   - Color-coded pie chart (red=ineligible, green=eligible)
   - Top eligible trials visualization

3. **🤖 Semantic Matches Tab**
   - Mode indicator (Advanced/Standard)
   - Similarity scores explanation
   - Results table with rankings
   - Color-coded bar chart with legend
   - Quality assessment indicators

4. **📊 Analytics Tab**
   - Structured vs Semantic comparison
   - Overlap analysis between approaches
   - Match quality metrics
   - Performance insights and recommendations

#### Step 5: Download Advanced Report
- Click the "📥 Download Advanced PDF Report" button
- Receive a comprehensive PDF with:
  - Complete patient data and analysis
  - All visualizations (pie charts, bar charts)
  - Summary statistics and insights
  - Timestamp and analysis mode information

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
- **Structured Data**: All extracted features displayed in organized sections
- **Lab Values Table**: Numerical values presented in clear table format
- **Missing Data Handling**: Graceful handling of missing or unclear information
- **Extraction Details**: Optional detailed view of extraction process

### Matching Results

#### Structured Matching
- **Eligibility Status**: Clear indication of eligible/ineligible trials
- **Explanation List**: Detailed reasons for each match decision
- **Visual Summary**: Color-coded pie chart (red=ineligible, green=eligible)
- **Metrics Dashboard**: Total trials, eligible count, and eligibility rate

#### Semantic Matching
- **Similarity Scores**: Cosine similarity scores (0-1 scale, lower is better)
- **Ranked Results**: Trials ordered by similarity
- **Visual Comparison**: Color-coded bar chart with quality indicators
- **Mode Information**: Clear indication of Advanced vs Standard mode

### Advanced Analytics
- **Comparison Metrics**: Structured vs semantic approach analysis
- **Overlap Detection**: Trials appearing in both approaches
- **Quality Assessment**: Match quality indicators and recommendations
- **Performance Insights**: Average similarity, best matches, and trends

### Enhanced PDF Report Contents
1. **Executive Summary**: Overview with timestamp and analysis mode
2. **Detailed Features**: Complete breakdown of medical data
3. **Structured Matches**: Eligibility results with explanations
4. **Semantic Matches**: Similarity-based recommendations
5. **Visualizations**: High-quality charts and graphs
6. **Summary Statistics**: Key metrics and insights
7. **Trial Details**: Comprehensive trial information

## 🔍 Technical Details

### AI/ML Components

#### Sentence Transformers
- **Model**: all-MiniLM-L6-v2
- **Purpose**: Convert text to high-dimensional vectors
- **Performance**: Fast inference with good semantic understanding
- **Output**: 384-dimensional embeddings

#### ChromaDB Vector Database (Advanced Mode)
- **Purpose**: Store and query trial embeddings with rich metadata
- **Features**: Efficient similarity search, metadata storage, advanced filtering
- **Configuration**: Anonymized telemetry disabled for privacy
- **Fallback**: Automatic switch to standard mode if unavailable

#### Scikit-learn (Standard Mode)
- **Purpose**: Efficient similarity computation and machine learning
- **Features**: Cosine similarity, reliable deployment compatibility
- **Performance**: Fast computation with minimal dependencies

### Data Processing Pipeline

1. **Text Preprocessing**: Case normalization, whitespace handling
2. **Feature Extraction**: Regular expression-based parsing
3. **Data Validation**: Type checking and value validation
4. **Vector Generation**: AI model inference for embeddings
5. **Similarity Computation**: Cosine distance calculations
6. **Result Ranking**: Sorting and filtering of matches
7. **Analytics Generation**: Comprehensive insights and metrics

### Performance Considerations

- **Processing Speed**: Typically 2-5 seconds for complete analysis
- **Memory Usage**: Efficient vector storage and retrieval
- **Scalability**: Modular design allows for easy expansion
- **Accuracy**: Combines rule-based precision with AI-powered similarity
- **Reliability**: Smart fallback system ensures consistent operation

## 🛠️ Customization and Extension

### Adding New Features
1. **Medical Features**: Extend `feature_extraction.py` with new extraction functions
2. **Trial Criteria**: Modify `trial_data.py` to include additional eligibility factors
3. **Matching Rules**: Update `structured_matching.py` with new matching logic
4. **UI Components**: Enhance `app.py` with additional visualization options
5. **Semantic Models**: Integrate different embedding models in semantic matching

### Integration Possibilities
- **EHR Integration**: Connect to electronic health record systems
- **Real Trial Data**: Replace synthetic data with actual clinical trial databases
- **API Endpoints**: Expose matching functionality via REST API
- **Multi-language Support**: Extend for international clinical trials
- **Advanced Analytics**: Integrate with business intelligence tools

## 🔒 Privacy and Security

### Data Handling
- **Local Processing**: All data processing occurs locally
- **No External Storage**: Patient data is not stored or transmitted
- **Session-based**: Data persists only during the current session
- **Secure Downloads**: PDF reports generated locally with safe encoding

### Compliance Considerations
- **HIPAA Awareness**: Designed with healthcare privacy in mind
- **Data Minimization**: Only processes necessary medical information
- **Audit Trail**: Clear logging of processing steps
- **Secure Encoding**: Safe text handling for PDF generation

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
   python -c "import feature_extraction, trial_data, structured_matching, semantic_matching_advanced"
   ```

4. **ChromaDB Issues**
   - The system automatically falls back to standard mode
   - Check if ChromaDB is properly installed
   - Advanced features will be disabled but app continues working

5. **PDF Generation Errors**
   - System includes fallback PDF generation
   - Unicode characters are automatically handled
   - Check browser console for detailed error messages

### Performance Optimization
- **Large Reports**: Consider breaking very long reports into sections
- **Multiple Users**: Each user session is independent
- **Resource Usage**: Monitor system resources during processing
- **Advanced Mode**: Disable if experiencing performance issues

## 📈 Future Enhancements

### Planned Features
- **Multi-language Support**: International clinical trial matching
- **Advanced NLP**: More sophisticated text understanding
- **Real-time Updates**: Live trial database integration
- **Mobile Interface**: Responsive design for mobile devices
- **Batch Processing**: Handle multiple patients simultaneously
- **Advanced Visualizations**: Interactive charts and 3D visualizations

### Research Opportunities
- **Improved Embeddings**: Domain-specific medical language models
- **Federated Learning**: Privacy-preserving model training
- **Explainable AI**: Better understanding of matching decisions
- **Clinical Validation**: Real-world accuracy assessment
- **Performance Optimization**: Faster processing and better scalability

## 📞 Support and Contact

### Getting Help
- **Documentation**: This README provides comprehensive guidance
- **Code Comments**: Well-documented source code for technical questions
- **Error Messages**: Clear error reporting in the application
- **Fallback Systems**: Automatic recovery from common issues

### Contributing
- **Code Quality**: Follow existing code style and patterns
- **Testing**: Test new features thoroughly
- **Documentation**: Update documentation for any changes
- **Error Handling**: Include proper fallback mechanisms

## 📄 License

This project is designed for educational and research purposes. Please ensure compliance with local healthcare regulations and privacy laws when using in clinical settings.

---

**Note**: This system uses synthetic data for demonstration purposes. For clinical use, replace with real clinical trial databases and ensure proper validation and regulatory compliance.

**Advanced Features**: The system includes both standard and advanced semantic matching capabilities. Advanced features require ChromaDB and may not be available in all deployment environments, but the system gracefully falls back to standard mode to ensure reliable operation.