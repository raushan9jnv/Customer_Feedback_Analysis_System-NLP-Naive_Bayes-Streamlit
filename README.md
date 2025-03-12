# Customer_Feedback_Analysis_System-NLP-Naive_Bayes-Streamlit
This system provides automated classification of customer feedback into Complaints, Suggestions, and Praises using Natural Language Processing (NLP). It features a real-time dashboard with interactive visualizations for business insights.
## How to Run

### Prerequisites
- Python 3.8+
- Required packages:
  ```text
  streamlit==1.29.0
  pandas==2.1.4
  scikit-learn==1.3.2
  nltk==3.8.1
  matplotlib==3.8.2
  seaborn==0.13.0
  wordcloud==1.9.3
  joblib==1.3.2

### Setup Instructions
    
    ```text
    1. Clone Repository
    2. Install Dependencies
       - pip install -r requirements.txt
    3. Place your Customer Feedback Analysis.csv in the project root
    4. Train Model
       - python train_model.py
    5. Launch Dashboard
       - streamlit run app.py

## Methodology
### 1. Data Preprocessing
- Missing Values: Remove rows with empty reviews/ratings
- Rating Conversion:
    ```text
    Rating | Category
    ≤2     → Complaint
    3      → Suggestion
    ≥4     → Praise
    
- Text Cleaning Pipeline:
    ```
    1. Lowercase conversion
    2. Remove punctuation/numbers
    3. Tokenization
    4. Stopword removal (English)
    5. Filter short words (<3 characters)

### 2. Machine Learning Model
    ```
    Vectorization: TF-IDF
    Classifier: Multinomial Naive Bayes
    Model Persistence: Saved as .pkl files for production use

## Features
1. Real-Time Analysis
   - Instant feedback classification
   - Color-coded results (Red=Complaint, Yellow=Suggestion, Green=Praise)
   - Actionable response templates
2. Visual Analytics
   - Interactive rating distribution histogram
   - Category proportion pie chart
   - Comparative word clouds across categories
3. Production-Ready
   - Model caching for fast reloads
   - Error handling for missing data
   - Mobile-responsive design

# Thanks!
