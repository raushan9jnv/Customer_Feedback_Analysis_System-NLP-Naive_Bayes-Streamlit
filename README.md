# Customer_Feedback_Analysis_System-NLP-Naive_Bayes-Streamlit
This system provides automated classification of customer feedback into Complaints, Suggestions, and Praises using Natural Language Processing (NLP). It features a real-time dashboard with interactive visualizations for business insights.

 ![image](https://github.com/user-attachments/assets/6e9a3691-de31-4e81-b83f-73b546525eab)

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

### 3. Model Evaluation Metrics
Key Performance Indicators
| Metric       | Formula                          | Purpose                   | Ideal Value       |
|--------------|----------------------------------|---------------------------|-------------------|
| **Accuracy** | `(Correct Predictions) / (Total Predictions)` | Overall correctness       | > 0.85           |
| **Precision**| `TP / (TP + FP)`                 | Avoid false positives     | Per-class > 0.75 |
| **Recall**   | `TP / (TP + FN)`                 | Capture all positives     | Per-class > 0.70 |
| **F1 Score** | `2*(Precision*Recall)/(Precision+Recall)` | Balance Precision/Recall | > 0.80           |

**Legend:**  
- TP = True Positives  
- FP = False Positives  
- FN = False Negatives

Output Result:

  ![image](https://github.com/user-attachments/assets/75bc17ee-dfa4-4afc-a193-fce3bd09caef)


## Features
1. Real-Time Analysis
   - Instant feedback classification
   - Color-coded results (Red=Complaint, Yellow=Suggestion, Green=Praise)
   - Actionable response templates

     ![image](https://github.com/user-attachments/assets/613282fe-1ac4-457c-acbc-c4428a9c1682)

2. Visual Analytics
   - Interactive rating distribution histogram
   - 
     ![image](https://github.com/user-attachments/assets/c74bbea6-2d32-40b6-88e4-74de9cb41a2d)

   - Category proportion pie chart
     
     ![image](https://github.com/user-attachments/assets/a5a65d84-4c53-45b0-ae86-0ec46c8661ea)

   - Comparative word clouds across categories
     
     ![image](https://github.com/user-attachments/assets/dda2b6e3-3351-49ed-9eb8-62eb55a0023a)

3. Production-Ready
   - Model caching for fast reloads
   - Error handling for missing data
   - Mobile-responsive design

# Thanks!
