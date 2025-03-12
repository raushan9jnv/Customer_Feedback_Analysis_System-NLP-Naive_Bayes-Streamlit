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
1. Clone Repository
2. Install Dependencies
   - pip install -r requirements.txt
3. Place your Customer Feedback Analysis.csv in the project root
4. Train Model
   - python train_model.py
5. Launch Dashboard
   - streamlit run app.py
