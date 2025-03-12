# Core data processing and ML libraries
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
# ML components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, 
                            confusion_matrix, classification_report)

from sklearn.model_selection import train_test_split

# Download stopwords list for text cleaning
nltk.download('stopwords')

def clean_text(text):
    """Process raw text into clean tokens for ML modeling"""
    # Handle missing values gracefully
    if pd.isna(text): 
        return ""
    
    # Standardize text format
    text = str(text).lower()  # Convert to lowercase for consistency
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\d+|\s+', ' ', text).strip()  # Eliminate numbers/extra spaces
    
    # Filter meaningful terms
    stop_words = set(stopwords.words('english'))
    words = [
        word for word in text.split() 
        if (word not in stop_words) and  # Remove common stopwords
           (len(word) > 2) and  # Keep only substantive words
           (not word.startswith(('http', 'www')))  # Exclude URLs
    ]
    return ' '.join(words)

def main():
    # Load raw customer feedback data
    data = pd.read_csv('Customer Feedback Analysis.csv')
    print(f"\nInitial dataset size: {len(data)} rows")
    
    # Data Quality Cleaning
    # Remove entries with missing reviews or ratings
    data = data.dropna(subset=['Reviews', 'Rating'])
    # Convert ratings to numeric type, coercing errors
    data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')  
    # Drop rows where rating conversion failed
    data = data.dropna(subset=['Rating'])  
    
    # Categorize ratings using industry-standard thresholds:
    # <=2: Complaint, 3: Suggestion, >=4: Praise
    data['Category'] = data['Rating'].apply(
        lambda x: 'Complaint' if x <=2 else 'Suggestion' if x==3 else 'Praise'
    )
    
    # Apply text cleaning pipeline
    data['Cleaned_Review'] = data['Reviews'].apply(clean_text)
    # Remove rows with empty reviews post-cleaning
    data = data[data['Cleaned_Review'].str.strip() != '']  
    
    # Split data into training (80%) and test (20%) sets
    # random_state=42 for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        data['Cleaned_Review'],  # Features (cleaned text)
        data['Category'],        # Target (category labels)
        test_size=0.2,           # Standard validation split
        random_state=42          # Seed for consistent shuffling
    )
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    # Convert text to numerical features using TF-IDF
    # max_features=5000 balances performance and memory usage
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)  # Fit on training data
    X_test_vec = vectorizer.transform(X_test)        # Transform test data
    
    # Initialize and train Naive Bayes classifier
    # Chosen for efficiency with text data
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)  # Train model
    
    # Generate predictions for evaluation
    y_pred = clf.predict(X_test_vec)
    
    # Model Performance Metrics
    print("\nModel Evaluation Metrics:")
    # Overall accuracy
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")  
    # Macro-averaged precision (handles class imbalance)
    print(f"Precision (Macro): {precision_score(y_test, y_pred, average='macro'):.2f}")  
    # Macro-averaged recall
    print(f"Recall (Macro): {recall_score(y_test, y_pred, average='macro'):.2f}")  
    # Macro-averaged F1-score
    print(f"F1 Score (Macro): {f1_score(y_test, y_pred, average='macro'):.2f}")  
    
    # Detailed class-wise metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Complaint', 'Suggestion', 'Praise']))
    
    # Prediction error matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Model Persistence
    # Save vectorizer and classifier for later use
    joblib.dump(vectorizer, 'models/vectorizer.pkl')  # Text processing pipeline
    joblib.dump(clf, 'models/classifier.pkl')         # Trained classification model
    print("\nModels saved to /models directory!")

if __name__ == "__main__":
    main()