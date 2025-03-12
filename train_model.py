import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')

# Text cleaning function
def clean_text(text):
    """Full text preprocessing pipeline"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    
    # Tokenization and filtering
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if 
            (word not in stop_words) and 
            (len(word) > 2) and 
            (not word.startswith(('http', 'www')))]
    return ' '.join(words)

def main():
    # Load and preprocess data
    data = pd.read_csv('.\Customer Feedback Analysis.csv')
    print(f"Initial rows: {len(data)}")
    
    # Clean data
    data = data.dropna(subset=['Reviews', 'Rating'])
    data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')
    data = data.dropna(subset=['Rating'])
    data['Category'] = data['Rating'].apply(
        lambda x: 'Complaint' if x <= 2 else 'Suggestion' if x == 3 else 'Praise'
    )
    data['Cleaned_Review'] = data['Reviews'].apply(clean_text)
    data = data[data['Cleaned_Review'].str.strip() != '']
    print(f"Final training samples: {len(data)}")
    
    # Train and save model
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['Cleaned_Review'])
    clf = MultinomialNB()
    clf.fit(X, data['Category'])
    
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    joblib.dump(clf, 'models/classifier.pkl')
    print("Models saved to /models directory!")

if __name__ == "__main__":
    main()