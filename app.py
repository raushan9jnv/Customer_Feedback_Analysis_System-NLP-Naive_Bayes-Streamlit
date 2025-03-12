# Core libraries
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re

# Configure Streamlit page settings
st.set_page_config(
    page_title="Live Feedback Analyzer", 
    layout="wide",  # Utilize full browser width
    page_icon="📊"   # Browser tab icon
)

# Model Loading with Caching
@st.cache_resource  # Prevent reloading on every interaction
def load_models():
    """Load persisted models from disk"""
    return (
        joblib.load('models/vectorizer.pkl'),  # TF-IDF vectorizer
        joblib.load('models/classifier.pkl')   # Trained classifier
    )

# Data Loading with Caching 
@st.cache_data  # Maintain dataset consistency per session
def load_data():
    """Load and preprocess feedback data"""
    data = pd.read_csv('Customer Feedback Analysis.csv')
    # Mirror training data preprocessing steps:
    data = data.dropna(subset=['Reviews', 'Rating'])
    data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')
    data['Category'] = data['Rating'].apply(
        lambda x: 'Complaint' if x <=2 else 'Suggestion' if x==3 else 'Praise')
    return data.dropna(subset=['Rating'])

# Initialize application components
vectorizer, model = load_models()
data = load_data()

# Configure page header
st.header("🔮 Real-Time Feedback Analyzer", divider="rainbow")

# Custom CSS styling for chat interface
st.markdown("""
<style>
    /* Chat input box sizing */
    .stChatInput textarea {min-height: 150px!important;}
    
    /* Feedback category styling */
    .feedback-box {
        border-left: 5px solid; 
        padding: 15px; 
        margin: 10px 0; 
        border-radius: 5px;
    }
    .complaint {border-color: #ff4444; background: #fff5f5;}  /* Red for complaints */
    .suggestion {border-color: #ffd700; background: #fffee5;}  /* Yellow for suggestions */
    .praise {border-color: #00c851; background: #f5fff7;}      /* Green for praise */
</style>
""", unsafe_allow_html=True)

# Session state management for chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Real-time feedback processing
user_input = st.chat_input("Paste customer feedback here...")
if user_input:
    # Clean input using same pipeline as training
    clean_input = re.sub(r'[^\w\s]', '', user_input.lower())
    
    # Vectorize and predict
    vec = vectorizer.transform([clean_input])
    pred = model.predict(vec)[0]
    
    # Generate response messages
    response = {
        'Complaint': "🚨 **Urgent Attention Needed** - Escalated to support team!",
        'Suggestion': "💡 **Innovation Alert** - Added to product roadmap!",
        'Praise': "🌟 **Customer Love** - Shared with executive team!"
    }[pred]
    
    # Update chat history
    st.session_state.history.append({
        "type": "user", 
        "content": user_input,
        "prediction": pred
    })
    st.session_state.history.append({
        "type": "bot", 
        "content": response
    })

# Layout configuration
col1, col2 = st.columns([0.7, 0.3])  # 70-30 split

# Chat History Display
with col1:
    for msg in st.session_state.history[-4:]:  # Show last 4 messages
        if msg['type'] == 'user':
            # Color-coded feedback display
            color_class = {
                'Complaint': 'complaint',
                'Suggestion': 'suggestion',
                'Praise': 'praise'
            }[msg['prediction']]
            
            # Dynamic styling based on category
            st.markdown(
                f"<div class='feedback-box {color_class}'>"
                f"<h4 style='margin-top:0; color: {'#cc0000' if msg['prediction'] == 'Complaint' else '#b38f00' if msg['prediction'] == 'Suggestion' else '#007E33'}'>"
                f"📩 {msg['prediction']}</h4>"
                f"<div style='font-size: 14px;'>{msg['content']}</div></div>",
                unsafe_allow_html=True
            )
        else:
            st.success(msg['content'])  # Bot response styling

# Analytics Panel
with col2:
    tab1, tab2 = st.tabs(["📊 Statistics", "🗣️ Word Analysis"])
    
    # Statistics Tab
    with tab1:
        st.subheader("Feedback Distribution")
        
        # Pie Chart - Category Proportions
        fig1, ax1 = plt.subplots(figsize=(6,6))
        category_counts = data['Category'].value_counts()
        colors = ['#ff4444', '#ffd700', '#00c851']  # Match chat colors
        ax1.pie(category_counts, 
                labels=category_counts.index, 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=colors,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax1.axis('equal')
        st.pyplot(fig1)
        
        # Histogram - Rating Distribution
        st.subheader("Rating Distribution")
        fig2, ax2 = plt.subplots(figsize=(8,3))
        sns.histplot(data=data, x='Rating', bins=5, 
                    kde=True, color='#4CAF50')  # Green for positive skew
        ax2.set_xlabel("")
        ax2.set_ylabel("")
        st.pyplot(fig2)
    
    # Word Analysis Tab
    with tab2:
        st.subheader("Most Frequent Words")
        
        def generate_wordcloud(text, title, color):
            """Visualize prominent terms in feedback"""
            wc = WordCloud(width=400, height=200,
                          background_color='white',
                          colormap=color).generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            ax.set_title(title, fontsize=12)
            return fig
        
        # Three-column layout for category word clouds
        col_a, col_b, col_c = st.columns(3)
        
        # Complaint Word Cloud
        with col_a:
            complaint_text = ' '.join(data[data['Category'] == 'Complaint']['Reviews'])
            st.pyplot(generate_wordcloud(complaint_text, "Complaints", 'Reds'))
        
        # Suggestion Word Cloud
        with col_b:
            suggestion_text = ' '.join(data[data['Category'] == 'Suggestion']['Reviews'])
            st.pyplot(generate_wordcloud(suggestion_text, "Suggestions", 'Oranges'))
        
        # Praise Word Cloud  
        with col_c:
            praise_text = ' '.join(data[data['Category'] == 'Praise']['Reviews'])
            st.pyplot(generate_wordcloud(praise_text, "Praises", 'Greens'))