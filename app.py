import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re

# Configuration
st.set_page_config(
    page_title="Live Feedback Analyzer", 
    layout="wide",
    page_icon="📊"
)

# Load models/data
@st.cache_resource
def load_models():
    return (joblib.load('models/vectorizer.pkl'), 
            joblib.load('models/classifier.pkl'))

vectorizer, model = load_models()

@st.cache_data
def load_data():
    data = pd.read_csv('.\Customer Feedback Analysis.csv')
    data = data.dropna(subset=['Reviews', 'Rating'])
    data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')
    data['Category'] = data['Rating'].apply(
        lambda x: 'Complaint' if x <=2 else 'Suggestion' if x==3 else 'Praise')
    return data.dropna(subset=['Rating'])

data = load_data()

# Main page layout
st.header("🔮 Real-Time Feedback Analyzer", divider="rainbow")
st.markdown("""
<style>
.stChatInput textarea {min-height: 150px!important;}
.feedback-box {border-left: 5px solid; padding: 15px; margin: 10px 0; border-radius: 5px;}
.complaint {border-color: #ff4444; background: #fff5f5;}
.suggestion {border-color: #ffd700; background: #fffee5;}
.praise {border-color: #00c851; background: #f5fff7;}
</style>
""", unsafe_allow_html=True)

# Chat input
if 'history' not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Paste customer feedback here...")
if user_input:
    clean_input = re.sub(r'[^\w\s]', '', user_input.lower())
    vec = vectorizer.transform([clean_input])
    pred = model.predict(vec)[0]
    
    response = {
        'Complaint': "🚨 **Urgent Attention Needed** - We've flagged this to our team!",
        'Suggestion': "💡 **Innovation Alert** - Added to our improvement roadmap!",
        'Praise': "🌟 **Customer Love** - Sharing with the whole team!"
    }[pred]
    
    st.session_state.history.append({
        "type": "user", 
        "content": user_input,
        "prediction": pred
    })
    st.session_state.history.append({
        "type": "bot", 
        "content": response
    })

# Chat display
col1, col2 = st.columns([0.7, 0.3])
with col1:
    for msg in st.session_state.history[-4:]:
        if msg['type'] == 'user':
            # Color-coded feedback boxes
            color_class = {
                'Complaint': 'complaint',
                'Suggestion': 'suggestion',
                'Praise': 'praise'
            }[msg['prediction']]
            
            st.markdown(
                f"<div class='feedback-box {color_class}'>"
                f"<h4 style='margin-top:0; color: {'#cc0000' if msg['prediction'] == 'Complaint' else '#b38f00' if msg['prediction'] == 'Suggestion' else '#007E33'}'>"
                f"📩 {msg['prediction']}</h4>"
                f"<div style='font-size: 14px;'>{msg['content']}</div></div>",
                unsafe_allow_html=True
            )
        else:
            st.success(msg['content'])

# Visualization tabs
with col2:
    tab1, tab2 = st.tabs(["📊 Statistics", "🗣️ Word Analysis"])
    
    with tab1:
        st.subheader("Feedback Distribution")
        
        # Pie Chart
        fig1, ax1 = plt.subplots(figsize=(6,6))
        category_counts = data['Category'].value_counts()
        colors = ['#ff4444', '#ffd700', '#00c851']
        ax1.pie(category_counts, labels=category_counts.index, 
                autopct='%1.1f%%', startangle=90, colors=colors,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax1.axis('equal')
        st.pyplot(fig1)
        
        # Rating Histogram
        st.subheader("Rating Distribution")
        fig2, ax2 = plt.subplots(figsize=(8,3))
        sns.histplot(data=data, x='Rating', bins=5, 
                    kde=True, color='#4CAF50')
        ax2.set_xlabel("")
        ax2.set_ylabel("")
        st.pyplot(fig2)
    
    with tab2:
        st.subheader("Most Frequent Words")
        
        def generate_wordcloud(text, title, color):
            wc = WordCloud(width=400, height=200,
                          background_color='white',
                          colormap=color).generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            ax.set_title(title, fontsize=12)
            return fig
        
        # Three columns for word clouds
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            complaint_text = ' '.join(data[data['Category'] == 'Complaint']['Reviews'])
            st.pyplot(generate_wordcloud(complaint_text, "Complaints", 'Reds'))
        
        with col_b:
            suggestion_text = ' '.join(data[data['Category'] == 'Suggestion']['Reviews'])
            st.pyplot(generate_wordcloud(suggestion_text, "Suggestions", 'Oranges'))
        
        with col_c:
            praise_text = ' '.join(data[data['Category'] == 'Praise']['Reviews'])
            st.pyplot(generate_wordcloud(praise_text, "Praises", 'Greens'))