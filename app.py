import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from transformers import pipeline
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import io
import base64
from typing import List, Dict, Any
import json
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import string
import warnings
warnings.filterwarnings('ignore')

# Streamlit page configuration
st.set_page_config(
    page_title="Enhanced Multilingual Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

# Language configuration with Indonesian model
LANGUAGES = {
    'id': {
        'name': 'Indonesian',
        'flag': '🇮🇩',
        'model': 'mdhugol/indonesia-bert-sentiment-classification',
        'gemini_model': 'gemini-2.0-flash-exp',
        'label_mapping': {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}
    },
    'en': {
        'name': 'English',
        'flag': '🇺🇸',
        'model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'gemini_model': 'gemini-2.0-flash-exp',
        'label_mapping': {'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'}
    }
}

# Initialize session state for language and mode
if 'language' not in st.session_state:
    st.session_state.language = 'id'
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = 'unlabeled'

def set_language(lang):
    """Set application language"""
    st.session_state.language = lang
    st.rerun()

def set_analysis_mode(mode):
    """Set analysis mode"""
    st.session_state.analysis_mode = mode
    st.rerun()

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        return True
    except:
        return False

# Initialize Indonesian text preprocessing tools
@st.cache_resource
def init_indonesian_tools():
    """Initialize Indonesian stemmer and stopword remover"""
    try:
        # Sastrawi Stemmer
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        
        # Sastrawi Stopword Remover
        stopword_factory = StopWordRemoverFactory()
        stopword_remover = stopword_factory.create_stop_word_remover()
        
        return stemmer, stopword_remover
    except Exception as e:
        st.error(f"Error initializing Indonesian tools: {str(e)}")
        return None, None

# Initialize English text preprocessing tools
@st.cache_resource
def init_english_tools():
    """Initialize English stopwords and stemmer"""
    try:
        english_stopwords = set(stopwords.words('english'))
        porter_stemmer = PorterStemmer()
        return english_stopwords, porter_stemmer
    except Exception as e:
        st.error(f"Error initializing English tools: {str(e)}")
        return None, None

# Cache for loading models
@st.cache_resource
def load_sentiment_model(language):
    """Load sentiment classifier based on language"""
    try:
        model_name = LANGUAGES[language]['model']
        with st.spinner(f"Loading AI model {language.upper()}..."):
            classifier = pipeline(
                "sentiment-analysis", 
                model=model_name,
                return_all_scores=True
            )
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to setup Gemini AI
def setup_gemini(api_key: str, language: str):
    """Setup Gemini AI with API key using Flash 2.0"""
    try:
        genai.configure(api_key=api_key)
        model_name = LANGUAGES[language]['gemini_model']
        return genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"Error setting up Gemini: {str(e)}")
        return None

# Function to detect text and numeric columns
def detect_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect text and numeric columns"""
    text_columns = []
    numeric_columns = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_values = df[col].dropna().head(10)
            if len(sample_values) > 0:
                avg_length = sample_values.astype(str).str.len().mean()
                if avg_length > 10:
                    text_columns.append(col)
        elif df[col].dtype in ['int64', 'float64']:
            numeric_columns.append(col)
    
    return {'text': text_columns, 'numeric': numeric_columns}

# Function to convert numeric labels to sentiment
def convert_numeric_to_sentiment(value, scale_type='1-5'):
    """Convert numeric rating to sentiment category"""
    if pd.isna(value):
        return 'neutral'
    
    value = int(value)
    
    if scale_type == '1-5':
        if value in [1, 2]:
            return 'negative'
        elif value == 3:
            return 'neutral'
        elif value in [4, 5]:
            return 'positive'
    elif scale_type == '1-10':
        if value in [1, 2, 3, 4]:
            return 'negative'
        elif value in [5, 6]:
            return 'neutral'
        elif value in [7, 8, 9, 10]:
            return 'positive'
    
    return 'neutral'

# Comprehensive text preprocessing function
def comprehensive_text_preprocessing(text: str, 
                                   language: str = 'id',
                                   stemmer=None, 
                                   stopword_remover=None,
                                   english_stopwords=None,
                                   porter_stemmer=None,
                                   remove_urls=True,
                                   remove_mentions=True,
                                   remove_hashtags=True,
                                   remove_numbers=True,
                                   remove_punctuation=True,
                                   to_lowercase=True,
                                   remove_extra_spaces=True,
                                   remove_stopwords=True,
                                   apply_stemming=True,
                                   min_word_length=2) -> str:
    """
    Comprehensive text preprocessing with various options for Indonesian and English
    """
    if pd.isna(text) or not text:
        return ""
    
    text = str(text)
    original_text = text
    
    # 1. Remove URLs
    if remove_urls:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 2. Remove mentions (@username)
    if remove_mentions:
        text = re.sub(r'@\w+', '', text)
    
    # 3. Remove hashtags (#hashtag)
    if remove_hashtags:
        text = re.sub(r'#\w+', '', text)
    
    # 4. Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # 5. Convert to lowercase
    if to_lowercase:
        text = text.lower()
    
    # 6. Remove punctuation
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', ' ', text)
    
    # 7. Remove extra spaces
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text).strip()
    
    # 8. Remove stopwords
    if remove_stopwords:
        if language == 'id' and stopword_remover:
            try:
                text = stopword_remover.remove(text)
            except:
                # Fallback: manual Indonesian stopword removal
                indonesian_stopwords = {
                    'yang', 'dan', 'di', 'ke', 'dari', 'dalam', 'untuk', 'pada', 'dengan', 'adalah',
                    'ini', 'itu', 'tidak', 'atau', 'juga', 'akan', 'sudah', 'ada', 'dapat', 'bisa',
                    'saya', 'anda', 'kamu', 'dia', 'mereka', 'kita', 'kami', 'nya', 'mu', 'ku',
                    'sangat', 'sekali', 'lebih', 'paling', 'seperti', 'karena', 'jika', 'kalau',
                    'tetapi', 'tapi', 'namun', 'sehingga', 'lalu', 'kemudian', 'setelah', 'sebelum'
                }
                words = text.split()
                words = [word for word in words if word not in indonesian_stopwords]
                text = ' '.join(words)
        elif language == 'en' and english_stopwords:
            words = text.split()
            words = [word for word in words if word not in english_stopwords]
            text = ' '.join(words)
    
    # 9. Apply stemming
    if apply_stemming:
        if language == 'id' and stemmer:
            try:
                text = stemmer.stem(text)
            except:
                pass
        elif language == 'en' and porter_stemmer:
            try:
                words = text.split()
                words = [porter_stemmer.stem(word) for word in words]
                text = ' '.join(words)
            except:
                pass
    
    # 10. Filter words by minimum length
    if min_word_length > 0:
        words = text.split()
        words = [word for word in words if len(word) >= min_word_length]
        text = ' '.join(words)
    
    # 11. Final cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Return original text if preprocessing result is empty
    if not text and original_text:
        return original_text.strip()
    
    return text

# Function to map sentiment labels
def map_sentiment_label(label: str, language: str) -> str:
    """Map label from model to desired format"""
    label_mapping = LANGUAGES[language]['label_mapping']
    return label_mapping.get(label, 'neutral')

# Function for sentiment analysis
def analyze_sentiment(classifier, texts: List[str], language: str) -> List[Dict]:
    """Sentiment analysis using AI Model"""
    results = []
    
    progress_bar = st.progress(0)
    
    for i, text in enumerate(texts):
        if not text or len(text.strip()) < 3:
            continue
        
        try:
            prediction = classifier(text)
            all_scores = {item['label']: item['score'] for item in prediction[0]}
            
            scores = []
            for pred in prediction[0]:
                sentiment = map_sentiment_label(pred['label'], language)
                scores.append((sentiment, pred['score']))
            
            # Sort by highest score
            scores.sort(key=lambda x: x[1], reverse=True)
            
            best_sentiment, confidence = scores[0]
            
            results.append({
                "text": text,
                "sentiment": best_sentiment,
                "confidence": confidence,
                "all_scores": all_scores
            })
            
        except Exception as e:
            st.warning(f"Error analyzing text: {str(e)}")
            continue
        
        progress = (i + 1) / len(texts)
        progress_bar.progress(progress)
    
    return results

# Function to generate summary with Gemini Flash 2.0
def generate_summary_with_gemini(model, sentiment_counts: Dict, wordcloud_images: Dict, language: str, is_labeled: bool = False) -> str:
    """Generate summary and insights using Gemini Flash 2.0"""
    try:
        if not model:
            return "Error: Gemini model not initialized"
        
        # Convert matplotlib images to format that can be sent to Gemini
        image_parts = []
        for sentiment, fig in wordcloud_images.items():
            if fig:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                
                image_parts.append({
                    "inlineData": {
                        "data": base64.b64encode(buf.getvalue()).decode('utf-8'),
                        "mimeType": "image/png"
                    }
                })
        
        analysis_type = "labeled dataset analysis" if is_labeled else "AI model prediction results"
        
        prompt = f"""
        Analyze the following customer review sentiment results based on sentiment distribution and word cloud visualizations from {analysis_type}:
        
        Sentiment Distribution:
        - Positive: {sentiment_counts.get('positive', 0)} reviews
        - Negative: {sentiment_counts.get('negative', 0)} reviews  
        - Neutral: {sentiment_counts.get('neutral', 0)} reviews
        
        Based on the word clouds shown for each sentiment category, please provide analysis and insights in the following format:
        
        ## 📊 Analysis Summary
        [Summary of overall sentiment and dominant words visible in word clouds]
        
        ## 💡 Key Insights
        [3-5 important insights based on sentiment distribution and word patterns]
        
        ## 🎯 Action Recommendations
        [Concrete recommendations based on identified word patterns and sentiments]
        
        ## ⚠️ Areas of Concern
        [Things that need attention from word patterns and sentiment]
        
        Use professional and easily understandable English.
        """
        
        # Create multimodal prompt with text and images
        parts = [{"text": prompt}]
        parts.extend(image_parts)
        
        # Generate content with text and images
        response = model.generate_content(
            parts,
            generation_config={"temperature": 0.7}
        )
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Function to create wordcloud
def create_wordcloud(texts: List[str], sentiment: str) -> plt.Figure:
    """Create wordcloud for specific sentiment"""
    if not texts:
        return None
        
    combined_text = " ".join(texts)
    
    if len(combined_text.strip()) < 10:
        return None
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis' if sentiment == 'positive' else 'Reds' if sentiment == 'negative' else 'Blues',
        max_words=100,
        relative_scaling=0.5,
        collocations=False
    ).generate(combined_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud - {sentiment.title()} Sentiment', fontsize=16, fontweight='bold')
    
    return fig

# Function for sentiment visualization
def create_sentiment_chart(sentiment_counts: Dict[str, int]) -> go.Figure:
    """Create pie chart for sentiment distribution"""
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())
    
    colors = {
        'positive': '#2E8B57',  # Green
        'negative': '#DC143C',  # Red
        'neutral': '#4682B4'    # Blue
    }
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=[colors.get(label, '#808080') for label in labels]),
            textinfo='label+percent',
            textfont_size=12,
            hole=0.4
        )
    ])
    
    fig.update_layout(
        title="Sentiment Analysis Distribution",
        font=dict(size=14),
        showlegend=True
    )
    
    return fig

# Function for confidence distribution chart
def create_confidence_chart(results_df: pd.DataFrame) -> go.Figure:
    """Create histogram of confidence score distribution"""
    fig = px.histogram(
        results_df, 
        x='confidence', 
        color='sentiment',
        nbins=20,
        title='Confidence Score Distribution per Sentiment',
        labels={'confidence': 'Confidence Score', 'count': 'Number of Reviews'}
    )
    
    fig.update_layout(
        xaxis_title="Confidence Score",
        yaxis_title="Number of Reviews",
        showlegend=True
    )
    
    return fig

# Main application
def main():
    st.title("📊 Enhanced Multilingual Sentiment Analyzer")
    st.markdown("Upload CSV file for sentiment analysis with **comprehensive preprocessing** + **AI Model** + insights from **Gemini AI**")
    
    # Initialize NLTK tools
    download_nltk_data()
    
    # Language selection
    st.markdown("### Select Language for Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(f"{LANGUAGES['id']['flag']} {LANGUAGES['id']['name']}", 
                     type="primary" if st.session_state.language == 'id' else "secondary",
                     use_container_width=True):
            set_language('id')
    
    with col2:
        if st.button(f"{LANGUAGES['en']['flag']} {LANGUAGES['en']['name']}", 
                     type="primary" if st.session_state.language == 'en' else "secondary",
                     use_container_width=True):
            set_language('en')
    
    # Analysis mode selection
    st.markdown("### Analysis Mode")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Labeled Analysis", 
                     type="primary" if st.session_state.analysis_mode == 'labeled' else "secondary",
                     use_container_width=True):
            set_analysis_mode('labeled')
    
    with col2:
        if st.button("🤖 Unlabeled Analysis (AI Prediction)", 
                     type="primary" if st.session_state.analysis_mode == 'unlabeled' else "secondary",
                     use_container_width=True):
            set_analysis_mode('unlabeled')
    
    # Show mode description
    if st.session_state.analysis_mode == 'labeled':
        st.info("📋 **Labeled Mode**: Use existing sentiment labels in your dataset for analysis and visualization")
    else:
        st.info("🤖 **Unlabeled Mode**: Use AI model to predict sentiment from text content")
    
    # Initialize language-specific tools
    if st.session_state.language == 'id':
        stemmer, stopword_remover = init_indonesian_tools()
        english_stopwords, porter_stemmer = None, None
    else:
        stemmer, stopword_remover = None, None
        english_stopwords, porter_stemmer = init_english_tools()
    
    # Load sentiment model only for unlabeled mode
    classifier = None
    if st.session_state.analysis_mode == 'unlabeled':
        classifier = load_sentiment_model(st.session_state.language)
        if classifier is None:
            st.error("❌ Failed to load AI model. Please check internet connection.")
            return
        st.success("✅ AI Model loaded successfully!")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Gemini API Key input
    api_key = st.sidebar.text_input(
        "Gemini AI API Key (Optional)",
        type="password",
        help="To get AI-powered insights and summary"
    )
    
    gemini_model = None
    if api_key:
        gemini_model = setup_gemini(api_key, st.session_state.language)
        if gemini_model:
            st.sidebar.success("✅ Gemini AI ready for insights")
    
    # Preprocessing settings (only for unlabeled mode)
    if st.session_state.analysis_mode == 'unlabeled':
        st.sidebar.subheader("🧹 Preprocessing Settings")
        
        preprocess_options = {
            'remove_urls': st.sidebar.checkbox("Remove URLs", value=True),
            'remove_mentions': st.sidebar.checkbox("Remove Mentions (@)", value=True),
            'remove_hashtags': st.sidebar.checkbox("Remove Hashtags (#)", value=True),
            'remove_numbers': st.sidebar.checkbox("Remove Numbers", value=True),
            'remove_punctuation': st.sidebar.checkbox("Remove Punctuation", value=True),
            'to_lowercase': st.sidebar.checkbox("Convert to Lowercase", value=True),
            'remove_stopwords': st.sidebar.checkbox("Remove Stopwords", value=True),
            'apply_stemming': st.sidebar.checkbox("Apply Stemming", value=True),
        }
        
        min_word_length = st.sidebar.slider(
            "Minimum word length",
            min_value=1,
            max_value=5,
            value=2,
            help="Words shorter than this will be removed"
        )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload CSV file containing customer reviews"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Data preview
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head())
                st.write("**Dataset Info:**")
                st.write(f"- Number of rows: {len(df)}")
                st.write(f"- Number of columns: {len(df.columns)}")
                st.write(f"- Columns: {', '.join(df.columns)}")
            
            # Detect columns
            column_info = detect_columns(df)
            
            if st.session_state.analysis_mode == 'labeled':
                # Labeled mode configuration
                st.subheader("📋 Labeled Analysis Configuration")
                
                # Select text columns
                if not column_info['text']:
                    st.error("❌ No text columns detected for analysis")
                    return
                
                selected_text_columns = st.multiselect(
                    "Select text columns for analysis:",
                    column_info['text'],
                    help="Select columns containing review text"
                )
                
                # Select target column
                if not column_info['numeric']:
                    st.error("❌ No numeric columns detected for sentiment labels")
                    return
                
                target_column = st.selectbox(
                    "Select target column (sentiment labels):",
                    column_info['numeric'],
                    help="Select column containing sentiment labels"
                )
                
                # Determine scale type
                if target_column:
                    unique_values = sorted(df[target_column].dropna().unique())
                    st.write(f"**Unique values in target column:** {unique_values}")
                    
                    if max(unique_values) <= 5:
                        scale_type = '1-5'
                        st.info("**Scale Detection:** 1-5 scale detected\n- 1,2 = Negative\n- 3 = Neutral\n- 4,5 = Positive")
                    else:
                        scale_type = '1-10'
                        st.info("**Scale Detection:** 1-10 scale detected\n- 1,2,3,4 = Negative\n- 5,6 = Neutral\n- 7,8,9,10 = Positive")
                
                if not selected_text_columns or not target_column:
                    st.warning("⚠️ Please select both text columns and target column for analysis")
                    return
                
            else:
                # Unlabeled mode configuration
                if not column_info['text']:
                    st.error("❌ No text columns detected for sentiment analysis")
                    return
                
                selected_text_columns = st.multiselect(
                    "Select columns for sentiment analysis:",
                    column_info['text'],
                    help="Select columns containing review text"
                )
                
                if not selected_text_columns:
                    st.warning("⚠️ Please select at least one column for analysis")
                    return
            
            # Start analysis button
            if st.button("🚀 Start Analysis"):
                if st.session_state.analysis_mode == 'labeled':
                    # Labeled analysis
                    with st.spinner("📝 Processing labeled data..."):
                        # Combine text from selected columns
                        all_texts = []
                        all_sentiments = []
                        
                        for idx, row in df.iterrows():
                            # Combine text from all selected columns
                            combined_text = ' '.join([str(row[col]) for col in selected_text_columns if pd.notna(row[col])])
                            if combined_text.strip():
                                all_texts.append(combined_text)
                                # Convert numeric label to sentiment
                                sentiment = convert_numeric_to_sentiment(row[target_column], scale_type)
                                all_sentiments.append(sentiment)
                        
                        if not all_texts:
                            st.error("❌ No valid text for analysis")
                            return
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame({
                            'text': all_texts,
                            'sentiment': all_sentiments
                        })
                        
                        # Count sentiments
                        sentiment_counts = results_df['sentiment'].value_counts().to_dict()
                        
                else:
                    # Unlabeled analysis
                    with st.spinner("📝 Processing and cleaning text..."):
                        # Process each selected column
                        all_texts = []
                        original_sample_texts = []
                        processed_sample_texts = []
                        
                        for col in selected_text_columns:
                            texts_from_col = df[col].astype(str).tolist()
                            for text in texts_from_col:
                                processed_text = comprehensive_text_preprocessing(
                                    text,
                                    language=st.session_state.language,
                                    stemmer=stemmer,
                                    stopword_remover=stopword_remover,
                                    english_stopwords=english_stopwords,
                                    porter_stemmer=porter_stemmer,
                                    **preprocess_options,
                                    min_word_length=min_word_length
                                )
                                if processed_text:
                                    all_texts.append(processed_text)
                                    # Take some examples for display
                                    if len(original_sample_texts) < 5:
                                        original_sample_texts.append(text)
                                        processed_sample_texts.append(processed_text)
                        
                        if not all_texts:
                            st.error("❌ No valid text for analysis")
                            return
                        
                        # Show preprocessing examples
                        with st.expander("🧹 Preprocessing Results"):
                            st.subheader("📝 Preprocessing Examples")
                            for i in range(len(original_sample_texts)):
                                st.markdown(f"**Original {i+1}:** {original_sample_texts[i]}")
                                st.markdown(f"**Processed {i+1}:** {processed_sample_texts[i]}")
                                st.markdown("---")
                        
                        # Sentiment analysis
                        with st.spinner(f"📊 Analyzing {len(all_texts)} texts with AI Model..."):
                            results = analyze_sentiment(classifier, all_texts, st.session_state.language)
                            
                            # Convert results to DataFrame
                            results_df = pd.DataFrame(results)
                            
                            # Count sentiments
                            sentiment_counts = results_df['sentiment'].value_counts().to_dict()
                
                # Display results
                st.header("📈 Analysis Results")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Reviews", len(results_df))
                with col2:
                    st.metric("Positive", sentiment_counts.get('positive', 0))
                with col3:
                    st.metric("Negative", sentiment_counts.get('negative', 0))
                with col4:
                    st.metric("Neutral", sentiment_counts.get('neutral', 0))
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Sentiment Distribution")
                    sentiment_chart = create_sentiment_chart(sentiment_counts)
                    st.plotly_chart(sentiment_chart, use_container_width=True)
                
                with col2:
                    if st.session_state.analysis_mode == 'unlabeled':
                        st.subheader("📈 Confidence Distribution")
                        confidence_chart = create_confidence_chart(results_df)
                        st.plotly_chart(confidence_chart, use_container_width=True)
                    else:
                        st.subheader("📊 Label Distribution")
                        # Create a simple bar chart for labeled data
                        fig = px.bar(
                            x=list(sentiment_counts.keys()),
                            y=list(sentiment_counts.values()),
                            title="Sentiment Label Distribution",
                            labels={'x': 'Sentiment', 'y': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Generate AI Insights if Gemini is available
                if gemini_model:
                    st.header("🤖 AI-Powered Insights")
                    with st.spinner("Generating insights with Gemini AI..."):
                        wordcloud_images = {}
                        for sentiment_type in ['positive', 'negative', 'neutral']:
                            sentiment_texts = results_df[results_df['sentiment'] == sentiment_type]['text'].tolist()
                            wordcloud_images[sentiment_type] = create_wordcloud(sentiment_texts, sentiment_type)
                        
                        # Generate summary with Gemini
                        summary = generate_summary_with_gemini(
                            gemini_model, 
                            sentiment_counts, 
                            wordcloud_images, 
                            st.session_state.language,
                            is_labeled=(st.session_state.analysis_mode == 'labeled')
                        )
                        
                        # Display word clouds
                        st.subheader("☁️ Word Clouds by Sentiment")
                        for sentiment_type in ['positive', 'negative', 'neutral']:
                            if wordcloud_images[sentiment_type] is not None:
                                st.markdown(f"**{sentiment_type.title()} Sentiment**")
                                st.pyplot(wordcloud_images[sentiment_type])
                                plt.close()
                        
                        # Display AI-generated insights
                        st.markdown(summary)
                
                # Display detailed results
                st.header("📋 Detailed Results")
                
                # Filter controls
                col1, col2 = st.columns(2)
                with col1:
                    sentiment_filter = st.selectbox(
                        "Filter by sentiment:",
                        ['All'] + list(sentiment_counts.keys())
                    )
                
                with col2:
                    if st.session_state.analysis_mode == 'unlabeled':
                        confidence_threshold = st.slider(
                            "Minimum confidence:",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.5,
                            step=0.1
                        )
                
                # Apply filters
                filtered_df = results_df.copy()
                if sentiment_filter != 'All':
                    filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_filter]
                
                if st.session_state.analysis_mode == 'unlabeled':
                    filtered_df = filtered_df[filtered_df['confidence'] >= confidence_threshold]
                
                # Display filtered results
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download results
                st.header("💾 Download Results")
                
                # Convert results to CSV
                csv_buffer = io.StringIO()
                filtered_df.to_csv(csv_buffer, index=False)
                csv_string = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_string,
                    file_name=f"sentiment_analysis_results_{st.session_state.language}_{st.session_state.analysis_mode}.csv",
                    mime="text/csv"
                )
                
                # Statistics summary
                st.header("📊 Statistics Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Sentiment Statistics")
                    total_reviews = len(results_df)
                    
                    for sentiment in ['positive', 'negative', 'neutral']:
                        count = sentiment_counts.get(sentiment, 0)
                        percentage = (count / total_reviews) * 100 if total_reviews > 0 else 0
                        st.write(f"**{sentiment.title()}:** {count} reviews ({percentage:.1f}%)")
                
                with col2:
                    if st.session_state.analysis_mode == 'unlabeled':
                        st.subheader("🎯 Confidence Statistics")
                        avg_confidence = results_df['confidence'].mean()
                        min_confidence = results_df['confidence'].min()
                        max_confidence = results_df['confidence'].max()
                        
                        st.write(f"**Average Confidence:** {avg_confidence:.3f}")
                        st.write(f"**Min Confidence:** {min_confidence:.3f}")
                        st.write(f"**Max Confidence:** {max_confidence:.3f}")
                        
                        # High confidence predictions
                        high_conf_count = len(results_df[results_df['confidence'] > 0.8])
                        high_conf_percentage = (high_conf_count / total_reviews) * 100
                        st.write(f"**High Confidence (>0.8):** {high_conf_count} reviews ({high_conf_percentage:.1f}%)")
                
                # Advanced analytics
                if st.session_state.analysis_mode == 'unlabeled':
                    st.header("🔍 Advanced Analytics")
                    
                    # Sentiment by confidence ranges
                    st.subheader("📊 Sentiment Distribution by Confidence Range")
                    
                    # Create confidence ranges
                    results_df['confidence_range'] = pd.cut(
                        results_df['confidence'],
                        bins=[0, 0.5, 0.7, 0.9, 1.0],
                        labels=['Low (0-0.5)', 'Medium (0.5-0.7)', 'High (0.7-0.9)', 'Very High (0.9-1.0)']
                    )
                    
                    # Create cross-tabulation
                    conf_sentiment_crosstab = pd.crosstab(
                        results_df['confidence_range'],
                        results_df['sentiment']
                    )
                    
                    # Display as heatmap
                    fig = px.imshow(
                        conf_sentiment_crosstab.values,
                        labels=dict(x="Sentiment", y="Confidence Range", color="Count"),
                        x=conf_sentiment_crosstab.columns,
                        y=conf_sentiment_crosstab.index,
                        color_continuous_scale="Blues",
                        title="Sentiment Distribution by Confidence Range"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Text length analysis
                st.header("📏 Text Length Analysis")
                
                results_df['text_length'] = results_df['text'].str.len()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Text length by sentiment
                    fig = px.box(
                        results_df,
                        x='sentiment',
                        y='text_length',
                        title='Text Length Distribution by Sentiment'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Text length statistics
                    st.subheader("📊 Text Length Statistics")
                    
                    for sentiment in ['positive', 'negative', 'neutral']:
                        sentiment_data = results_df[results_df['sentiment'] == sentiment]
                        if len(sentiment_data) > 0:
                            avg_length = sentiment_data['text_length'].mean()
                            st.write(f"**{sentiment.title()} avg length:** {avg_length:.1f} chars")
                
                # Success message
                st.success("✅ Analysis completed successfully!")
                
                # Tips for improvement
                st.header("💡 Tips for Better Analysis")
                
                if st.session_state.analysis_mode == 'unlabeled':
                    st.markdown("""
                    **For better sentiment analysis results:**
                    
                    1. **Text Quality:** Ensure your text data is clean and meaningful
                    2. **Language Consistency:** Use the correct language model for your data
                    3. **Preprocessing:** Adjust preprocessing settings based on your data characteristics
                    4. **Confidence Threshold:** Filter results by confidence score for more reliable predictions
                    5. **Sample Size:** Larger datasets generally provide more reliable insights
                    """)
                else:
                    st.markdown("""
                    **For better labeled analysis:**
                    
                    1. **Label Consistency:** Ensure sentiment labels are consistent across your dataset
                    2. **Scale Understanding:** Verify that the detected scale mapping matches your data
                    3. **Text Quality:** Clean text data provides better word cloud visualizations
                    4. **Data Balance:** Consider if your dataset has balanced sentiment distribution
                    5. **Validation:** Cross-check results with known sentiment patterns in your domain
                    """)
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.error("Please check your file format and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>Enhanced Multilingual Sentiment Analyzer</p>
            <p>Powered by 🤖 AI Models + 🔥 Gemini AI + 📊 Advanced Analytics</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
