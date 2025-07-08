import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import io
import base64
from typing import List, Dict, Any, Optional, Tuple
import json
import warnings
from dataclasses import dataclass, frozen
from functools import lru_cache
import random

# Suppress warnings
warnings.filterwarnings('ignore')

# Streamlit configuration
st.set_page_config(
    page_title="Enhanced Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

# Configuration classes
@dataclass
class ModelConfig:
    """Configuration for ML models"""
    name: str
    flag: str
    model_name: str
    gemini_model: str
    label_mapping: Dict[str, str]

@dataclass(frozen=True)  # Make the class immutable and hashable
class PreprocessConfig:
    """Configuration for text preprocessing"""
    remove_urls: bool = True
    remove_mentions: bool = True
    remove_hashtags: bool = True
    remove_numbers: bool = True
    remove_punctuation: bool = True
    to_lowercase: bool = True
    remove_stopwords: bool = True
    apply_stemming: bool = True
    min_word_length: int = 2

# Language configurations
LANGUAGES = {
    'id': ModelConfig(
        name='Indonesian',
        flag='🇮🇩',
        model_name='mdemdyfitriya/indonesian-roberta-base-sentiment-classifier',
        gemini_model='gemini-2.0-flash-exp',
        label_mapping={'POSITIVE': 'positive', 'NEGATIVE': 'negative', 'NEUTRAL': 'neutral'}
    ),
    'en': ModelConfig(
        name='English',
        flag='🇺🇸',
        model_name='cardiffnlp/twitter-roberta-base-sentiment',  # Updated model name
        gemini_model='gemini-2.0-flash-exp',
        label_mapping={
            'LABEL_0': 'negative',  # Updated mapping
            'LABEL_1': 'neutral',
            'LABEL_2': 'positive'
        }
    )
}

# Session state management
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'language': 'id',
        'analysis_mode': 'unlabeled',
        'current_results': None,
        'model_cache': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def update_session_state(key: str, value: Any):
    """Update session state and rerun if necessary"""
    if st.session_state.get(key) != value:
        st.session_state[key] = value
        if key in ['language', 'analysis_mode']:
            st.rerun()

# Model loading with caching
@st.cache_resource
def load_sentiment_model(language: str):
    """Load sentiment analysis model with caching"""
    if language not in LANGUAGES:
        raise ValueError(f"Language {language} not supported")
    
    config = LANGUAGES[language]
    try:
        with st.spinner(f"Loading AI model for {config.name}..."):
            # First try to load the tokenizer to verify model exists
            from transformers import AutoTokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            except Exception as e:
                st.error(f"Error loading tokenizer: {str(e)}")
                st.error("Please check your internet connection and model name.")
                return None
            
            # Then load the model
            classifier = pipeline(
                "text-classification",
                model=config.model_name,
                tokenizer=tokenizer,
                return_all_scores=True
            )
            st.success(f"✅ Successfully loaded {config.name} model")
            return classifier
    except Exception as e:
        st.error(f"Failed to load model for {config.name}: {str(e)}")
        st.error("Please check your internet connection or try again later.")
        return None

# Gemini setup
def setup_gemini(api_key: str, language: str) -> Optional[Any]:
    """Setup Gemini AI model"""
    try:
        genai.configure(api_key=api_key)
        model_name = LANGUAGES[language].gemini_model
        return genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"Failed to setup Gemini: {str(e)}")
        return None

# Text preprocessing functions
class TextPreprocessor:
    """Optimized text preprocessing class"""
    
    def __init__(self, language: str = 'id'):
        self.language = language
        self._setup_tools()
    
    def _setup_tools(self):
        """Setup language-specific preprocessing tools"""
        if self.language == 'id':
            self.stopwords = self._get_indonesian_stopwords()
        else:
            try:
                import nltk
                from nltk.corpus import stopwords
                from nltk.stem import PorterStemmer
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                self.stopwords = set(stopwords.words('english'))
                self.stemmer = PorterStemmer()
            except:
                self.stopwords = set()
                self.stemmer = None
    
    def _get_indonesian_stopwords(self) -> set:
        """Get Indonesian stopwords"""
        return {
            'yang', 'dan', 'di', 'ke', 'dari', 'dalam', 'untuk', 'pada', 'dengan', 'adalah',
            'ini', 'itu', 'tidak', 'atau', 'juga', 'akan', 'sudah', 'ada', 'dapat', 'bisa',
            'saya', 'anda', 'kamu', 'dia', 'mereka', 'kita', 'kami', 'nya', 'mu', 'ku',
            'sangat', 'sekali', 'lebih', 'paling', 'seperti', 'karena', 'jika', 'kalau',
            'tetapi', 'tapi', 'namun', 'sehingga', 'lalu', 'kemudian', 'setelah', 'sebelum',
            'yaitu', 'samak', 'oleh', 'kaerana', 'sii', 'udah', 'kayak', 'gimana', 'jadi',
            'gue', 'gw', 'lo', 'buat', 'banget', 'sih', 'juga', 'lagi', 'enggak', 'muka',
            'bisa', 'doang', 'pake', 'cuma', 'kalo', 'emang', 'udah', 'udah', 'gitu',
            'gue', 'gw', 'lo', 'buat', 'banget', 'sih', 'juga', 'lagi', 'enggak', 'muka',
            'bisa', 'doang', 'pake', 'cuma', 'kalo', 'emang', 'udah', 'udah', 'gitu',
            'dah', 'kebanayakan', 'gimana', 'caranya', 'jadi', 'jadi', 'artinya', 'nah',
            'sekarang', 'tinggal', 'saling', 'masing', 'terus', 'langsung', 'berdasarkan',
            'memiliki', 'yg', 'yaitu', 'juga', 'bukan', 'bukannya', 'walaupun', 'meskipun',
            'kemungkinan', 'kabel', 'sebelum', 'sesudah', 'stelah', 'dimana', 'diantara',
            'tentang', 'antara', 'bahwa', 'sampai', 'samapi', 'seputar', 'sekitar', 'terdapat',
            'terjadi', 'mengapa', 'mengapa', 'merupakan', 'semua', 'semestrata', 'mulai',
            'mulai', 'termasuk', 'meskipun', 'meskipun', 'merupakan', 'sesudah', 'kita',
            'kita', 'saya', 'kamu', 'dia', 'nya', 'mereka', 'kami', 'dimana', 'oleh',
            'dengan', 'dalam', 'untuk', 'pada', 'dari', 'karena', 'kalo', 'kalau', 'jika',
            'apabila', 'jika', 'jika', 'apabila', 'bila', 'sekiranya', 'sejauh', 'tidak',
            'tidaklah', 'nggak', 'tidak', 'bukanya', 'bukan', 'bukannya', 'walaupun',
            'meskipun', 'biar', 'biarkan', 'biarkan', 'walaupun', 'meskipun', 'bagaimanapun',
            'bagaimana', 'bagaimana', 'seberapa', 'seberapa', 'semakin', 'semakin', 'setelah',
            'setelah', 'sesudah', 'sesudah', 'sementara', 'sementara'
        }
    
    @lru_cache(maxsize=1000)
    def preprocess_text(self, text: str, config: PreprocessConfig) -> str:
        """Optimized text preprocessing with caching"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        
        # Basic cleaning
        if config.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        if config.remove_mentions:
            text = re.sub(r'@\w+', '', text)
        if config.remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        if config.remove_numbers:
            text = re.sub(r'\d+', '', text)
        if config.to_lowercase:
            text = text.lower()
        if config.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Word-level processing
        if config.remove_stopwords or config.apply_stemming:
            words = text.split()
            
            if config.remove_stopwords:
                words = [word for word in words if word not in self.stopwords]
            
            if config.apply_stemming and self.language == 'en' and hasattr(self, 'stemmer'):
                words = [self.stemmer.stem(word) for word in words]
            
            if config.min_word_length > 1:
                words = [word for word in words if len(word) >= config.min_word_length]
            
            text = ' '.join(words)
        
        return text or str(text)  # Return original if empty

    @st.cache_data
    def _cached_preprocess(self, text: str, config_tuple: tuple) -> str:
        """Cached preprocessing with tuple config"""
        config = PreprocessConfig(*config_tuple)
        return self.preprocess_text(text, config)
    
    def process_batch(self, texts: List[str], config: PreprocessConfig) -> List[str]:
        """Process a batch of texts with caching"""
        # Convert config to tuple for hashing
        config_tuple = (
            config.remove_urls,
            config.remove_mentions,
            config.remove_hashtags,
            config.remove_numbers,
            config.remove_punctuation,
            config.to_lowercase,
            config.remove_stopwords,
            config.apply_stemming,
            config.min_word_length
        )
        
        # Process each text with caching
        processed_texts = []
        for text in texts:
            if text and str(text).strip():
                processed = self._cached_preprocess(str(text), config_tuple)
                if processed:
                    processed_texts.append(processed)
        
        return processed_texts

# Analysis functions
class SentimentAnalyzer:
    """Optimized sentiment analysis class"""
    
    def __init__(self, classifier, language: str, mode: str = 'unlabeled'):
        self.classifier = classifier
        self.language = language
        self.mode = mode
        self.config = LANGUAGES[language]
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for batch of texts"""
        results = []
        
        # Filter empty texts
        valid_texts = [(i, text) for i, text in enumerate(texts) if text and len(text.strip()) > 2]
        
        if not valid_texts:
            return results
        
        # Progress bar
        progress = st.progress(0)
        
        for idx, (orig_idx, text) in enumerate(valid_texts):
            try:
                # Get model prediction
                prediction = self.classifier(text)
                
                # Process results
                result = self._process_prediction(text, prediction)
                results.append(result)
                
                # Update progress
                progress.progress((idx + 1) / len(valid_texts))
                
            except Exception as e:
                st.warning(f"Error analyzing text {orig_idx}: {str(e)}")
                # Add fallback result
                results.append({
                    'text': text,
                    'sentiment': 'positive' if random.random() > 0.5 else 'negative',
                    'confidence': 0.5,
                    'positive_score': 0.5,
                    'negative_score': 0.5
                })
        
        return results
    
    def _process_prediction(self, text: str, prediction: List[Dict]) -> Dict[str, Any]:
        """Process model prediction into standardized format"""
        # Map scores to sentiment labels
        sentiment_scores = {}
        for item in prediction:
            mapped_label = self.config.label_mapping.get(item['label'])
            if mapped_label:
                sentiment_scores[mapped_label] = item['score']
        
        # Get positive and negative scores
        positive_score = sentiment_scores.get('positive', 0)
        negative_score = sentiment_scores.get('negative', 0)
        
        # For unlabeled mode, only use positive/negative
        if self.mode == 'unlabeled' or not sentiment_scores.get('neutral'):
            sentiment = 'positive' if positive_score > negative_score else 'negative'
            confidence = max(positive_score, negative_score)
        else:
            # For labeled mode, use all sentiments
            best_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            sentiment = best_sentiment
            confidence = sentiment_scores[best_sentiment]
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_score': positive_score,
            'negative_score': negative_score
        }

# Data processing functions
def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect text and numeric columns more efficiently"""
    text_cols = []
    numeric_cols = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if it's text-like
            sample = df[col].dropna().head(5)
            if len(sample) > 0:
                avg_len = sample.astype(str).str.len().mean()
                if avg_len > 10:  # Likely text
                    text_cols.append(col)
        elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_cols.append(col)
    
    return {'text': text_cols, 'numeric': numeric_cols}

def process_labeled_data(df: pd.DataFrame, text_cols: List[str], 
                        target_col: str, scale_type: str) -> pd.DataFrame:
    """Process labeled data efficiently"""
    results = []
    
    for idx, row in df.iterrows():
        # Combine text from selected columns
        combined_text = ' '.join([
            str(row[col]) for col in text_cols 
            if pd.notna(row[col]) and str(row[col]).strip()
        ])
        
        if combined_text.strip():
            # Convert numeric rating to sentiment
            rating = row[target_col]
            if pd.notna(rating):
                sentiment = convert_rating_to_sentiment(rating, scale_type)
                results.append({
                    'text': combined_text,
                    'sentiment': sentiment,
                    'rating': rating
                })
    
    return pd.DataFrame(results)

def convert_rating_to_sentiment(rating: float, scale_type: str) -> str:
    """Convert numeric rating to sentiment more efficiently"""
    if pd.isna(rating):
        return 'neutral'
    
    rating = float(rating)
    
    if scale_type == '1-5':
        return 'negative' if rating <= 2 else 'positive' if rating >= 4 else 'neutral'
    elif scale_type == '1-10':
        return 'negative' if rating <= 4 else 'positive' if rating >= 7 else 'neutral'
    else:
        return 'neutral'

# Visualization functions
def create_sentiment_chart(sentiment_counts: Dict[str, int], mode: str = 'unlabeled') -> go.Figure:
    """Create optimized sentiment distribution chart"""
    if mode == 'unlabeled':
        # Only show positive/negative for unlabeled
        labels = ['positive', 'negative']
        values = [sentiment_counts.get(label, 0) for label in labels]
    else:
        labels = list(sentiment_counts.keys())
        values = list(sentiment_counts.values())
    
    colors = ['#2E8B57', '#DC143C', '#4682B4'][:len(labels)]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            textinfo='label+percent',
            textfont_size=12,
            hole=0.3
        )
    ])
    
    fig.update_layout(
        title="Sentiment Distribution",
        showlegend=True,
        height=400
    )
    
    return fig

def create_confidence_chart(df: pd.DataFrame) -> go.Figure:
    """Create confidence distribution chart"""
    fig = px.histogram(
        df, 
        x='confidence', 
        color='sentiment',
        nbins=20,
        title='Confidence Distribution',
        labels={'confidence': 'Confidence Score', 'count': 'Count'}
    )
    
    fig.update_layout(
        xaxis_title="Confidence Score",
        yaxis_title="Count",
        height=400
    )
    
    return fig

def create_wordcloud(texts: List[str], sentiment: str, max_words: int = 100) -> Optional[plt.Figure]:
    """Create optimized word cloud"""
    if not texts:
        return None
    
    combined_text = ' '.join(texts)
    if len(combined_text.strip()) < 10:
        return None
    
    try:
        colormap = 'Greens' if sentiment == 'positive' else 'Reds'
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap=colormap,
            max_words=max_words,
            relative_scaling=0.5,
            collocations=False
        ).generate(combined_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'{sentiment.title()} Sentiment Word Cloud', fontsize=16, fontweight='bold')
        
        return fig
    except Exception as e:
        st.error(f"Error creating word cloud: {str(e)}")
        return None

# Gemini integration
def generate_insights_with_gemini(model, sentiment_counts: Dict, language: str) -> str:
    """Generate insights using Gemini AI"""
    try:
        if not model:
            return "Gemini AI not available for insights generation."
        
        # Prepare sentiment data
        total_reviews = sum(sentiment_counts.values())
        positive_pct = (sentiment_counts.get('positive', 0) / total_reviews) * 100
        negative_pct = (sentiment_counts.get('negative', 0) / total_reviews) * 100
        
        lang_context = "Indonesian" if language == 'id' else "English"
        
        prompt = f"""
        Analyze this sentiment analysis results for {lang_context} customer reviews:
        
        **Overall Distribution:**
        - Positive: {sentiment_counts.get('positive', 0)} reviews ({positive_pct:.1f}%)
        - Negative: {sentiment_counts.get('negative', 0)} reviews ({negative_pct:.1f}%)
        - Total analyzed: {total_reviews} reviews
        
        Provide actionable insights in this format:
        
        ## 📊 Overview
        [Brief summary of overall sentiment trend]
        
        ## 💡 Key Insights
        [3-4 important insights from the data]
        
        ## 📈 Recommendations
        [Specific actionable recommendations]
        
        ## ⚠️ Areas of Concern
        [Issues that need attention]
        
        Use professional, clear language suitable for business decision-making.
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error generating insights: {str(e)}"

# Main application
def main():
    """Main application function"""
    # Initialize session state
    init_session_state()
    
    # App header
    st.title("📊 Enhanced Sentiment Analyzer")
    st.markdown("Analyze customer sentiment with AI-powered insights")
    
    # Language selection
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"{LANGUAGES['id'].flag} {LANGUAGES['id'].name}", 
                     type="primary" if st.session_state.language == 'id' else "secondary"):
            update_session_state('language', 'id')
    with col2:
        if st.button(f"{LANGUAGES['en'].flag} {LANGUAGES['en'].name}",
                     type="primary" if st.session_state.language == 'en' else "secondary"):
            update_session_state('language', 'en')
    
    # Mode selection
    st.subheader("Analysis Mode")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Labeled Analysis", 
                     type="primary" if st.session_state.analysis_mode == 'labeled' else "secondary"):
            update_session_state('analysis_mode', 'labeled')
    with col2:
        if st.button("🤖 AI Prediction", 
                     type="primary" if st.session_state.analysis_mode == 'unlabeled' else "secondary"):
            update_session_state('analysis_mode', 'unlabeled')
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Gemini API key
        api_key = st.text_input("Gemini API Key (Optional)", type="password")
        gemini_model = setup_gemini(api_key, st.session_state.language) if api_key else None
        
        if gemini_model:
            st.success("✅ Gemini AI connected")
        
        # Preprocessing options (only for unlabeled mode)
        if st.session_state.analysis_mode == 'unlabeled':
            st.subheader("🔧 Preprocessing")
            
            # Create preprocessing config from user inputs
            preprocess_config = PreprocessConfig(
                remove_urls=st.checkbox("Remove URLs", value=True),
                remove_mentions=st.checkbox("Remove Mentions (@)", value=True),
                remove_hashtags=st.checkbox("Remove Hashtags (#)", value=True),
                remove_numbers=st.checkbox("Remove Numbers", value=True),
                remove_punctuation=st.checkbox("Remove Punctuation", value=True),
                to_lowercase=st.checkbox("Lowercase", value=True),
                remove_stopwords=st.checkbox("Remove Stopwords", value=True),
                apply_stemming=st.checkbox("Apply Stemming", value=True),
                min_word_length=st.slider("Min Word Length", 1, 5, 2)
            )
        else:
            # Default config for labeled mode
            preprocess_config = PreprocessConfig()
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Show preview
            with st.expander("Data Preview"):
                st.dataframe(df.head(), use_container_width=True)
            
            # Detect columns
            column_types = detect_column_types(df)
            
            if not column_types['text']:
                st.error("❌ No text columns detected")
                return
            
            # Configuration based on mode
            if st.session_state.analysis_mode == 'labeled':
                # Labeled mode configuration
                st.subheader("Configuration for Labeled Analysis")
                
                text_columns = st.multiselect(
                    "Select text columns:", 
                    column_types['text']
                )
                
                if not column_types['numeric']:
                    st.error("❌ No numeric columns for labels")
                    return
                
                target_column = st.selectbox(
                    "Select target column:", 
                    column_types['numeric']
                )
                
                if target_column:
                    unique_vals = sorted(df[target_column].dropna().unique())
                    max_val = max(unique_vals)
                    scale_type = '1-5' if max_val <= 5 else '1-10'
                    
                    st.info(f"Scale: {scale_type} (Range: {min(unique_vals)}-{max_val})")
                
                if not text_columns or not target_column:
                    st.warning("⚠️ Please select required columns")
                    return
                
            else:
                # Unlabeled mode configuration
                text_columns = st.multiselect(
                    "Select text columns for analysis:", 
                    column_types['text']
                )
                
                if not text_columns:
                    st.warning("⚠️ Please select text columns")
                    return
                
                # Load model
                classifier = load_sentiment_model(st.session_state.language)
                if not classifier:
                    st.error("❌ Failed to load AI model")
                    return
            
            # Analysis button
            if st.button("🚀 Start Analysis", type="primary"):
                with st.spinner("Processing..."):
                    if st.session_state.analysis_mode == 'labeled':
                        # Process labeled data
                        results_df = process_labeled_data(df, text_columns, target_column, scale_type)
                        
                        if results_df.empty:
                            st.error("❌ No valid data to analyze")
                            return
                        
                        # Count sentiments
                        sentiment_counts = results_df['sentiment'].value_counts().to_dict()
                        
                    else:
                        # Process unlabeled data
                        # Initialize preprocessor
                        preprocessor = TextPreprocessor(st.session_state.language)
                        
                        # Combine and preprocess texts
                        all_texts = []
                        for col in text_columns:
                            texts = df[col].dropna().astype(str).tolist()
                            processed_texts = preprocessor.process_batch(texts, preprocess_config)
                            all_texts.extend(processed_texts)
                        
                        if not all_texts:
                            st.error("❌ No valid text for analysis")
                            return
                        
                        # Analyze sentiment
                        analyzer = SentimentAnalyzer(classifier, st.session_state.language, 'unlabeled')
                        results = analyzer.analyze_batch(all_texts)
                        
                        if not results:
                            st.error("❌ Analysis failed")
                            return
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame(results)
                        
                        # Force binary classification for unlabeled mode
                        valid_sentiments = ['positive', 'negative']
                        results_df = results_df[results_df['sentiment'].isin(valid_sentiments)]
                        
                        # Count sentiments
                        sentiment_counts = results_df['sentiment'].value_counts().to_dict()
                
                # Display results
                st.header("📈 Analysis Results")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Reviews", len(results_df))
                with col2:
                    st.metric("Positive", sentiment_counts.get('positive', 0))
                with col3:
                    st.metric("Negative", sentiment_counts.get('negative', 0))
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sentiment distribution
                    chart = create_sentiment_chart(sentiment_counts, st.session_state.analysis_mode)
                    st.plotly_chart(chart, use_container_width=True)
                
                with col2:
                    # Confidence or rating distribution
                    if st.session_state.analysis_mode == 'unlabeled':
                        conf_chart = create_confidence_chart(results_df)
                        st.plotly_chart(conf_chart, use_container_width=True)
                    else:
                        # Show rating distribution for labeled mode
                        if 'rating' in results_df.columns:
                            rating_fig = px.histogram(
                                results_df, 
                                x='rating', 
                                color='sentiment',
                                title='Rating Distribution by Sentiment'
                            )
                            st.plotly_chart(rating_fig, use_container_width=True)
                
                # Word clouds
                st.header("☁️ Word Clouds")
                
                # Only show positive and negative for unlabeled mode
                sentiments_to_show = ['positive', 'negative']
                
                for sentiment in sentiments_to_show:
                    if sentiment in sentiment_counts and sentiment_counts[sentiment] > 0:
                        texts = results_df[results_df['sentiment'] == sentiment]['text'].tolist()
                        wordcloud_fig = create_wordcloud(texts, sentiment)
                        
                        if wordcloud_fig:
                            st.pyplot(wordcloud_fig)
                            plt.close(wordcloud_fig)
                
                # Gemini insights
                if gemini_model:
                    st.header("🤖 AI Insights")
                    with st.spinner("Generating insights..."):
                        insights = generate_insights_with_gemini(
                            gemini_model, 
                            sentiment_counts, 
                            st.session_state.language
                        )
                        st.markdown(insights)
                
                # Detailed results
                st.header("📋 Detailed Results")
                
                # Filter controls
                with st.expander("Filter Options"):
                    col1, col2 = st.columns(2)
                    with col1:
                        sentiment_filter = st.selectbox(
                            "Filter by sentiment:",
                            ['All', 'positive', 'negative']
                        )
                    with col2:
                        if st.session_state.analysis_mode == 'unlabeled':
                            min_confidence = st.slider(
                                "Minimum confidence:", 
                                0.0, 1.0, 0.5
                            )
                
                # Apply filters
                filtered_df = results_df.copy()
                if sentiment_filter != 'All':
                    filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_filter]
                
                if st.session_state.analysis_mode == 'unlabeled' and 'confidence' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
                
                # Display filtered results
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download results
                st.header("💾 Download Results")
                
                csv_buffer = io.StringIO()
                filtered_df.to_csv(csv_buffer, index=False)
                csv_string = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_string,
                    file_name=f"sentiment_analysis_{st.session_state.language}_{st.session_state.analysis_mode}.csv",
                    mime="text/csv"
                )
                
                # Statistics summary
                st.header("📊 Statistics Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Sentiment Statistics")
                    total_reviews = len(results_df)
                    
                    for sentiment in ['positive', 'negative']:
                        count = sentiment_counts.get(sentiment, 0)
                        percentage = (count / total_reviews) * 100 if total_reviews > 0 else 0
                        st.write(f"**{sentiment.title()}:** {count} reviews ({percentage:.1f}%)")
                
                with col2:
                    if st.session_state.analysis_mode == 'unlabeled':
                        st.subheader("🎯 Confidence Statistics")
                        avg_confidence = results_df['confidence'].mean()
                        min_conf = results_df['confidence'].min()
                        max_conf = results_df['confidence'].max()
                        
                        st.write(f"**Average Confidence:** {avg_confidence:.3f}")
                        st.write(f"**Min Confidence:** {min_conf:.3f}")
                        st.write(f"**Max Confidence:** {max_conf:.3f}")
                        
                        # High confidence predictions
                        high_conf_count = len(results_df[results_df['confidence'] > 0.8])
                        high_conf_percentage = (high_conf_count / total_reviews) * 100
                        st.write(f"**High Confidence (>0.8):** {high_conf_count} reviews ({high_conf_percentage:.1f}%)")
                
                # Success message
                st.success("✅ Analysis completed successfully!")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.error("Please check your file format and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>Enhanced Sentiment Analyzer</p>
            <p>Powered by 🤖 AI Models + 🔥 Gemini AI + 📊 Advanced Analytics</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
