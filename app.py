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
from dataclasses import dataclass
from functools import lru_cache
import random

# Suppress warnings
warnings.filterwarnings('ignore')

# Streamlit configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="📊",
    layout="wide"
)

# Configuration classes
@dataclass(frozen=True)  # Use frozen parameter directly
class ModelConfig:
    """Configuration for ML models"""
    name: str
    flag: str
    model_name: str
    gemini_model: str
    label_mapping: Dict[str, str]

@dataclass(frozen=True)  # Use frozen parameter directly
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
        'model_cache': {},
        'scale_type': None # Added for labeled mode scale type
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
        with st.spinner(f"Loading sentiment model for {config.name}..."):
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
            st.success(f"✅ Successfully loaded {config.name} sentiment model")
            return classifier
    except Exception as e:
        st.error(f"Failed to load sentiment model for {config.name}: {str(e)}")
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
    
    @st.cache_data(show_spinner=False)
    def _preprocess_cached(_self, text: str,  # Changed self to _self for caching
                         remove_urls: bool,
                         remove_mentions: bool,
                         remove_hashtags: bool,
                         remove_numbers: bool,
                         remove_punctuation: bool,
                         to_lowercase: bool,
                         remove_stopwords: bool,
                         apply_stemming: bool,
                         min_word_length: int) -> str:
        """Cached version of text preprocessing using individual parameters"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        
        # Basic cleaning
        if remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        if remove_mentions:
            text = re.sub(r'@\w+', '', text)
        if remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        if to_lowercase:
            text = text.lower()
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Word-level processing
        if remove_stopwords or apply_stemming:
            words = text.split()
            
            if remove_stopwords:
                words = [word for word in words if word not in _self.stopwords]
            
            if apply_stemming and _self.language == 'en' and hasattr(_self, 'stemmer'):
                words = [_self.stemmer.stem(word) for word in words]
            
            if min_word_length > 1:
                words = [word for word in words if len(word) >= min_word_length]
            
            text = ' '.join(words)
        
        return text or str(text)
    
    def preprocess_text(self, text: str, config: PreprocessConfig) -> str:
        """Preprocess text using config object"""
        return self._preprocess_cached(
            text,
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
    
    def process_batch(self, texts: List[str], config: PreprocessConfig) -> List[str]:
        """Process a batch of texts"""
        processed_texts = []
        total = len(texts)
        
        # Add progress bar
        progress_bar = st.progress(0)
        
        for idx, text in enumerate(texts):
            if text and str(text).strip():
                processed = self.preprocess_text(text, config)
                if processed:
                    processed_texts.append(processed)
            
            # Update progress
            progress = (idx + 1) / total
            progress_bar.progress(progress)
        
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
        """Analyze sentiment for a batch of texts"""
        results = []
        
        # Filter empty texts
        valid_texts = [(i, text) for i, text in enumerate(texts) if text and len(str(text).strip()) > 2]
        
        if not valid_texts:
            return results
        
        # Progress bar
        progress_bar = st.progress(0)
        total = len(valid_texts)
        
        for idx, (orig_idx, text) in enumerate(valid_texts):
            try:
                # Get model prediction
                prediction = self.classifier(text)
                
                # Process results
                result = self._process_prediction(text, prediction[0])
                results.append(result)
                
                # Update progress
                progress = (idx + 1) / total
                progress_bar.progress(progress)
                
            except Exception as e:
                st.warning(f"Error analyzing text {orig_idx}: {str(e)}")
                # Add fallback result
                results.append({
                    'text': text,
                    'sentiment': 'positive' if random.random() > 0.5 else 'negative',
                    'confidence': 0.5,
                    'positive_score': 0.5,
                    'negative_score': 0.5,
                    'neutral_score': 0.0,
                    'final_scores': {
                        'positive': 0.5,
                        'negative': 0.5,
                        'neutral': 0.0
                    }
                })
        
        return results
    
    def _process_prediction(self, text: str, prediction: List[Dict]) -> Dict[str, Any]:
        """Process model prediction into standardized format"""
        try:
            # Map scores to sentiment labels
            sentiment_scores = {}
            for item in prediction:
                mapped_label = self.config.label_mapping.get(item['label'])
                if mapped_label:
                    sentiment_scores[mapped_label] = item['score']
            
            # Get scores with defaults
            positive_score = sentiment_scores.get('positive', 0)
            negative_score = sentiment_scores.get('negative', 0)
            neutral_score = sentiment_scores.get('neutral', 0)
            
            # For unlabeled mode, convert neutral to positive/negative
            if self.mode == 'unlabeled':
                # If neutral is highest, look at positive vs negative
                if neutral_score > positive_score and neutral_score > negative_score:
                    # Compare positive and negative scores
                    if positive_score > negative_score:
                        sentiment = 'positive'
                        confidence = positive_score + (positive_score - negative_score) * 0.2
                    else:
                        sentiment = 'negative'
                        confidence = negative_score + (negative_score - positive_score) * 0.2
                else:
                    # Use the highest non-neutral score
                    if positive_score > negative_score:
                        sentiment = 'positive'
                        confidence = positive_score
                    else:
                        sentiment = 'negative'
                        confidence = negative_score
                
                # Normalize confidence
                confidence = min(max(confidence, 0.0), 1.0)
                
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
                'negative_score': negative_score,
                'neutral_score': neutral_score,
                'final_scores': sentiment_scores
            }
            
        except Exception as e:
            st.warning(f"Error in prediction processing: {str(e)}")
            # Fallback to simple positive/negative based on text characteristics
            # This is a very basic fallback mechanism
            word_count = len(str(text).split())
            if word_count > 0:
                # Simple rule-based fallback
                negative_words = ['tidak', 'bukan', 'kurang', 'buruk', 'jelek', 'poor', 'bad', 'worst', 'terrible']
                has_negative = any(word in str(text).lower() for word in negative_words)
                sentiment = 'negative' if has_negative else 'positive'
                confidence = 0.51  # Low confidence for fallback
            else:
                sentiment = 'positive'
                confidence = 0.5
            
            return {
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence,
                'positive_score': 0.5,
                'negative_score': 0.5,
                'neutral_score': 0.0,
                'final_scores': {
                    'positive': 0.5,
                    'negative': 0.5,
                    'neutral': 0.0
                }
            }

# Data processing functions
def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect column types without strict limitations"""
    text_cols = []
    target_cols = []
    
    for col in df.columns:
        # Check if column contains mostly text
        sample = df[col].dropna().astype(str).head(5)
        if len(sample) > 0:
            avg_len = sample.str.len().mean()
            if avg_len > 10:  # Likely text content
                text_cols.append(col)
            else:
                target_cols.append(col)  # Any column that's not long text can be a target
    
    return {
        'text': text_cols,
        'target': target_cols
    }

def process_labeled_data(df: pd.DataFrame, text_cols: List[str], 
                        target_col: str, scale_type: str = None) -> pd.DataFrame:
    """Process labeled data with flexible target handling"""
    results = []
    
    for idx, row in df.iterrows():
        # Combine text from selected columns
        combined_text = ' '.join([
            str(row[col]) for col in text_cols 
            if pd.notna(row[col]) and str(row[col]).strip()
        ])
        
        if combined_text.strip():
            target_value = row[target_col]
            if pd.notna(target_value):
                # Handle numeric values with scale_type
                if pd.api.types.is_numeric_dtype(df[target_col]) and scale_type:
                    try:
                        rating = float(target_value)
                        sentiment = convert_rating_to_sentiment(rating, scale_type)
                    except:
                        sentiment = 'neutral'
                else:
                    # Handle string/categorical values
                    target_str = str(target_value).lower()
                    # Map common sentiment terms
                    sentiment_map = {
                        'positif': 'positive',
                        'negatif': 'negative',
                        'netral': 'neutral',
                        'positive': 'positive',
                        'negative': 'negative',
                        'neutral': 'neutral',
                    }
                    sentiment = sentiment_map.get(target_str, target_str)
                
                results.append({
                    'text': combined_text,
                    'target_value': target_value,
                    'sentiment': sentiment
                })
    
    return pd.DataFrame(results)

def convert_rating_to_sentiment(rating: float, scale_type: str) -> str:
    """Convert numeric rating to sentiment based on scale type"""
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
    """Create optimized sentiment distribution chart with consistent colors"""
    color_map = {
        'positive': '#2E8B57',  # Green
        'negative': '#DC143C',  # Red
        'neutral': '#FFD700'    # Gold/Yellow
    }
    
    if mode == 'unlabeled':
        # Only show positive/negative for unlabeled
        labels = ['positive', 'negative']
        values = [sentiment_counts.get(label, 0) for label in labels]
    else:
        labels = list(sentiment_counts.keys())
        values = list(sentiment_counts.values())
    
    colors = [color_map.get(label, '#4682B4') for label in labels]
    
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
    """Create optimized word cloud with consistent colors"""
    if not texts:
        return None
    
    combined_text = ' '.join(texts)
    if len(combined_text.strip()) < 10:
        return None
    
    try:
        # Consistent color scheme
        color_map = {
            'positive': 'Greens',
            'negative': 'Reds',
            'neutral': 'YlOrBr'  # Yellow-Orange-Brown for neutral
        }
        
        colormap = color_map.get(sentiment, 'viridis')
        
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
def generate_insights_with_gemini(model, results_df: pd.DataFrame, sentiment_counts: Dict, language: str) -> str:
    """Generate business insights using Gemini AI"""
    try:
        if not model:
            return "Gemini AI not available for insights generation."
        
        # Sample text data for domain identification
        sample_texts = results_df['text'].head(10).tolist()
        sample_text = ' '.join(sample_texts)
        
        # Prepare sentiment data
        total_reviews = sum(sentiment_counts.values())
        positive_pct = (sentiment_counts.get('positive', 0) / total_reviews) * 100 if total_reviews > 0 else 0
        negative_pct = (sentiment_counts.get('negative', 0) / total_reviews) * 100 if total_reviews > 0 else 0
        neutral_pct = (sentiment_counts.get('neutral', 0) / total_reviews) * 100 if total_reviews > 0 else 0
        
        lang_context = "Indonesian" if language == 'id' else "English"
        
        # Enhanced prompt for business insights
        prompt = f"""
        Analyze these customer reviews in {lang_context} and provide comprehensive business insights:

        **Sample Review Content:**
        {sample_text[:1000]}...

        **Sentiment Analysis Results:**
        - Positive: {sentiment_counts.get('positive', 0)} reviews ({positive_pct:.1f}%)
        - Negative: {sentiment_counts.get('negative', 0)} reviews ({negative_pct:.1f}%)
        - Neutral: {sentiment_counts.get('neutral', 0)} reviews ({neutral_pct:.1f}%)
        - Total analyzed: {total_reviews} reviews

        Based on the review content and sentiment distribution, provide insights in this format:

        ## 🏢 Business Domain Analysis
        [Identify the business domain/industry based on review content and provide context]

        ## 📊 Sentiment Overview
        [Comprehensive analysis of sentiment distribution and what it means for the business]

        ## 🔍 Key Insights
        [4-5 critical insights extracted from the sentiment analysis that impact business performance]

        ## 📈 Strategic Recommendations
        [Specific, actionable strategies the company can implement immediately to improve customer satisfaction]

        ## ⚠️ Priority Actions
        [Urgent issues that require immediate attention based on negative sentiment analysis]

        ## 💡 Competitive Advantages
        [Positive aspects the company should leverage and amplify]

        ## 📋 Implementation Roadmap
        [Step-by-step action plan with priorities and timelines]

        Provide practical, data-driven recommendations that can be implemented by management teams immediately. Focus on actionable insights rather than generic advice.
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
    st.title("📊 Sentiment Analysis App")
    st.markdown("Analyze customer sentiment with AI-powered insights")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Language selection
        st.subheader("🌍 Language")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"{LANGUAGES['id'].flag} ID", 
                         type="primary" if st.session_state.language == 'id' else "secondary"):
                update_session_state('language', 'id')
        with col2:
            if st.button(f"{LANGUAGES['en'].flag} EN",
                         type="primary" if st.session_state.language == 'en' else "secondary"):
                update_session_state('language', 'en')
        
        # Analysis mode selection
        st.subheader("📊 Analysis Mode")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Labeled", 
                         type="primary" if st.session_state.analysis_mode == 'labeled' else "secondary"):
                update_session_state('analysis_mode', 'labeled')
        with col2:
            if st.button("🤖 Unlabeled", 
                         type="primary" if st.session_state.analysis_mode == 'unlabeled' else "secondary"):
                update_session_state('analysis_mode', 'unlabeled')
        
        # Gemini API key
        st.subheader("🤖 AI Insights")
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
                    "Select text columns to analyze:", 
                    column_types['text']
                )
                
                target_column = st.selectbox(
                    "Select target column:",
                    column_types['target']
                )
                
                # Check if selected column is numeric
                is_numeric = False
                if target_column:
                    sample_value = df[target_column].dropna().iloc[0] if not df[target_column].empty else None
                    try:
                        float(sample_value)
                        is_numeric = True
                    except (ValueError, TypeError):
                        is_numeric = False
                
                # Show scale selection only for numeric columns
                scale_type = None
                if is_numeric:
                    st.write("Since you selected a numeric column, please choose the rating scale:")
                    scale_col1, scale_col2 = st.columns(2)
                    
                    with scale_col1:
                        if st.button("Scale 1-5", 
                            help="Negative: ≤2, Neutral: 3, Positive: ≥4"):
                            scale_type = "1-5"
                            st.session_state.scale_type = "1-5"
                    
                    with scale_col2:
                        if st.button("Scale 1-10",
                            help="Negative: ≤4, Neutral: 5-6, Positive: ≥7"):
                            scale_type = "1-10"
                            st.session_state.scale_type = "1-10"
                    
                    # Get scale_type from session state if exists
                    scale_type = st.session_state.get('scale_type', None)
                    
                    if scale_type:
                        st.info(f"""
                        Selected scale {scale_type}:
                        - Negative: {'≤2' if scale_type == '1-5' else '≤4'}
                        - Neutral: {'3' if scale_type == '1-5' else '5-6'}
                        - Positive: {'≥4' if scale_type == '1-5' else '≥7'}
                        """)
                    else:
                        st.warning("Please select a scale to proceed")
                
                # Run Analysis button
                if st.button("Run Labeled Analysis", type="primary"):
                    if not text_columns:
                        st.error("❌ Please select at least one text column")
                        return
                    
                    if is_numeric and not scale_type:
                        st.warning("⚠️ Please select a scale type for numeric analysis")
                        return
                    
                    # Process labeled data
                    with st.spinner("Processing labeled data..."):
                        processed_df = process_labeled_data(
                            df, 
                            text_columns,
                            target_column,
                            scale_type if is_numeric else None
                        )
                        
                        if len(processed_df) == 0:
                            st.error("❌ No valid data to analyze")
                            return
                        
                        # Store results
                        st.session_state.current_results = processed_df
                        
                        # Display results
                        st.success(f"✅ Analyzed {len(processed_df)} items")
                        
                        # Calculate sentiment counts
                        sentiment_counts = processed_df['sentiment'].value_counts().to_dict()
                        
                        # Create visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.plotly_chart(
                                create_sentiment_chart(sentiment_counts, mode='labeled'),
                                use_container_width=True
                            )
                        
                        with col2:
                            if is_numeric:
                                # Rating distribution with consistent colors
                                fig = px.histogram(
                                    processed_df,
                                    x='rating',
                                    color='sentiment',
                                    title='Rating Distribution',
                                    labels={'rating': 'Rating', 'count': 'Count'},
                                    color_discrete_map={
                                        'positive': '#2E8B57',  # Green
                                        'negative': '#DC143C',  # Red
                                        'neutral': '#FFD700'    # Gold/Yellow
                                    }
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # Categorical sentiment distribution
                                st.plotly_chart(
                                    create_sentiment_chart(sentiment_counts, mode='labeled'),
                                    use_container_width=True
                                )
                        
                        # Word clouds based on target column type
                        st.subheader("Word Clouds")
                        
                        if target_column in column_types['numeric']:
                            # For numeric ratings, group texts by sentiment categories
                            sentiment_texts = {
                                'negative': [],
                                'neutral': [],
                                'positive': []
                            }
                            
                            # Group texts based on rating ranges
                            for idx, row in df.iterrows():
                                rating = row[target_column]
                                if pd.notna(rating):
                                    rating = float(rating)
                                    # Determine sentiment based on scale type
                                    if scale_type == '1-5':
                                        if rating <= 2:
                                            sentiment = 'negative'
                                        elif rating >= 4:
                                            sentiment = 'positive'
                                        else:
                                            sentiment = 'neutral'
                                    else:  # 1-10 scale
                                        if rating <= 4:
                                            sentiment = 'negative'
                                        elif rating >= 7:
                                            sentiment = 'positive'
                                        else:
                                            sentiment = 'neutral'
                                    
                                    # Combine text from selected columns
                                    text = ' '.join([
                                        str(row[col]) for col in text_columns 
                                        if pd.notna(row[col]) and str(row[col]).strip()
                                    ])
                                    if text.strip():
                                        sentiment_texts[sentiment].append(text)
                            
                            # Create word cloud for each sentiment category
                            for sentiment, texts in sentiment_texts.items():
                                if texts:  # Only create word cloud if there are texts
                                    st.subheader(f"Word Cloud for {sentiment.title()} Reviews")
                                    wordcloud_fig = create_wordcloud(
                                        texts,
                                        sentiment
                                    )
                                    if wordcloud_fig:
                                        st.pyplot(wordcloud_fig)
                        
                        else:
                            # For object/string columns, use unique values
                            unique_categories = df[target_column].dropna().unique()
                            
                            # Color mapping for common sentiment terms
                            color_map = {
                                'positive': 'positive',
                                'positif': 'positive',
                                'negative': 'negative',
                                'negatif': 'negative',
                                'neutral': 'neutral',
                                'netral': 'neutral'
                            }
                            
                            for category in unique_categories:
                                category_texts = []
                                category_str = str(category).lower()
                                
                                # Get texts for this category
                                for idx, row in df.iterrows():
                                    if pd.notna(row[target_column]) and str(row[target_column]).lower() == category_str:
                                        text = ' '.join([
                                            str(row[col]) for col in text_columns 
                                            if pd.notna(row[col]) and str(row[col]).strip()
                                        ])
                                        if text.strip():
                                            category_texts.append(text)
                                
                                if category_texts:
                                    st.subheader(f"Word Cloud for '{category}' Category")
                                    # Use sentiment color mapping if available, otherwise use neutral
                                    sentiment_type = color_map.get(category_str, 'neutral')
                                    wordcloud_fig = create_wordcloud(
                                        category_texts,
                                        sentiment_type
                                    )
                                    if wordcloud_fig:
                                        st.pyplot(wordcloud_fig)
                        
                        # Generate insights if Gemini is available
                        if gemini_model:
                            st.subheader("🤖 AI-Generated Business Insights")
                            with st.spinner("Generating insights..."):
                                insights = generate_insights_with_gemini(
                                    gemini_model,
                                    processed_df,
                                    sentiment_counts,
                                    st.session_state.language
                                )
                                st.markdown(insights)
                        
                        # Show detailed results
                        st.subheader("Detailed Results")
                        st.dataframe(
                            processed_df[['text', 'target_value', 'sentiment']],
                            use_container_width=True
                        )
            
            else:
                # Unlabeled mode configuration
                st.subheader("Configuration for Unlabeled Analysis")
                
                text_columns = st.multiselect(
                    "Select text columns:",
                    column_types['text']
                )
                
                if st.button("Run Unlabeled Analysis", type="primary"):
                    if not text_columns:
                        st.error("❌ Please select at least one text column")
                        return
                    
                    # Load model
                    model = load_sentiment_model(st.session_state.language)
                    if not model:
                        return
                    
                    # Initialize preprocessor
                    preprocessor = TextPreprocessor(st.session_state.language)
                    
                    # Combine text columns
                    with st.spinner("Preprocessing texts..."):
                        combined_texts = []
                        for _, row in df.iterrows():
                            text = ' '.join([
                                str(row[col]) for col in text_columns 
                                if pd.notna(row[col]) and str(row[col]).strip()
                            ])
                            if text.strip():
                                combined_texts.append(text)
                        
                        # Preprocess texts
                        processed_texts = preprocessor.process_batch(
                            combined_texts,
                            preprocess_config
                        )
                    
                    if not processed_texts:
                        st.error("❌ No valid texts to analyze")
                        return
                    
                    # Analyze sentiments
                    with st.spinner("Analyzing sentiments..."):
                        analyzer = SentimentAnalyzer(
                            model,
                            st.session_state.language,
                            mode='unlabeled'
                        )
                        results = analyzer.analyze_batch(processed_texts)
                    
                    if not results:
                        st.error("❌ Analysis failed")
                        return
                    
                    # Convert results to DataFrame
                    results_df = pd.DataFrame(results)
                    
                    # Store results
                    st.session_state.current_results = results_df
                    
                    # Display results
                    st.success(f"✅ Analyzed {len(results_df)} texts")
                    
                    # Calculate sentiment counts
                    sentiment_counts = results_df['sentiment'].value_counts().to_dict()
                    
                    # Create visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(
                            create_sentiment_chart(sentiment_counts, mode='unlabeled'),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.plotly_chart(
                            create_confidence_chart(results_df),
                            use_container_width=True
                        )
                    
                    # Word clouds by sentiment
                    st.subheader("Word Clouds by Sentiment")
                    
                    for sentiment in ['positive', 'negative']:
                        sentiment_texts = results_df[
                            results_df['sentiment'] == sentiment
                        ]['text'].tolist()
                        
                        if sentiment_texts:
                            wordcloud_fig = create_wordcloud(
                                sentiment_texts,
                                sentiment
                            )
                            if wordcloud_fig:
                                st.pyplot(wordcloud_fig)
                    
                    # Generate insights if Gemini is available
                    if gemini_model:
                        st.subheader("🤖 AI-Generated Business Insights")
                        with st.spinner("Generating insights..."):
                            insights = generate_insights_with_gemini(
                                gemini_model,
                                results_df,
                                sentiment_counts,
                                st.session_state.language
                            )
                            st.markdown(insights)
                    
                    # Show detailed results
                    st.subheader("Detailed Results")
                    st.dataframe(
                        results_df[[
                            'text',
                            'sentiment',
                            'confidence',
                            'positive_score',
                            'negative_score',
                            'neutral_score'
                        ]],
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.error("Please check your file format and try again")

if __name__ == "__main__":
    main()
