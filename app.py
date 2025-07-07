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

# Konfigurasi halaman
st.set_page_config(
    page_title="Enhanced Multilingual Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

# Language configuration with updated Indonesian model
LANGUAGES = {
    'id': {
        'name': 'Bahasa Indonesia',
        'flag': '🇮🇩',
        'model': 'mdhugol/indonesia-bert-sentiment-classification',  # Updated working model
        'gemini_model': 'gemini-2.0-flash-exp',
        'label_mapping': {'LABEL_0': 'positif', 'LABEL_1': 'netral', 'LABEL_2': 'negatif'}
    },
    'en': {
        'name': 'English',
        'flag': '🇺🇸',
        'model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'gemini_model': 'gemini-2.0-flash-exp',
        'label_mapping': {'LABEL_0': 'negatif', 'LABEL_1': 'netral', 'LABEL_2': 'positif'}
    }
}

# Translations
TRANSLATIONS = {
    'id': {
        'title': "📊 Enhanced Customer Review Sentiment Analyzer",
        'subtitle': "Upload CSV file untuk analisis sentiment dengan **preprocessing lengkap** + **AI Model** + insight dari **Gemini AI**",
        'language_select': "Pilih Bahasa / Select Language",
        'config_header': "⚙️ Konfigurasi",
        'api_key_label': "Gemini AI API Key (Opsional)",
        'api_key_help': "Untuk mendapatkan AI-powered insights dan summary",
        'gemini_ready': "✅ Gemini AI siap untuk insights",
        'preprocessing_header': "🧹 Pengaturan Preprocessing",
        'analysis_header': "📈 Pengaturan Analisis",
        'upload_label': "Upload CSV file",
        'upload_help': "Upload file CSV yang berisi customer review",
        'file_success': "✅ File berhasil diupload! Shape: {shape}",
        'preview_data': "📋 Preview Data",
        'info_dataset': "**Info Dataset:**",
        'rows_count': "- Jumlah baris: {count}",
        'cols_count': "- Jumlah kolom: {count}",
        'columns_list': "- Kolom: {columns}",
        'no_text_cols': "❌ Tidak ada kolom teks yang terdeteksi untuk analisis sentiment",
        'select_cols': "Pilih kolom untuk analisis sentiment:",
        'select_cols_help': "Pilih kolom yang berisi teks review",
        'select_warning': "⚠️ Silakan pilih minimal satu kolom untuk analisis",
        'start_analysis': "🚀 Mulai Analisis Sentiment",
        'processing_text': "📝 Memproses dan membersihkan teks...",
        'no_valid_text': "❌ Tidak ada teks valid untuk dianalisis",
        'preprocessing_spinner': "Melakukan preprocessing teks...",
        'preprocessing_results': "🧹 Hasil Preprocessing",
        'preprocessing_examples': "📝 Contoh Hasil Preprocessing",
        'analyzing_text': "📊 Menganalisis {count} teks dengan AI Model...",
        'analysis_failed': "❌ Gagal melakukan analisis sentiment",
        'no_results': "⚠️ Tidak ada hasil dengan confidence >= {threshold}",
        'results_header': "📈 Hasil Analisis",
        'total_reviews': "Total Review",
        'positive': "Positif",
        'negative': "Negatif",
        'neutral': "Netral",
        'sentiment_dist': "📊 Distribusi Sentiment",
        'confidence_dist': "📈 Distribusi Confidence",
        'ai_insights': "🤖 AI-Powered Insights",
        'generating_insights': "Generating insights with Gemini AI...",
        'wordcloud_header': "☁️ Word Cloud per Sentiment",
        'detail_results': "📋 Detail Hasil Analisis",
        'filter_sentiment': "Filter berdasarkan sentiment:",
        'download_header': "💾 Download Hasil",
        'download_button': "📥 Download Hasil CSV",
        'summary_stats': "📊 Statistik Summary",
        'sentiment_distribution': "**Distribusi Sentiment:**",
        'confidence_stats': "**Confidence Score:**",
        'model_loaded': "✅ Model AI berhasil dimuat!",
        'model_failed': "❌ Gagal memuat model AI. Pastikan koneksi internet stabil.",
        'error_processing': "❌ Error memproses file: {error}",
        'file_validation': "Pastikan file CSV valid dan berisi kolom teks yang dapat dianalisis"
    },
    'en': {
        'title': "📊 Enhanced Customer Review Sentiment Analyzer",
        'subtitle': "Upload CSV file for sentiment analysis with **comprehensive preprocessing** + **AI Model** + insights from **Gemini AI**",
        'language_select': "Select Language / Pilih Bahasa",
        'config_header': "⚙️ Configuration",
        'api_key_label': "Gemini AI API Key (Optional)",
        'api_key_help': "To get AI-powered insights and summary",
        'gemini_ready': "✅ Gemini AI ready for insights",
        'preprocessing_header': "🧹 Preprocessing Settings",
        'analysis_header': "📈 Analysis Settings",
        'upload_label': "Upload CSV file",
        'upload_help': "Upload CSV file containing customer reviews",
        'file_success': "✅ File uploaded successfully! Shape: {shape}",
        'preview_data': "📋 Data Preview",
        'info_dataset': "**Dataset Info:**",
        'rows_count': "- Number of rows: {count}",
        'cols_count': "- Number of columns: {count}",
        'columns_list': "- Columns: {columns}",
        'no_text_cols': "❌ No text columns detected for sentiment analysis",
        'select_cols': "Select columns for sentiment analysis:",
        'select_cols_help': "Select columns containing review text",
        'select_warning': "⚠️ Please select at least one column for analysis",
        'start_analysis': "🚀 Start Sentiment Analysis",
        'processing_text': "📝 Processing and cleaning text...",
        'no_valid_text': "❌ No valid text for analysis",
        'preprocessing_spinner': "Performing text preprocessing...",
        'preprocessing_results': "🧹 Preprocessing Results",
        'preprocessing_examples': "📝 Preprocessing Examples",
        'analyzing_text': "📊 Analyzing {count} texts with AI Model...",
        'analysis_failed': "❌ Failed to perform sentiment analysis",
        'no_results': "⚠️ No results with confidence >= {threshold}",
        'results_header': "📈 Analysis Results",
        'total_reviews': "Total Reviews",
        'positive': "Positive",
        'negative': "Negative",
        'neutral': "Neutral",
        'sentiment_dist': "📊 Sentiment Distribution",
        'confidence_dist': "📈 Confidence Distribution",
        'ai_insights': "🤖 AI-Powered Insights",
        'generating_insights': "Generating insights with Gemini AI...",
        'wordcloud_header': "☁️ Word Cloud per Sentiment",
        'detail_results': "📋 Detailed Analysis Results",
        'filter_sentiment': "Filter by sentiment:",
        'download_header': "💾 Download Results",
        'download_button': "📥 Download Results CSV",
        'summary_stats': "📊 Summary Statistics",
        'sentiment_distribution': "**Sentiment Distribution:**",
        'confidence_stats': "**Confidence Score:**",
        'model_loaded': "✅ AI Model loaded successfully!",
        'model_failed': "❌ Failed to load AI model. Please check internet connection.",
        'error_processing': "❌ Error processing file: {error}",
        'file_validation': "Please ensure CSV file is valid and contains text columns for analysis"
    }
}

# Initialize session state for language
if 'language' not in st.session_state:
    st.session_state.language = 'id'

def get_text(key):
    """Get translated text based on current language"""
    return TRANSLATIONS[st.session_state.language][key]

def set_language(lang):
    """Set application language"""
    st.session_state.language = lang
    st.rerun()

# Language selector buttons
st.markdown("### " + get_text('language_select'))
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

# Download NLTK data yang diperlukan
@st.cache_resource
def download_nltk_data():
    """Download NLTK data yang diperlukan"""
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
        # Sastrawi stemmer
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        
        # Sastrawi stopword remover
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

# Cache untuk model loading
@st.cache_resource
def load_sentiment_model(language):
    """Load sentiment classifier based on language"""
    try:
        model_name = LANGUAGES[language]['model']
        with st.spinner(f"Loading {language.upper()} AI model..."):
            classifier = pipeline(
                "sentiment-analysis", 
                model=model_name,
                return_all_scores=True
            )
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Fungsi untuk setup Gemini AI dengan Flash 2.0
def setup_gemini(api_key: str, language: str):
    """Setup Gemini AI dengan API key untuk summary menggunakan Flash 2.0"""
    try:
        genai.configure(api_key=api_key)
        model_name = LANGUAGES[language]['gemini_model']
        return genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"Error setting up Gemini: {str(e)}")
        return None

# Fungsi untuk deteksi kolom teks
def detect_text_columns(df: pd.DataFrame) -> List[str]:
    """Deteksi kolom yang berisi teks untuk analisis sentiment"""
    text_columns = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_values = df[col].dropna().head(10)
            if len(sample_values) > 0:
                avg_length = sample_values.astype(str).str.len().mean()
                if avg_length > 10:
                    text_columns.append(col)
    
    return text_columns

# Fungsi preprocessing teks yang lengkap
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
    Comprehensive text preprocessing dengan berbagai opsi untuk Indonesian dan English
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
    
    # Return original if preprocessing results in empty string
    if not text and original_text:
        return original_text.strip()
    
    return text

# Fungsi untuk mapping label sentiment
def map_sentiment_label(label: str, language: str) -> str:
    """Map label dari model ke format yang diinginkan"""
    label_mapping = LANGUAGES[language]['label_mapping']
    return label_mapping.get(label, 'netral')

# Fungsi untuk analisis sentiment
def analyze_sentiment(classifier, texts: List[str], language: str) -> List[Dict]:
    """Analisis sentiment menggunakan AI Model"""
    results = []
    
    progress_bar = st.progress(0)
    
    for i, text in enumerate(texts):
        if not text or len(text.strip()) < 3:
            results.append({
                "text": text,
                "sentiment": "netral",
                "confidence": 0.5,
                "all_scores": {}
            })
            continue
        
        try:
            prediction = classifier(text)
            all_scores = {item['label']: item['score'] for item in prediction}
            best_pred = max(prediction, key=lambda x: x['score'])
            
            sentiment = map_sentiment_label(best_pred['label'], language)
            confidence = best_pred['score']
            
            results.append({
                "text": text,
                "sentiment": sentiment,
                "confidence": confidence,
                "all_scores": all_scores
            })
            
        except Exception as e:
            results.append({
                "text": text,
                "sentiment": "netral",
                "confidence": 0.5,
                "all_scores": {}
            })
        
        progress = (i + 1) / len(texts)
        progress_bar.progress(progress)
    
    return results

# Fungsi untuk generate summary dengan Gemini Flash 2.0
def generate_summary_with_gemini(model, sentiment_counts: Dict, wordcloud_images: Dict, language: str) -> str:
    """Generate summary dan insight menggunakan Gemini Flash 2.0"""
    try:
        if not model:
            return "Error: Gemini model not initialized"
        
        # Convert matplotlib figures to PIL images for Gemini
        image_parts = []
        for sentiment, fig in wordcloud_images.items():
            if fig:
                # Save figure to bytes buffer
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                
                # Add image part with description
                image_parts.append({
                    "inlineData": {
                        "data": base64.b64encode(buf.getvalue()).decode('utf-8'),
                        "mimeType": "image/png"
                    }
                })
        
        if language == 'id':
            prompt = f"""
            Analisis hasil sentiment customer review berikut berdasarkan distribusi sentiment dan visualisasi word cloud yang ditampilkan:
            
            Distribusi Sentiment:
            - Positif: {sentiment_counts.get('positif', 0)} review
            - Negatif: {sentiment_counts.get('negatif', 0)} review  
            - Netral: {sentiment_counts.get('netral', 0)} review
            
            Berdasarkan word cloud untuk setiap kategori sentiment yang ditampilkan dalam gambar, berikan analisis dan insight dalam format berikut:
            
            ## 📊 Ringkasan Analisis
            [Ringkasan kondisi umum sentiment dan kata-kata dominan yang terlihat dalam word cloud]
            
            ## 💡 Key Insights
            [3-5 insight penting berdasarkan distribusi sentiment dan pola kata dalam word cloud]
            
            ## 🎯 Rekomendasi Aksi
            [Rekomendasi konkret berdasarkan pola kata dan sentiment yang teridentifikasi]
            
            ## ⚠️ Area Perhatian
            [Hal-hal yang perlu diperhatikan dari pola kata dan sentiment]
            
            Gunakan bahasa Indonesia yang profesional dan mudah dipahami.
            """
        else:  # English
            prompt = f"""
            Analyze the following customer review sentiment results based on sentiment distribution and word cloud visualizations:
            
            Sentiment Distribution:
            - Positive: {sentiment_counts.get('positif', 0)} reviews
            - Negative: {sentiment_counts.get('negatif', 0)} reviews  
            - Neutral: {sentiment_counts.get('netral', 0)} reviews
            
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
        parts = [
            {"text": prompt}
        ]
        parts.extend(image_parts)
        
        # Generate content with both text and images
        response = model.generate_content(
            parts,
            generation_config={"temperature": 0.7}
        )
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Fungsi untuk membuat wordcloud
def create_wordcloud(texts: List[str], sentiment: str) -> plt.Figure:
    """Membuat wordcloud untuk sentiment tertentu"""
    if not texts:
        return None
        
    combined_text = " ".join(texts)
    
    if len(combined_text.strip()) < 10:
        return None
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis' if sentiment == 'positif' else 'Reds' if sentiment == 'negatif' else 'Blues',
        max_words=100,
        relative_scaling=0.5,
        collocations=False
    ).generate(combined_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud - Sentiment {sentiment.title()}', fontsize=16, fontweight='bold')
    
    return fig

# Fungsi untuk visualisasi sentiment
def create_sentiment_chart(sentiment_counts: Dict[str, int]) -> go.Figure:
    """Membuat pie chart untuk distribusi sentiment"""
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())
    
    colors = {
        'positif': '#2E8B57',
        'negatif': '#DC143C',
        'netral': '#4682B4'
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
        title="Distribusi Sentiment Analysis",
        font=dict(size=14),
        showlegend=True
    )
    
    return fig

# Fungsi untuk confidence distribution chart
def create_confidence_chart(results_df: pd.DataFrame) -> go.Figure:
    """Membuat histogram distribusi confidence score"""
    fig = px.histogram(
        results_df, 
        x='confidence', 
        color='sentiment',
        nbins=20,
        title='Distribusi Confidence Score per Sentiment',
        labels={'confidence': 'Confidence Score', 'count': 'Jumlah Review'}
    )
    
    fig.update_layout(
        xaxis_title="Confidence Score",
        yaxis_title="Jumlah Review",
        showlegend=True
    )
    
    return fig

# Main app
def main():
    st.title(get_text('title'))
    st.markdown(get_text('subtitle'))
    
    # Initialize tools
    download_nltk_data()
    
    # Initialize language-specific tools
    if st.session_state.language == 'id':
        stemmer, stopword_remover = init_indonesian_tools()
        english_stopwords, porter_stemmer = None, None
    else:
        stemmer, stopword_remover = None, None
        english_stopwords, porter_stemmer = init_english_tools()
    
    # Load model
    classifier = load_sentiment_model(st.session_state.language)
    if classifier is None:
        st.error(get_text('model_failed'))
        return
    
    st.success(get_text('model_loaded'))
    
    # Sidebar untuk konfigurasi
    st.sidebar.header(get_text('config_header'))
    
    # Input API Key Gemini
    api_key = st.sidebar.text_input(
        get_text('api_key_label'),
        type="password",
        help=get_text('api_key_help')
    )
    
    gemini_model = None
    if api_key:
        gemini_model = setup_gemini(api_key, st.session_state.language)
        if gemini_model:
            st.sidebar.success(get_text('gemini_ready'))
    
    # Pengaturan preprocessing
    st.sidebar.subheader(get_text('preprocessing_header'))
    
    preprocess_options = {
        'remove_urls': st.sidebar.checkbox("Remove URLs", value=True),
        'remove_mentions': st.sidebar.checkbox("Remove Mentions (@)", value=True),
        'remove_hashtags': st.sidebar.checkbox("Remove Hashtags (#)", value=True),
        'remove_numbers': st.sidebar.checkbox("Remove Numbers", value=True),
        'remove_punctuation': st.sidebar.checkbox("Remove Punctuation", value=True),
        'to_lowercase': st.sidebar.checkbox("Lowercase", value=True),
        'remove_stopwords': st.sidebar.checkbox("Remove Stopwords", value=True),
        'apply_stemming': st.sidebar.checkbox("Stemming", value=True),
    }
    
    min_word_length = st.sidebar.slider(
        "Minimum word length",
        min_value=1,
        max_value=5,
        value=2,
        help="Words shorter than this will be removed"
    )
    
    # Pengaturan analisis
    st.sidebar.subheader(get_text('analysis_header'))
    
    max_length = st.sidebar.slider(
        "Maximum text length (characters)",
        min_value=50,
        max_value=1000,
        value=512,
        help="Text will be truncated if longer than this"
    )
    
    min_confidence = st.sidebar.slider(
        "Minimum confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Filter results by confidence score"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        get_text('upload_label'),
        type=['csv'],
        help=get_text('upload_help')
    )
    
    if uploaded_file is not None:
        try:
            # Baca CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(get_text('file_success').format(shape=df.shape))
            
            # Preview data
            with st.expander(get_text('preview_data')):
                st.dataframe(df.head())
                st.write(get_text('info_dataset'))
                st.write(get_text('rows_count').format(count=len(df)))
                st.write(get_text('cols_count').format(count=len(df.columns)))
                st.write(get_text('columns_list').format(columns=', '.join(df.columns)))
            
            # Deteksi kolom teks
            text_columns = detect_text_columns(df)
            
            if not text_columns:
                st.error(get_text('no_text_cols'))
                return
            
            # Pilih kolom untuk analisis
            selected_columns = st.multiselect(
                get_text('select_cols'),
                text_columns,
                help=get_text('select_cols_help')
            )
            
            if not selected_columns:
                st.warning(get_text('select_warning'))
                return
            
            # Start analysis button
            if st.button(get_text('start_analysis')):
                with st.spinner(get_text('processing_text')):
                    # Process each selected column
                    all_texts = []
                    for col in selected_columns:
                        texts = df[col].astype(str).tolist()
                        # Preprocess texts
                        processed_texts = [
                            comprehensive_text_preprocessing(
                                text[:max_length],
                                language=st.session_state.language,
                                stemmer=stemmer,
                                stopword_remover=stopword_remover,
                                english_stopwords=english_stopwords,
                                porter_stemmer=porter_stemmer,
                                **preprocess_options,
                                min_word_length=min_word_length
                            ) for text in texts
                        ]
                        all_texts.extend([text for text in processed_texts if text])
                    
                    if not all_texts:
                        st.error(get_text('no_valid_text'))
                        return
                    
                    # Show preprocessing examples
                    with st.expander(get_text('preprocessing_results')):
                        st.subheader(get_text('preprocessing_examples'))
                        for i, (original, processed) in enumerate(zip(texts[:5], processed_texts[:5])):
                            st.markdown(f"**Original {i+1}:** {original}")
                            st.markdown(f"**Processed {i+1}:** {processed}")
                            st.markdown("---")
                    
                    # Analyze sentiment
                    with st.spinner(get_text('analyzing_text').format(count=len(all_texts))):
                        results = analyze_sentiment(classifier, all_texts, st.session_state.language)
                        
                        # Convert results to DataFrame
                        results_df = pd.DataFrame(results)
                        
                        # Filter by confidence
                        results_df = results_df[results_df['confidence'] >= min_confidence]
                        
                        if len(results_df) == 0:
                            st.warning(get_text('no_results').format(threshold=min_confidence))
                            return
                        
                        # Calculate sentiment counts
                        sentiment_counts = results_df['sentiment'].value_counts().to_dict()
                        
                        # Display results
                        st.header(get_text('results_header'))
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(get_text('total_reviews'), len(results_df))
                        with col2:
                            st.metric(get_text('positive'), sentiment_counts.get('positif', 0))
                        with col3:
                            st.metric(get_text('negative'), sentiment_counts.get('negatif', 0))
                        with col4:
                            st.metric(get_text('neutral'), sentiment_counts.get('netral', 0))
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader(get_text('sentiment_dist'))
                            sentiment_chart = create_sentiment_chart(sentiment_counts)
                            st.plotly_chart(sentiment_chart, use_container_width=True)
                        
                        with col2:
                            st.subheader(get_text('confidence_dist'))
                            confidence_chart = create_confidence_chart(results_df)
                            st.plotly_chart(confidence_chart, use_container_width=True)
                        
                        # Generate AI Insights if Gemini is available
                        if gemini_model:
                            st.header(get_text('ai_insights'))
                            with st.spinner(get_text('generating_insights')):
                                # Create word clouds for each sentiment
                                wordcloud_images = {}
                                for sentiment in ['positif', 'negatif', 'netral']:
                                    sentiment_texts = results_df[results_df['sentiment'] == sentiment]['text'].tolist()
                                    wordcloud_images[sentiment] = create_wordcloud(sentiment_texts, sentiment)
                                
                                # Generate and display insights
                                insights = generate_summary_with_gemini(
                                    gemini_model,
                                    sentiment_counts,
                                    wordcloud_images,
                                    st.session_state.language
                                )
                                st.markdown(insights)
                        
                        # Word Clouds
                        st.header(get_text('wordcloud_header'))
                        for sentiment in ['positif', 'negatif', 'netral']:
                            sentiment_texts = results_df[results_df['sentiment'] == sentiment]['text'].tolist()
                            wordcloud_fig = create_wordcloud(sentiment_texts, sentiment)
                            if wordcloud_fig:
                                st.pyplot(wordcloud_fig)
                        
                        # Detailed Results
                        st.header(get_text('detail_results'))
                        
                        # Filter by sentiment
                        selected_sentiment = st.selectbox(
                            get_text('filter_sentiment'),
                            ['all'] + list(sentiment_counts.keys())
                        )
                        
                        filtered_df = results_df if selected_sentiment == 'all' else \
                                    results_df[results_df['sentiment'] == selected_sentiment]
                        
                        st.dataframe(filtered_df[['text', 'sentiment', 'confidence']])
                        
                        # Download results
                        st.header(get_text('download_header'))
                        
                        csv = filtered_df.to_csv(index=False)
                        st.download_button(
                            get_text('download_button'),
                            csv,
                            "sentiment_analysis_results.csv",
                            "text/csv",
                            key='download-csv'
                        )
                        
                        # Summary statistics
                        st.header(get_text('summary_stats'))
                        
                        st.write(get_text('sentiment_distribution'))
                        for sentiment, count in sentiment_counts.items():
                            percentage = (count / len(results_df)) * 100
                            st.write(f"- {sentiment.title()}: {count} ({percentage:.1f}%)")
                        
                        st.write(get_text('confidence_stats'))
                        st.write(f"- Mean: {results_df['confidence'].mean():.3f}")
                        st.write(f"- Median: {results_df['confidence'].median():.3f}")
                        st.write(f"- Min: {results_df['confidence'].min():.3f}")
                        st.write(f"- Max: {results_df['confidence'].max():.3f}")
                        
        except Exception as e:
            st.error(get_text('error_processing').format(error=str(e)))
            st.info(get_text('file_validation'))

if __name__ == "__main__":
    main()
