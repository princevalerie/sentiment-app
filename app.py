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

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Enhanced Multilingual Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

# Konfigurasi bahasa dengan model Bahasa Indonesia yang diperbarui
LANGUAGES = {
    'id': {
        'name': 'Bahasa Indonesia',
        'flag': '🇮🇩',
        'model': 'mdhugol/indonesia-bert-sentiment-classification',  # Model Bahasa Indonesia yang berfungsi
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

# Terjemahan teks untuk UI
TRANSLATIONS = {
    'id': {
        'title': "📊 Enhanced Customer Review Sentiment Analyzer",
        'subtitle': "Upload file CSV untuk analisis sentimen dengan **preprocessing lengkap** + **Model AI** + insight dari **Gemini AI**",
        'language_select': "Pilih Bahasa / Select Language",
        'config_header': "⚙️ Konfigurasi",
        'api_key_label': "Gemini AI API Key (Opsional)",
        'api_key_help': "Untuk mendapatkan insight dan ringkasan berbasis AI",
        'gemini_ready': "✅ Gemini AI siap untuk insight",
        'preprocessing_header': "🧹 Pengaturan Preprocessing",
        'analysis_header': "📈 Pengaturan Analisis", # This header is now unused in UI but kept in translations
        'upload_label': "Upload file CSV",
        'upload_help': "Upload file CSV yang berisi ulasan pelanggan",
        'file_success': "✅ File berhasil diupload! Bentuk: {shape}",
        'preview_data': "📋 Pratinjau Data",
        'info_dataset': "**Info Dataset:**",
        'rows_count': "- Jumlah baris: {count}",
        'cols_count': "- Jumlah kolom: {count}",
        'columns_list': "- Kolom: {columns}",
        'no_text_cols': "❌ Tidak ada kolom teks yang terdeteksi untuk analisis sentimen",
        'select_cols': "Pilih kolom untuk analisis sentimen:",
        'select_cols_help': "Pilih kolom yang berisi teks ulasan",
        'select_warning': "⚠️ Silakan pilih minimal satu kolom untuk analisis",
        'start_analysis': "🚀 Mulai Analisis Sentimen",
        'processing_text': "📝 Memproses dan membersihkan teks...",
        'no_valid_text': "❌ Tidak ada teks valid untuk dianalisis",
        'preprocessing_spinner': "Melakukan preprocessing teks...",
        'preprocessing_results': "🧹 Hasil Preprocessing",
        'preprocessing_examples': "📝 Contoh Hasil Preprocessing",
        'analyzing_text': "📊 Menganalisis {count} teks dengan Model AI...",
        'analysis_failed': "❌ Gagal melakukan analisis sentimen",
        'no_results': "⚠️ Tidak ada hasil dengan confidence >= {threshold}",
        'results_header': "📈 Hasil Analisis",
        'total_reviews': "Total Ulasan",
        'positive': "Positif",
        'negative': "Negatif",
        'neutral': "Netral",
        'sentiment_dist': "📊 Distribusi Sentimen",
        'confidence_dist': "📈 Distribusi Confidence",
        'ai_insights': "🤖 Insight Berbasis AI",
        'generating_insights': "Menghasilkan insight dengan Gemini AI...",
        'wordcloud_header': "☁️ Word Cloud per Sentimen",
        'detail_results': "📋 Detail Hasil Analisis",
        'filter_sentiment': "Filter berdasarkan sentimen:",
        'download_header': "💾 Unduh Hasil",
        'download_button': "📥 Unduh Hasil CSV",
        'summary_stats': "📊 Statistik Ringkasan",
        'sentiment_distribution': "**Distribusi Sentimen:**",
        'confidence_stats': "**Skor Confidence:**",
        'model_loaded': "✅ Model AI berhasil dimuat!",
        'model_failed': "❌ Gagal memuat model AI. Pastikan koneksi internet stabil.",
        'error_processing': "❌ Error memproses file: {error}",
        'file_validation': "Pastikan file CSV valid dan berisi kolom teks yang dapat dianalisis",
        'neutral_range_label': "Rentang Confidence Netral (0-1)", # This label is now unused in UI but kept in translations
        'neutral_range_help': "Jika skor confidence tertinggi berada dalam rentang ini, sentimen akan dianggap netral." # This help text is now unused in UI but kept in translations
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
        'analysis_header': "📈 Analysis Settings", # This header is now unused in UI but kept in translations
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
        'file_validation': "Please ensure CSV file is valid and contains text columns for analysis",
        'neutral_range_label': "Neutral Confidence Range (0-1)", # This label is now unused in UI but kept in translations
        'neutral_range_help': "If the highest confidence score falls within this range, the sentiment will be considered neutral." # This help text is now unused in UI but kept in translations
    }
}

# Inisialisasi status sesi untuk bahasa
if 'language' not in st.session_state:
    st.session_state.language = 'id'

def get_text(key):
    """Mendapatkan teks terjemahan berdasarkan bahasa saat ini"""
    return TRANSLATIONS[st.session_state.language][key]

def set_language(lang):
    """Mengatur bahasa aplikasi"""
    st.session_state.language = lang
    st.rerun()

# Tombol pemilih bahasa
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

# Mengunduh data NLTK yang diperlukan
@st.cache_resource
def download_nltk_data():
    """Mengunduh data NLTK yang diperlukan"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        return True
    except:
        return False

# Menginisialisasi alat preprocessing teks Bahasa Indonesia
@st.cache_resource
def init_indonesian_tools():
    """Menginisialisasi stemmer dan penghapus stopword Bahasa Indonesia"""
    try:
        # Stemmer Sastrawi
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        
        # Penghapus stopword Sastrawi
        stopword_factory = StopWordRemoverFactory()
        stopword_remover = stopword_factory.create_stop_word_remover()
        
        return stemmer, stopword_remover
    except Exception as e:
        st.error(f"Error menginisialisasi alat Bahasa Indonesia: {str(e)}")
        return None, None

# Menginisialisasi alat preprocessing teks Bahasa Inggris
@st.cache_resource
def init_english_tools():
    """Menginisialisasi stopword dan stemmer Bahasa Inggris"""
    try:
        english_stopwords = set(stopwords.words('english'))
        porter_stemmer = PorterStemmer()
        return english_stopwords, porter_stemmer
    except Exception as e:
        st.error(f"Error menginisialisasi alat Bahasa Inggris: {str(e)}")
        return None, None

# Cache untuk memuat model
@st.cache_resource
def load_sentiment_model(language):
    """Memuat pengklasifikasi sentimen berdasarkan bahasa"""
    try:
        model_name = LANGUAGES[language]['model']
        with st.spinner(f"Memuat model AI {language.upper()}..."):
            classifier = pipeline(
                "sentiment-analysis", 
                model=model_name,
                return_all_scores=True
            )
        return classifier
    except Exception as e:
        st.error(f"Error memuat model: {str(e)}")
        return None

# Fungsi untuk setup Gemini AI dengan Flash 2.0
def setup_gemini(api_key: str, language: str):
    """Setup Gemini AI dengan API key untuk ringkasan menggunakan Flash 2.0"""
    try:
        genai.configure(api_key=api_key)
        model_name = LANGUAGES[language]['gemini_model']
        return genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"Error mengatur Gemini: {str(e)}")
        return None

# Fungsi untuk deteksi kolom teks
def detect_text_columns(df: pd.DataFrame) -> List[str]:
    """Mendeteksi kolom yang berisi teks untuk analisis sentimen"""
    text_columns = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_values = df[col].dropna().head(10)
            if len(sample_values) > 0:
                avg_length = sample_values.astype(str).str.len().mean()
                if avg_length > 10: # Ambil kolom yang rata-rata panjang teksnya lebih dari 10 karakter
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
    Preprocessing teks komprehensif dengan berbagai opsi untuk Bahasa Indonesia dan Inggris
    """
    if pd.isna(text) or not text:
        return ""
    
    text = str(text)
    original_text = text
    
    # 1. Hapus URL
    if remove_urls:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 2. Hapus sebutan (@username)
    if remove_mentions:
        text = re.sub(r'@\w+', '', text)
    
    # 3. Hapus hashtag (#hashtag)
    if remove_hashtags:
        text = re.sub(r'#\w+', '', text)
    
    # 4. Hapus angka
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # 5. Konversi ke huruf kecil
    if to_lowercase:
        text = text.lower()
    
    # 6. Hapus tanda baca
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', ' ', text)
    
    # 7. Hapus spasi berlebih
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text).strip()
    
    # 8. Hapus stopword
    if remove_stopwords:
        if language == 'id' and stopword_remover:
            try:
                text = stopword_remover.remove(text)
            except:
                # Fallback: penghapusan stopword Bahasa Indonesia manual
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
    
    # 9. Terapkan stemming
    if apply_stemming:
        if language == 'id' and stemmer:
            try:
                text = stemmer.stem(text)
            except:
                pass # Lewati jika ada error pada stemming
        elif language == 'en' and porter_stemmer:
            try:
                words = text.split()
                words = [porter_stemmer.stem(word) for word in words]
                text = ' '.join(words)
            except:
                pass # Lewati jika ada error pada stemming
    
    # 10. Filter kata berdasarkan panjang minimum
    if min_word_length > 0:
        words = text.split()
        words = [word for word in words if len(word) >= min_word_length]
        text = ' '.join(words)
    
    # 11. Pembersihan akhir
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Kembalikan teks asli jika hasil preprocessing berupa string kosong
    if not text and original_text:
        return original_text.strip()
    
    return text

# Fungsi untuk memetakan label sentimen
def map_sentiment_label(label: str, language: str) -> str:
    """Memetakan label dari model ke format yang diinginkan"""
    label_mapping = LANGUAGES[language]['label_mapping']
    return label_mapping.get(label, 'netral') # Default ke netral jika label tidak ditemukan

# Fungsi untuk analisis sentimen
def analyze_sentiment(classifier, texts: List[str], language: str, neutral_lower_bound: float = 0.45, neutral_upper_bound: float = 0.55) -> List[Dict]:
    """Analisis sentimen menggunakan Model AI dengan logika netralisasi berbasis rentang"""
    results = []
    
    progress_bar = st.progress(0)
    
    for i, text in enumerate(texts):
        if not text or len(text.strip()) < 3:
            continue  # Lewati teks kosong atau terlalu pendek
        
        try:
            prediction = classifier(text)
            all_scores = {item['label']: item['score'] for item in prediction[0]}
            
            scores = []
            for pred in prediction[0]:
                sentiment = map_sentiment_label(pred['label'], language)
                scores.append((sentiment, pred['score']))
            
            # Urutkan berdasarkan skor tertinggi
            scores.sort(key=lambda x: x[1], reverse=True)
            
            best_sentiment_raw, confidence_raw = scores[0] # Sentimen dan confidence tertinggi dari model
            
            final_sentiment = best_sentiment_raw
            final_confidence = confidence_raw

            # --- Logika Netralisasi: Jika confidence tertinggi berada dalam rentang netral ---
            # Menggunakan nilai default 0.45 dan 0.55 jika tidak disediakan
            if (confidence_raw >= neutral_lower_bound and confidence_raw <= neutral_upper_bound):
                 final_sentiment = 'netral'
                 # final_confidence tetap confidence_raw untuk menunjukkan skor asli model
            # --- Akhir Logika Netralisasi ---

            # Logika lama untuk memprioritaskan non-netral jika skor sangat dekat
            # Ini akan dievaluasi setelah logika netralisasi berbasis rentang
            # Dihapus karena logika netralisasi berbasis rentang sudah cukup komprehensif
            # if best_sentiment_raw != 'netral' and len(scores) > 1 and abs(scores[0][1] - scores[1][1]) < 0.1:
            #     if 'netral' in [scores[0][0], scores[1][0]]:
            #         pass 
            
            results.append({
                "text": text,
                "sentiment": final_sentiment, # Gunakan sentimen yang sudah disesuaikan
                "confidence": final_confidence, # Gunakan confidence yang sudah disesuaikan
                "all_scores": all_scores
            })
            
        except Exception as e:
            st.warning(f"Error menganalisis teks: {str(e)}")
            continue
        
        progress = (i + 1) / len(texts)
        progress_bar.progress(progress)
    
    return results

# Fungsi untuk menghasilkan ringkasan dengan Gemini Flash 2.0
def generate_summary_with_gemini(model, sentiment_counts: Dict, wordcloud_images: Dict, language: str) -> str:
    """Menghasilkan ringkasan dan insight menggunakan Gemini Flash 2.0"""
    try:
        if not model:
            return "Error: Model Gemini tidak diinisialisasi"
        
        # Konversi gambar matplotlib ke format yang bisa dikirim ke Gemini
        image_parts = []
        for sentiment, fig in wordcloud_images.items():
            if fig:
                # Simpan gambar ke buffer byte
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                
                # Tambahkan bagian gambar dengan deskripsi
                image_parts.append({
                    "inlineData": {
                        "data": base64.b64encode(buf.getvalue()).decode('utf-8'),
                        "mimeType": "image/png"
                    }
                })
        
        if language == 'id':
            prompt = f"""
            Analisis hasil sentimen ulasan pelanggan berikut berdasarkan distribusi sentimen dan visualisasi word cloud yang ditampilkan:
            
            Distribusi Sentimen:
            - Positif: {sentiment_counts.get('positif', 0)} ulasan
            - Negatif: {sentiment_counts.get('negatif', 0)} ulasan  
            - Netral: {sentiment_counts.get('netral', 0)} ulasan
            
            Berdasarkan word cloud untuk setiap kategori sentimen yang ditampilkan dalam gambar, berikan analisis dan insight dalam format berikut:
            
            ## 📊 Ringkasan Analisis
            [Ringkasan kondisi umum sentimen dan kata-kata dominan yang terlihat dalam word cloud]
            
            ## 💡 Key Insights
            [3-5 insight penting berdasarkan distribusi sentimen dan pola kata dalam word cloud]
            
            ## 🎯 Rekomendasi Aksi
            [Rekomendasi konkret berdasarkan pola kata dan sentimen yang teridentifikasi]
            
            ## ⚠️ Area Perhatian
            [Hal-hal yang perlu diperhatikan dari pola kata dan sentimen]
            
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
        
        # Buat prompt multimodal dengan teks dan gambar
        parts = [
            {"text": prompt}
        ]
        parts.extend(image_parts)
        
        # Hasilkan konten dengan teks dan gambar
        response = model.generate_content(
            parts,
            generation_config={"temperature": 0.7}
        )
        return response.text
    except Exception as e:
        return f"Error menghasilkan ringkasan: {str(e)}"

# Fungsi untuk membuat wordcloud
def create_wordcloud(texts: List[str], sentiment: str) -> plt.Figure:
    """Membuat wordcloud untuk sentimen tertentu"""
    if not texts:
        return None
        
    combined_text = " ".join(texts)
    
    if len(combined_text.strip()) < 10: # Hindari membuat wordcloud dari teks yang terlalu pendek
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
    ax.set_title(f'Word Cloud - Sentimen {sentiment.title()}', fontsize=16, fontweight='bold')
    
    return fig

# Fungsi untuk visualisasi sentimen
def create_sentiment_chart(sentiment_counts: Dict[str, int]) -> go.Figure:
    """Membuat pie chart untuk distribusi sentimen"""
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())
    
    colors = {
        'positif': '#2E8B57', # Hijau
        'negatif': '#DC143C', # Merah
        'netral': '#4682B4'   # Biru
    }
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=[colors.get(label, '#808080') for label in labels]),
            textinfo='label+percent',
            textfont_size=12,
            hole=0.4 # Membuat donat chart
        )
    ])
    
    fig.update_layout(
        title="Distribusi Analisis Sentimen",
        font=dict(size=14),
        showlegend=True
    )
    
    return fig

# Fungsi untuk chart distribusi confidence
def create_confidence_chart(results_df: pd.DataFrame) -> go.Figure:
    """Membuat histogram distribusi skor confidence"""
    fig = px.histogram(
        results_df, 
        x='confidence', 
        color='sentiment',
        nbins=20, # Jumlah bin untuk histogram
        title='Distribusi Skor Confidence per Sentimen',
        labels={'confidence': 'Skor Confidence', 'count': 'Jumlah Ulasan'}
    )
    
    fig.update_layout(
        xaxis_title="Skor Confidence",
        yaxis_title="Jumlah Ulasan",
        showlegend=True
    )
    
    return fig

# Aplikasi utama
def main():
    st.title(get_text('title'))
    st.markdown(get_text('subtitle'))
    
    # Inisialisasi alat NLTK
    download_nltk_data()
    
    # Inisialisasi alat spesifik bahasa
    if st.session_state.language == 'id':
        stemmer, stopword_remover = init_indonesian_tools()
        english_stopwords, porter_stemmer = None, None
    else:
        stemmer, stopword_remover = None, None
        english_stopwords, porter_stemmer = init_english_tools()
    
    # Muat model sentimen
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
        'remove_urls': st.sidebar.checkbox("Hapus URL", value=True),
        'remove_mentions': st.sidebar.checkbox("Hapus Sebutan (@)", value=True),
        'remove_hashtags': st.sidebar.checkbox("Hapus Hashtag (#)", value=True),
        'remove_numbers': st.sidebar.checkbox("Hapus Angka", value=True),
        'remove_punctuation': st.sidebar.checkbox("Hapus Tanda Baca", value=True),
        'to_lowercase': st.sidebar.checkbox("Huruf Kecil", value=True),
        'remove_stopwords': st.sidebar.checkbox("Hapus Stopword", value=True),
        'apply_stemming': st.sidebar.checkbox("Stemming", value=True),
    }
    
    min_word_length = st.sidebar.slider(
        "Panjang kata minimum",
        min_value=1,
        max_value=5,
        value=2,
        help="Kata yang lebih pendek dari ini akan dihapus"
    )

    # Hardcode the neutral confidence range as the sidebar control is removed
    neutral_lower_bound = 0.45
    neutral_upper_bound = 0.55
    
    # Upload file
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
            
            # Pratinjau data
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
            
            # Tombol mulai analisis
            if st.button(get_text('start_analysis')):
                with st.spinner(get_text('processing_text')):
                    # Proses setiap kolom yang dipilih
                    all_texts = []
                    # Simpan contoh teks asli dan hasil preprocessing untuk ditampilkan
                    original_sample_texts = []
                    processed_sample_texts = []

                    for col in selected_columns:
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
                                # Ambil beberapa contoh untuk ditampilkan
                                if len(original_sample_texts) < 5: # Hanya ambil 5 contoh
                                    original_sample_texts.append(text)
                                    processed_sample_texts.append(processed_text)
                    
                    if not all_texts:
                        st.error(get_text('no_valid_text'))
                        return
                    
                    # Tampilkan contoh preprocessing
                    with st.expander(get_text('preprocessing_results')):
                        st.subheader(get_text('preprocessing_examples'))
                        for i in range(len(original_sample_texts)):
                            st.markdown(f"**Asli {i+1}:** {original_sample_texts[i]}")
                            st.markdown(f"**Diproses {i+1}:** {processed_sample_texts[i]}")
                            st.markdown("---")
                    
                    # Analisis sentimen
                    with st.spinner(get_text('analyzing_text').format(count=len(all_texts))):
                        # Pass the hardcoded neutral bounds to the analysis function
                        results = analyze_sentiment(classifier, all_texts, st.session_state.language, neutral_lower_bound, neutral_upper_bound)
                        
                        # Konversi hasil ke DataFrame
                        results_df = pd.DataFrame(results)
                        
                        # Hitung jumlah sentimen
                        sentiment_counts = results_df['sentiment'].value_counts().to_dict()
                        
                        # Tampilkan hasil
                        st.header(get_text('results_header'))
                        
                        # Metrik
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(get_text('total_reviews'), len(results_df))
                        with col2:
                            st.metric(get_text('positive'), sentiment_counts.get('positif', 0))
                        with col3:
                            st.metric(get_text('negative'), sentiment_counts.get('negatif', 0))
                        with col4:
                            st.metric(get_text('neutral'), sentiment_counts.get('netral', 0))
                        
                        # Visualisasi
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader(get_text('sentiment_dist'))
                            sentiment_chart = create_sentiment_chart(sentiment_counts)
                            st.plotly_chart(sentiment_chart, use_container_width=True)
                        
                        with col2:
                            st.subheader(get_text('confidence_dist'))
                            confidence_chart = create_confidence_chart(results_df)
                            st.plotly_chart(confidence_chart, use_container_width=True)
                        
                        # Hasilkan Insight AI jika Gemini tersedia
                        if gemini_model:
                            st.header(get_text('ai_insights'))
                            with st.spinner(get_text('generating_insights')):
                                # Buat word cloud untuk setiap sentimen
                                wordcloud_images = {}
                                for sentiment_type in ['positif', 'negatif', 'netral']:
                                    sentiment_texts = results_df[results_df['sentiment'] == sentiment_type]['text'].tolist()
                                    wordcloud_images[sentiment_type] = create_wordcloud(sentiment_texts, sentiment_type)
                                
                                # Hasilkan dan tampilkan insight
                                insights = generate_summary_with_gemini(
                                    gemini_model,
                                    sentiment_counts,
                                    wordcloud_images,
                                    st.session_state.language
                                )
                                st.markdown(insights)
                        
                        # Word Clouds
                        st.header(get_text('wordcloud_header'))
                        for sentiment_type in ['positif', 'negatif', 'netral']:
                            sentiment_texts = results_df[results_df['sentiment'] == sentiment_type]['text'].tolist()
                            wordcloud_fig = create_wordcloud(sentiment_texts, sentiment_type)
                            if wordcloud_fig:
                                st.pyplot(wordcloud_fig)
                        
                        # Hasil Detail
                        st.header(get_text('detail_results'))
                        
                        # Filter berdasarkan sentimen
                        selected_sentiment = st.selectbox(
                            get_text('filter_sentiment'),
                            ['all'] + list(sentiment_counts.keys())
                        )
                        
                        filtered_df = results_df if selected_sentiment == 'all' else \
                                    results_df[results_df['sentiment'] == selected_sentiment]
                        
                        st.dataframe(filtered_df[['text', 'sentiment', 'confidence']])
                        
                        # Unduh hasil
                        st.header(get_text('download_header'))
                        
                        csv = filtered_df.to_csv(index=False)
                        st.download_button(
                            get_text('download_button'),
                            csv,
                            "sentiment_analysis_results.csv",
                            "text/csv",
                            key='download-csv'
                        )
                        
                        # Statistik Ringkasan
                        st.header(get_text('summary_stats'))
                        
                        st.write(get_text('sentiment_distribution'))
                        for sentiment_type, count in sentiment_counts.items():
                            percentage = (count / len(results_df)) * 100
                            st.write(f"- {sentiment_type.title()}: {count} ({percentage:.1f}%)")
                        
                        st.write(get_text('confidence_stats'))
                        st.write(f"- Rata-rata: {results_df['confidence'].mean():.3f}")
                        st.write(f"- Median: {results_df['confidence'].median():.3f}")
                        st.write(f"- Min: {results_df['confidence'].min():.3f}")
                        st.write(f"- Maks: {results_df['confidence'].max():.3f}")
                        
        except Exception as e:
            st.error(get_text('error_processing').format(error=str(e)))
            st.info(get_text('file_validation'))

if __name__ == "__main__":
    main()
