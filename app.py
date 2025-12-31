import streamlit as st
import io
import csv
import pandas as pd
import re
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import pickle
import os
import platform

# --- KONFIGURASI HALAMAN & TESSERACT ---
st.set_page_config(page_title="OCR & Sentiment AI", page_icon="🧠", layout="wide")

if platform.system() == "Windows":
    # Sesuaikan path ini jika berbeda
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Scope untuk Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# --- 1. SETUP MODEL AI (SENTIMEN) ---
@st.cache_resource
def load_sentiment_model():
    """
    Load model IndoBERT untuk Analisis Sentimen.
    Cached agar tidak download ulang setiap klik.
    """
    try:
        model = pipeline(
            "sentiment-analysis", 
            model="w11wo/indonesian-roberta-base-sentiment-classifier",
            tokenizer="w11wo/indonesian-roberta-base-sentiment-classifier"
        )
        return model
    except Exception as e:
        st.error(f"Gagal memuat model AI. Pastikan internet lancar dan 'transformers' terinstall. Error: {e}")
        return None

def analyze_text_sentiment(text, model):
    """Fungsi helper untuk prediksi per kalimat"""
    if not isinstance(text, str) or not text.strip() or len(text) < 3:
        return "Netral"
    
    try:
        # Batasi panjang teks (max 512 token) untuk menghindari error model
        result = model(text[:512])[0]
        label = result['label']
        
        # Mapping label model ke Bahasa Indonesia yang rapi
        if label == 'positive': return 'Positif'
        elif label == 'negative': return 'Negatif'
        else: return 'Netral'
    except:
        return "Netral"

# --- 2. FUNGSI GOOGLE DRIVE & OCR ---
def authenticate_google_drive():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if os.path.exists('credentials.json'):
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            else:
                return None
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return creds

def extract_folder_id(url):
    if 'folders/' in url:
        return url.split('folders/')[1].split('?')[0]
    elif 'id=' in url:
         return url.split('id=')[1].split('&')[0]
    elif url and '/' not in url:
        return url
    return None

def get_images_from_folder(service, folder_id):
    query = f"'{folder_id}' in parents and (mimeType contains 'image/') and trashed = false"
    try:
        results = service.files().list(
            q=query, fields="files(id, name, mimeType)", pageSize=1000,
            includeItemsFromAllDrives=True, supportsAllDrives=True
        ).execute()
        return results.get('files', [])
    except Exception as e:
        st.error(f"Error Drive: {str(e)}")
        return []

def download_image(service, file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return Image.open(fh)

def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image, lang='ind+eng')
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        return "Error: Tesseract belum terinstall/path salah."
    except Exception as e:
        return f"Error: {str(e)}"

# --- 3. FUNGSI PEMBERSIHAN DATA ---
def clean_text_data(text, options):
    if pd.isna(text): return ""
    text = str(text)

    # A. Filtering Baris Sampah
    if options.get('remove_certificate_junk') or options.get('remove_admin_chat'):
        lines = text.split('\n')
        cleaned_lines = []
        junk_keywords = []
        
        if options.get('remove_certificate_junk'):
            junk_keywords.extend([
                "SERTIFIKAT", "SELEKSI KOMPETENSI", "KOMPETENSI DASAR", 
                "Tes Wawasan", "Tes Intelejensia", "Tes Karakteristik", 
                "Tanggal Ujian", "Diberikan kepada", "NO :", "Total", 
                "Screenshot_", ".jpg", ".png", "Nama File"
            ])
            
        if options.get('remove_admin_chat'):
            junk_keywords.extend([
                "Admin pengen tau", "prediksi kita", "materi apa aja", 
                "sesuai dengan prediksi", "Anda", "ketik pesan", "online"
            ])

        for line in lines:
            is_junk = False
            for keyword in junk_keywords:
                if keyword.lower() in line.lower():
                    is_junk = True
                    break
            if not is_junk:
                cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)

    # B. Regex Cleaning
    if options.get('remove_wa_meta'):
        text = re.sub(r'(\+62|62|08)\d{8,15}', '', text)
        text = re.sub(r'\b\d{1,2}[:.]\d{2}\b', '', text)
        text = re.sub(r'\d{1,3}%', '', text)
        text = re.sub(r'[<>]', '', text)
        text = re.sub(r'(online|mengetik|terakhir dilihat|pesan diteruskan)', '', text, flags=re.IGNORECASE)

    if options.get('remove_symbols'):
        text = re.sub(r'[^\w\s.,?!@-]', ' ', text)

    if options.get('custom_regex'):
        try:
            text = re.sub(options['custom_regex'], '', text)
        except:
            pass

    if options.get('remove_newlines'):
        text = text.replace('\n', ' ').replace('\r', ' ')

    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- UI UTAMA ---

st.title("🧠 OCR, Cleaning & Sentiment AI Dashboard")
st.markdown("Convert gambar testimoni, bersihkan datanya, lalu analisis sentimennya menggunakan AI.")

# Inisialisasi Session State
if 'ocr_data' not in st.session_state:
    st.session_state['ocr_data'] = None
if 'sentiment_results' not in st.session_state:
    st.session_state['sentiment_results'] = None

# Tab Layout (3 TABS)
tab1, tab2, tab3 = st.tabs(["📥 1. Ekstraksi OCR", "🧼 2. Pembersihan Data", "🤖 3. Analisis Sentimen"])

# ==========================================
# TAB 1: EKSTRAKSI
# ==========================================
with tab1:
    st.header("Ambil Gambar dari Google Drive")
    drive_url = st.text_input("URL Google Drive Folder:", value="https://drive.google.com/drive/folders/1N_woz7qp2gCJFLc5I8jExh3BKB_31WdW")
    
    if st.button("🚀 Mulai Proses OCR", type="primary"):
        folder_id = extract_folder_id(drive_url)
        if not folder_id:
            st.error("URL Folder Salah.")
            st.stop()
            
        try:
            with st.spinner("Autentikasi Google Drive..."):
                creds = authenticate_google_drive()
                if not creds:
                    st.error("Gagal login. Cek credentials.json")
                    st.stop()
                service = build('drive', 'v3', credentials=creds)
            
            with st.spinner("Mencari gambar..."):
                images = get_images_from_folder(service, folder_id)
            
            if not images:
                st.warning("Tidak ada gambar ditemukan. Cek akses folder.")
                st.stop()
            
            st.info(f"Ditemukan {len(images)} gambar. Memulai OCR...")
            
            progress_bar = st.progress(0)
            results = []
            
            for idx, img_file in enumerate(images):
                try:
                    image = download_image(service, img_file['id'])
                    text = extract_text_from_image(image)
                    results.append({'Nama File': img_file['name'], 'Teks Asli': text})
                except Exception as e:
                    results.append({'Nama File': img_file['name'], 'Teks Asli': f"Error: {e}"})
                
                prog = int((idx + 1) / len(images) * 100)
                progress_bar.progress(prog)
            
            df = pd.DataFrame(results)
            st.session_state['ocr_data'] = df
            st.success("✅ OCR Selesai! Silakan pindah ke Tab 2.")
            
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

# ==========================================
# TAB 2: PEMBERSIHAN
# ==========================================
with tab2:
    st.header("Bersihkan Hasil OCR")
    
    if st.session_state['ocr_data'] is None:
        st.warning("⚠️ Belum ada data OCR. Upload CSV manual jika ada.")
        uploaded_file = st.file_uploader("Upload CSV OCR:", type=['csv'], key="up1")
        if uploaded_file:
            temp_df = pd.read_csv(uploaded_file)
            if 'Teks Hasil OCR' in temp_df.columns:
                temp_df = temp_df.rename(columns={'Teks Hasil OCR': 'Teks Asli'})
            elif 'Teks Asli' not in temp_df.columns and len(temp_df.columns) >= 2:
                 temp_df = temp_df.rename(columns={temp_df.columns[1]: 'Teks Asli'})
            st.session_state['ocr_data'] = temp_df
            st.rerun()
    else:
        df = st.session_state['ocr_data'].copy()
        
        col_view, col_clean = st.columns([1, 1])
        with col_view:
            st.subheader("Preview Data Mentah")
            st.dataframe(df.head(3), use_container_width=True)

        with col_clean:
            st.subheader("⚙️ Filter Testimoni")
            with st.form("cleaning_form"):
                opt_cert = st.checkbox("Hapus Teks Sertifikat / Nilai", value=True)
                opt_admin = st.checkbox("Hapus Chat Admin", value=True)
                opt_wa = st.checkbox("Hapus Metadata WA", value=True)
                opt_newline = st.checkbox("Gabungkan Baris (Satu Paragraf)", value=True)
                opt_symbols = st.checkbox("Hapus Simbol Noise", value=True)
                custom_regex = st.text_input("Regex Kustom:", placeholder="")
                submit_clean = st.form_submit_button("✨ Bersihkan")

        if submit_clean:
            options = {
                'remove_certificate_junk': opt_cert, 'remove_admin_chat': opt_admin,
                'remove_wa_meta': opt_wa, 'remove_newlines': opt_newline,
                'remove_symbols': opt_symbols, 'custom_regex': custom_regex
            }
            
            df['Teks Bersih'] = df['Teks Asli'].apply(lambda x: clean_text_data(x, options))
            
            # Filter yang kosong / terlalu pendek
            df_clean = df[df['Teks Bersih'].str.len() > 5].copy()
            
            st.session_state['ocr_data'] = df_clean # Update session state dengan data bersih
            st.success("✅ Pembersihan Selesai! Data di-update.")
            st.dataframe(df_clean[['Nama File', 'Teks Bersih']].head())

# ==========================================
# TAB 3: SENTIMEN ANALISIS (BARU)
# ==========================================
with tab3:
    st.header("🤖 Analisis Sentimen (IndoBERT)")
    
    # Cek Data
    data_ready = False
    target_col = ""
    
    if st.session_state['ocr_data'] is not None:
        df_sent = st.session_state['ocr_data'].copy()
        
        # Prioritaskan kolom 'Teks Bersih' (dari Tab 2), kalau tidak ada pakai 'Teks Asli'
        if 'Teks Bersih' in df_sent.columns:
            target_col = 'Teks Bersih'
            data_ready = True
            st.info(f"Menggunakan data dari kolom: **{target_col}** (Hasil Tab 2)")
        elif 'Teks Asli' in df_sent.columns:
            target_col = 'Teks Asli'
            data_ready = True
            st.warning(f"Menggunakan data mentah: **{target_col}**. Disarankan bersihkan dulu di Tab 2.")
    else:
        st.warning("Belum ada data. Silakan proses di Tab 1 & 2 atau upload CSV di bawah.")
        upl_sent = st.file_uploader("Upload CSV Testimoni:", type=['csv'], key="up_sent")
        if upl_sent:
            df_sent = pd.read_csv(upl_sent, sep=';', on_bad_lines='skip') # Coba separator titik koma
            if len(df_sent.columns) < 2: # Kalau gagal, coba koma
                 upl_sent.seek(0)
                 df_sent = pd.read_csv(upl_sent, sep=',')
            
            # Cari kolom teks
            possible_cols = ['Teks Bersih', 'Teks Asli', 'feedback', 'review', 'Teks']
            found_col = next((c for c in possible_cols if c in df_sent.columns), None)
            
            if found_col:
                target_col = found_col
                data_ready = True
                st.session_state['ocr_data'] = df_sent
                st.success(f"File dimuat. Kolom target: {target_col}")
            else:
                st.error("Tidak ditemukan kolom teks (Teks Bersih/feedback).")

    # PROSES ANALISIS
    if data_ready:
        if st.button("🚀 Jalankan Analisis AI"):
            # Load Model
            with st.spinner("Memuat Model IndoBERT... (Pertama kali mungkin agak lama)"):
                model = load_sentiment_model()
            
            if model:
                # Progress bar
                prog_bar = st.progress(0)
                sentiments = []
                total = len(df_sent)
                
                status_text = st.empty()
                
                for i, row in df_sent.iterrows():
                    text_content = row[target_col]
                    # Prediksi
                    sentiment = analyze_text_sentiment(text_content, model)
                    sentiments.append(sentiment)
                    
                    # Update progress
                    percent = int((i + 1) / total * 100)
                    prog_bar.progress(percent)
                    status_text.text(f"Menganalisis {i+1}/{total}...")
                
                df_sent['Sentimen'] = sentiments
                st.session_state['sentiment_results'] = df_sent
                st.success("Analisis Selesai!")
                prog_bar.empty()

    # HASIL VISUALISASI
    if st.session_state['sentiment_results'] is not None:
        final_df = st.session_state['sentiment_results']
        
        st.divider()
        
        # 1. METRIK RINGKASAN
        c1, c2, c3, c4 = st.columns(4)
        total_data = len(final_df)
        pos = len(final_df[final_df['Sentimen'] == 'Positif'])
        neg = len(final_df[final_df['Sentimen'] == 'Negatif'])
        net = len(final_df[final_df['Sentimen'] == 'Netral'])
        
        c1.metric("Total Testimoni", total_data)
        c2.metric("Positif 😊", pos)
        c3.metric("Negatif 😡", neg)
        c4.metric("Netral 😐", net)
        
        st.divider()

        # 2. GRAFIK (PIE & BAR)
        col_chart1, col_chart2 = st.columns(2)
        
        color_map = {'Positif': '#66b3ff', 'Negatif': '#ff9999', 'Netral': '#99ff99'}
        
        with col_chart1:
            st.subheader("Proporsi Sentimen")
            counts = final_df['Sentimen'].value_counts()
            if not counts.empty:
                # Resize figure to be smaller
                fig1, ax1 = plt.subplots(figsize=(3, 3))
                colors = [color_map.get(x, '#ccc') for x in counts.index]
                ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
                st.pyplot(fig1, use_container_width=False)

        with col_chart2:
            st.subheader("Jumlah per Kategori")
            if not final_df.empty:
                # Resize figure to be smaller
                fig2, ax2 = plt.subplots(figsize=(3, 3))
                sns.countplot(x='Sentimen', data=final_df, palette=color_map, ax=ax2)
                st.pyplot(fig2, use_container_width=False)

        # 3. TABEL DATA & DOWNLOAD
        st.subheader("📋 Data Hasil Analisis")
        st.dataframe(final_df[[target_col, 'Sentimen']], use_container_width=True)
        
        # Download Button
        # Clean newline for CSV safety
        final_df[target_col] = final_df[target_col].astype(str).str.replace(r'[\r\n]+', ' ', regex=True)
        
        csv_result = final_df.to_csv(index=False, sep=';', quoting=csv.QUOTE_ALL).encode('utf-8-sig')
        
        st.download_button(
            label="📥 Download Laporan Sentimen (CSV)",
            data=csv_result,
            file_name="laporan_sentimen_ai.csv",
            mime="text/csv",
            type="primary"
        )