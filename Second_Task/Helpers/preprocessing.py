from nltk.tokenize import word_tokenize
import nltk
import time
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

# NLTK veri indirme
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    print("NLTK veri indirme hatası, basit tokenizer kullanılacak")


#@method: simple_tokenize
# Basit tokenizer fonksiyonu, NLTK hatası durumunda kullanılacak
#@input:text (str): Tokenize edilecek metin
# @output : List[str]: Tokenize edilmiş kelimeler listesi
# @description: Basit bir regex ile metni tokenize eder, NLTK hatası durumunda kullanılır.
def simple_tokenize(text):
    """Basit tokenizer (NLTK hatası durumunda)"""
    if pd.isna(text):
        return []
    # Basit regex ile tokenize
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [token for token in tokens if len(token) > 2]

def clean_text(text):
    """Türkçe metin temizleme"""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\sçğıöşüÇĞIİÖŞÜ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_tr(text):
    """Türkçe tokenize (hata durumunda basit tokenizer)"""
    cleaned_text = clean_text(text)
    try:
        tokens = word_tokenize(cleaned_text)
        return [token for token in tokens if len(token) > 2]
    except:
        # NLTK hatası durumunda basit tokenizer kullan
        return simple_tokenize(cleaned_text)

def vectorize_text(text, model, vector_size):
    """Metni Word2Vec vektörüne çevir"""
    tokens = tokenize_tr(text)
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)
