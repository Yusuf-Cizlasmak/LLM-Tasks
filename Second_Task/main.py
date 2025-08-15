# sentiment_word2vec_clean.py
import pandas as pd
import numpy as np
import joblib
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import nltk
import time
import re
from collections import Counter
import warnings


from Helpers.preprocessing import *


warnings.filterwarnings('ignore')



def load_data():
    """Turkish movie sentiment dataset'i yükle"""
    try:
        df = pd.read_csv("turkish_movie_sentiment_dataset.csv")
        
        # Sütun isimlerini kontrol et
        print(f"✓ Veri yüklendi: {len(df)} örnek")
        print(f"📊 Sütunlar: {list(df.columns)}")
        
        # NaN değerleri temizle
        df = df.dropna(subset=['comment', 'point'])
        
        # 'point' sütununu 'label'a çevir ve sentiment kategorilerine dönüştür
        # point: 1.0-2.5 = negatif (0), 3.0-3.5 = nötr (1), 4.0-5.0 = pozitif (2)
        df['point'] = df['point'].astype(str).str.replace(',', '.').astype(float)
        df['label'] = df['point'].apply(lambda x: 0 if x <= 2.5 else (1 if x <= 3.5 else 2))
        
        # 'comment' sütununu 'text' olarak kullan
        df['text'] = df['comment']
        
        # Sadece gerekli sütunları al
        df = df[['text', 'label']].dropna()
        
        # Çok kısa metinleri filtrele
        df = df[df['text'].str.len() > 10]
        
        print(f"📊 Temizlenen veri: {len(df)} örnek")
        print(f"📊 İlk 3 örnek:")
        for i in range(min(3, len(df))):
            print(f"  {i+1}. Text: {df.iloc[i]['text'][:100]}...")
            print(f"     Label: {df.iloc[i]['label']}")
        
        return df
    except Exception as e:
        print(f"❌ turkish_movie_sentiment_dataset.csv yüklenemedi: {e}")
        return None

def calculate_token_stats(texts):
    """Token istatistiklerini hesapla"""
    all_tokens = []
    total_sentences = len(texts)
    
    for text in texts:
        tokens = tokenize_tr(text)
        all_tokens.extend(tokens)
    
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    token_counter = Counter(all_tokens)
    
    print(f"  📊 Token İstatistikleri:")
    print(f"     Toplam cümle: {total_sentences}")
    print(f"     Toplam token: {total_tokens}")
    print(f"     Benzersiz token: {unique_tokens}")
    print(f"     Ortalama token/cümle: {total_tokens/total_sentences:.2f}")
    
    # En sık kullanılan 10 kelime
    most_common = token_counter.most_common(10)
    print(f"     En sık 10 kelime: {most_common}")
    
    return {
        'total_sentences': total_sentences,
        'total_tokens': total_tokens,
        'unique_tokens': unique_tokens,
        'avg_tokens_per_sentence': total_tokens/total_sentences,
        'most_common_words': most_common
    }

def calculate_tfidf_scores(texts, top_n=20):
    """TF-IDF skorlarını hesapla"""
    # Metinleri temizle
    cleaned_texts = [clean_text(text) for text in texts]
    
    # TF-IDF vektörizer
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words=None
    )
    
    # TF-IDF matrisini hesapla
    tfidf_matrix = tfidf.fit_transform(cleaned_texts)
    feature_names = tfidf.get_feature_names_out()
    
    # Ortalama TF-IDF skorları
    mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
    
    # En yüksek skorlu kelimeleri bul
    indices = np.argsort(mean_scores)[::-1][:top_n]
    top_words_scores = [(feature_names[i], mean_scores[i]) for i in indices]
    
    print(f"  📈 TF-IDF En Yüksek {top_n} Kelime:")
    for word, score in top_words_scores:
        print(f"     {word}: {score:.4f}")
    
    return tfidf, top_words_scores

def train_word2vec(texts, vector_size):
    """Word2Vec modeli eğit"""
    sentences = [tokenize_tr(t) for t in texts]
    
    start_time = time.time()
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=5,
        min_count=2,
        workers=4,
        sg=1,
        epochs=10
    )
    train_time = time.time() - start_time
    
    print(f"  ⚡ Word2Vec eğitildi - Süre: {train_time:.2f}s, Kelime: {len(model.wv.key_to_index)}")
    return model

def vectorize_data(texts, model, vector_size):
    """Veriyi vektörleştir"""
    return np.array([vectorize_text(t, model, vector_size) for t in texts])

def train_classifiers(X_train, y_train):
    """Sınıflandırıcıları eğit"""
    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42)
    }
    
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
    
    return classifiers

def evaluate_model(clf, X_val, y_val, clf_name):
    """Model performansını değerlendir"""
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")
    
    success = "✅" if accuracy > 0.80 and f1 > 0.75 else "❌"
    print(f"    {clf_name}: Acc={accuracy:.4f}, F1={f1:.4f} {success}")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'y_pred': y_pred
    }

def save_model(w2v_model, best_clf, vector_size):
    """Modeli kaydet"""
    filename = f"sentiment_w2v_{vector_size}d.joblib"
    joblib.dump((w2v_model, best_clf), filename)
    print(f"  💾 Model kaydedildi: {filename}")

def analyze_wrong_predictions(X_val, y_val, y_pred):
    """Hatalı tahminleri analiz et"""
    X_val_df = pd.DataFrame({'text': X_val})
    X_val_df['y_pred'] = y_pred
    X_val_df['y_val'] = y_val
    wrong_predictions = X_val_df[X_val_df['y_pred'] != X_val_df['y_val']]
    
    print(f"    Hatalı tahmin: {len(wrong_predictions)}/{len(X_val_df)}")
    return wrong_predictions

def compare_word2vec_tfidf(w2v_model, tfidf_top_words):
    """Word2Vec ve TF-IDF kelimelerini karşılaştır"""
    print(f"  🔄 Word2Vec vs TF-IDF Karşılaştırması:")
    
    found_in_w2v = 0
    for word, tfidf_score in tfidf_top_words[:10]:  # İlk 10'u kontrol et
        if word in w2v_model.wv.key_to_index:
            found_in_w2v += 1
            # Word2Vec'teki en benzer kelimeleri bul
            try:
                similar_words = w2v_model.wv.most_similar(word, topn=3)
                similar_str = ", ".join([f"{w}({s:.3f})" for w, s in similar_words])
                print(f"     '{word}' (TF-IDF: {tfidf_score:.4f}) → Benzer: {similar_str}")
            except:
                print(f"     '{word}' (TF-IDF: {tfidf_score:.4f}) → Benzer kelime bulunamadı")
        else:
            print(f"     '{word}' (TF-IDF: {tfidf_score:.4f}) → Word2Vec'te yok")
    
    print(f"     📈 TF-IDF top 10'dan {found_in_w2v} tanesi Word2Vec'te bulundu")

def main():
    """Ana fonksiyon"""
    print("🚀 Türkçe Film Yorumları Sentiment Analizi - Word2Vec + TF-IDF")
    
    # Veriyi yükle
    df = load_data()
    if df is None:
        return
    
    # Veriyi hazırla (performans için sınırla)
    df = df.head(15000)
    
    print(f"📊 Label dağılımı (0:negatif, 1:nötr, 2:pozitif):\n{df['label'].value_counts()}")
    
    # Token istatistiklerini hesapla
    print(f"\n📈 GENEL TOKEN ANALİZİ")
    token_stats = calculate_token_stats(df['text'])
    
    # TF-IDF hesapla
    print(f"\n📈 TF-IDF ANALİZİ")
    tfidf_vectorizer, tfidf_top_words = calculate_tfidf_scores(df['text'])
    
    # Veriyi böl
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    
    # Vektör boyutları
    vector_sizes = [100, 150, 200]
    all_results = {}
    
    for vector_size in vector_sizes:
        print(f"\n{'='*50}")
        print(f"🔄 Vektör boyutu: {vector_size}")
        print(f"{'='*50}")
        
        # Token istatistikleri (eğitim verisi için)
        train_token_stats = calculate_token_stats(X_train)
        
        # Word2Vec eğit
        w2v_model = train_word2vec(X_train, vector_size)
        
        # Word2Vec vs TF-IDF karşılaştırması
        compare_word2vec_tfidf(w2v_model, tfidf_top_words)
        
        # Vektörleştir
        X_train_vec = vectorize_data(X_train, w2v_model, vector_size)
        X_val_vec = vectorize_data(X_val, w2v_model, vector_size)
        
        # Sınıflandırıcıları eğit
        classifiers = train_classifiers(X_train_vec, y_train)
        
        # Değerlendirme
        print(f"  🎯 Model Performansları:")
        results = {}
        for clf_name, clf in classifiers.items():
            result = evaluate_model(clf, X_val_vec, y_val, clf_name)
            results[clf_name] = result
        
        # En iyi modeli bul ve kaydet
        best_clf_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_clf = classifiers[best_clf_name]
        save_model(w2v_model, best_clf, vector_size)
        
        # Hatalı tahminleri göster (en iyi model için)
        best_y_pred = results[best_clf_name]['y_pred']
        wrong_preds = analyze_wrong_predictions(X_val, y_val, best_y_pred)
        
        all_results[vector_size] = {
            'models': results,
            'token_stats': train_token_stats,
            'w2v_vocab_size': len(w2v_model.wv.key_to_index)
        }
    
    # Sonuçları özetle
    print(f"\n{'='*60}")
    print("📋 SONUÇLAR ÖZETİ")
    print(f"{'='*60}")
    
    best_overall = 0
    best_config = None
    
    for vector_size, data in all_results.items():
        results = data['models']
        token_stats = data['token_stats']
        vocab_size = data['w2v_vocab_size']
        
        print(f"\nVektör boyutu {vector_size}:")
        print(f"  📊 Eğitim token: {token_stats['total_tokens']}, W2V kelime: {vocab_size}")
        
        for clf_name, result in results.items():
            accuracy = result['accuracy']
            f1 = result['f1_score']
            success = "✅" if accuracy > 0.80 and f1 > 0.75 else "❌"
            print(f"  {clf_name}: Acc={accuracy:.4f}, F1={f1:.4f} {success}")
            
            if accuracy > best_overall:
                best_overall = accuracy
                best_config = (vector_size, clf_name)
    
    if best_config:
        vector_size, classifier = best_config
        print(f"\n🏆 EN İYİ SONUÇ:")
        print(f"  Accuracy: {best_overall:.4f}")
        print(f"  Vektör boyutu: {vector_size}")
        print(f"  Sınıflandırıcı: {classifier}")
        
    # TF-IDF özet
    print(f"\n📈 TF-IDF ÖZETİ:")
    print(f"  En yüksek 5 TF-IDF skoru:")
    for word, score in tfidf_top_words[:5]:
        print(f"    {word}: {score:.4f}")

if __name__ == "__main__":
    main()