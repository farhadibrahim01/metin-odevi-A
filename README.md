Metin ödevi A

Bu projede, üç farklı metin vektörleme yöntemi ile (TF-IDF, LDA, BERT) belge sınıflandırması gerçekleştirilmiştir. Veriler PDF formatında olup, "alcohol", "dvi", ve "paternity" konularını içermektedir.

Klasör Yapısı

Metin-A/
├── data/
│   ├── raw/                  # Alcohol.pdf, DVI.pdf, Forensic Paternity.pdf
│   └── processed/            # train_initial.csv, train_clean.csv
├── outputs/                  # Model ve görselleştirme çıktıları
├── src/                      # Kodlar
│   ├── extract_data.py
│   ├── preprocess.py
│   ├── tfidf_features.py
│   ├── tfidf_logreg_final.py
│   ├── train_lda.py
│   ├── lda_classifier.py
│   ├── lda_vis.py
│   ├── bert_features.py
│   └── bert_classifier.py
└── README.md

Ortam Kurulumu

run:

python -m venv .venv
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
python -m nltk.downloader stopwords


> Eğer `requirements.txt` dosyası yoksa, manuel kur:
run:

pip install pandas numpy scikit-learn nltk joblib pdfminer.six unidecode gensim tqdm sentence-transformers pyldavis


Adım Adım Çalıştırma

1. Veri Hazırlığı

run:

python src/extract_data.py
python src/preprocess.py

2. TF-IDF ile Sınıflandırma

run:

python src/tfidf_features.py
python src/tfidf_logreg_final.py

Sonuç: `outputs/tfidf_top_terms.csv` ve `model_tfidf_logreg_balanced.joblib`

3. LDA ile Sınıflandırma

run:
python src/train_lda.py
python src/lda_classifier.py

Sonuç: `lda_logreg_model.joblib`, `lda_model`, `lda_dict.dict`, `lda_corpus.mm`

4. LDA Görselleştirme

run:

python src/lda_vis.py

Sonuç: `lda_viz.html` olarak kayıt edilir, tarayıcıda aç.

5. BERT ile Sınıflandırma

run:

python src/bert_features.py
python src/bert_classifier.py

Sonuç: `bert_embeddings.npy`, `bert_logreg_model.joblib`

Açıklama

- Her yöntem farklı bir vektörleme tekniğiyle metinleri temsilleyip, Logistic Regression sınıflandırıcısı ile eğitilmiştir.
- TF-IDF → %99 doğruluk
- LDA → %75 doğruluk
- BERT → %98 doğruluk
