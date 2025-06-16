Metin-A: Çok Sınıflı Metin Sınıflandırma Projesi

Bu proje, alcohol, dvi ve forensic sınıflarına ait metin paragraflarını sınıflandırmak amacıyla çeşitli metin işleme ve makine öğrenmesi tekniklerini uygular: TF-IDF, LDA ve BERT.

Veri Hazırlama

Veri dosyası: data/paragraphs.csv
Bu dosya, prepare_data.py ile PDF dosyalarından otomatik olarak oluşturulmuştur.

Etiketleri 3 sınıfa indirmek için aşağıdaki komut çalıştırılır:
python clean_labels_to_3class.py
Bu işlem sonucunda BERT için kullanılacak data/paragraphs_3class.csv dosyası oluşturulur.

TF-IDF Aşaması

Amaç: Basit kelime sıklığı temelli bir yaklaşım ile metin sınıflandırması yapmaktır.

Çalıştırılacak script:
python train_model_tf-idf.py

Bu adımda:

TF-IDF vektörleri çıkarılır.

Kural tabanlı bir sınıflandırıcı uygulanır.

Sonuç: %100 doğruluk, ancak bu yüksek overfitting riski taşır.

LDA Aşaması

Amaç: Konu modellemesi ile her paragrafın hangi temaya ait olduğunu anlamaya çalışmak ve bunu sınıflandırmada kullanmaktır.

3.1. LDA Özelliklerini Hazırla
Çalıştırılacak script:
python prepare_lda_features.py

Bu script, aşağıdaki dosyaları üretir:

data/lda_features.csv

data/paragraphs_with_lda_topics.csv

3.2. LDA + TF-IDF ile Sınıflandır
Çalıştırılacak script:
python train_model_lda_classifier.py

Bu adımda:

LDA konuları + TF-IDF vektörleri birlikte kullanılır.

Kategorilere göre sınıflandırma yapılır.

BERT + Logistic Regression Aşaması

Amaç: BERT modelinden alınan anlam temelli [CLS] vektörleri ile metinleri sınıflandırmaktır.

4.1. BERT Özelliklerini Çıkar
Çalıştırılacak script:
python extract_bert_features.py
Girdi: data/paragraphs_3class.csv
Çıktı: data/bert_features.csv

4.2. PCA + Logistic Regression ile Sınıflandır
Çalıştırılacak script:
python train_model_bert.py

Bu adımda:

768 boyutlu BERT vektörleri PCA ile 100 boyuta indirgenir.

Logistic Regression uygulanır.

Ortalama doğruluk: %80–90

Kurulum

Gerekli tüm kütüphaneleri yüklemek için aşağıdaki komut kullanılır:
pip install -r requirements.txt

requirements.txt içeriği:
pandas
numpy
tqdm
scikit-learn
gensim
pyLDAvis
transformers
torch

Klasör Yapısı

Metin-A/
├── data/
│ ├── Alcohol.pdf
│ ├── DVI.pdf
│ ├── Forensic Paternity.pdf
│ ├── paragraphs.csv
│ ├── paragraphs_3class.csv
│ ├── paragraphs_with_lda_topics.csv
│ ├── lda_features.csv
│ ├── bert_features.csv
├── prepare_data.py
├── clean_labels_to_3class.py
├── prepare_lda_features.py
├── extract_bert_features.py
├── train_model_tf-idf.py
├── train_model_lda_classifier.py
├── train_model_lda.py
├── train_model_bert.py
├── requirements.txt
└── README.md

Yazar:
Ferhad Ibrahimov
İstanbul Ticaret Üniversitesi – 2025
İleri Makine Öğrenmesi Dönem Projesi