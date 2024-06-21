from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

import pandas as pd

# Step 1: Membaca dataset dari Excel
file_path = 'dataset.xlsx'
df = pd.read_excel(file_path)
# Asumsikan kolom text adalah 'review' dan kolom label adalah 'category'
X = df['text'].astype(str)  # Kolom teks
y = df['category']  # Kolom label

# Step 2: Praproses teks (contoh menggunakan TF-IDF Vectorizer)
vectorizer = TfidfVectorizer(max_features=1000)  # Sesuaikan jumlah fitur TF-IDF sesuai kebutuhan
X = vectorizer.fit_transform(X)

# Step 3: Membagi data menjadi data pelatihan dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Melatih model KNN
knn_model = KNeighborsClassifier(n_neighbors=5)  # Contoh: menggunakan 5 tetangga terdekat
knn_model.fit(X_train, y_train)

# Step 5: Mengevaluasi model
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi: {accuracy:.2f}')

print('Laporan Klasifikasi:')
print(classification_report(y_test, y_pred))
