import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

class KaeNeNTesting:
    def ngetest(self, text) :
        # Step 1: Membaca dataset dari Excel
        file_path = 'dataset.xlsx'
        df = pd.read_excel(file_path)
        # Asumsikan kolom text adalah 'review' dan kolom label adalah 'category'
        X = df['text'].astype(str)  # Kolom teks
        y = df['category']  # Kolom label

        # Step 2: Praproses teks pada data pelatihan
        vectorizer = TfidfVectorizer(max_features=1000)  # Sesuaikan jumlah fitur TF-IDF sesuai kebutuhan
        X = vectorizer.fit_transform(X)

        # Step 3: Melatih model KNN
        knn_model = KNeighborsClassifier(n_neighbors=7)  # Contoh: menggunakan 7 tetangga terdekat
        knn_model.fit(X, y)  # Menggunakan seluruh data untuk pelatihan

        
        # Step 4: Praproses teks baru
        new_text_vectorized = vectorizer.transform([text])

        # Step 5: Melakukan prediksi kategori dengan model KNN
        predicted_category = knn_model.predict(new_text_vectorized)
        predicted_proba = knn_model.predict_proba(new_text_vectorized)

        # Mendapatkan tetangga terdekat
        distances, indices = knn_model.kneighbors(new_text_vectorized)

        # print(f'Teks: {new_text}')
        # print(f'Kategori Prediksi: {predicted_category}')
        # Menampilkan hasil
        # print(f'Teks: {text}')
        # print(f'Kategori Prediksi: {predicted_category}')
        # print(f'Probabilitas Prediksi: {predicted_proba}')
        # print(f'distances: {distances}')
        # print(f'indices: {indices}')

        # print('\nTetangga Terdekat:')
        # for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        #     # print(f'{i+1}. Teks: idx: {idx} - Jarak: {dist}')
        #     print(f'{i+1}. Teks: {df.iloc[idx]["text"]} (Label: {df.iloc[idx]["category"]}) - Jarak: {dist}')

        return predicted_category[0]