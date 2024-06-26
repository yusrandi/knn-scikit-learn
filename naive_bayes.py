import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class NaiveBayesTesting:
    def ngetest(self, text) :
        print("Naive Bayes")

        # Step 1: Membaca dataset dari Excel
        file_path = 'dataset.xlsx'
        df = pd.read_excel(file_path)

        text_column = 'text'  # Ubah sesuai nama kolom teks di file Excel
        label_column = 'category'  # Ubah sesuai nama kolom label di file Excel

        # Periksa apakah kolom yang diharapkan ada di DataFrame
        if text_column not in df.columns or label_column not in df.columns:
            raise KeyError(f"Salah satu atau kedua kolom '{text_column}' dan '{label_column}' tidak ditemukan di DataFrame.")

        X = df[text_column].astype(str)  # Kolom teks
        y = df[label_column]  # Kolom label

        # Step 2: Inisialisasi dan melatih model Naive Bayes
        nb_model = Pipeline([
            ('vect', CountVectorizer()),  # CountVectorizer untuk mengubah teks menjadi vektor fitur
            ('clf', MultinomialNB()),     # Multinomial Naive Bayes sebagai klasifier
        ])

        nb_model.fit(X, y)  # Melatih model menggunakan seluruh data

       
        # Step 3: Melakukan prediksi kategori dengan model Naive Bayes
        predicted_category = nb_model.predict([text])[0]

        # Menampilkan hasil prediksi
        # print(f'Teks: {text}')
        # print(f'Kategori Prediksi: {predicted_category}')
        return predicted_category

