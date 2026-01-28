# Laporan Proyek Machine Learning - Naomi Sitanggang

## Project Overview

Dalam era digital yang semakin berkembang, akses terhadap informasi literatur menjadi semakin luas dan mudah dijangkau oleh masyarakat. Tersedianya ribuan judul buku secara daring melalui berbagai platform membuka peluang bagi siapa saja untuk meningkatkan minat baca serta memperluas wawasan. Namun demikian, tingginya jumlah buku yang tersedia justru dapat menimbulkan permasalahan tersendiri, yakni kesulitan dalam memilih buku yang relevan dan sesuai dengan preferensi individu (Lops et al., 2011).

Salah satu solusi yang dapat diimplementasikan untuk mengatasi permasalahan tersebut adalah pengembangan sistem rekomendasi buku. Sistem ini bertujuan untuk mempermudah pengguna dalam menemukan buku yang sesuai dengan selera mereka, berdasarkan histori interaksi atau karakteristik konten buku. Dua pendekatan utama yang umum digunakan dalam pengembangan sistem rekomendasi adalah _Collaborative Filtering_ dan _Content Based Filtering_. _Collaborative Filtering_ memanfaatkan pola interaksi antar pengguna dan item, sedangkan _Content Based Filtering_ berfokus pada atribut dari item itu sendiri (Ricci et al., 2015).

Proyek ini menggunakan dataset Goodbooks-10k, yang terdiri dari 10.000 judul buku dan lebih dari 6 juta data penilaian dari pengguna. Dataset ini memberikan representasi yang komprehensif mengenai interaksi pembaca terhadap buku, sehingga memungkinkan pengembangan sistem rekomendasi yang lebih akurat dan terpersonalisasi.

Dengan mengimplementasikan model _deep learning_ berbasis _embedding_ melalui pendekatan _Collaborative Filtering_ dan _Content Based Filtering_, sistem ini diharapkan mampu memberikan rekomendasi buku yang relevan dan meningkatkan kepuasan pengguna dalam menemukan buku yang sesuai. Secara lebih luas, keberadaan sistem seperti ini juga dapat berkontribusi terhadap peningkatan budaya literasi di masyarakat, terutama dengan memberikan pengalaman membaca yang lebih efisien dan terarah.


## Business Understanding

### Problem Statements

1. Bagaimana membantu pengguna menemukan buku yang sesuai dengan preferensi mereka secara efisien di tengah banyaknya pilihan yang tersedia secara daring?
2. Bagaimana memanfaatkan data interaksi pengguna (seperti rating) untuk membangun sistem rekomendasi buku yang relevan dan terpersonalisasi?
3. Sejauh mana model berbasis _deep learning_ dengan pendekatan _Collaborative Filtering_ atau _Content Based Filtering_  mampu meningkatkan akurasi rekomendasi dibanding metode konvensional?
4. Bagaimana mengoptimalkan performa sistem rekomendasi agar tetap efisien dalam menangani data skala besar seperti Goodbooks-10k?

### Goals

1. Mengembangkan sistem rekomendasi buku berbasis _deep learning_ yang dapat memberikan rekomendasi buku yang relevan kepada pengguna.
2. Mengimplementasikan pendekatan  _Collaborative Filtering_ berbasis _embedding_  dan _Content Based Filtering_ untuk mempelajari pola interaksi pengguna dan buku secara efisien.
3. Mengevaluasi performa model dengan menggunakan metrik seperti _Root Mean Squared Error_ (RMSE) dan _Precision_ untuk menilai kualitas prediksi.
4. Meningkatkan pengalaman pengguna dalam menemukan buku dengan mengurangi waktu pencarian dan meningkatkan kepuasan terhadap hasil rekomendasi.

### Solution Statements
- Menggunakan pendekatan _Collaborative Filtering_ berbasis _embedding_ dengan memanfaatkan jaringan saraf  untuk mempelajari representasi laten pengguna dan buku.
- Mengimplementasikan _Content Based Filtering_ sebagai pembanding atau pelengkap, dengan mempertimbangkan fitur-fitur buku seperti penulis dan judul buku.
- Melakukan pelatihan dan validasi model menggunakan dataset Goodbooks-10k yang memuat jutaan interaksi pengguna terhadap ribuan buku.
- Mengevaluasi performa sistem menggunakan _Root Mean Squared Error_  dan _Precision_ lalu menyajikan rekomendasi buku bagi pengguna berdasarkan hasil prediksi model.
- Menyediakan visualisasi metrik pelatihan model serta contoh rekomendasi buku yang dihasilkan oleh sistem.


## Data Understanding
Proyek ini menggunakan data yang diambil dari situs Kaggle dengan judul  [goodbooks-10k](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k/data). Dataset tersebut memuat sebanyak **lima file** dengan informasi yang dimiliki berbeda di setiap file.

### Struktur dan variabel data
1. **book_tags**

   File ini berisi data label buku

   | No | Variabel                     | Tipe Data |
   |----|------------------------------|-----------|
   | 0  | goodreads_book_id            | int64   |
   | 1  | tag_id                       | int64   |
   | 2  | count                        | int64   |

   - goodreads_id : ID dari goodreads
   - tag_id : ID tag (genre)
   - count : Jumlah goodreads
   
   Kondisi file :
   - Jumlah baris dan kolom : (999912, 3)
   - Missing value : 0
   - Data duplikat  :
        - goodreads_book_id: 989912 duplikat
        - tag_id: 965660 duplikat
        - count: 990511 duplikat

   Tipe data yang dimiliki dari semua variabel merupakan tipe data numerik.
   

2. **books**

   File ini berisi informasi mengenai buku

   | No | Variabel            | Tipe Data |
   |----|------------------ |-----------|
   | 0 | id                 | int64   |
   | 1 | book_id            | int64   |
   | 2 | best_book_id       | int64   |
   | 3 | work_id            | int64   |
   | 4 | books_count        | int64   |
   | 5 | isbn               | object  |
   | 6 | isbn13             | float64 |
   | 7 | authors            | object  |
   | 8 | original_publication_year      | float64 |
   | 9 | original_title     | object  |
   | 10 | title             | object  |
   | 11 | language_code     | object  |
   | 12 | average_rating    | float64 |
   | 13 | ratings_count     | int64   |
   | 14 | work_ratings_count| int64   |
   | 15 | work_text_reviews_count       | int64   |
   | 16 | ratings_1         | int64   |
   | 17 | ratings_2         | int64   |
   | 18 | ratings_3         | int64   |
   | 19 | ratings_4         | int64   |
   | 20 | ratings_5         | int64   |
   | 21 | image_url         | object  |
   | 22 | small_image_url   | object  | 
   - id : ID dari file books
   - book_id : ID buku
   - best_book_id : ID dari buku populer
   - work_id : ID karya
   - books_count : jumlah edisi buku tertentu
   - isbn : nomor isbn
   - authors : nama penulis
   - original_publication_year : tahun terbit buku
   - original_title : judul asli buku
   - title : Judul versi final atau yang digunakan di katalog
   - language_code : Kode bahasa
   - average_rating : Rata-rata rating dari semua pengguna
   - ratings_count :  Total jumlah rating yang diterima buku
   - work_ratings_count : Jumlah rating untuk karya
   - work_text_reviews_count : Jumlah ulasan berbasis teks terhadap karya
   - ratings_1 hingga ratings_5 : Jumlah pengguna yang memberi rating 1 hingga 5 bintang
   - image_url : URL gambar sampul buku
   - small_image_url : URL gambar sampul berukuran kecil

   Kondisi file :
   - Jumlah baris dan kolom : (10000, 23)
   - Missing value : 
        - isbn : 700
        - isbn13 : 585
        - original_publication_year : 21
        - original_title : 585
        - language_code : 1084
   - Data duplikat  :
        - books_count: 9403 duplikat
        - isbn: 699 duplikat
        - isbn13: 846 duplikat
        - authors: 5336 duplikat
        - original_publication_year: 9706 duplikat
        - original_title: 725 duplikat
        - title: 36 duplikat
        - language_code: 9974 duplikat
        - average_rating: 9816 duplikat
        - ratings_count: 997 duplikat
        - work_ratings_count: 947 duplikat
        - work_text_reviews_count: 5419 duplikat
        - ratings_1: 7370 duplikat
        - ratings_2: 5883 duplikat
        - ratings_3: 3028 duplikat
        - ratings_4: 2238 duplikat
        - ratings_5: 1897 duplikat
        - image_url: 3331 duplikat
        - small_image_url: 3331 duplikat

    Tipe data yang dimiliki dari semua variabel merupakan tipe data numerik dan kategorik

3. **ratings**

   File ini berisi rating buku sesuai id pengguna

   | No | Variabel     | Tipe Data |
   |----|--------------|-----------|
   | 0  | book_id      | int64     |
   | 1  | user_id      | int64     |
   | 2  | rating       | int64     |
   - book_id : ID buku
   - user_id : ID Pengguna
   - rating : rating buku

   Kondisi file :
   - Jumlah baris dan kolom : (981756, 3)
   - Missing value : 0
   - Data duplikat  :
        - book_id: 971756 duplikat
        - user_id: 928332 duplikat
        - rating: 981751 duplikat

   Tipe data yang dimiliki dari semua variabel merupakan tipe data numerik

4. **tags**

   File ini berisi tentang id-nama tag

   | No | Variabel        | Tipe Data |
   |----|-----------------|-----------|
   | 0 | tag_id           | int64     |
   | 1 | tag_name         | object    |
 
   - tag_id : ID tag (genre)
   - tag_name : Nama tag (genre)

   Kondisi file :
   - Jumlah baris dan kolom : (34252, 2)
   - Missing value : 0
   - Data duplikat  : 0
   
   Tipe data yang dimiliki dari semua variabel merupakan tipe data numerik dan kategorik

5. **to_read**

   File ini berisi daftar buku yang ditandai oleh pengguna untuk dibaca

   | No | Variabel        | Tipe Data |
   |----|-----------------|-----------|
   | 0  | user_id         | int64     |
   | 1  | book_id         | int64    |
 
   - user_id : ID pengguna/pembaca
   - book_id : ID buku

   Kondisi file :
   - Jumlah baris dan kolom : (912705, 2)
   - Missing value : 0
   - Data duplikat  :
        - user_id: 863834 duplikat
        - book_id: 902719 duplikat

  Tipe data yang dimiliki dari semua variabel merupakan tipe data numerik
      
### Mendeteksi missing values pada data detail_ulasan

   Pada proses _Content Based Filtering_ kita hanya akan menggunakan book_id, user_id, rating, authors, dan title.
  | No | Variabel             | Jumlah missing value |
  |----|----------------------|----------------------|
  | 0 | book_id                     | 0        |
  | 1 | user_id                     | 0        |
  | 2 | rating                      | 0        |
  | 3 | authors                     | 88860317 |
  | 4 | title                       | 88860317 |
  | 5 | original_publication_year   | 88870317 |

   Terdapat missing value pada variabel authors sebanyak 88860317, title sebanyak 88860317, dan original_publication_year sebanyak 88870317.


### Mendeteksi data duplikat pada data detail_ulasan

Pada proses _Content Based Filtering_ kita hanya akan menggunakan book_id, user_id, rating, authors, dan title. 

![Duplikat](https://raw.githubusercontent.com/glimpseofnaomi/image/main/books.png)

Berdasarkan book_id terdapat 7837104 data duplikat 

### Exploratory data analysis

1. Distribusi rating buku

   ![Distribusi Histogram](https://raw.githubusercontent.com/glimpseofnaomi/image/main/rating.png)

   Berdasarkan grafik distribusi rating buku, terlihat bahwa sebagian besar pengguna memberikan rating yang tinggi, terutama pada skor 4.0, diikuti oleh 5.0 dan 3.0. Sementara itu, rating rendah seperti 1.0 dan 2.0 relatif jarang diberikan. Hal ini mengindikasikan bahwa pengguna cenderung memberikan penilaian positif terhadap buku yang mereka baca, yang bisa disebabkan oleh preferensi terhadap bacaan tertentu atau karena mereka hanya memberi rating pada buku yang benar-benar mereka sukai.

2. Distribusi top 10 tag yang paling sering digunakan

   ![Distribusi Histogram](https://raw.githubusercontent.com/glimpseofnaomi/image/main/tag.png)

    Berdasarkan grafik top 10 tag menunjukkan bahwa tag to-read merupakan yang paling sering digunakan oleh pengguna, jauh melampaui tag lainnya seperti currently-reading, favorites, dan fiction. Dominasi tag to-read ini menunjukkan bahwa banyak pengguna menggunakan fitur tagging untuk menandai buku-buku yang ingin mereka baca di masa depan. Hal ini mencerminkan perilaku eksploratif dan minat tinggi pengguna terhadap buku-buku yang belum mereka baca.

## Data Preparation
Pada tahap ini, dilakukan beberapa teknik untuk menyiapkan data sebelum masuk ke proses pemodelan. Urutan dan penjelasan tiap langkah sebagai berikut:

1. Menangani missing values

   Setelah melakukan pengecekan _missing values_, terdeteksi terdapat _missing values_ pada 3 variabel dari variabel book_id, user_id, rating, authors, dan title pada data detail_ulasan. Hal ini perlu diatasi dengan melakukan penghapusan data dengan missing value menggunakan `dropna()` untuk menghilangkan data yang bernilai kosong. Alasan menghapus _missing values_ ini agar model tidak _error_ saat pelatihan dan dapat meningkatkan kualitas data. Teknik mengatasi _missing values_ ini diterapkan untuk data _preparation_ pada model _Content Based Filtering_.


2. Menangani data duplikat

   Setelah melakukan pengecekan data duplikat pada data detail_ulasan, terdeteksi terdapat data duplikat berdasarkan book_id, sehingga data duplikat tersebut perlu diatasi dengan melakukan penghapusan data dengan `drop_duplicates()`. Alasan menghapus data supaya menghindari bias dalam pelatihan model dan menjaga konsistensi dalam evaluasi. Teknik mengatasi data duplikat ini diterapkan untuk data _preparation_ pada model _Content Based Filtering_.

3. Encoding data

   Variabel user_id dan book_id dilakukan proses encoding ke bentuk indeks integer yang akan digunakan pada model _collaborative filtering_. Alasan dilakukan _encoding_ agar bisa diproses oleh model _machine learning_ atau _deep learning_, yang umumnya hanya dapat memahami input dalam bentuk angka.

4. Membagi data

   Dataset dibagi menjadi 80% data latih dan 20% data uji. Alasan melakukan pembagian data ini agar model memiliki cukup data untuk belajar pola secara optimal, sekaligus dapat dievaluasi performanya pada data yang belum pernah dilihat. Rasio ini umum digunakan karena memberikan keseimbangan antara akurasi pelatihan dan kemampuan generalisasi, sehingga model tidak _overfitting_ dan tetap andal saat digunakan pada data baru.

## Modeling

Pada tahap _modeling_, digunakan 2 metode yaitu _Content Based Filtering_ dan _Collaborative Filtering_

1. **Content Based Filtering**

    Pendekatan _Content Based Filtering_ dikembangkan dengan tujuan merekomendasikan buku yang memiliki kemiripan karakteristik dengan buku yang disukai oleh pengguna. Dalam implementasinya, sistem ini memanfaatkan informasi dari judul dan nama penulis buku untuk membangun representasi fitur menggunakan teknik TF-IDF (_Term Frequency-Inverse Document Frequency_) pada judul buku dan penulis. Setelah itu, digunakan metode _cosine similarity_ untuk mengukur tingkat kemiripan antar buku berdasarkan representasi vektor tersebut. Sistem ini kemudian menghasilkan lima rekomendasi teratas (_**top-5 recommendation**_) yang paling mirip dengan buku yang diberikan sebagai input. 

    Kelebihan dari pendekatan ini adalah kemampuannya dalam memberikan rekomendasi meskipun hanya berdasarkan satu buku input dari pengguna, tanpa perlu bergantung pada data pengguna lain. Hal ini membuat _Content Based Filtering_ sangat cocok untuk menangani masalah _cold start_ bagi pengguna baru. Namun, kelemahannya terletak pada ruang rekomendasi yang sempit, karena sistem cenderung hanya menyarankan buku-buku yang sangat mirip dengan yang sudah dikenal pengguna, sehingga dapat mengurangi keragaman dalam hasil rekomendasi.

    Modelling menggunakan _Content Based Filtering_ menghasilkan rekomendasi top 5 buku dari judul buku yang dimasukkan pada sistem.

    Hasil Pencarian Informasi Berdasarkan Judul Buku

   |    | IDBuku |           judul_buku |            penulis | thn_terbit |
   |----|--------|----------------------|--------------------|------------|
   |326 |   4406 | East of Eden         | John Steinbeck     |     1952.0 |

    Hasil Rekomendasi Top 5 Buku dari Judul Buku yang Dimasukkan
   
   |   |                                        judul_buku |       penulis  |
   |---|---------------------------------------------------|----------------|
   | 0 |                              Of Mice and Men      | John Steinbeck |
   | 1 |                        The Grapes of Wrath        | John Steinbeck |
   | 2 |                          The Pearl                | John Steinbeck |
   | 3 | Travels with Charley: In Search of America	     | John Steinbeck |
   | 4 |                                    The Summons    |   John Grisham |
   

2. **Collaborative Filtering**

   Pendekatan _Collaborative Filtering_ dirancang untuk memberikan rekomendasi berdasarkan pola perilaku pengguna lain yang memiliki kesamaan dalam memberikan rating. Dalam proyek ini, digunakan arsitektur _deep learning_ sederhana melalui kelas RecommenderNet yang mengandalkan _embedding layer_ untuk merepresentasikan hubungan antara user_id dan book_id. Model ini dilatih menggunakan data interaksi pengguna dan buku dari file ratings.csv, dengan tujuan memprediksi seberapa besar kemungkinan seorang pengguna akan menyukai suatu buku yang belum pernah ia baca. Hasil akhir dari model ini adalah sepuluh rekomendasi teratas (_**top-10 recommendation**_) dengan prediksi rating tertinggi bagi pengguna tertentu.

   Pendekatan ini unggul dalam hal personalisasi karena dapat menangkap preferensi implisit pengguna dari data historis rating. Dengan demikian, sistem dapat merekomendasikan buku yang mungkin tidak memiliki kesamaan konten, tetapi disukai oleh pengguna lain dengan pola preferensi serupa. Namun, kelemahan dalam _Collaborative Filtering_ adalah _cold start problem_, terutama ketika sistem dihadapkan pada pengguna atau buku yang belum memiliki cukup data interaksi. Selain itu, model ini membutuhkan jumlah data yang besar agar bisa belajar secara optimal dan menghasilkan prediksi yang akurat.

   _Modelling_ menggunakan _Collaborative Filtering_ menghasilkan rekomendasi top 10 buku dengan rating tinggi
   
   | penulis                                                       | judul_buku                                                |
   |---------------------------------------------------------------|-----------------------------------------------------------|
   | J.K. Rowling, Mary GrandPré, Rufus Beck                       | Harry Potter and the Prisoner of Azkaban (Harry Potter#3) |
   | Homer, Robert Fagles, E.V. Rieu, Frédéric Mugler, Bernard Knox| The Odyssey                                               |
   | José Saramago, Margaret Jull Costa                            | All the Names                                             |
   | E.M. Forster                                                  | A Room with a View                                        |
   | John Grisham                                                  | The Innocent Man: Murder and Injustice in a Small Town    |
   | Steven Pinker                                                 | The Language Instinct: How the Mind Creates Language      |
   | Patricia Cornwell                                             | Predator (Kay Scarpetta, #14)                             |
   | Tom Robbins                                                   | Half Asleep in Frog Pajamas                               |
   | Tom Robbins                                                   | Another Roadside Attraction                               |
   | D.H. Lawrence                                                 |  Women in Love (Brangwen Family, #2)                      |
   
## Evaluation

### Metrik evaluasi yang digunakan:

**Precision**

_Precision_ digunakan untuk mengukur seberapa relevan rekomendasi yang dihasilkan oleh model _Content Based Filtering_. Metrik ini menunjukkan proporsi item yang benar-benar relevan dari seluruh item yang direkomendasikan. Semakin tinggi precision, semakin baik model dalam memberikan rekomendasi yang sesuai dengan preferensi pengguna.

Precision = Precision = jumlah rekomendasi relevan / jumlah total rekomendasi

Sebagai contoh, sistem merekomendasikan 5 buku berdasarkan penulis John Steinbeck, dan 4 di antaranya ditulis oleh penulis tersebut. Maka precision = 4/5 = 0.8, atau 80%. Artinya, 80% rekomendasi yang diberikan sistem terbukti relevan.

_**Root Mean Squared Error**_

_Root Mean Squared Error_ (RMSE) mengukur selisih antara rating yang diprediksi dengan rating sebenarnya dalam  _Collaborative Filtering_. Nilai RMSE yang kecil menandakan bahwa model memiliki prediksi yang akurat. Metrik ini penting karena memperhitungkan besarnya kesalahan dan memberikan penalti lebih besar untuk prediksi yang meleset jauh. Perhitungan RMSE ditunjukkan pada rumus berikut.

$$RMSE = \sqrt{\Sigma_{i=1}^{n}{\frac{(ŷ_i - y_i)^{2}}{n}}}$$

Keterangan: 

RMSE = nilai root mean square error

y = nilai hasil observasi

ŷ = nilai hasil prediksi

i = urutan data

n = jumlah data

Nilai RMSE rendah menunjukkan bahwa variasi nilai yang dihasilkan oleh suatu model prakiraan mendekati variasi nilai obeservasinya. RMSE menghitung seberapa bedanya seperangkat nilai. Semakin kecil nilai RMSE, semakin dekat nilai yang diprediksi dan diamati. 

Visualisasi evaluasi metrik menggunakan RMSE setelah pelatihan untuk model  _Collaborative Filtering_

![nilai](https://raw.githubusercontent.com/glimpseofnaomi/image/main/metrik.png)

Grafik menunjukkan perkembangan RMSE pada data latih dan validasi selama proses pelatihan model. Terlihat bahwa nilai RMSE untuk kedua dataset terus menurun seiring bertambahnya jumlah epoch dengan nilai error akhir sebesar sekitar 0.2136 dan error pada data validasi sebesar 0.2163. Hal ini menandakan bahwa model semakin baik dalam memprediksi rating, baik pada data pelatihan maupun data yang tidak terlihat sebelumnya (validasi). Selain itu, jarak antara kurva latih dan validasi cukup kecil dan konsisten, menunjukkan bahwa model tidak mengalami _overfitting_. Dengan kata lain, model berhasil belajar dengan baik tanpa menghafal data pelatihan secara berlebihan.

## Kesimpulan

Proyek ini berhasil membangun sistem rekomendasi buku yang efektif dengan menggabungkan pendekatan _Content Based Filtering_ dan  _Collaborative Filtering_. Pada model _Content Based Filtering_, sistem mampu memberikan rekomendasi yang relevan berdasarkan kemiripan penulis, dengan nilai precision mencapai 80%. Ini menunjukkan bahwa 4 dari 5 buku yang direkomendasikan benar-benar sesuai dengan preferensi pengguna.

Untuk model _Collaborative Filtering_, performa dievaluasi menggunakan _Root Mean Squared Error_ (RMSE) dimana nilai error akhir sebesar sekitar 0.2136 dan error pada data validasi sebesar 0.2163 . Hal ini menandakan bahwa model semakin akurat dalam memprediksi rating pengguna tanpa mengalami _overfitting_.

Secara keseluruhan, hasil evaluasi membuktikan bahwa sistem rekomendasi yang dikembangkan dalam proyek ini mampu bekerja secara optimal dan memberikan rekomendasi yang akurat serta relevan.

**Referensi:**

Lops, P., de Gemmis, M., & Semeraro, G. (2011). _Content-based recommender systems: State of the art and trends_. In Recommender Systems Handbook (pp. 73–105). Springer.

Ricci, F., Rokach, L., & Shapira, B. (2015). _Recommender Systems: Introduction and Challenges_. In Recommender Systems Handbook (pp. 1–34). Springer.