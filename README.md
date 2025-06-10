# 📚 LiberAI — A Book Recommender System

LiberAI is an AI-powered book recommendation system that suggests books based on genre, author, and description using content-based filtering. Built with `pandas`, `scikit-learn` (TF-IDF Vectorizer, TruncatedSVD, K-Means Clustering, and Cosine Similarity), and deployed via `Streamlit`.

## 🔗 Live Demo
🌐 [Try the App Here](https://liberai.streamlit.app)

---

## 🧠 How It Works

The recommendation system follows these main steps:

1. **Data Preparation**: 
   - Cleans and filters raw book metadata.
   - Handles missing values and duplicates.
   - Extracts key features such as `genres`, `authors`, and `title`.

2. **Modeling**: 
   - Converts textual features into vectors using **TF-IDF**.
   - Reduces dimensionality with **TruncatedSVD**.
   - Clusters books into groups using **KMeans** to enhance similarity precision.
   - Computes book similarity using **cosine similarity** within clusters.

3. **Validation & Evaluation**:
   - Uses manual validation (by checking top recommendations).
   - Visual evaluation with clustering plots and similarity analysis.

4. **Frontend**:
   - Built with `Streamlit` to allow users to search for a book and get smart recommendations in real-time.
   - Users can input 3 book titles to receive personalized suggestions.
   - Users can input a genre to see all books within that genre.

---

## 🗂️ Project Structure
```
├── app.py # Streamlit app frontend
├── main.ipynb # Model training & development notebook
├── books_1.Best_Books_Ever.csv # Raw dataset
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

---

## ▶️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/ceciliasx/liberai.git
cd liberai
```

### 2. Create Environment and Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Run the App Locally
```bash
streamlit run app.py
```

---

## 🧪 Model Notebook (main.ipynb)
Explore the full model development process in main.ipynb. Key steps include:
1. Data cleaning & preprocessing
2. TF-IDF vectorization of genres, authors, and descriptions
3. Dimensionality reduction with TruncatedSVD
4. Clustering books with KMeans
5. Calculating cosine similarity for recommendations

---

## 💡 Features
1. Recommend similar books by title input
2. Content-based filtering (no user ratings needed)
3. Lightweight and fast via Streamlit interface
4. Uses clustering to group and improve recommendations
5. View all books within a specific genre

---

### 📌 Dependencies
1. pandas
2. numpy
3. scikit-learn
4. matplotlib
5. seaborn
6. scipy
7. streamlit

Install all dependencies with:
```bash
pip install -r requirements.txt
```
