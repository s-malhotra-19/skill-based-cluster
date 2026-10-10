# 🎯 SkillCluster AI: Skill-Based Resume Clustering & Talent Intelligence

An enterprise-ready, unsupervised machine learning system that analyzes, clusters, and visualizes candidate skill archetypes across 2,480+ professional resumes. Built with **TF-IDF**, **K-Means**, **PCA**, and a modern **Streamlit** dashboard.

---

## 🌟 Key Features

1. **📊 2D Interactive Cluster Map & Analytics:**
   - Visualizes resume groupings projected into 2D space via Principal Component Analysis (PCA).
   - Dynamic exploration of 8 mathematically validated skill archetypes with top TF-IDF keywords and category distributions.

2. **🔍 Real-Time Resume Skill Matcher & Gap Analysis:**
   - Paste or upload candidate resumes (`.pdf`, `.txt`) to calculate cosine similarity against cluster centroids.
   - Provides confidence scores, detected skill tokens, and targeted skill gap recommendations to improve ATS visibility.

3. **📁 Batch Resume Cohort Triage (CSV):**
   - Upload bulk candidate CSV files for instant multi-profile categorization.
   - Generates cohort distribution visualizations and exports categorized datasets with cluster IDs and confidence scores.

4. **📐 Rigorous Model Evaluation & Diagnostics:**
   - Multi-metric validation sweeping $K \in [2, 8]$ using **WCSS (Inertia)**, **Silhouette Score**, and **Davies-Bouldin Index**.
   - Proves why $K=8$ establishes the globally optimal balance of intra-cluster cohesion and inter-cluster separation.

---

## 🏗️ Architecture & Pipeline

```mermaid
graph LR
    A[Raw Resume Corpus (2,484 Docs)] --> B[Text Preprocessing & Domain Stopwords]
    B --> C[TF-IDF Vectorization (1,000 Features)]
    C --> D[K-Means Cluster Optimization (K=8)]
    C --> E[PCA 2D Spatial Projection]
    D --> F[Cluster Centroid Keyword Extraction]
    E --> G[Interactive Streamlit Dashboard]
    F --> G
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/s-malhotra-19/skill-based-cluster.git
cd skill-based-cluster
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application
```bash
streamlit run app.py
```

---

## ☁️ Streamlit Community Cloud Deployment

- **Repository:** `s-malhotra-19/skill-based-cluster`
- **Branch:** `main`
- **Main file path:** `app.py`
- **Python version:** `3.11` (or `3.10` / `3.12`)

All cluster models and metadata are self-contained in `clusters.json` (~214 KB), enabling sub-second cold starts and minimal memory consumption on Streamlit Community Cloud.
