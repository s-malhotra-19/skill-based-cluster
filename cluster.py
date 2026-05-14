import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score, davies_bouldin_score

df = pd.read_csv("Resume/Resume.csv")

print(df.columns)
print(df.head(2))


resumes = df["Resume_str"].astype(str).tolist()


def preprocess(text):
    return text.lower()

resumes = [preprocess(r) for r in resumes]


custom_stopwords = list(ENGLISH_STOP_WORDS.union([
    "experience", "company", "team", "management",
    "customer", "work", "project", "skills",
    "city", "state", "resume", "knowledge"
]))

vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words=custom_stopwords
)

X = vectorizer.fit_transform(resumes)

print("Shape:", X.shape)


# ── Find best K using WCSS, Silhouette, and Davies-Bouldin ────────────────────
k_range = range(2, 9)
wcss      = []
sil_scores = []
db_scores  = []

print("\nEvaluating K values...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbls   = kmeans.fit_predict(X)
    wcss.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X, lbls))
    db_scores.append(davies_bouldin_score(X.toarray(), lbls))
    print(f"  K={k}  WCSS={kmeans.inertia_:.1f}  Sil={sil_scores[-1]:.4f}  DB={db_scores[-1]:.4f}")

# Best K = highest silhouette score (primary) + lowest DB index (tiebreaker)
best_k = list(k_range)[int(np.argmax(sil_scores))]
print(f"\nBest K = {best_k}  (Silhouette={max(sil_scores):.4f})")


# ── Plot elbow + silhouette + DB ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(k_range, wcss, "o-", color="#e8a87c", lw=2)
axes[0].set(title="Elbow — WCSS", xlabel="K", ylabel="Inertia")
axes[0].axvline(best_k, color="red", ls="--", alpha=0.6)
axes[0].grid(alpha=0.3)

axes[1].plot(k_range, sil_scores, "o-", color="#7eb8a4", lw=2)
axes[1].set(title="Silhouette Score", xlabel="K", ylabel="Score ↑")
axes[1].axvline(best_k, color="red", ls="--", alpha=0.6, label=f"Best K={best_k}")
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(k_range, db_scores, "o-", color="#8faec9", lw=2)
axes[2].set(title="Davies-Bouldin Index", xlabel="K", ylabel="Score ↓")
axes[2].axvline(best_k, color="red", ls="--", alpha=0.6)
axes[2].grid(alpha=0.3)

plt.suptitle("Cluster Evaluation Metrics", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("cluster_eval.png", dpi=130, bbox_inches="tight")
plt.show()
print("Saved: cluster_eval.png")


# ── Final KMeans with best K ──────────────────────────────────────────────────
model  = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = model.fit_predict(X)

unique, counts = np.unique(labels, return_counts=True)

print("\nCluster Results:")
for u, c in zip(unique, counts):
    print(f"Cluster {u}: {c} resumes")


pca    = PCA(n_components=2)
coords = pca.fit_transform(X.toarray())


feature_names = vectorizer.get_feature_names_out()

cluster_data = []

for cluster in set(labels):

    indices = np.where(labels == cluster)[0]

    cluster_vectors = X[indices]

    mean_vector = cluster_vectors.mean(axis=0)

    top_indices = np.argsort(mean_vector.A1)[-5:]

    keywords = []

    for i in reversed(top_indices):
        keywords.append({
            "word": feature_names[i],
            "score": float(mean_vector.A1[i])
        })

    if cluster == -1:
        label = "Noise / Unclassified"
    else:
        label = " / ".join([k["word"] for k in keywords[:3]])

    samples = []
    for idx in indices[:3]:
        samples.append({
            "category": str(df.iloc[idx]["Category"]),
            "snippet": resumes[idx][:300]
        })

    category_counts = (
        df.iloc[indices]["Category"]
        .value_counts()
        .head(5)
    )

    top_categories = []
    for cat, cnt in category_counts.items():   # fixed: was 'n', shadowed range var
        top_categories.append({
            "cat": str(cat),
            "n": int(cnt)
        })

    x_coords = coords[indices, 0].tolist()
    y_coords = coords[indices, 1].tolist()

    cluster_obj = {
        "id":             int(cluster),
        "label":          label,
        "count":          int(len(indices)),
        "keywords":       keywords,
        "top_categories": top_categories,
        "samples":        samples,
        "x":              x_coords,
        "y":              y_coords
    }

    cluster_data.append(cluster_obj)


result = {
    "total_resumes": int(len(resumes)),
    "n_clusters":    int(len(set(labels)) - (1 if -1 in labels else 0)),
    "noise_count":   int(np.sum(labels == -1)),
    "best_k":        best_k,
    "eval_metrics": {
        "k_values":     list(k_range),
        "wcss":         wcss,
        "silhouette":   [round(s, 4) for s in sil_scores],
        "davies_bouldin":[round(d, 4) for d in db_scores],
    },
    "clusters": cluster_data
}

with open("clusters.json", "w") as f:
    json.dump(result, f, indent=4)

print("\nclusters.json generated successfully!")