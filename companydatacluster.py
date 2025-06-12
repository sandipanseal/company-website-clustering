import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from sentence_transformers import SentenceTransformer
from umap.umap_ import UMAP

# Paths to your data
data_dir = "./data"
output_dir = "./output"
EXTRACTED_JSON = os.path.join(data_dir, "extracted.json")
KODES_JSON     = os.path.join(data_dir, "kodes.json")
CACHE_DIR      = os.path.join(data_dir, "website_cache")

# 1. Load JSONs
with open(EXTRACTED_JSON, "r", encoding="utf-8") as f:
    raw_companies = json.load(f)
with open(KODES_JSON, "r", encoding="utf-8") as f:
    kode_map = json.load(f)

# 2. Map sector codes
code2desc = {code: entry.get("description", "") for code, entry in kode_map.items()}

# 3. Sample data
target = raw_companies[:200]
texts, ids, sectors = [], [], []
for rec in target:
    cid = rec.get("crefonummer")
    zweck = rec.get("zweck", "")
    desc = rec.get("desc", "")
    docs = [zweck, desc]
    folder = os.path.join(CACHE_DIR, cid)
    for hf in glob.glob(os.path.join(folder, "*.html")):
        with open(hf, "r", encoding="utf-8", errors="ignore") as fh:
            soup = BeautifulSoup(fh, "html.parser")
            docs.append(soup.get_text(" ", strip=True))
    full = " ".join(d for d in docs if d.strip())
    texts.append(full)
    ids.append(cid)
    sectors.append(code2desc.get(rec.get("code", ""), rec.get("code", "")))

# 4. Feature extraction
# 4.1 TF–IDF
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), token_pattern=r"(?u)\b\w+\b")
X_tfidf = tfidf.fit_transform(texts).toarray()
# 4.2 BERT embeddings
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
X_bert = model.encode(texts, batch_size=32, show_progress_bar=True)
# 4.3 One-hot sectors
enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_sector = enc.fit_transform(np.array(sectors).reshape(-1,1))

# 5. Combine features
X = np.hstack((X_tfidf, X_bert, X_sector))

# 6. PCA for DBSCAN/dendrogram
pca50 = PCA(n_components=50, random_state=42)
X50 = pca50.fit_transform(X)

# 7. Clustering
results = {}
# K-Means
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
results['kmeans'] = kmeans.fit_predict(X)
# DBSCAN
db = DBSCAN(eps=2.0, min_samples=5, n_jobs=-1)
results['dbscan'] = db.fit_predict(X50)
# Hierarchical (Ward)
hier = AgglomerativeClustering(n_clusters=6, linkage='ward')
results['hierarchical'] = hier.fit_predict(X50)

# 8. Evaluation and purity
metrics = {}
for method, labels in results.items():
    feat = X50 if method in ['dbscan','hierarchical'] else X
    mask = labels != -1 if method == 'dbscan' else np.ones_like(labels, dtype=bool)
    sil = silhouette_score(feat[mask], labels[mask]) if mask.sum() > 1 else -1
    dfm = pd.DataFrame({'sector': sectors, 'label': labels})
    valid = dfm[dfm.label != -1]
    purity = valid.groupby('label')['sector'] .agg(lambda x: x.value_counts().iloc[0] / len(x)) .mean()
    metrics[method] = {'silhouette': sil, 'purity': purity}

# 8.1 Noise count
noise_count = int((results['dbscan'] == -1).sum())

# 8.2 Sector composition per cluster for DBSCAN
df_db = pd.DataFrame({'sector': sectors, 'label': results['dbscan']})
comp_db = df_db[df_db.label != -1] .groupby(['label', 'sector']).size() .unstack(fill_value=0)

# 8.3 Print metrics and noise count
print(pd.DataFrame(metrics).T)
print(f"DBSCAN noise count: {noise_count}\n")

# 8.4 List-style sector composition per DBSCAN cluster
composition_lines = []
composition_lines.append("DBSCAN sector composition (per cluster):")
for cluster_id, row in comp_db.iterrows():
    components = [f"{sector} ({count})" for sector, count in row.items() if count > 0]
    components_str = ", ".join(components)
    line = f"- Cluster {cluster_id}: {components_str}"
    print(line)
    composition_lines.append(line)

# Write outputs to file
output_path = os.path.join(output_dir, "dbscan_results.txt")
with open(output_path, "w", encoding="utf-8") as outf:
    # write metrics table
    outf.write(pd.DataFrame(metrics).T.to_string())
    outf.write("\n")
    outf.write(f"DBSCAN noise count: {noise_count}")
    outf.write("\n")
    # write composition
    outf.write("".join(composition_lines))

# 9. Visualizations
# PCA 2D
pca2 = PCA(n_components=2, random_state=42)
X2 = pca2.fit_transform(X)
colors = ['gold','darkorange','crimson','magenta','cyan','teal']
for method, labels in results.items():
    plt.figure(figsize=(6,5))
    for cid, col in enumerate(colors):
        mask = labels == cid
        plt.scatter(X2[mask,0], X2[mask,1], c=col, label=f'{method} C{cid}', marker='x')
    if method == 'dbscan':
        noise_mask = labels == -1
        plt.scatter(X2[noise_mask,0], X2[noise_mask,1], c='lightgrey', label='noise', marker='.')
    plt.title(f'PCA 2D: {method}')
    plt.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.show()

# UMAP 2D for DBSCAN
umap2 = UMAP(n_components=2, random_state=42)
Xu = umap2.fit_transform(X)
plt.figure(figsize=(6,5))
for cid, col in enumerate(colors):
    mask = results['dbscan'] == cid
    plt.scatter(Xu[mask,0], Xu[mask,1], c=col, marker='x', label=f'C{cid}')
plt.scatter(Xu[results['dbscan'] == -1,0], Xu[results['dbscan'] == -1,1], c='lightgrey', marker='.', label='noise')
plt.title('UMAP 2D: DBSCAN')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.show()

# Dendrogram
dist_link = linkage(X50, method='ward')
plt.figure(figsize=(8,4))
dendrogram(dist_link, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()
