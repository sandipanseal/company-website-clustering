import json
import os
import glob
import re
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from sentence_transformers import SentenceTransformer
from umap.umap_ import UMAP

# Ensure NLTK German stopwords are available
nltk.download('stopwords')
de_stopwords = stopwords.words('german')

# --- Configuration ---
data_dir       = "./data"
output_dir     = "./output"
EXTRACTED_JSON = os.path.join(data_dir, "extracted.json")
KODES_JSON     = os.path.join(data_dir, "kodes.json")
CACHE_DIR      = os.path.join(data_dir, "website_cache")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# --- Helper Function ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zäöüß\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- 1. Load Data ---
with open(EXTRACTED_JSON, "r", encoding="utf-8") as f:
    raw_companies = json.load(f)
with open(KODES_JSON, "r", encoding="utf-8") as f:
    kode_map = json.load(f)

# --- 2. Map Sector Codes ---
code2desc = {code: entry.get("description", "") for code, entry in kode_map.items()}

# --- 3. Sample and Extract Text ---
target = raw_companies[:200]
texts, ids, sectors = [], [], []
for rec in target:
    cid   = rec.get("crefonummer")
    zweck = rec.get("zweck", "")
    desc  = rec.get("desc", "")
    docs  = [zweck, desc]
    folder = os.path.join(CACHE_DIR, cid)
    for hf in glob.glob(os.path.join(folder, "*.html")):
        with open(hf, "r", encoding="utf-8", errors="ignore") as fh:
            soup = BeautifulSoup(fh, "html.parser")
            docs.append(soup.get_text(" ", strip=True))
    full = " ".join(d for d in docs if d.strip())
    texts.append(preprocess_text(full))
    ids.append(cid)
    sectors.append(code2desc.get(rec.get("code",""), rec.get("code","")))

# --- 4. Feature Extraction ---
# 4.1 TF–IDF
tfidf_vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words=de_stopwords)
X_tfidf   = tfidf_vec.fit_transform(texts).toarray()
# 4.2 BERT Embeddings
bert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
X_bert    = bert_model.encode(texts, batch_size=32, show_progress_bar=True)

# Combine features (no one-hot sectors)
X = np.hstack((X_tfidf, X_bert))

# --- 5. PCA Reduction ---
pca50 = PCA(n_components=50, random_state=42)
X50   = pca50.fit_transform(X)

# --- 6. Clustering ---
results = {}
results['kmeans']      = KMeans(n_clusters=6, random_state=42, n_init=10).fit_predict(X)
results['dbscan']      = DBSCAN(eps=2.0, min_samples=5, n_jobs=-1).fit_predict(X50)
results['hierarchical'] = AgglomerativeClustering(n_clusters=6, linkage='ward').fit_predict(X50)

# --- 7. Evaluation (Silhouette & Purity) ---
metrics = {}
for method, labels in results.items():
    feat = X50 if method in ['dbscan','hierarchical'] else X
    uniq = np.unique(labels[labels!=-1] if method=='dbscan' else labels)
    if len(uniq) > 1:
        mask = labels != -1 if method=='dbscan' else np.ones_like(labels, dtype=bool)
        sil = silhouette_score(feat[mask], labels[mask])
    else:
        sil = -1
    dfm = pd.DataFrame({'sector': sectors, 'label': labels})
    valid = dfm[dfm.label != -1]
    purity = valid.groupby('label')['sector'] \
                  .apply(lambda x: x.value_counts().iloc[0] / len(x)) \
                  .mean()
    metrics[method] = {'silhouette': sil, 'purity': purity}
pd.DataFrame(metrics).T.to_csv(os.path.join(output_dir, 'metrics.csv'))

# --- Noise count ---
db_lbls = results['dbscan']
noise_count = int((db_lbls == -1).sum())
with open(os.path.join(output_dir, 'noise_count.txt'), 'w') as f:
    f.write(f"DBSCAN noise count: {noise_count}\n")

# --- 8. DBSCAN Sensitivity ---
eps_vals = [0.5,1.0,1.5,2.0,2.5]
ms_vals  = [3,5,7,10]
rows = []
for eps, ms in itertools.product(eps_vals, ms_vals):
    lbl = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1).fit_predict(X50)
    uniq = np.unique(lbl[lbl!=-1])
    sil = silhouette_score(X50[lbl!=-1], lbl[lbl!=-1]) if len(uniq)>1 else -1
    noise_pct = (lbl == -1).mean()
    rows.append({'eps':eps, 'min_samples':ms, 'silhouette':sil, 'noise_pct':noise_pct})
sens_df = pd.DataFrame(rows)
sens_df.to_csv(os.path.join(output_dir, 'dbscan_sensitivity.csv'), index=False)

# --- 9. Best DBSCAN Composition ---
best = sens_df.sort_values('silhouette', ascending=False).iloc[0]
best_lbls = DBSCAN(eps=best.eps, min_samples=int(best.min_samples), n_jobs=-1).fit_predict(X50)
df_db  = pd.DataFrame({'sector': sectors, 'label': best_lbls})
comp_db = df_db[df_db.label!=-1].groupby(['label','sector']).size().unstack(fill_value=0)
comp_db.to_csv(os.path.join(output_dir, 'dbscan_best_composition.csv'))

# --- 10. Visualizations ---
pca2 = PCA(n_components=2, random_state=42)
X2   = pca2.fit_transform(X)
for method, labels in results.items():
    plt.figure(figsize=(6,5))
    for cid in np.unique(labels):
        mask = labels==cid
        color = f'C{cid}' if cid!=-1 else 'lightgrey'
        plt.scatter(X2[mask,0], X2[mask,1], c=color, label=f"{method} {'noise' if cid==-1 else f'C{cid}'}", s=10)
    plt.title(f"PCA 2D: {method}")
    plt.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'pca2d_{method}.png'))
    plt.show()
    plt.close()

# UMAP 2D: show all DBSCAN clusters in distinct colors
umap2 = UMAP(n_components=2, random_state=42)
Xu    = umap2.fit_transform(X)
plt.figure(figsize=(6,5))
unique = np.unique(db_lbls)
for idx, cid in enumerate(unique):
    mask = db_lbls==cid
    color = f'C{idx}' if cid!=-1 else 'lightgrey'
    label = 'noise' if cid==-1 else f'C{cid}'
    plt.scatter(Xu[mask,0], Xu[mask,1], c=color, s=10, label=label)
plt.title('UMAP 2D: DBSCAN')
plt.legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'umap2d_dbscan.png'))
plt.show()
plt.close()

# --- 11. Dendrogram ---
dist_link = linkage(X50, method='ward')
plt.figure(figsize=(8,4))
dendrogram(dist_link, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index or Cluster Size')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'dendrogram.png'))
plt.show()
plt.close()
