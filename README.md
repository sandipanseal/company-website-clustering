# Website Data Clustering

This repository contains code for clustering companies based on website content, using TF–IDF and BERT features with PCA and various clustering algorithms.

## Features

- **HTML Preprocessing**: Extracts titles and paragraphs from company websites.  
- **Feature Extraction**:  
  - TF–IDF vectors (5,000-term vocabulary)  
  - Multilingual MiniLM BERT embeddings (384 dimensions)  
- **Dimensionality Reduction**: PCA to 50 components for clustering; PCA or UMAP for 2D visualization.  
- **Clustering Pipelines**:  
  - K-means  
  - DBSCAN (grid search over ε and min_samples)  
  - Agglomerative (Ward linkage)  
- **Evaluation & Outputs**:  
  - Silhouette scores, purity, noise statistics  
  - Sector-wise composition tables  
  - Visualization plots (dendrograms, PCA/UMAP projections)
  
## Requirements

Company data: extracted.json (provided by the University)

Business sector codes: kodes.json (provided by the University)

Website cache: directory of cached HTML files (provided by the University)

##Data & Outputs
Sector codes: German WZ 2008 classifications in kodes.json.

Metrics CSV: silhouette, purity, noise counts per method.

Composition CSVs: sector-wise breakdown for each clustering run.

Figures: dendrograms, PCA/UMAP scatterplots in the output folder.

## Running the Code

Simply execute:

```bash
python companydatacluster.py



