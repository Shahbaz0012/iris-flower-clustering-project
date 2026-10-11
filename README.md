# Iris Flower Clustering Project

An unsupervised machine-learning mini project using the Iris dataset. It demonstrates data scaling, K-Means clustering, elbow-method selection of K, PCA dimensionality reduction, model saving, and a comparison of discovered clusters with the dataset's known species labels.

## What this project does

- Loads the Iris dataset bundled with scikit-learn.
- Standardizes all four numeric flower measurements.
- Produces an elbow plot for K values from 1 through 10.
- Trains K-Means with `k=3`.
- Uses PCA to reduce the measurements to two dimensions for visualization.
- Saves a scatter plot, elbow plot, fitted pipeline, and metric reports.
- Reports Adjusted Rand Index and Normalized Mutual Information against the known species labels. These labels are used only for evaluation, not training.

## Project structure

```text
.
├── .github/workflows/test.yml       # GitHub Actions test workflow
├── models/                          # Generated model file
├── reports/figures/                 # Generated charts
├── src/train_iris_clustering.py     # Main experiment script
├── tests/test_clustering.py         # Basic workflow tests
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the project

```bash
python src/train_iris_clustering.py
```

This creates the following ignored local outputs:

- `models/iris_kmeans_pipeline.joblib`
- `reports/iris_clustering_metrics.json`
- `reports/iris_cluster_vs_species.csv`
- `reports/figures/iris_elbow_curve.png`
- `reports/figures/iris_pca_clusters.png`

## Options

```bash
python src/train_iris_clustering.py --k 3 --max-k 10 --random-state 42
```

Use `--help` for the full list of options.

## Evaluation notes

K-Means cluster numbers are arbitrary: cluster `0` does not inherently match a particular Iris species. Therefore, the project evaluates clustering with label-invariant metrics:

- **Adjusted Rand Index (ARI)** measures agreement between clusters and true species, corrected for chance.
- **Normalized Mutual Information (NMI)** measures shared information between the two label assignments.
- **Silhouette score** describes how well-separated the discovered clusters are using the feature data alone.

## Test

```bash
pytest
```

## License

Licensed under the MIT License. See [LICENSE](LICENSE).
