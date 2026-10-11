"""Cluster the Iris dataset with scaled K-Means and visualize it with PCA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_MODEL = Path("models/iris_kmeans_pipeline.joblib")
DEFAULT_METRICS = Path("reports/iris_clustering_metrics.json")
DEFAULT_COMPARISON = Path("reports/iris_cluster_vs_species.csv")
DEFAULT_ELBOW = Path("reports/figures/iris_elbow_curve.png")
DEFAULT_SCATTER = Path("reports/figures/iris_pca_clusters.png")


def load_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load feature data and true species labels from scikit-learn's Iris dataset."""
    iris = load_iris(as_frame=True)
    return iris.data, iris.target, list(iris.target_names)


def build_pipeline(k: int, random_state: int) -> Pipeline:
    """Create a scaling and K-Means pipeline."""
    if k < 2:
        raise ValueError("k must be at least 2 for clustering.")
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, random_state=random_state, n_init=20)),
        ]
    )


def calculate_elbow_values(features: pd.DataFrame, max_k: int, random_state: int) -> list[float]:
    """Return K-Means inertia values after scaling for K values from 1 through max_k."""
    if max_k < 1:
        raise ValueError("max_k must be at least 1.")
    scaled = StandardScaler().fit_transform(features)
    return [
        float(KMeans(n_clusters=k, random_state=random_state, n_init=20).fit(scaled).inertia_)
        for k in range(1, max_k + 1)
    ]


def run_experiment(k: int = 3, max_k: int = 10, random_state: int = 42):
    """Fit the project pipeline and calculate clustering data and evaluation metrics."""
    features, species, species_names = load_data()
    pipeline = build_pipeline(k, random_state)
    clusters = pipeline.fit_predict(features)
    scaled = pipeline.named_steps["scaler"].transform(features)
    pca = PCA(n_components=2, random_state=random_state)
    components = pca.fit_transform(scaled)
    metrics = {
        "k": k,
        "adjusted_rand_index": float(adjusted_rand_score(species, clusters)),
        "normalized_mutual_information": float(normalized_mutual_info_score(species, clusters)),
        "silhouette_score": float(silhouette_score(scaled, clusters)),
        "pca_explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
    }
    comparison = pd.crosstab(
        pd.Series(clusters, name="cluster"),
        pd.Series(species.map(dict(enumerate(species_names))), name="true_species"),
    )
    return pipeline, components, clusters, metrics, comparison, calculate_elbow_values(features, max_k, random_state)


def save_elbow_plot(inertias: list[float], output_path: Path) -> None:
    """Save an elbow chart showing K-Means inertia by K."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ks = range(1, len(inertias) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(ks, inertias, marker="o", color="teal")
    plt.xticks(list(ks))
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Iris K-Means Elbow Method")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_cluster_plot(components, clusters, output_path: Path) -> None:
    """Save a PCA scatter plot colored by K-Means cluster."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(components[:, 0], components[:, 1], c=clusters, cmap="viridis", s=45, alpha=0.8)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Iris Clusters in PCA Space")
    plt.colorbar(scatter, label="K-Means cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3, help="Number of clusters for the final K-Means model.")
    parser.add_argument("--max-k", type=int, default=10, help="Maximum K value shown in the elbow plot.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--comparison-output", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--elbow-output", type=Path, default=DEFAULT_ELBOW)
    parser.add_argument("--scatter-output", type=Path, default=DEFAULT_SCATTER)
    args = parser.parse_args()

    pipeline, components, clusters, metrics, comparison, inertias = run_experiment(args.k, args.max_k, args.random_state)
    for output in (args.model_output, args.metrics_output, args.comparison_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_output)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    comparison.to_csv(args.comparison_output)
    save_elbow_plot(inertias, args.elbow_output)
    save_cluster_plot(components, clusters, args.scatter_output)

    print(f"Adjusted Rand Index: {metrics['adjusted_rand_index']:.3f}")
    print(f"Normalized Mutual Information: {metrics['normalized_mutual_information']:.3f}")
    print(f"Silhouette score: {metrics['silhouette_score']:.3f}")
    print(f"Saved model: {args.model_output}")
    print(f"Saved figures: {args.elbow_output}, {args.scatter_output}")


if __name__ == "__main__":
    main()
