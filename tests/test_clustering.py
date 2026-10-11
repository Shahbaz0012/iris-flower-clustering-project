import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_iris_clustering import calculate_elbow_values, load_data, run_experiment


def test_iris_data_has_expected_shape():
    features, target, names = load_data()
    assert features.shape == (150, 4)
    assert target.shape == (150,)
    assert len(names) == 3


def test_elbow_values_decrease_with_more_clusters():
    features, _, _ = load_data()
    inertias = calculate_elbow_values(features, max_k=4, random_state=42)
    assert len(inertias) == 4
    assert all(first > second for first, second in zip(inertias, inertias[1:]))


def test_complete_clustering_experiment_returns_valid_metrics():
    _, components, clusters, metrics, comparison, inertias = run_experiment(k=3, max_k=5)
    assert components.shape == (150, 2)
    assert len(clusters) == 150
    assert set(np.unique(clusters)) == {0, 1, 2}
    assert 0.0 <= metrics["adjusted_rand_index"] <= 1.0
    assert comparison.to_numpy().sum() == 150
    assert len(inertias) == 5
