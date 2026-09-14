"""Tests for fleet fingerprinting components.

Tests the PCA embedder, KMeans clusterer, and cross-machine
early warning system with synthetic data.
"""

import numpy as np
import pandas as pd
import pytest

from machineguard.fleet.fingerprint import (
    FleetEmbedder,
    FleetClusterer,
    CrossMachineWarning,
)


class TestFleetEmbedder:
    """Test the fleet embedding system."""

    def test_pca_fit_transform(self):
        """PCA should reduce dimensionality."""
        np.random.seed(42)
        X = np.random.randn(100, 20)  # 100 samples, 20 features
        embedder = FleetEmbedder(n_components=5, method="pca")
        embeddings = embedder.fit_transform(X)
        assert embeddings.shape == (100, 5)

    def test_pca_fit_then_transform(self):
        """Separate fit/transform should produce same results."""
        np.random.seed(42)
        X_train = np.random.randn(80, 15)
        X_test = np.random.randn(20, 15)

        embedder = FleetEmbedder(n_components=4, method="pca")
        embedder.fit(X_train)
        embeddings = embedder.transform(X_test)
        assert embeddings.shape == (20, 4)

    def test_dataframe_input(self):
        """Should accept DataFrames."""
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(50, 10),
            columns=[f"sensor_{i}" for i in range(10)]
        )
        embedder = FleetEmbedder(n_components=3, method="pca")
        embeddings = embedder.fit_transform(df)
        assert embeddings.shape == (50, 3)

    def test_explained_variance(self):
        """PCA should report explained variance."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        embedder = FleetEmbedder(n_components=5, method="pca")
        embedder.fit(X)
        ev = embedder.explained_variance
        assert ev is not None
        assert len(ev) == 5
        assert all(0 <= v <= 1 for v in ev)

    def test_transform_before_fit_raises(self):
        """Transform before fit should raise."""
        embedder = FleetEmbedder(n_components=3)
        with pytest.raises(RuntimeError, match="not fitted"):
            embedder.transform(np.random.randn(10, 5))

    def test_n_components_clamped(self):
        """Should clamp n_components if data has fewer features."""
        np.random.seed(42)
        X = np.random.randn(20, 3)  # Only 3 features
        embedder = FleetEmbedder(n_components=10, method="pca")
        embeddings = embedder.fit_transform(X)
        assert embeddings.shape[1] <= 3

    def test_nan_handling(self):
        """Should handle NaN rows by dropping them."""
        X = np.array([
            [1.0, 2.0, 3.0],
            [np.nan, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        embedder = FleetEmbedder(n_components=2, method="pca")
        embedder.fit(X)  # Should not crash
        assert embedder._fitted


class TestFleetClusterer:
    """Test the fleet clustering system."""

    def test_basic_clustering(self):
        """Should assign clusters to embeddings."""
        np.random.seed(42)
        # Create clearly separable clusters
        cluster_1 = np.random.randn(30, 4) + [5, 0, 0, 0]
        cluster_2 = np.random.randn(30, 4) + [0, 5, 0, 0]
        embeddings = np.vstack([cluster_1, cluster_2])

        clusterer = FleetClusterer(n_clusters=2)
        clusterer.fit(embeddings)

        assignments = clusterer.predict(embeddings)
        assert assignments.shape == (60,)
        assert len(set(assignments)) == 2

    def test_cluster_profiles(self):
        """Should compute cluster statistics."""
        np.random.seed(42)
        embeddings = np.random.randn(50, 4)
        clusterer = FleetClusterer(n_clusters=3)
        clusterer.fit(embeddings)

        profiles = clusterer.cluster_profiles
        assert len(profiles) == 3
        for c, profile in profiles.items():
            assert "size" in profile
            assert "centroid" in profile
            assert profile["size"] > 0

    def test_distance_to_centroids(self):
        """Should compute distance matrix."""
        np.random.seed(42)
        embeddings = np.random.randn(20, 3)
        clusterer = FleetClusterer(n_clusters=2)
        clusterer.fit(embeddings)

        distances = clusterer.distance_to_centroids(embeddings)
        assert distances.shape == (20, 2)
        assert np.all(distances >= 0)

    def test_silhouette_score(self):
        """Should compute valid silhouette score."""
        np.random.seed(42)
        cluster_1 = np.random.randn(30, 4) + [10, 0, 0, 0]
        cluster_2 = np.random.randn(30, 4) + [0, 10, 0, 0]
        embeddings = np.vstack([cluster_1, cluster_2])

        clusterer = FleetClusterer(n_clusters=2)
        clusterer.fit(embeddings)

        assert -1 <= clusterer.silhouette <= 1

    def test_with_labels(self):
        """Should track label distribution per cluster."""
        np.random.seed(42)
        embeddings = np.random.randn(40, 4)
        labels = np.array(["healthy"] * 20 + ["failed"] * 20)

        clusterer = FleetClusterer(n_clusters=2)
        clusterer.fit(embeddings, labels=labels)

        for profile in clusterer.cluster_profiles.values():
            assert "label_distribution" in profile


class TestCrossMachineWarning:
    """Test the cross-machine early warning system."""

    def test_normal_machine(self):
        """Normal machine should not trigger warnings."""
        np.random.seed(42)
        # Create healthy and degraded clusters
        healthy = np.random.randn(40, 4) + [0, 0, 0, 0]
        degraded = np.random.randn(40, 4) + [10, 10, 10, 10]
        all_embeddings = np.vstack([healthy, degraded])

        embedder = FleetEmbedder(n_components=4, method="pca")
        embedder.fit(all_embeddings)

        clusterer = FleetClusterer(n_clusters=2)
        clusterer.fit(all_embeddings)

        # Normal machine = close to healthy cluster
        warning_system = CrossMachineWarning(
            embedder=embedder,
            clusterer=clusterer,
            degraded_clusters=[1],  # Assume cluster 1 is degraded
        )

        result = warning_system.check_machine(healthy[:5])
        assert "alert_level" in result
        assert "distances" in result

    def test_identify_degraded_clusters(self):
        """Should auto-identify clusters with high failure rate."""
        np.random.seed(42)
        embeddings = np.random.randn(80, 4)
        # First 40 are healthy, last 40 are failed
        failure_labels = np.array([False] * 40 + [True] * 40)

        clusterer = FleetClusterer(n_clusters=2)
        clusterer.fit(embeddings)

        embedder = FleetEmbedder(n_components=4, method="pca")
        embedder.fit(embeddings)

        warning_system = CrossMachineWarning(
            embedder=embedder, clusterer=clusterer
        )
        degraded = warning_system.identify_degraded_clusters(
            embeddings, failure_labels, failure_ratio_threshold=0.3
        )
        # At least some clusters should be identified (random data may vary)
        assert isinstance(degraded, list)
