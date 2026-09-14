"""Fleet fingerprinting — embed machines into a shared latent space.

Maps sensor trajectories to fixed-size embeddings using either:
1. PCA for lightweight, interpretable embeddings
2. Autoencoder for nonlinear, capacity-rich embeddings

Once embedded, machines can be clustered and compared to detect
when a "healthy" machine starts drifting toward a cluster of
failed machines — the cross-machine early warning system.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from machineguard.config import MODELS_DIR

logger = logging.getLogger(__name__)


class FleetEmbedder:
    """Maps machine sensor data to fixed-size embeddings.

    Two modes:
    - PCA: lightweight, interpretable, good baseline
    - Autoencoder: learns nonlinear manifold structure (future extension)

    The embedder is fleet-agnostic: it normalizes different sensor schemas
    into a common feature space before embedding.
    """

    def __init__(self, n_components: int = 8, method: str = "pca"):
        """Initialize the embedder.

        Args:
            n_components: Embedding dimensionality.
            method: "pca" or "autoencoder".
        """
        self.n_components = n_components
        self.method = method
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components) if method == "pca" else None
        self._fitted = False
        self._feature_names: list[str] = []

    def fit(self, feature_matrix: pd.DataFrame | np.ndarray) -> "FleetEmbedder":
        """Fit the embedder on historical sensor data.

        Args:
            feature_matrix: (n_samples, n_features) array of sensor readings.

        Returns:
            Self for chaining.
        """
        if isinstance(feature_matrix, pd.DataFrame):
            self._feature_names = feature_matrix.columns.tolist()
            X = feature_matrix.values
        else:
            X = feature_matrix

        # Drop any NaN rows
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]

        # Fit scaler and reducer
        X_scaled = self.scaler.fit_transform(X)

        if self.method == "pca" and self.pca is not None:
            # Clamp n_components to available features
            actual_components = min(self.n_components, X_scaled.shape[1], X_scaled.shape[0])
            if actual_components < self.n_components:
                logger.warning(
                    f"Reducing n_components from {self.n_components} to {actual_components} "
                    f"(data shape: {X_scaled.shape})"
                )
                self.pca = PCA(n_components=actual_components)
                self.n_components = actual_components
            self.pca.fit(X_scaled)
            explained = sum(self.pca.explained_variance_ratio_) * 100
            logger.info(
                f"PCA fitted: {actual_components} components, "
                f"{explained:.1f}% variance explained"
            )

        self._fitted = True
        return self

    def transform(self, feature_matrix: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform sensor data into embeddings.

        Args:
            feature_matrix: (n_samples, n_features) array.

        Returns:
            (n_samples, n_components) embedding array.
        """
        if not self._fitted:
            raise RuntimeError("Embedder not fitted. Call fit() first.")

        if isinstance(feature_matrix, pd.DataFrame):
            X = feature_matrix.values
        else:
            X = feature_matrix

        X_scaled = self.scaler.transform(X)

        if self.method == "pca" and self.pca is not None:
            return self.pca.transform(X_scaled)

        return X_scaled[:, :self.n_components]

    def fit_transform(self, feature_matrix: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(feature_matrix).transform(feature_matrix)

    @property
    def explained_variance(self) -> list[float] | None:
        """Get explained variance ratios (PCA only)."""
        if self.pca is not None and self._fitted:
            return self.pca.explained_variance_ratio_.tolist()
        return None


class FleetClusterer:
    """Clusters machine embeddings to find operational regimes.

    Groups machines by similarity in the embedding space, enabling:
    - Identification of normal vs degraded operational clusters
    - Cross-machine early warning when a machine drifts toward a
      degraded cluster
    """

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )
        self._fitted = False
        self._cluster_profiles: dict[int, dict] = {}

    def fit(self, embeddings: np.ndarray, labels: np.ndarray | None = None) -> "FleetClusterer":
        """Fit clusters on machine embeddings.

        Args:
            embeddings: (n_samples, n_components) from FleetEmbedder.
            labels: Optional known labels (e.g., failure/healthy) for
                    cluster interpretation.

        Returns:
            Self for chaining.
        """
        self.kmeans.fit(embeddings)
        cluster_assignments = self.kmeans.labels_
        self._fitted = True

        # Compute cluster profiles
        for c in range(self.n_clusters):
            mask = cluster_assignments == c
            cluster_embeddings = embeddings[mask]

            profile: dict[str, Any] = {
                "size": int(mask.sum()),
                "centroid": self.kmeans.cluster_centers_[c].tolist(),
                "mean_distance_to_centroid": float(
                    np.mean(np.linalg.norm(
                        cluster_embeddings - self.kmeans.cluster_centers_[c], axis=1
                    ))
                ) if mask.sum() > 0 else 0.0,
            }

            # If we have labels, compute label distribution per cluster
            if labels is not None:
                cluster_labels = labels[mask]
                unique, counts = np.unique(cluster_labels, return_counts=True)
                profile["label_distribution"] = {
                    str(u): int(c) for u, c in zip(unique, counts)
                }

            self._cluster_profiles[c] = profile

        # Silhouette score (quality measure)
        if embeddings.shape[0] > self.n_clusters:
            self._silhouette = float(silhouette_score(embeddings, cluster_assignments))
            logger.info(f"Clustering fitted: {self.n_clusters} clusters, silhouette={self._silhouette:.3f}")
        else:
            self._silhouette = 0.0

        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Assign new embeddings to clusters."""
        if not self._fitted:
            raise RuntimeError("Clusterer not fitted. Call fit() first.")
        return self.kmeans.predict(embeddings)

    def distance_to_centroids(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute distance from each embedding to all cluster centroids.

        Returns:
            (n_samples, n_clusters) distance matrix.
        """
        centroids = self.kmeans.cluster_centers_
        distances = np.zeros((embeddings.shape[0], self.n_clusters))
        for c in range(self.n_clusters):
            distances[:, c] = np.linalg.norm(embeddings - centroids[c], axis=1)
        return distances

    @property
    def cluster_profiles(self) -> dict[int, dict]:
        return self._cluster_profiles

    @property
    def silhouette(self) -> float:
        return getattr(self, "_silhouette", 0.0)


class CrossMachineWarning:
    """Cross-machine early warning system.

    When a machine's embedding drifts toward a cluster dominated by
    failed machines, generate an early warning alert.
    """

    def __init__(
        self,
        embedder: FleetEmbedder,
        clusterer: FleetClusterer,
        degraded_clusters: list[int] | None = None,
        drift_threshold: float = 0.3,
    ):
        self.embedder = embedder
        self.clusterer = clusterer
        self.degraded_clusters = degraded_clusters or []
        self.drift_threshold = drift_threshold

    def check_machine(
        self,
        sensor_readings: pd.DataFrame | np.ndarray,
    ) -> dict:
        """Check if a machine is drifting toward degraded clusters.

        Args:
            sensor_readings: Recent sensor data for the machine.

        Returns:
            Warning dict with drift assessment.
        """
        # Embed
        embedding = self.embedder.transform(sensor_readings)
        if embedding.ndim == 2:
            embedding = embedding.mean(axis=0, keepdims=True)

        # Get distances to all clusters
        distances = self.clusterer.distance_to_centroids(embedding)[0]
        assigned_cluster = int(self.clusterer.predict(embedding)[0])

        # Compute drift toward degraded clusters
        warnings = []
        for deg_cluster in self.degraded_clusters:
            distance_to_degraded = distances[deg_cluster]
            min_distance = distances.min()

            # Relative proximity to degraded cluster
            if min_distance > 0:
                relative_proximity = min_distance / max(distance_to_degraded, 1e-8)
            else:
                relative_proximity = 1.0

            if relative_proximity > self.drift_threshold:
                warnings.append({
                    "degraded_cluster": deg_cluster,
                    "distance": float(distance_to_degraded),
                    "relative_proximity": float(relative_proximity),
                    "severity": "high" if relative_proximity > 0.7 else "medium",
                })

        return {
            "assigned_cluster": assigned_cluster,
            "distances": distances.tolist(),
            "is_in_degraded_cluster": assigned_cluster in self.degraded_clusters,
            "drift_warnings": warnings,
            "alert_level": (
                "critical" if assigned_cluster in self.degraded_clusters
                else "warning" if len(warnings) > 0
                else "normal"
            ),
        }

    def identify_degraded_clusters(
        self,
        embeddings: np.ndarray,
        failure_labels: np.ndarray,
        failure_ratio_threshold: float = 0.3,
    ) -> list[int]:
        """Auto-identify which clusters are dominated by failed machines.

        Args:
            embeddings: All machine embeddings.
            failure_labels: Boolean array (True = failed).
            failure_ratio_threshold: Minimum failure ratio to mark as degraded.

        Returns:
            List of cluster indices dominated by failures.
        """
        assignments = self.clusterer.predict(embeddings)
        degraded = []

        for c in range(self.clusterer.n_clusters):
            mask = assignments == c
            if mask.sum() == 0:
                continue
            failure_ratio = failure_labels[mask].mean()
            if failure_ratio > failure_ratio_threshold:
                degraded.append(c)
                logger.info(
                    f"Cluster {c}: {failure_ratio:.1%} failure rate "
                    f"({mask.sum()} machines) → DEGRADED"
                )

        self.degraded_clusters = degraded
        return degraded
