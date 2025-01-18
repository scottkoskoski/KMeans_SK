import numpy as np
import pandas as pd
import random

class KMeans:
    def __init__(self, K=5, max_iterations=100):
        if K <= 0:
            raise ValueError("Number of clusters (K) must be a positive integer.")
        if max_iterations <= 0:
            raise ValueError("Max iterations must be a positive integer.")
        
        self._K = K
        self._max_iterations = max_iterations
    
    @property
    def K(self):
        return self._K
    
    @K.setter
    def K(self, K):
        if K <= 0:
            raise ValueError("Number of clusters (K) must be a positive integer.")
        self._K = K
    
    @property
    def max_iterations(self):
        return self._max_iterations
    
    @max_iterations.setter
    def max_iterations(self, max_iterations):
        if max_iterations <= 0:
            raise ValueError("Max iterations must be a positive integer.")
        self._max_iterations = max_iterations

    def get_initial_centroids(self, data):
        if data.empty:
            raise ValueError("Input data is empty. Please provide a non-empty dataset.")
        if data.shape[0] < self._K:
            raise ValueError(f"Number of data points ({data.shape[0]}) must be greater than or equal to K ({self._K}).")

        rows = data.shape[0]
        centroids = {}
        for i in range(self._K):
            idx = random.randint(0, rows - 1)
            centroid = data.iloc[idx]
            centroids[i] = centroid
        return pd.DataFrame(centroids)
    
    def assign_clusters(self, data, centroids):
        if centroids.empty:
            raise ValueError("Centroids are empty. Ensure initial centroids are properly initialized.")
        dist = centroids.apply(lambda x: np.sqrt((data - x) ** 2).sum(axis=1))
        return dist.idxmin(axis=1)
    
    def get_cluster_means(self, data, clusterlabels):
        if data.empty:
            raise ValueError("Data is empty. Cannot calculate cluster means.")
        if clusterlabels.empty:
            raise ValueError("Cluster labels are empty. Ensure clustering assignments are valid.")
        return data.groupby(clusterlabels).apply(lambda x: np.exp(np.log(x).mean())).T
    
    def scale_data(self, data, scalar=1, offset=0):
        if data.empty:
            raise ValueError("Data is empty. Cannot scale empty data.")
        return ((data - data.min()) / (data.max() - data.min())) * scalar + offset
    
    def fit_model(self, data):
        if data.empty:
            raise ValueError("Input data is empty. Please provide a non-empty dataset.")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        if not np.issubdtype(data.dtypes.values[0], np.number):
            raise TypeError("Input data must contain numeric values.")

        centroids = self.get_initial_centroids(data)
        prev_centroids = pd.DataFrame()

        clusters = self.assign_clusters(data, centroids)

        iter = 0
        while iter < self._max_iterations and not centroids.equals(prev_centroids):
            prev_centroids = centroids.copy()

            centroids = self.get_cluster_means(data, clusters)
            clusters = self.assign_clusters(data, centroids)
            iter += 1
        
        if iter == self._max_iterations:
            print("Max iterations have been reached and the model did not converge.")
        else:
            print(f"The model converged on iteration {iter + 1}.")
        return centroids, clusters


if __name__ == "__main__":
    try:
        # Example usage
        np.random.seed(42)
        data = pd.DataFrame({
            "Feature1": np.random.rand(100) * 100,
            "Feature2": np.random.rand(100) * 100
        })

        kmeans = KMeans(K=3, max_iterations=50)
        final_centroids, cluster_labels = kmeans.fit_model(data)

        print("\nFinal Centroids:")
        print(final_centroids)

        print("\nCluster Assignments:")
        print(cluster_labels.value_counts())

    except Exception as e:
        print(f"An error occurred: {e}")
