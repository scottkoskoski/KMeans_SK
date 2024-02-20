import numpy as np
import pandas as pd
import random

class KMeans:
    def __init__(self, K=5, max_iterations=100):
        self._K = K
        self._max_iterations = max_iterations
    
    @property
    def K(self):
        return self._K
    
    @K.setter
    def K(self, K):
        self._K = K
    
    @property
    def max_iterations(self):
        return self._max_iterations
    
    @max_iterations.setter
    def max_iterations(self, max_iterations):
        self._max_iterations = max_iterations

    def get_initial_centroids(self, data):
        rows = data.shape[0]

        centroids = {}
        for i in range(self._K):
            idx = random.randint(0, rows - 1)
            centroid = data.iloc[idx]
            centroids[i] = centroid
        return pd.DataFrame(centroids)
    
    def assign_clusters(self, data, centroids):
        # calculating distance between each centroid and each observation
        dist = centroids.apply(lambda x: np.sqrt((data - x) ** 2).sum(axis=1))
        return dist.idxmin(axis=1)
    
    def get_cluster_means(self, data, clusterlabels):
        """
        Returns the geometric mean of the cluster.
        """
        return data.groupby(clusterlabels).apply(lambda x: np.exp(np.log(x).mean())).T
    
    def scale_data(self, data, scalar=1, offset=0):
        return ((data - data.min()) / (dta.max() - data.min())) * scalar + offset
    
    def fit_model(self, data):
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
