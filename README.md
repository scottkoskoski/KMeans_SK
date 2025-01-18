# K-Means Clustering Implementation in Python

This repository contains a custom implementation of the K-Means clustering algorithm using Python. The script is designed for educational purposes and provides insights into the mechanics of K-Means clustering. It was created as a simple educational exercise to build the algorithm from scratch, done purely for fun.

## Features

-   **Customizable Parameters**:
    -   Number of clusters (`K`)
    -   Maximum iterations for convergence (`max_iterations`)
-   **Random Centroid Initialization**: Randomly selects initial centroids from the dataset.
-   **Cluster Assignment**: Assigns data points to the nearest cluster based on Euclidean distance.
-   **Centroid Updates**: Computes geometric means for updated cluster centroids.
-   **Data Scaling**: Includes a method for scaling data to a specified range.
-   **Error Handling**: Robust error checking for invalid inputs, empty datasets, and more.

## Requirements

The script requires the following Python libraries:

-   `numpy`
-   `pandas`

Install the dependencies using:

```bash
pip install numpy pandas
```

## Usage

1. Clone the repository:

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Run the script directly:

    ```bash
    python kmeans.py
    ```

3. Modify the `if __name__ == "__main__"` block to test the implementation with your dataset.

### Example

```python
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
```

This will create a random dataset with 100 samples and cluster it into 3 groups.

## Methods

### Class Initialization

-   `KMeans(K=5, max_iterations=100)`
    -   `K`: Number of clusters (default: 5).
    -   `max_iterations`: Maximum iterations for convergence (default: 100).

### Key Methods

#### `get_initial_centroids(data)`

Randomly selects `K` initial centroids from the dataset.

#### `assign_clusters(data, centroids)`

Assigns data points to the nearest cluster using Euclidean distance.

#### `get_cluster_means(data, clusterlabels)`

Calculates geometric means for each cluster.

#### `scale_data(data, scalar=1, offset=0)`

Normalizes data to a specific range (default: 0 to 1).

#### `fit_model(data)`

Executes the K-Means clustering algorithm. Returns the final centroids and cluster labels.

## Error Handling

The script includes checks for:

-   Invalid `K` or `max_iterations` values.
-   Empty or non-numeric datasets.
-   Insufficient data points for the specified number of clusters.

## Output

-   **Final Centroids**: A DataFrame containing the coordinates of the final centroids.
-   **Cluster Assignments**: A Series indicating the cluster label for each data point.

## Contributing

Feel free to submit issues or pull requests if you have suggestions for improvement or find any bugs.

## License

This project is not licensed. It is shared as-is for educational and personal use.
