class KMeans:
    def __init__(self, k=2, max_iters=100):
        self.k = k                            # Number of clusters
        self.max_iters = max_iters            # Max iterations to update centroids
        self.centroids = []                   # List to store current centroids

    def fit(self, X):
        """
        Train K-Means on data X
        """
        # Randomly pick first k points as initial centroids
        self.centroids = [X[i] for i in range(self.k)]

        for _ in range(self.max_iters):
            # Create empty clusters
            clusters = [[] for _ in range(self.k)]

            # Assign each point to the nearest centroid
            for x in X:
                distances = [self.euclidean_distance(x, centroid) for centroid in self.centroids]
                cluster_index = distances.index(min(distances))
                clusters[cluster_index].append(x)

            # Store old centroids to check convergence
            old_centroids = self.centroids[:]

            # Update centroids as the mean of points in each cluster
            self.centroids = [self.compute_centroid(cluster) for cluster in clusters]

            # If centroids do not change, break early
            if self.centroids == old_centroids:
                break

    def predict(self, x):
        """
        Assign a new point to the closest cluster
        """
        distances = [self.euclidean_distance(x, centroid) for centroid in self.centroids]
        return distances.index(min(distances))

    def compute_centroid(self, cluster):
        """
        Compute mean (centroid) of a cluster
        """
        if len(cluster) == 0:
            return [0 for _ in range(len(self.centroids[0]))]  # Handle empty cluster

        n_features = len(cluster[0])
        centroid = [0] * n_features

        for point in cluster:
            for i in range(n_features):
                centroid[i] += point[i]

        for i in range(n_features):
            centroid[i] /= len(cluster)

        return centroid

    def euclidean_distance(self, a, b):
        """
        Compute Euclidean distance between two points
        """
        return sum((a[i] - b[i]) ** 2 for i in range(len(a))) ** 0.5
