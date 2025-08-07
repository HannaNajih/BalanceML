# Define the KNN classifier class
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k  # Number of nearest neighbors to consider
        self.train_data = []  # List to store training samples and their labels

    # Train the classifier by storing the training data
    def fit(self, X, y):
        self.train_data = list(zip(X, y))  # Combine features and labels into one list

    # Predict the label for a single test instance x
    def predict(self, x):
        # Step 1: Calculate the distance from x to every training sample
        distances = []
        for features, label in self.train_data:
            dist = self.euclidean_distance(x, features)  # Compute distance between x and training point
            distances.append((dist, label))  # Store distance and label as a tuple

        # Step 2: Sort distances in ascending order and pick the first k
        distances.sort(key=lambda pair: pair[0])  # Sort by distance
        k_nearest = distances[:self.k]  # Take the k closest points

        # Step 3: Count the frequency of each class label among the k nearest neighbors
        class_votes = {}
        for _, label in k_nearest:
            if label in class_votes:
                class_votes[label] += 1
            else:
                class_votes[label] = 1

        # Step 4: Return the label with the highest number of votes (majority class)
        return max(class_votes, key=class_votes.get)

    # Predict the labels for a list of test instances
    def predict_batch(self, X_test):
        predictions = []
        for x in X_test:
            label = self.predict(x)  # Predict label for each test point
            predictions.append(label)
        return predictions

    # Helper function to compute Euclidean distance between two vectors
    def euclidean_distance(self, a, b):
        distance = 0.0
        for i in range(len(a)):
            distance += (a[i] - b[i]) ** 2  # Square of difference
        return distance ** 0.5  # Square root of sum of squares
