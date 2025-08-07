class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}        # P(class)
        self.feature_probs = {}      # P(feature=value | class)
        self.classes = set()         # All possible class labels

    def fit(self, X, y):
        """
        Train the classifier using training features X and labels y
        """
        total_samples = len(y)
        self.classes = set(y)

        # Count occurrences of each class
        class_counts = {}
        for label in y:
            class_counts[label] = class_counts.get(label, 0) + 1

        # Calculate prior probabilities for each class
        for label in class_counts:
            self.class_probs[label] = class_counts[label] / total_samples

        # Initialize feature_probs dictionary for each class
        for label in self.classes:
            self.feature_probs[label] = {}

        # Count feature occurrences per class
        for features, label in zip(X, y):
            for i, value in enumerate(features):
                if i not in self.feature_probs[label]:
                    self.feature_probs[label][i] = {}
                if value not in self.feature_probs[label][i]:
                    self.feature_probs[label][i][value] = 0
                self.feature_probs[label][i][value] += 1

        # Convert counts to conditional probabilities
        for label in self.classes:
            for i in self.feature_probs[label]:
                total = sum(self.feature_probs[label][i].values())
                for value in self.feature_probs[label][i]:
                    self.feature_probs[label][i][value] /= total

    def predict(self, x):
        """
        Predict the class for a single input sample x
        """
        class_scores = {}

        for label in self.classes:
            # Start with the prior probability
            prob = self.class_probs[label]

            # Multiply by the conditional probability for each feature
            for i, value in enumerate(x):
                value_probs = self.feature_probs[label].get(i, {})
                cond_prob = value_probs.get(value, 1e-6)  # Use small value for unseen values
                prob *= cond_prob

            class_scores[label] = prob

        # Return the class with the highest probability
        return max(class_scores, key=class_scores.get)

    def predict_batch(self, X_test):
        """
        Predict labels for a list of samples
        """
        return [self.predict(x) for x in X_test]
