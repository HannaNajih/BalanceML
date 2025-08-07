# This class represents a node in the decision tree.
class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, label=None):
        self.feature_index = feature_index  # index of the feature to split on
        self.threshold = threshold          # value of the threshold to split
        self.left = left                    # left child node (for <= threshold)
        self.right = right                  # right child node (for > threshold)
        self.label = label                  # class label if this is a leaf node

# This is the main Decision Tree Classifier
class DecisionTreeClassifier:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth  # max depth of the tree to prevent overfitting
        self.root = None            # the root of the tree will be set after training

    # The function to train the decision tree
    def fit(self, X, y):
        data = list(zip(X, y))                    # Combine features and labels into pairs
        self.root = self.build_tree(data, depth=0)  # Start building tree from depth 0

    # Recursive function to build the tree
    def build_tree(self, data, depth):
        labels = [label for _, label in data]  # extract all labels in the current node

        # Base case 1: if all labels are the same, return a leaf node
        if labels.count(labels[0]) == len(labels):
            return DecisionNode(label=labels[0])

        # Base case 2: if depth limit reached or only one sample left, return majority label
        if depth >= self.max_depth or len(data) <= 1:
            return DecisionNode(label=self.majority_class(labels))

        # Find the best feature and threshold to split on
        best_feature, best_threshold = self.best_split(data)

        # Base case 3: if no good split found, return majority class
        if best_feature is None:
            return DecisionNode(label=self.majority_class(labels))

        # Split the data into two groups
        left_data, right_data = self.split_data(data, best_feature, best_threshold)

        # Recursively build left and right subtrees
        left_child = self.build_tree(left_data, depth + 1)
        right_child = self.build_tree(right_data, depth + 1)

        # Return the decision node with children
        return DecisionNode(best_feature, best_threshold, left_child, right_child)

    # Predict the class for a single sample x
    def predict(self, x):
        node = self.root  # Start from the root
        while node.label is None:  # Traverse until reaching a leaf
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.label  # Return the class label at the leaf

    # Helper: get the majority class from a list of labels
    def majority_class(self, labels):
        count = {}
        for label in labels:
            count[label] = count.get(label, 0) + 1  # count occurrences
        return max(count, key=count.get)  # return label with max count

    # Find the best feature and threshold to split on
    def best_split(self, data):
        best_gini = 1                # Initialize with worst possible Gini
        best_feature = None
        best_threshold = None
        n_features = len(data[0][0])  # Number of features in the data

        # Try every feature
        for feature_index in range(n_features):
            values = set([x[feature_index] for x, _ in data])  # Unique values for this feature
            for threshold in values:
                # Split data on current feature and threshold
                left, right = self.split_data(data, feature_index, threshold)
                if not left or not right:  # Skip if split is invalid (all to one side)
                    continue
                gini = self.gini_index([left, right])  # Compute Gini impurity of split
                if gini < best_gini:  # Update best split if better Gini found
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold  # Return best split found

    # Split data based on feature and threshold
    def split_data(self, data, feature_index, threshold):
        left = []
        right = []
        for x, label in data:
            if x[feature_index] <= threshold:  # Send to left if value <= threshold
                left.append((x, label))
            else:                              # Otherwise to right
                right.append((x, label))
        return left, right

    # Compute Gini index (impurity) for a split (list of groups)
    def gini_index(self, groups):
        total_samples = sum(len(group) for group in groups)  # total number of samples
        gini = 0.0

        # Compute Gini for each group
        for group in groups:
            if len(group) == 0:
                continue
            score = 0.0
            labels = [label for _, label in group]  # Get labels in this group
            for cls in set(labels):  # For each unique class
                p = labels.count(cls) / len(group)  # proportion of this class
                score += p * p                      # square of proportion
            gini += (1 - score) * (len(group) / total_samples)  # weighted impurity

        return gini
