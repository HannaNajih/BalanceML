def k_fold_split(data, k):
    """
    Splits data into k folds
    """
    fold_size = len(data) // k                 # Calculate number of samples per fold
    folds = []

    for i in range(k):
        start = i * fold_size                  # Start index of fold
        end = start + fold_size                # End index of fold
        folds.append(data[start:end])          # Add the fold to the list

    return folds


def cross_validate(model_class, data, k=5):
    """
    Perform k-fold cross-validation.
    
    model_class: the class of the model (not instance) with .fit() and .predict()
    data: list of (features, label) tuples
    """
    folds = k_fold_split(data, k)              # Split data into k folds
    scores = []

    for i in range(k):
        # Use fold i as test, the rest as training data
        test_data = folds[i]
        train_data = []

        for j in range(k):
            if j != i:
                train_data.extend(folds[j])    # Add all folds except i to training data

        X_train, y_train = zip(*train_data)
        X_test, y_test = zip(*test_data)

        # Train and predict using the model
        model = model_class()
        model.fit(X_train, y_train)
        y_pred = [model.predict(x) for x in X_test]

        # Evaluate
        score = accuracy(y_test, y_pred)
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    return avg_score


def accuracy(y_true, y_pred):
    """
    Computes accuracy: (correct predictions / total samples)
    """
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    return correct / len(y_true)
