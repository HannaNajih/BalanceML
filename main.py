from src.utils import load_data, train_test_split, precision_recall_f1
from src.decision_tree import DecisionTreeClassifier
from src.knn import KNNClassifier
from src.naive_bayes import NaiveBayesClassifier

def evaluate_model(model, X_test, y_test):
    y_pred = [model.predict(x) for x in X_test]
    return precision_recall_f1(y_test, y_pred)

def print_results_table(all_results):
    print("+-------------------+--------+------------+--------+----------+")
    print("| Model             | Class  | Precision  | Recall | F1-Score |")
    print("+===================+========+============+========+==========+")
    for model_name, metrics in all_results.items():
        for cls, scores in metrics.items():
            precision = f"{scores['precision']:.2f}"
            recall = f"{scores['recall']:.2f}"
            f1 = f"{scores['f1']:.2f}"
            print(f"| {model_name:<17} | {cls:<6} | {precision:<10} | {recall:<6} | {f1:<8} |")
        print("+-------------------+--------+------------+--------+----------+")

def main():
    # Load dataset
    data = load_data("dataset/balance-scale.data")
    train, test = train_test_split(data, ratio=0.7)
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    all_results = {}

    # 1. Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    dt_metrics = evaluate_model(dt_model, X_test, y_test)
    all_results["Decision Tree"] = dt_metrics

    # 2. KNN
    knn_model = KNNClassifier(k=3)
    knn_model.fit(X_train, y_train)
    knn_metrics = evaluate_model(knn_model, X_test, y_test)
    all_results["KNN (k=3)"] = knn_metrics

    # 3. Naive Bayes
    nb_model = NaiveBayesClassifier()
    nb_model.fit(X_train, y_train)
    nb_metrics = evaluate_model(nb_model, X_test, y_test)
    all_results["Naive Bayes"] = nb_metrics

    # Print final evaluation table
    print_results_table(all_results)
if __name__ == "__main__":
    main()