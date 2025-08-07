# Balance Scale Classification using Pure Python

This project implements fundamental machine learning algorithms **from scratch in pure Python** (no external libraries), focusing on the **Balance Scale Dataset** from the UCI Machine Learning Repository. It aims to classify the position of a balance scale (Left, Right, Balanced) based on weight and distance features using classical classification and clustering techniques.

---

## 📂 Project Structure

BalanceML/
│

├── main.py # Main execution file

├── dataset/

│ └── balance-scale.data # UCI Balance Scale dataset

└── src/

├── init.py

├── utils.py # Data loading, metrics, train/test split

├── decision_tree.py # ID3 Decision Tree implementation

├── knn.py # k-Nearest Neighbors implementation

├── naive_bayes.py # Naive Bayes classifier

├── kmeans.py # K-Means clustering (unsupervised)

└── cross_validation.py # K-fold cross-validation


---

## 📊 Dataset Description

- **Name:** Balance Scale Dataset  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Balance+Scale)  
- **Samples:** 625  
- **Features:**
  - Left weight (Integer: 1–5)
  - Left distance (Integer: 1–5)
  - Right weight (Integer: 1–5)
  - Right distance (Integer: 1–5)
- **Classes:**
  - `L` → Tilts Left  
  - `B` → Balanced  
  - `R` → Tilts Right

---

## 🔍 Implemented Algorithms

- ✅ **ID3 Decision Tree** (`decision_tree.py`)  
- ✅ **k-Nearest Neighbors** (`knn.py`)  
- ✅ **Naive Bayes Classifier** (`naive_bayes.py`)  
- ✅ **K-Means Clustering** (Unsupervised, `kmeans.py`)

All algorithms are implemented using only built-in Python functionality (no `pandas`, `numpy`, `sklearn`, etc.).

---

## 🛠 How to Run

### 1. Clone the repo


git clone [(https://github.com/HannaNajih/BalanceML.git)]
cd balance-scale-classifier

## 📂 2. Prepare Dataset

Place the `balance-scale.data` file in the `dataset/` folder.

You can download it from [UCI Repository](https://archive.ics.uci.edu/ml/datasets/balance+scale) or any provided link.

---

## ▶️ 3. Run the Project

bash: 

python main.py


---
## 📈 Evaluation Metrics

The following performance metrics are calculated:

- **Precision**  
- **Recall**  
- **F1-Score**

These metrics are implemented manually in `utils.py` for educational purposes and a deeper understanding of model evaluation.

## 🔁 Cross-Validation

`cross_validation.py` provides **k-fold cross-validation** capability to assess model generalization.

---

## 📚 Educational Objective

This project is part of an academic assignment focused on:

- Understanding core classification and clustering algorithms  
- Implementing from scratch without any libraries  
- Learning performance evaluation in machine learning  
- Practicing clean code and modular design  

---

## 📜 License

This project is open for **educational use**. If you use this code in any public project or course, please **give credit**.

---

## 👨‍🎓 Author

**Hana Najih**  
*MSc Student in Computer Science*  
**Focus:** Machine Learning & Software Engineering
