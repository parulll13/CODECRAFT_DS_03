# CODECRAFT_DS_03

## 🌳 Bank Marketing Decision Tree Classifier

This project builds a Decision Tree Classifier using the Bank Marketing dataset to predict whether a customer will subscribe to a term deposit. It includes data preprocessing, model training, evaluation, and visualization of the decision tree.

---

## 📁 Files

- `Decision_tree.py` – Main Python script containing all preprocessing, model training, evaluation, and visualization steps.
- `bank-additional-full.csv` – The dataset used (must be downloaded and placed locally or in a `data/` folder).

---

## 📊 Dataset Overview

- 📂 **Source**: UCI Machine Learning Repository — [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- 🧾 **Description**: Marketing campaign data including attributes like job, marital status, age, balance, duration, etc.
- 🎯 **Target Variable**: `y` — whether the client subscribed to a term deposit (`yes` or `no`)

---

## 🧹 Data Preprocessing Steps

- Loaded `.csv` using `;` delimiter
- Encoded all categorical features using `LabelEncoder`
- Split the dataset into training and testing sets (80/20 split)

---

## 🤖 Model

- **Algorithm**: Decision Tree Classifier  
- **Criterion**: Entropy (information gain)  
- **Max Depth**: 5  
- **Library**: `sklearn.tree.DecisionTreeClassifier`

---

## 🧪 Evaluation Metrics

- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-score)
- **Confusion Matrix**

---

## 🌳 Tree Visualization

- Plotted using `plot_tree()` from `sklearn`
- Clearly shows decision paths based on feature splits

---

## 🛠️ Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn`

Install them via:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn

