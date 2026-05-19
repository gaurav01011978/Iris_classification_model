# Iris Flower Classification

A Machine Learning project on the classic **Iris dataset** using a **Random Forest Classifier**. The project was developed on Google Colab and covers a complete end-to-end ML pipeline — from data loading to model evaluation.

## Project Overview

The Iris dataset is the "Hello World" of machine learning. It contains 150 flower samples across 3 species (Setosa, Versicolor, Virginica), with 4 measurement features per sample.

**Goal:** Predict the flower species based on these 4 features.

## Workflow

1. Load the dataset (`Iris.csv`)
2. Data exploration (`head`, `info`, `shape`)
3. Separate features (X) and target (y)
4. Train-test split (80-20)
5. Feature scaling using `StandardScaler`
6. Train a Random Forest Classifier
7. Evaluate predictions and accuracy
8. Visualize confusion matrix and feature importance

## Dataset

| Feature        | Description                  |
|----------------|------------------------------|
| SepalLengthCm  | Sepal length (in cm)         |
| SepalWidthCm   | Sepal width (in cm)          |
| PetalLengthCm  | Petal length (in cm)         |
| PetalWidthCm   | Petal width (in cm)          |
| Species        | Target class (3 categories)  |

**Classes:** Iris-setosa, Iris-versicolor, Iris-virginica (50 samples each)

## Project Structure
iris-classification/
├── iris_classification.ipynb   # Main Jupyter notebook
├── iris_classifier.py          # Standalone Python script
├── Iris.csv                    # Dataset
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
## Installation

Clone the repo:
```bash
git clone https://github.com/<your-username>/iris-classification.git
cd iris-classification
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook
```bash
jupyter notebook iris_classification.ipynb
```

### Standalone Script
```bash
python iris_classifier.py
```

### Google Colab
Open the notebook in Colab and uncomment the `Iris.csv` upload cell.

## Results

The Random Forest Classifier achieved **100% test accuracy**:
## Installation

Clone the repo:
```bash
git clone https://github.com/<your-username>/iris-classification.git
cd iris-classification
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook
```bash
jupyter notebook iris_classification.ipynb
```

### Standalone Script
```bash
python iris_classifier.py
```

### Google Colab
Open the notebook in Colab and uncomment the `Iris.csv` upload cell.

## Results

The Random Forest Classifier achieved **100% test accuracy**:
Model Accuracy: 100.00%
Classification Report:
                  precision    recall   f1-score   support
Iris-setosa       1.00         1.00      1.00        10
Iris-versicolor   1.00         1.00      1.00         9
Iris-virginica    1.00         1.00      1.00        11
accuracy                                 1.00        30
**Most important features** (based on Random Forest feature importance): Petal length and petal width — these are the most discriminative features for separating Setosa from Versicolor/Virginica.

> **Note:** The `Id` column should be dropped — it's just a row number and not a useful feature. If left in, the model will incorrectly treat it as important.

## Tech Stack

- **Python 3.8+**
- **pandas** — data manipulation
- **numpy** — numerical computing
- **scikit-learn** — Random Forest, train/test split, scaling, metrics
- **matplotlib** & **seaborn** — visualizations

## Future Improvements

- Compare multiple models (SVM, KNN, Logistic Regression)
- Hyperparameter tuning with `GridSearchCV`
- Add cross-validation scores
- Build a Streamlit web app for live predictions

## Author

**Gaurav Kumar - gk8131831@gmail.com**

## License

MIT License
