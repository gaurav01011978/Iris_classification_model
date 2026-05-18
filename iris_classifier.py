"""
Iris Flower Classification
==========================
A standalone Python script that trains a Random Forest Classifier on the
Iris dataset and reports predictions and metrics.

Usage:
    python iris_classifier.py
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def load_data(path: str = "Iris.csv") -> pd.DataFrame:
    """Load the Iris dataset and drop the Id column if present."""
    df = pd.read_csv(path)
    if "Id" in df.columns:
        df = df.drop("Id", axis=1)
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    return df


def preprocess(df: pd.DataFrame):
    """Split features/target, encode labels, scale, and create train/test sets."""
    X = df.drop("Species", axis=1)   # all features
    y = df["Species"]                 # target (flower type)

    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape:   {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size:  {X_test.shape[0]}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns


def train_model(X_train, y_train):
    """Train a Random Forest Classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("\nModel training complete!")
    return model


def evaluate_model(model, X_test, y_test):
    """Make predictions and print the accuracy and classification report."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    return y_pred


def plot_confusion_matrix(y_test, y_pred):
    """Plot and save the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=100)
    print("Saved: confusion_matrix.png")
    plt.close()


def plot_feature_importance(model, feature_names):
    """Plot and save the feature importance chart."""
    plt.figure(figsize=(10, 5))
    plt.barh(feature_names, model.feature_importances_)
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=100)
    print("Saved: feature_importance.png")
    plt.close()


def main():
    df = load_data("Iris.csv")
    X_train, X_test, y_train, y_test, feature_names = preprocess(df)
    model = train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model, feature_names)
    print("\nPROJECT COMPLETE!")


if __name__ == "__main__":
    main()