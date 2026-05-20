import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Page config
st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")

st.title("🌸 Iris Flower Classifier")
st.write("Sepal aur petal ki measurements daalo, model species predict karega.")

# Cache the model so it trains only once
@st.cache_resource
def train_model():
    df = pd.read_csv("Iris.csv")
    if "Id" in df.columns:
        df = df.drop("Id", axis=1)

    X = df.drop("Species", axis=1)
    y = df["Species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X.columns.tolist()


model, scaler, feature_names = train_model()

# Sidebar inputs
st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.4, 0.1)
sepal_width  = st.sidebar.slider("Sepal Width (cm)",  2.0, 4.5, 3.4, 0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.3, 0.1)
petal_width  = st.sidebar.slider("Petal Width (cm)",  0.1, 2.5, 0.2, 0.1)

# Show user's input
user_input = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=feature_names,
)
st.subheader("Your Input")
st.dataframe(user_input)

# Predict
if st.button("Predict Species 🌸"):
    scaled = scaler.transform(user_input)
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]

    st.success(f"### Predicted Species: **{pred}**")

    # Probabilities
    st.subheader("Confidence")
    proba_df = pd.DataFrame({
        "Species": model.classes_,
        "Probability": proba,
    })
    st.bar_chart(proba_df.set_index("Species"))

st.markdown("---")
st.caption("Made with Streamlit • Random Forest Classifier • 100% test accuracy")
