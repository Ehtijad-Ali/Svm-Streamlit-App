import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =============================
# Train the model once at startup
# =============================
@st.cache_resource
def train_model():
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_classes=2,
        n_clusters_per_class=2,
        n_redundant=0,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=10
    )

    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": [1, 0.1, 0.01, 0.001],
        "kernel": ["linear"]
    }

    grid = GridSearchCV(SVC(), param_grid=param_grid, refit=True, cv=5)
    grid.fit(X_train, y_train)

    return grid, X, y, X_train, X_test, y_train, y_test

model, X, y, X_train, X_test, y_train, y_test = train_model()

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="SVM Classifier App", page_icon="🤖", layout="wide")

st.title("🤖 Support Vector Machine Classifier")
st.markdown("Interactive **SVM Classifier** with GridSearchCV tuning. "
            "Enter your own values and see predictions in real-time.")

# Sidebar for input
st.sidebar.header("🔢 Enter Feature Values")
x0 = st.sidebar.number_input("Feature 1 (x0)", value=0.0, step=0.1)
x1 = st.sidebar.number_input("Feature 2 (x1)", value=0.0, step=0.1)

if st.sidebar.button("Predict"):
    pred = model.predict([[x0, x1]])[0]
    st.sidebar.success(f"✅ Predicted Class: {pred}")

# Show scatter plot
st.subheader("📊 Data Visualization")
fig, ax = plt.subplots(figsize=(6, 4))
df = pd.DataFrame(X, columns=["x0", "x1"])
df["y"] = y
sns.scatterplot(x="x0", y="x1", hue="y", data=df, palette="deep", ax=ax)
st.pyplot(fig)

# Model performance
st.subheader("📈 Model Performance")
y_pred = model.predict(X_test)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
with col2:
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix")
    st.dataframe(pd.DataFrame(cm))
with col3:
    st.write("Classification Report")
    st.text(classification_report(y_test, y_pred))
