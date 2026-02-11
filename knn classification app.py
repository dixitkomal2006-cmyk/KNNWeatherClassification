import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="KNN Weather Classifier")

st.title("🌤 KNN Weather Classification")

# ---------------- Dataset ----------------
X = np.array([
    [25, 80],
    [27, 60],
    [31, 65],
    [23, 85],
    [20, 75]
])

y = np.array([0, 1, 1, 0, 0])

label_map = {0: "Sunny", 1: "Rainy"}

# ---------------- Sidebar ----------------
st.sidebar.header("Input Features")
temp = st.sidebar.slider("Temperature", 10, 60, 26)
hum = st.sidebar.slider("Humidity", 50, 95, 78)

# ---------------- Model ----------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

new_data = np.array([[temp, hum]])
prediction = knn.predict(new_data)[0]

st.write(f"### Predicted Weather: **{label_map[prediction]}**")

# ---------------- Graph ----------------
st.subheader("Training Data Distribution")

df = pd.DataFrame({
    "Temperature": X[:, 0],
    "Humidity": X[:, 1],
    "Weather": ["Sunny" if i == 0 else "Rainy" for i in y]
})

st.scatter_chart(df, x="Temperature", y="Humidity")
