import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

st.title("ML Model Comparison App")
st.write("Upload a CSV file and compare multiple ML algorithms automatically.")

# Upload file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📌 Preview of Dataset")
    st.dataframe(df.head())

    # Select target column
    target = st.selectbox("Select Target Column", df.columns)

    if st.button("Run Models"):
        X = df.drop(columns=[target])
        y = df[target]

        # Check for non-numeric data
        X = X.select_dtypes(include=['int64', 'float64'])

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=200),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier()
        }

        results = {}

        # Train + Evaluate
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc

        # Show Accuracy Table
        st.write("### 📌 Model Accuracy Comparison")
        result_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
        st.dataframe(result_df)

        # Plot Accuracy Graph
        st.write("### 📊 Accuracy Graph")
        fig, ax = plt.subplots()
        ax.bar(result_df["Model"], result_df["Accuracy"])
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Best Model
        best_model = max(results, key=results.get)
        st.success(f"🏆 Best Model: **{best_model}** with accuracy {results[best_model]:.2f}")


