import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("XGBoost Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully")

    # Handle missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna(method="ffill", inplace=True)

    # Encode categorical columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.factorize(df[col])[0]

    # Target = last column
    target_column = df.columns[-1]

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write("### Model Accuracy")
    st.success(f"{accuracy * 100:.2f}%")

    # Save model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    st.success("Model trained and saved as model.pkl")

else:
    st.warning("Please upload a CSV file to continue")
