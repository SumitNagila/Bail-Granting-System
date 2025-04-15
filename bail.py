import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

@st.cache_data
def load_and_train():
    df = pd.read_csv('Balanced_Data1.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Identify categorical features (assuming columns 6 to 17 are categorical)
    categorical_indices = list(range(6, 18))
    cat_columns = [X.columns[i] for i in categorical_indices]

    # Store unique values for dropdowns
    unique_values = {col: sorted(df[col].dropna().unique()) for col in cat_columns}

    # ColumnTransformer for one-hot encoding
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_indices)],
                           remainder='passthrough')
    X_transformed = ct.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_encoded,
                                                        test_size=0.25, random_state=0)

    clf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    clf.fit(X_train, y_train)

    return clf, ct, X.columns, cat_columns, unique_values, label_encoder

clf, transformer, features, cat_columns, cat_values, label_encoder = load_and_train()

st.title("🔍 Bail Decision Predictor")
st.markdown("Enter the details below to predict whether bail should be granted.")

input_data = []

for feature in features:
    if feature in cat_columns:
        value = st.selectbox(f"{feature}", cat_values[feature])
    else:
        value = st.text_input(f"{feature}", "0")
    input_data.append(value)

# Predict button
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([input_data], columns=features)
        for col in features:
            if col not in cat_columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        if input_df.isnull().values.any():
            st.error("Please ensure all numerical inputs are valid numbers.")
        else:
            input_transformed = transformer.transform(input_df)
            prediction = clf.predict(input_transformed)
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            st.success(f"**Decision:** {predicted_label}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

