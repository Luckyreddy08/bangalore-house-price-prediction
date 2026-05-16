import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Page Title
st.title("Bangalore House Price Prediction")

st.write("Machine Learning House Price Predictor")

# Load Dataset
@st.cache_data
def load_data():

    df = pd.read_csv("bengaluru_house_prices.csv")

    return df

# Train Model
@st.cache_resource
def train_model():

    df = load_data()

    # Select useful columns
    df = df[['total_sqft', 'bath', 'price']]

    # Remove null values
    df.dropna(inplace=True)

    # Convert total_sqft into numeric
    df['total_sqft'] = pd.to_numeric(
        df['total_sqft'],
        errors='coerce'
    )

    # Remove invalid rows
    df.dropna(inplace=True)

    # Features
    X = df[['total_sqft', 'bath']]

    # Target
    y = df['price']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=10
    )

    # Linear Regression Model
    model = LinearRegression()

    model.fit(X_train, y_train)

    return model

# Train model
model = train_model()

# User Inputs
st.subheader("Enter House Details")

sqft = st.number_input(
    "Total Square Feet",
    min_value=100,
    max_value=10000,
    value=1200
)

bath = st.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

# Predict Button
if st.button("Predict Price"):

    prediction = model.predict([[sqft, bath]])

    st.success(
        f"Estimated House Price: ₹ {prediction[0]:.2f} Lakhs"
    )

# Show Dataset
if st.checkbox("Show Dataset"):

    st.write(load_data().head())