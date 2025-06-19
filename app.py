import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Streamlit UI setup
st.set_page_config(page_title="Stock Price Prediction", layout="centered")
st.title("📈 Stock Price Prediction")

# Load dataset
data = pd.read_csv("stock_data.csv")
st.success("✅ Dataset 'stock_data' successfully loaded.")

# Apply LabelEncoder to categorical columns
encoder = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = encoder.fit_transform(data[col])

# Define stock columns
stock_columns = [col for col in data.columns if col.startswith('Stock')]

# Model selection
model_name = st.selectbox("Select Prediction Model:", ["Decision Tree", "Linear Regression", "Random Forest"])

# Initialize model
if model_name == "Linear Regression":
    model = LinearRegression()
elif model_name == "Random Forest":
    model = RandomForestRegressor()
elif model_name == "Decision Tree":
    model = DecisionTreeRegressor()

for stock in stock_columns:
    # Features and target
    X = data.drop(stock, axis=1)
    y = data[stock]

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train model
    model.fit(x_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)

    # Show MSE
    st.subheader(f"📊 Model Evaluation for {stock}")
    st.write(f"**Mean Squared Error (MSE):** `{mse:.4f}`")

    # Plot Actual vs Predicted
    st.subheader(f"📈 Actual vs Predicted Plot for {stock}")
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    axs[0].plot(y_test.values, label="Actual", marker='', linestyle='-', alpha=0.7, color='blue')
    axs[0].set_title("Actual Stock Price")
    axs[0].set_xlabel("Test Sample Index")
    axs[0].set_ylabel("Stock Price")
    axs[0].legend()

    axs[1].plot(y_pred, label="Predicted", marker='', linestyle='-', alpha=0.7, color='red')
    axs[1].set_title("Predicted Stock Price")
    axs[1].set_xlabel("Test Sample Index")
    axs[1].set_ylabel("Stock Price")
    axs[1].legend()

    plt.tight_layout()
    st.pyplot(fig)
