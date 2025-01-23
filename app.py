import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

st.markdown(
    """
    <style>
    h1 { color: #4CAF50; text-align: center; font-family: Arial, sans-serif; }
    [data-testid="stSidebar"] { background-color: #f7f9fc; padding: 20px; }
    button { background-color: #4CAF50; border: none; color: white; padding: 10px 20px; text-align: center; 
             font-size: 16px; margin: 10px; border-radius: 5px; cursor: pointer; }
    button:hover { background-color: #45a049; }
    .plot-container { background-color: #ffffff; border-radius: 10px; padding: 20px; 
                      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); }
    </style>
    """,
    unsafe_allow_html=True,
)

gbr = joblib.load("gbr_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("\U0001F3E0 HDB Resale Price Prediction")

st.markdown(
    """
    Welcome to the **HDB Resale Price Prediction** tool. Use this app to estimate the resale price of HDB flats based on various property details.
    """
)

# Sidebar for user inputs
st.sidebar.header("\U0001F4CB Enter Property Details")

town = st.sidebar.selectbox(
    "Town",
    [
        "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH",
        "BUKIT PANJANG", "BUKIT TIMAH", "CENTRAL AREA", "CHOA CHU KANG",
        "CLEMENTI", "GEYLANG", "HOUGANG", "JURONG EAST", "JURONG WEST",
        "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "PUNGGOL",
        "QUEENSTOWN", "SEMBAWANG", "SENGKANG", "SERANGOON", "TAMPINES",
        "TOA PAYOH", "WOODLANDS", "YISHUN"
    ]
)

flat_type = st.sidebar.selectbox(
    "Flat Type (Number of Rooms)", [2, 3, 4, 5, 6], format_func=lambda x: f"{x} ROOM"
)

flat_model = st.sidebar.selectbox(
    "Flat Model",
    [
        "Improved", "New Generation", "DBSS", "Standard", "Apartment",
        "Simplified", "Model A", "Premium Apartment", "Adjoined flat",
        "Model A-Maisonette", "Maisonette", "Type S1", "Type S2",
        "Model A2", "Terrace", "Improved-Maisonette", "Premium Maisonette",
        "Multi Generation", "Premium Apartment Loft", "2-room", "3Gen"
    ]
)

floor_area_sqm = st.sidebar.number_input("Floor Area (sqm)", min_value=31.0, max_value=249.0, step=1.0)
flat_age = st.sidebar.number_input("Flat Age (years)", min_value=0, max_value=99, step=1)
remaining_lease_months = st.sidebar.number_input(
    "Remaining Lease (months)", min_value=0, max_value=1200, step=1
)

# Prepare input data for prediction
input_data = {
    "floor_area_sqm": floor_area_sqm,
    "flat_age": flat_age,
    "remaining_lease_months": remaining_lease_months,
    "flat_type": flat_type,
}

def plot_feature_importance(importance, names, model_type):
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    plt.figure(figsize=(10, 8))
    plt.barh(fi_df['feature_names'], fi_df['feature_importance'], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    plt.title('Feature Importance - ' + model_type)
    st.pyplot(plt)


# Create a DataFrame with all features and initialize with zeros
input_df = pd.DataFrame(columns=feature_names)
input_df.loc[0] = 0

# Fill in the numeric features
for key in input_data.keys():
    if key in input_df.columns:
        input_df.loc[0, key] = input_data[key]

if st.sidebar.button("Predict"):
    predicted_price = gbr.predict(input_df)[0]
    st.markdown(f"### \U0001F4B5 Predicted Resale Price: **${predicted_price:,.2f}**")

    # Calculate price per square meter (sqm)
    price_per_sqm = predicted_price / floor_area_sqm
    st.markdown(f"### \U0001F3E2 Price per Square Meter: **${price_per_sqm:,.2f}**")

    if hasattr(gbr, 'feature_importances_'):
        plot_feature_importance(gbr.feature_importances_, feature_names, "Gradient Boosting Regressor")
    else:
        st.write("This model does not support feature importance output.")