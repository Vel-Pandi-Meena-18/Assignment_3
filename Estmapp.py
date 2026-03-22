import streamlit as st
import pandas as pd
import pickle

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="EcoType Forest Cover Prediction",
    page_icon="🌲",
    layout="wide"
)

# -----------------------------------
# Load Saved Files
# -----------------------------------
with open("forest_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model_features.pkl", "rb") as f:
    model_features = pickle.load(f)

# -----------------------------------
# App Title
# -----------------------------------
st.title("🌲 EcoType: Forest Cover Type Prediction")
st.markdown("""
This app predicts the **forest cover type** based on cartographic and environmental input features.

Fill in the values below and click **Predict** to get the result.
""")

# -----------------------------------
# Instructions Block
# -----------------------------------
st.info("""
📌 **Input Guidelines**
- **Elevation:** usually between 1800 and 4000 meters
- **Aspect:** 0 to 360 degrees
- **Slope:** 0 to 60 degrees
- **Hillshade values:** 0 to 255
- **Distances:** enter in meters
- **Vertical Distance To Hydrology:** can be negative, zero, or positive
- **Wilderness Area:** choose from 1 to 4
- **Soil Type:** choose from 1 to 40
""")

# -----------------------------------
# About Section
# -----------------------------------
with st.expander("ℹ️ About this app"):
    st.write("""
    - **Model used:** Random Forest Classifier
    - **Problem type:** Multi-class classification
    - **Output:** Predicted forest cover class
    - **Features used:** elevation, aspect, slope, hydrology distances, roadways distance,
      hillshade values, fire points distance, wilderness area, and soil type
    """)

# -----------------------------------
# Input Section
# -----------------------------------
st.subheader("📥 Enter Feature Values")

col1, col2 = st.columns(2)

with col1:
    elevation = st.number_input(
        "Elevation",
        min_value=0,
        max_value=5000,
        value=2800,
        step=1,
        help="Height of land above sea level in meters"
    )

    aspect = st.number_input(
        "Aspect",
        min_value=0,
        max_value=360,
        value=120,
        step=1,
        help="Direction the slope faces in degrees"
    )

    slope = st.number_input(
        "Slope",
        min_value=0,
        max_value=90,
        value=10,
        step=1,
        help="Steepness of the terrain in degrees"
    )

    horizontal_distance_to_hydrology = st.number_input(
        "Horizontal Distance To Hydrology",
        min_value=0,
        max_value=5000,
        value=200,
        step=1,
        help="Horizontal distance to nearest water source in meters"
    )

    vertical_distance_to_hydrology = st.number_input(
        "Vertical Distance To Hydrology",
        min_value=-500,
        max_value=1000,
        value=20,
        step=1,
        help="Vertical distance to nearest water source in meters"
    )

    horizontal_distance_to_roadways = st.number_input(
        "Horizontal Distance To Roadways",
        min_value=0,
        max_value=10000,
        value=1000,
        step=1,
        help="Horizontal distance to nearest roadway in meters"
    )

with col2:
    hillshade_9am = st.number_input(
        "Hillshade 9am",
        min_value=0,
        max_value=255,
        value=220,
        step=1,
        help="Illumination index at 9:00 AM"
    )

    hillshade_noon = st.number_input(
        "Hillshade Noon",
        min_value=0,
        max_value=255,
        value=230,
        step=1,
        help="Illumination index at 12:00 PM"
    )

    hillshade_3pm = st.number_input(
        "Hillshade 3pm",
        min_value=0,
        max_value=255,
        value=140,
        step=1,
        help="Illumination index at 3:00 PM"
    )

    horizontal_distance_to_fire_points = st.number_input(
        "Horizontal Distance To Fire Points",
        min_value=0,
        max_value=10000,
        value=1500,
        step=1,
        help="Horizontal distance to nearest wildfire ignition point"
    )

    wilderness_area = st.selectbox(
        "Wilderness Area",
        options=[1, 2, 3, 4],
        index=0,
        help="Select wilderness area category"
    )

    soil_type = st.selectbox(
        "Soil Type",
        options=list(range(1, 41)),
        index=0,
        help="Select soil type category"
    )

st.markdown("---")

# -----------------------------------
# Validation + Prediction
# -----------------------------------
if st.button("🔍 Predict Forest Cover Type", use_container_width=True):

    if elevation <= 0:
        st.warning("⚠ Elevation must be greater than 0.")
    else:
        # Create input dataframe
        input_data = pd.DataFrame([{
            "Elevation": elevation,
            "Aspect": aspect,
            "Slope": slope,
            "Horizontal_Distance_To_Hydrology": horizontal_distance_to_hydrology,
            "Vertical_Distance_To_Hydrology": vertical_distance_to_hydrology,
            "Horizontal_Distance_To_Roadways": horizontal_distance_to_roadways,
            "Hillshade_9am": hillshade_9am,
            "Hillshade_Noon": hillshade_noon,
            "Hillshade_3pm": hillshade_3pm,
            "Horizontal_Distance_To_Fire_Points": horizontal_distance_to_fire_points,
            "Wilderness_Area": wilderness_area,
            "Soil_Type": soil_type
        }])

        # Apply same preprocessing structure used during training
        input_data = pd.get_dummies(input_data, drop_first=True)

        # Match training columns
        input_data = input_data.reindex(columns=model_features, fill_value=0)

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]

        st.success(f"✅ Predicted Forest Cover Type: **{prediction}**")

        # Prediction probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            class_labels = model.classes_

            prob_df = pd.DataFrame({
                "Cover Type": class_labels,
                "Probability": probs
            }).sort_values(by="Probability", ascending=False)

            prob_df["Probability"] = prob_df["Probability"].round(2)

            st.subheader("📊 Prediction Confidence")
            st.dataframe(prob_df, use_container_width=True)

            top_class = prob_df.iloc[0]["Cover Type"]
            top_prob = prob_df.iloc[0]["Probability"] * 100

            st.info(f"Top predicted class: **{top_class}** with confidence **{top_prob:.2f}%**")

st.markdown("---")
st.caption("Built with Streamlit for EcoType Forest Cover Classification")