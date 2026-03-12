import streamlit as st
import pickle
import pandas as pd
import os

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Personality Type Predictor",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------
# Feature columns
# Must match training exactly
# -------------------------------
feature_columns = [
    'social_energy',
    'alone_time_preference',
    'talkativeness',
    'deep_reflection',
    'group_comfort',
    'party_liking',
    'listening_skill',
    'empathy',
    'organization',
    'leadership',
    'risk_taking',
    'public_speaking_comfort',
    'curiosity',
    'routine_preference',
    'excitement_seeking',
    'friendliness',
    'planning',
    'spontaneity',
    'adventurousness',
    'reading_habit',
    'sports_interest',
    'online_social_usage',
    'travel_desire',
    'gadget_usage',
    'work_style_collaborative',
    'decision_speed'
]

# -------------------------------
# User-friendly labels
# -------------------------------
feature_labels = {
    'social_energy': 'Social Energy',
    'alone_time_preference': 'Alone Time Preference',
    'talkativeness': 'Talkativeness',
    'deep_reflection': 'Deep Reflection',
    'group_comfort': 'Group Comfort',
    'party_liking': 'Party Liking',
    'listening_skill': 'Listening Skill',
    'empathy': 'Empathy',
    'organization': 'Organization',
    'leadership': 'Leadership',
    'risk_taking': 'Risk Taking',
    'public_speaking_comfort': 'Public Speaking Comfort',
    'curiosity': 'Curiosity',
    'routine_preference': 'Routine Preference',
    'excitement_seeking': 'Excitement Seeking',
    'friendliness': 'Friendliness',
    'planning': 'Planning',
    'spontaneity': 'Spontaneity',
    'adventurousness': 'Adventurousness',
    'reading_habit': 'Reading Habit',
    'sports_interest': 'Sports Interest',
    'online_social_usage': 'Online Social Usage',
    'travel_desire': 'Travel Desire',
    'gadget_usage': 'Gadget Usage',
    'work_style_collaborative': 'Work Style Collaborative',
    'decision_speed': 'Decision Speed'
}

# -------------------------------
# Prediction label mapping
# Change these if needed
# Example:
# 0 = Introvert
# 1 = Extrovert
# -------------------------------
label_map = {
    0: "Introvert",
    1: "Extrovert"
}

# -------------------------------
# Load model and scaler
# -------------------------------
@st.cache_resource
def load_files():
    model_path = "personality_model.pkl"
    scaler_path = "scaler.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"'{model_path}' not found in the project folder.")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"'{scaler_path}' not found in the project folder.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


# -------------------------------
# Title and sidebar
# -------------------------------
st.title("🧠 Personality Type Predictor")
st.write("Adjust the values below and click the button to predict the personality type.")

st.sidebar.header("About")
st.sidebar.write(
    "This app predicts personality type using your trained Logistic Regression model."
)
st.sidebar.write("Choose values from 1 to 10 for each feature.")

# -------------------------------
# Load model safely
# -------------------------------
try:
    model, scaler = load_files()
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")
    st.stop()

# -------------------------------
# Input UI
# -------------------------------
st.subheader("Enter Feature Values")

col1, col2, col3 = st.columns(3)
user_input = {}

for i, feature in enumerate(feature_columns):
    label = feature_labels.get(feature, feature)

    if i % 3 == 0:
        with col1:
            user_input[feature] = st.slider(label, 1, 10, 5)
    elif i % 3 == 1:
        with col2:
            user_input[feature] = st.slider(label, 1, 10, 5)
    else:
        with col3:
            user_input[feature] = st.slider(label, 1, 10, 5)

st.markdown("---")

# -------------------------------
# Predict button
# -------------------------------
if st.button("Predict Personality Type", use_container_width=True):
    try:
        input_df = pd.DataFrame([user_input])

        # exact order
        input_df = input_df[feature_columns]

        # scale
        input_scaled = scaler.transform(input_df)

        # predict
        prediction = model.predict(input_scaled)[0]

        # convert numeric result to text
        if isinstance(prediction, str):
            prediction_text = prediction
        else:
            prediction_text = label_map.get(int(prediction), f"Unknown ({prediction})")

        st.success(f"🧠 Predicted Personality Type: **{prediction_text}**")

        # show probabilities if available
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_scaled)[0]
            class_names = model.classes_

            # convert class names to readable text if numeric
            readable_classes = []
            for cls in class_names:
                try:
                    readable_classes.append(label_map.get(int(cls), str(cls)))
                except:
                    readable_classes.append(str(cls))

            prob_df = pd.DataFrame({
                "Personality Type": readable_classes,
                "Probability": probabilities
            }).sort_values(by="Probability", ascending=False)

            prob_df["Probability"] = (prob_df["Probability"] * 100).round(2).astype(str) + "%"

            st.subheader("Prediction Probabilities")
            st.dataframe(prob_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------------------
# Show input data
# -------------------------------
with st.expander("Show Input Data"):
    st.dataframe(pd.DataFrame([user_input]), use_container_width=True)