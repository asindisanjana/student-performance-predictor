import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load dataset
data = pd.read_csv('university_student_preprocessed.csv')

# Ensure target is numeric
data['Final_CGPA'] = pd.to_numeric(data['Final_CGPA'], errors='coerce')

# Split features and target
X = data.drop('Final_CGPA', axis=1)
y = data['Final_CGPA']

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# --- Mappings (IMPORTANT) ---
gender_map = {'Male': 0, 'Female': 1}
major_map = {
    'Engineering': 0,
    'Business': 1,
    'Arts': 2,
    'Science': 3
}

# --- UI ---
st.title("🎓 University Student CGPA Predictor")

st.write("Enter student details:")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=16, max_value=60, value=20)
major = st.selectbox("Major", ["Engineering", "Business", "Arts", "Science"])

attendance = st.slider("Attendance (%)", 0, 100, 90)
study_hours = st.slider("Study Hours Per Day", 0, 12, 3)
previous_gpa = st.number_input("Previous GPA", 0.0, 4.0, 3.0)

sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
social_hours = st.slider("Social Hours per Week", 0, 40, 5)

# --- Predict ---
if st.button("Predict CGPA"):

    # Convert to numeric
    gender_val = gender_map[gender]
    major_val = major_map[major]

    input_df = pd.DataFrame({
        'Gender':[gender_val],
        'Age':[age],
        'Major':[major_val],
        'Attendance_Pct':[attendance],
        'Study_Hours_Per_Day':[study_hours],
        'Previous_GPA':[previous_gpa],
        'Sleep_Hours':[sleep_hours],
        'Social_Hours_Week':[social_hours]
    })

    prediction = model.predict(input_df)[0]

    st.success(f"🎯 Predicted CGPA: {round(prediction, 2)}")