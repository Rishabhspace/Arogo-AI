import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
@st.cache_resource
def load_data():
    data = pd.read_csv('survey.csv')
    data.replace("NA", pd.NA, inplace=True)
    data.drop(columns=['comments'], inplace=True)
    data.fillna(data.mode().iloc[0], inplace=True)
    
    # Encode categorical features
    categorical_columns = ['Gender', 'self_employed', 'family_history', 'work_interfere', 
                           'no_employees', 'remote_work', 'tech_company', 'benefits', 
                           'care_options', 'wellness_program', 'seek_help', 'anonymity', 
                           'leave', 'mental_health_consequence', 'phys_health_consequence', 
                           'coworkers', 'supervisor', 'mental_health_interview', 
                           'phys_health_interview', 'mental_vs_physical', 'obs_consequence']
    
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le
    
    # Encode target variable
    le_treatment = LabelEncoder()
    data['treatment'] = le_treatment.fit_transform(data['treatment'])
    
    return data, label_encoders, le_treatment

data, label_encoders, le_treatment = load_data()

# Train model
@st.cache_resource
def train_model():
    features = ['Age', 'Gender', 'family_history', 'work_interfere', 'benefits', 
                'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave', 
                'mental_health_consequence', 'phys_health_consequence', 'coworkers', 
                'supervisor', 'mental_health_interview', 'phys_health_interview', 
                'mental_vs_physical', 'obs_consequence']
    
    X = data[features]
    y = data['treatment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Streamlit UI
st.title("Mental Health Assessment Tool")
st.markdown("### Complete the following questionnaire for personalized analysis")

with st.form("health_assessment"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Personal Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", options=['Male', 'Female', 'Other'])
        family_history = st.radio("Family History of Mental Illness", ['Yes', 'No'])
        work_interfere = st.selectbox("Work Interference with Mental Health", 
                                    ['Never', 'Rarely', 'Sometimes', 'Often'])
        
        st.header("Company Benefits")
        benefits = st.radio("Mental Health Benefits Provided", 
                          ['Yes', 'No', 'Don\'t know'])
        care_options = st.selectbox("Knowledge of Care Options", 
                                  ['Yes', 'No', 'Not sure'])
        wellness_program = st.radio("Wellness Program Availability", 
                                  ['Yes', 'No', 'Don\'t know'])
        seek_help = st.selectbox("Ease of Seeking Help", 
                               ['Yes', 'No', 'Don\'t know'])
        anonymity = st.radio("Protected Anonymity", 
                           ['Yes', 'No', 'Don\'t know'])
        leave = st.selectbox("Mental Health Leave Accessibility", 
                           ['Very easy', 'Somewhat easy', 'Somewhat difficult', 
                            'Very difficult', 'Don\'t know'])
    
    with col2:
        st.header("Work Environment")
        mental_health_consequence = st.radio("Consequences for Mental Health Discussions", 
                                           ['Yes', 'No', 'Maybe'])
        phys_health_consequence = st.radio("Consequences for Physical Health Discussions", 
                                         ['Yes', 'No', 'Maybe'])
        coworkers = st.selectbox("Coworker Support", 
                               ['Yes', 'No', 'Some of them'])
        supervisor = st.selectbox("Supervisor Support", 
                                 ['Yes', 'No', 'Some of them'])
        
        st.header("Personal Perspectives")
        mental_health_interview = st.radio("Discuss Mental Health in Interviews", 
                                         ['Yes', 'No', 'Maybe'])
        phys_health_interview = st.radio("Discuss Physical Health in Interviews", 
                                       ['Yes', 'No', 'Maybe'])
        mental_vs_physical = st.radio("Mental vs Physical Health Priority", 
                                   ['Yes', 'No', 'Don\'t know'])
        obs_consequence = st.radio("Observed Negative Consequences", 
                                 ['Yes', 'No'])
    
    submitted = st.form_submit_button("Get Assessment")
    
    if submitted:
        # Prepare input data
        input_data = {
            'Age': age,
            'Gender': label_encoders['Gender'].transform([gender])[0],
            'family_history': label_encoders['family_history'].transform([family_history])[0],
            'work_interfere': label_encoders['work_interfere'].transform([work_interfere])[0],
            'benefits': label_encoders['benefits'].transform([benefits])[0],
            'care_options': label_encoders['care_options'].transform([care_options])[0],
            'wellness_program': label_encoders['wellness_program'].transform([wellness_program])[0],
            'seek_help': label_encoders['seek_help'].transform([seek_help])[0],
            'anonymity': label_encoders['anonymity'].transform([anonymity])[0],
            'leave': label_encoders['leave'].transform([leave])[0],
            'mental_health_consequence': label_encoders['mental_health_consequence'].transform([mental_health_consequence])[0],
            'phys_health_consequence': label_encoders['phys_health_consequence'].transform([phys_health_consequence])[0],
            'coworkers': label_encoders['coworkers'].transform([coworkers])[0],
            'supervisor': label_encoders['supervisor'].transform([supervisor])[0],
            'mental_health_interview': label_encoders['mental_health_interview'].transform([mental_health_interview])[0],
            'phys_health_interview': label_encoders['phys_health_interview'].transform([phys_health_interview])[0],
            'mental_vs_physical': label_encoders['mental_vs_physical'].transform([mental_vs_physical])[0],
            'obs_consequence': label_encoders['obs_consequence'].transform([obs_consequence])[0]
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        treatment_needed = le_treatment.inverse_transform(prediction)[0]
        
        st.divider()
        st.subheader("Assessment Result")
        if treatment_needed == 'Yes':
            st.error("Recommendation: Professional consultation suggested")
            st.write("""
            Based on your responses, our model suggests seeking professional consultation. 
            Consider reaching out to a mental health specialist or HR representative.
            """)
        else:
            st.success("Recommendation: No immediate treatment needed")
            st.write("""
            Based on your responses, our model doesn't detect immediate needs for treatment. 
            Maintain healthy habits and monitor your mental wellbeing.
            """)
        
        st.markdown("""
        **Disclaimer:**  
        This assessment is not a substitute for professional medical advice. 
        Always consult qualified healthcare providers for personal health decisions.
        """)


