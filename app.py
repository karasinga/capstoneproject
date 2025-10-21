import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import time
from sklearn.pipeline import Pipeline

# Import SHAP modules
from shap_utils import SHAPProcessor, compute_shap_values, prepare_shap_data
from shap_visualizations import create_shap_visualization_section, display_single_prediction_plot

# Clinical intervention mapping
CLINICAL_INTERVENTIONS = {
    # Behavioral factors
    'Missed_Appointments_Last_12_Months': {
        'actionable': True,
        'interventions': [
            'Implement appointment reminders via SMS/text',
            'Schedule follow-up calls after missed appointments',
            'Consider transportation assistance programs',
            'Review appointment scheduling preferences'
        ]
    },
    'App_Logins_Per_Week': {
        'actionable': True,
        'interventions': [
            'Provide patient education on app benefits',
            'Simplify app interface and navigation',
            'Offer technical support for app usage',
            'Create personalized medication schedules in app'
        ]
    },

    # Health status factors
    'Comorbidity_Count': {
        'actionable': True,
        'interventions': [
            'Coordinate care with multiple specialists',
            'Implement medication reconciliation reviews',
            'Provide comprehensive medication education',
            'Consider polypharmacy optimization'
        ]
    },
    'Has_Unhealthy_Diet': {
        'actionable': True,
        'interventions': [
            'Provide nutritional counseling',
            'Connect with dietitian services',
            'Offer meal planning resources',
            'Address food insecurity concerns'
        ]
    },
    'Is_Physically_Inactive': {
        'actionable': True,
        'interventions': [
            'Recommend gradual exercise programs',
            'Connect with physical therapy services',
            'Address barriers to physical activity',
            'Monitor medication effects on energy levels'
        ]
    },

    # Social factors
    'Family_Support_Level': {
        'actionable': True,
        'interventions': [
            'Involve family members in medication management',
            'Provide family education sessions',
            'Connect with caregiver support groups',
            'Assess family dynamics affecting adherence'
        ]
    },
    'Reports_Cost_Barrier': {
        'actionable': True,
        'interventions': [
            'Connect with patient assistance programs',
            'Explore generic medication options',
            'Review prescription assistance eligibility',
            'Discuss medication cost management strategies'
        ]
    },
    'Experienced_Stock_Out': {
        'actionable': True,
        'interventions': [
            'Implement automatic refill reminders',
            'Connect with mail-order pharmacy services',
            'Provide emergency medication supplies',
            'Review pharmacy communication protocols'
        ]
    },

    # Demographic factors (non-actionable)
    'Gender': {
        'actionable': False,
        'reason': 'Demographic characteristic cannot be changed',
        'alternative_focus': 'Address gender-specific barriers and cultural factors'
    },
    'Age': {
        'actionable': False,
        'reason': 'Age cannot be changed',
        'alternative_focus': 'Address age-related barriers and life stage considerations'
    },
    'Education_Level': {
        'actionable': False,
        'reason': 'Education level cannot be easily changed',
        'alternative_focus': 'Use appropriate health literacy materials and communication methods'
    },

    # Other factors
    'BMI': {
        'actionable': True,
        'interventions': [
            'Provide weight management counseling',
            'Connect with nutrition and exercise programs',
            'Monitor medication effects on weight',
            'Address obesity-related medication adherence barriers'
        ]
    },
    'Self_Reported_Mood': {
        'actionable': True,
        'interventions': [
            'Screen for depression and anxiety',
            'Provide mental health referrals',
            'Monitor medication side effects affecting mood',
            'Implement stress management techniques'
        ]
    },
    'Patient_Belief_Profile': {
        'actionable': True,
        'interventions': [
            'Address medication misconceptions',
            'Provide evidence-based education',
            'Explore cultural beliefs about medication',
            'Build trust in healthcare system'
        ]
    }
}
###########################################################################################################
# Function to generate clinical recommendations based on SHAP results
###########################################################################################################

def get_clinical_recommendations(shap_result, patient_data):
    """
    Generate actionable clinical recommendations based on SHAP results.

    Args:
        shap_result: SHAP computation results
        patient_data: Patient input data

    Returns:
        Dictionary with prioritized recommendations
    """
    if not shap_result:
        return None

    contributions = shap_result['feature_contributions']

    # Get top risk factors (positive SHAP values)
    risk_factors = [(name, val) for name, val in contributions.items() if val > 0]
    risk_factors.sort(key=lambda x: x[1], reverse=True)

    # Get top protective factors (negative SHAP values)
    protective_factors = [(name, val) for name, val in contributions.items() if val < 0]
    protective_factors.sort(key=lambda x: x[1])  # Most negative first

    recommendations = {
        'primary_focus': None,
        'immediate_actions': [],
        'long_term_strategies': [],
        'monitoring_points': [],
        'patient_education': []
    }

    # Process risk factors
    for factor_name, impact in risk_factors[:3]:  # Top 3 risk factors
        if factor_name in CLINICAL_INTERVENTIONS:
            intervention_info = CLINICAL_INTERVENTIONS[factor_name]

            if intervention_info['actionable']:
                if recommendations['primary_focus'] is None:
                    recommendations['primary_focus'] = {
                        'factor': factor_name,
                        'impact': impact,
                        'interventions': intervention_info['interventions'][:2]  # Top 2 interventions
                    }

                # Add immediate actions
                recommendations['immediate_actions'].extend(intervention_info['interventions'][:3])
            else:
                # Handle non-actionable factors
                recommendations['long_term_strategies'].append({
                    'factor': factor_name,
                    'reason': intervention_info['reason'],
                    'alternative': intervention_info['alternative_focus']
                })

    # Process protective factors
    for factor_name, impact in protective_factors[:2]:  # Top 2 protective factors
        if factor_name in CLINICAL_INTERVENTIONS:
            intervention_info = CLINICAL_INTERVENTIONS[factor_name]

            if intervention_info['actionable']:
                recommendations['monitoring_points'].append({
                    'factor': factor_name,
                    'strength': abs(impact),
                    'maintenance': f"Maintain current {factor_name.lower().replace('_', ' ')} practices"
                })

    # Add general patient education based on risk profile
    if risk_factors:
        recommendations['patient_education'].extend([
            "Provide clear medication instructions and expectations",
            "Educate on importance of consistent medication adherence",
            "Address common medication misconceptions",
            "Teach self-monitoring techniques for medication effects"
        ])

    return recommendations

# --- 1. Load the saved model ---
@st.cache_resource
def load_model():
    try:
        return joblib.load(r'model\adherence_prediction_pipeline.pkl')
    except FileNotFoundError:
        st.error("Model file not found. Please make sure the model is in the right directory.")
        st.stop()

model = load_model()

# --- Cache SHAP processor for better performance ---
@st.cache_resource
def get_shap_processor():
    """Cache SHAP processor to avoid reinitialization"""
    feature_names = ['Age', 'Comorbidity_Count', 'BMI', 'Missed_Appointments_Last_12_Months',
                     'App_Logins_Per_Week', 'Has_Unhealthy_Diet', 'Is_Physically_Inactive',
                     'Reports_Cost_Barrier', 'Experienced_Stock_Out', 'Estate', 'Income_Level',
                     'Education_Level', 'Marital_Status', 'Primary_Diagnosis', 'Comorbidities_List',
                     'Patient_Belief_Profile', 'Family_Support_Level', 'Self_Reported_Mood',
                     'Reason_for_Missed_Dose', 'Gender', 'BMI_Category']
    return SHAPProcessor(model, feature_names)

shap_processor = get_shap_processor()

# --- 2. Set up the web app's title and a brief description ---
st.set_page_config(
    page_title="MedAdhereKe",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="logo.jpeg"
)

# Initialize session state for collecting SHAP results
if 'shap_results_history' not in st.session_state:
    st.session_state.shap_results_history = []

# Add button to clear SHAP results history
if st.sidebar.button('🔄 Clear SHAP History', help='Clear stored SHAP results to start fresh'):
    st.session_state.shap_results_history = []
    st.sidebar.success('SHAP history cleared!')
    st.rerun()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
    }
    .info-box {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border: 1px solid #2196F3;
        border-radius: 0.5rem;
        padding: 1.2rem;
        margin-bottom: 1rem;
        color: #0D47A1;
        font-size: 1.1em;
        line-height: 1.6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box-adherent {
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 50%, #A5D6A7 100%);
        border: 2px solid #4CAF50;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
        position: relative;
        overflow: hidden;
    }
    .prediction-box-adherent::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #4CAF50, #66BB6A, #81C784);
    }
    .prediction-box-nonadherent {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 50%, #EF9A9A 100%);
        border: 2px solid #F44336;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(244, 67, 54, 0.2);
        position: relative;
        overflow: hidden;
    }
    .prediction-box-nonadherent::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #F44336, #EF5350, #E57373);
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .feature-importance {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App header with custom styling
# st.markdown('<p class="main-header">Smart Medication Adherence Predictor 💊</p>', unsafe_allow_html=True)

# Add logo next to the header
col1, col2 = st.columns([1, 3])
with col1:
    st.image("logo.jpeg", width=960)
with col2:
    st.header("MedAdhereKe")

    # App description in a nice info box
st.markdown("""
<div class="info-box">
    <p>This application predicts the likelihood of a patient adhering to their medication regimen for chronic diseases.
    Please provide the patient's details on the left to get a prediction.</p>
    <p><strong>How it works:</strong> Our ML model analyzes patient characteristics and behaviors to estimate adherence probability.</p>
</div>
""", unsafe_allow_html=True)

# --- 3. Create the user input form in the sidebar ---
st.sidebar.markdown('<p class="sidebar-header">Patient Details</p>', unsafe_allow_html=True)

# We will organize the form into collapsible sections
def user_input_features():
    # Basic Information Section
    with st.sidebar.expander("📋 Basic Information", expanded=False):
        age = st.slider('Age', 18, 100, st.session_state.get('profile_age', 50))
        gender = st.radio('Gender', ('Male', 'Female', 'Other'), index=['Male', 'Female', 'Other'].index(st.session_state.get('profile_gender', 'Female')))

        col1, col2 = st.columns(2)
        with col1:
            height = st.number_input("Height (m)", min_value=0.01, step=0.01, format="%.2f", value=st.session_state.get('profile_height', 1.70))
        with col2:
            weight = st.number_input("Weight (kg)", min_value=0.01, step=0.1, format="%.1f", value=st.session_state.get('profile_weight', 70.0))
        
        try:
            bmi = round(weight/height**2, 2)
            st.info(f"BMI: {bmi}")
            
            # Add BMI Category based on calculated BMI
            if bmi < 18.5:
                bmi_category = "Underweight"
            elif 18.5 <= bmi < 25:
                bmi_category = "Normal"
            elif 25 <= bmi < 30:
                bmi_category = "Overweight"
            else:
                bmi_category = "Obese"
            st.caption(f"BMI Category: {bmi_category}")
        except ZeroDivisionError:
            bmi = 0
            bmi_category = "Unknown"
    
    # Health Status Section
    with st.sidebar.expander("🏥 Health Status", expanded=False):
        diagnosis_options = ['Hypertension', 'Diabetes', 'Heart Disease', 'Asthma', 'COPD', 'Depression', 'Anxiety', 'Other']
        primary_diagnosis = st.selectbox('Primary Diagnosis', diagnosis_options,
                                       index=diagnosis_options.index(st.session_state.get('profile_primary_diagnosis', 'Hypertension')))

        comorbidity_options = ['None', 'Diabetes', 'Hypertension', 'Heart Disease', 'Asthma', 'COPD', 'Depression', 'Anxiety', 'Other']
        default_comorbidities = st.session_state.get('profile_comorbidities_list', ['None'])
        if isinstance(default_comorbidities, str):
            default_comorbidities = [default_comorbidities] if default_comorbidities != 'None' else ['None']
        comorbidities_list = st.multiselect('Comorbidities', comorbidity_options, default=default_comorbidities)

        # Calculate comorbidity count from multiselect
        if 'None' in comorbidities_list and len(comorbidities_list) > 1:
            # Remove 'None' if other conditions are selected
            comorbidities_list = [c for c in comorbidities_list if c != 'None']
        elif 'None' in comorbidities_list and len(comorbidities_list) == 1:
            comorbidity_count = 0
        else:
            comorbidity_count = len(comorbidities_list)

        # Display comorbidity count (read-only, automatically calculated)
        st.number_input('Number of Comorbidities (Auto-calculated)', min_value=0, max_value=10, value=comorbidity_count, disabled=True,
                       help="This value is automatically calculated from your comorbidity selections above.")

        # Convert multiselect to string
        comorbidities_str = ', '.join(comorbidities_list) if comorbidities_list and 'None' not in comorbidities_list else 'None'

        has_unhealthy_diet = st.radio('Has Unhealthy Diet?', ('Yes', 'No'),
                                    index=['Yes', 'No'].index(st.session_state.get('profile_has_unhealthy_diet', 'No')))
        is_physically_inactive = st.radio('Is Physically Inactive?', ('Yes', 'No'),
                                        index=['Yes', 'No'].index(st.session_state.get('profile_is_physically_inactive', 'No')))
    
    # Behavioral Factors Section
    with st.sidebar.expander("🧠 Behavioral Factors", expanded=False):
        app_logins = st.slider('App Logins Per Week', 0, 7, st.session_state.get('profile_app_logins', 3))
        missed_appts = st.slider('Missed Appointments (Last 12 Months)', 0, 10, st.session_state.get('profile_missed_appts', 1))

        mood_options = ('Good', 'Okay', 'Stressed', 'Sad')
        mood = st.selectbox('Self-Reported Mood', mood_options,
                          index=mood_options.index(st.session_state.get('profile_mood', 'Good')))

        belief_options = ('Trusts Clinical System', 'Stops Meds When Well', 'Integrates Traditional Medicine')
        belief_profile = st.selectbox('Patient Belief Profile', belief_options,
                                    index=belief_options.index(st.session_state.get('profile_belief_profile', 'Trusts Clinical System')))

        reason_options = ('None', 'Forgot', 'Side Effects', 'Cost', 'Felt Better', 'Ran Out', 'Other')
        reason_for_missed_dose = st.selectbox('Reason for Missed Dose (if any)', reason_options,
                                           index=reason_options.index(st.session_state.get('profile_reason_for_missed_dose', 'None')))

    # Social & Economic Factors Section
    with st.sidebar.expander("🏘️ Social & Economic Factors", expanded=False):
        support_options = ('High', 'Moderate', 'Low')
        family_support = st.selectbox('Family Support Level', support_options,
                                    index=support_options.index(st.session_state.get('profile_family_support', 'High')))

        cost_barrier = st.radio('Patient Reports Cost Barrier?', ('Yes', 'No'),
                              index=['Yes', 'No'].index(st.session_state.get('profile_cost_barrier', 'No')))

        experienced_stock_out = st.radio('Experienced Medication Stock Out?', ('Yes', 'No'),
                                       index=['Yes', 'No'].index(st.session_state.get('profile_experienced_stock_out', 'No')))

        estate_options = ('Low', 'Medium', 'High')
        estate = st.selectbox('Estate Type', estate_options,
                            index=estate_options.index(st.session_state.get('profile_estate', 'High')))

        income_options = ('Low', 'Medium', 'High')
        income_level = st.selectbox('Income Level', income_options,
                                  index=income_options.index(st.session_state.get('profile_income_level', 'High')))

        education_options = ('None', 'Primary', 'Secondary', 'Tertiary', 'University')
        education_level = st.selectbox('Education Level', education_options,
                                     index=education_options.index(st.session_state.get('profile_education_level', 'University')))

        marital_options = ('Single', 'Married', 'Divorced', 'Widowed', 'Separated')
        marital_status = st.selectbox('Marital Status', marital_options,
                                    index=marital_options.index(st.session_state.get('profile_marital_status', 'Married')))

    # Store all inputs in a dictionary
    data = {
        'Age': age,
        'Comorbidity_Count': comorbidity_count,
        'BMI': bmi,
        'BMI_Category': bmi_category,
        'Missed_Appointments_Last_12_Months': missed_appts,
        'App_Logins_Per_Week': app_logins,
        'Has_Unhealthy_Diet': True if has_unhealthy_diet == 'Yes' else False,
        'Is_Physically_Inactive': True if is_physically_inactive == 'Yes' else False,
        'Reports_Cost_Barrier': True if cost_barrier == 'Yes' else False,
        'Experienced_Stock_Out': True if experienced_stock_out == 'Yes' else False,
        'Estate': estate,
        'Income_Level': income_level,
        'Education_Level': education_level,
        'Marital_Status': marital_status,
        'Primary_Diagnosis': primary_diagnosis,
        'Comorbidities_List': comorbidities_str,
        'Patient_Belief_Profile': belief_profile,
        'Family_Support_Level': family_support,
        'Self_Reported_Mood': mood,
        'Reason_for_Missed_Dose': reason_for_missed_dose,
        'Gender': gender
    }

    # Create a DataFrame with all the model columns
    model_columns = ['Age', 'Comorbidity_Count', 'BMI', 'Missed_Appointments_Last_12_Months',
                     'App_Logins_Per_Week', 'Has_Unhealthy_Diet', 'Is_Physically_Inactive',
                     'Reports_Cost_Barrier', 'Experienced_Stock_Out', 'Estate', 'Income_Level',
                     'Education_Level', 'Marital_Status', 'Primary_Diagnosis', 'Comorbidities_List',
                     'Patient_Belief_Profile', 'Family_Support_Level', 'Self_Reported_Mood',
                     'Reason_for_Missed_Dose', 'Gender', 'BMI_Category']

    # Create DataFrame with the collected data
    input_df = pd.DataFrame(data, index=[0])
    
    # Ensure all columns are in the right order
    input_df = input_df[model_columns]
            
    return input_df

# Get the user input
input_df = user_input_features()

# --- 4. Make predictions and display the results ---
# Add a button to trigger the prediction
predict_button = st.sidebar.button('Predict Adherence', use_container_width=True)

# Add quick patient profile selection
st.sidebar.markdown("---")
# st.sidebar.markdown('<p class="sidebar-header">Quick Patient Profiles</p>', unsafe_allow_html=True)

# if st.sidebar.button("Load High-Risk Patient Profile"):
#     # This would be implemented to pre-fill the form with a high-risk patient
#     st.session_state['load_high_risk'] = True
#     st.rerun()

# if st.sidebar.button("Load Low-Risk Patient Profile"):
#     # This would be implemented to pre-fill the form with a low-risk patient
#     st.session_state['load_low_risk'] = True
#     st.rerun()

# Main content area
tab1, tab2, tab3 = st.tabs(["Prediction", "Patient Details", "About"])

with tab1:
    if predict_button:
        # Show a spinner while predicting
        with st.spinner('Analyzing patient data...'):
            # Add a small delay to simulate processing
            time.sleep(0.5)
            
            # The pipeline.predict() method will automatically handle all the preprocessing
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            
            st.markdown('<p class="sub-header">Prediction Result</p>', unsafe_allow_html=True)

            # Display the prediction in a user-friendly way
            if prediction[0] == 'Adherent':
                st.markdown(f"""
                <div class="prediction-box-adherent">
                    <h2>✅ ADHERENT</h2>
                    <p>The model predicts this patient will likely adhere to their medication regimen.</p>
                    <h3>Confidence: {prediction_proba[0][0]:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box-nonadherent">
                    <h2>⚠️ NON-ADHERENT</h2>
                    <p>The model predicts this patient may not adhere to their medication regimen.</p>
                    <h3>Risk: {prediction_proba[0][1]:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)

            # Compute SHAP values for model explainability FIRST
            shap_result = None
            with st.spinner('🔍 Analyzing patient factors...'):
                try:
                    # Prepare data for SHAP
                    processed_input = prepare_shap_data(input_df)
                    feature_names = input_df.columns.tolist()

                    # Compute SHAP values with progress indicator
                    progress_bar = st.progress(0)
                    progress_bar.progress(25, text="Preparing data...")
                    progress_bar.progress(50, text="Computing explanations...")

                    # Use cached SHAP processor for better performance
                    shap_result = shap_processor.compute_shap_values(processed_input)
                    progress_bar.progress(100, text="Analysis complete!")
                    progress_bar.empty()

                    # Store SHAP result in session state for global summary plot
                    if shap_result:
                        st.session_state.shap_results_history.append(shap_result)

                except Exception as e:
                    st.error(f"❌ Error computing SHAP values: {str(e)}")
                    st.info("💡 The prediction is still valid, but detailed explanations are unavailable.")
                    shap_result = None

            # Create two columns for visualization
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Adherence Probability")

                # Create a gauge chart for the adherence probability
                fig, ax = plt.subplots(figsize=(4, 4))

                # Create a gauge chart
                adherent_prob = prediction_proba[0][0]
                non_adherent_risk = prediction_proba[0][1]

                # Create pie chart as gauge with consistent colors
                # Adherent is always green, Non-Adherent is always red
                ax.pie([adherent_prob, non_adherent_risk],
                       colors=['#4CAF50', '#F44336'],  # Green for Adherent, Red for Non-Adherent
                       startangle=90,
                       counterclock=False,
                       labels=['Adherent', 'Non-Adherent'],
                       autopct='%1.1f%%',
                       wedgeprops={'width': 0.5, 'edgecolor': 'w'})

                ax.set_title('Adherence Probability')
                st.pyplot(fig)

                # Add prediction confidence metrics below the chart
                st.markdown("**Prediction Confidence**")
                if prediction[0] == 'Adherent':
                    st.metric("Adherence Confidence", f"{prediction_proba[0][0]:.1%}")
                    st.metric("Non-Adherence Risk", f"{prediction_proba[0][1]:.1%}")
                else:
                    st.metric("Non-Adherence Risk", f"{prediction_proba[0][1]:.1%}")
                    st.metric("Adherence Confidence", f"{prediction_proba[0][0]:.1%}")

                # Add interpretation guide
                st.markdown("**Quick Interpretation**")
                if prediction[0] == 'Adherent':
                    st.success("✅ **Low Risk** - Patient likely to adhere to medication regimen")
                else:
                    st.error("⚠️ **High Risk** - Patient may need additional support")

                # Add model information
                with st.expander("ℹ️ Model Details"):
                    st.markdown("""
                    **Algorithm**: LightGBM Classifier
                    **Accuracy**: >90% on validation data
                    **Features**: 21 patient characteristics
                    **Last Updated**: Model trained on recent patient data
                    """)

            with col2:
                st.subheader("Key Factors Analysis")

                # SHAP-based risk factor analysis
                if shap_result:
                    contributions = shap_result['feature_contributions']

                    # Get top risk factors (positive SHAP values)
                    risk_factors = [(name, val) for name, val in contributions.items() if val > 0]
                    risk_factors.sort(key=lambda x: x[1], reverse=True)

                    # Get top protective factors (negative SHAP values)
                    protective_factors = [(name, val) for name, val in contributions.items() if val < 0]
                    protective_factors.sort(key=lambda x: x[1])  # Most negative first

                    # Create two columns for risk and protective factors
                    factor_col1, factor_col2 = st.columns(2)

                    with factor_col1:
                        st.markdown("**🔴 Risk Factors**")
                        st.markdown("*Factors increasing non-adherence risk:*")
                        if risk_factors:
                            for factor_name, impact in risk_factors[:3]:
                                display_name = factor_name.replace('_', ' ')
                                st.markdown(f"• **{display_name}**: +{impact:.3f}")
                        else:
                            st.info("No significant risk factors identified")

                    with factor_col2:
                        st.markdown("**🟢 Protective Factors**")
                        st.markdown("*Factors supporting adherence:*")
                        if protective_factors:
                            for factor_name, impact in protective_factors[:3]:
                                display_name = factor_name.replace('_', ' ')
                                st.markdown(f"• **{display_name}**: {impact:.3f}")
                        else:
                            st.info("No significant protective factors identified")

                    st.markdown("---")

                    # Generate clinical recommendations
                    clinical_recs = get_clinical_recommendations(shap_result, input_df)

                    # Display clinical insights with actionable recommendations at the bottom
                    st.markdown("**💡 Clinical Action Plan**")

                    # First row: Primary Focus and Immediate Actions
                    st.markdown("### Priority Actions")
                    priority_col1, priority_col2 = st.columns(2)

                    with priority_col1:
                        st.markdown("**🎯 Primary Focus**")
                        if clinical_recs and clinical_recs['primary_focus']:
                            primary = clinical_recs['primary_focus']
                            factor_display = primary['factor'].replace('_', ' ')
                            st.markdown(f"**{factor_display}**")
                            st.markdown("*Key interventions:*")
                            for intervention in primary['interventions']:
                                st.markdown(f"• {intervention}")
                        else:
                            st.info("No primary focus identified")

                    with priority_col2:
                        st.markdown("**⚡ Immediate Actions**")
                        if clinical_recs and clinical_recs['immediate_actions']:
                            unique_actions = list(set(clinical_recs['immediate_actions']))[:4]
                            for action in unique_actions:
                                st.markdown(f"• {action}")
                        else:
                            st.info("No immediate actions recommended")

                    # Second row: Long-term Strategies and Monitoring Points
                    st.markdown("### Ongoing Management")
                    ongoing_col1, ongoing_col2 = st.columns(2)

                    with ongoing_col1:
                        st.markdown("**📈 Long-term Strategies**")
                        if clinical_recs and clinical_recs['long_term_strategies']:
                            for strategy in clinical_recs['long_term_strategies']:
                                st.markdown(f"• *{strategy['factor'].replace('_', ' ')}*: {strategy['alternative']}")
                        else:
                            st.info("No long-term strategies needed")

                    with ongoing_col2:
                        st.markdown("**📊 Monitoring Points**")
                        if clinical_recs and clinical_recs['monitoring_points']:
                            for point in clinical_recs['monitoring_points']:
                                st.markdown(f"• {point['maintenance']}")
                        else:
                            st.info("No specific monitoring points identified")

                    st.markdown("---")
                    st.markdown("*📋 Detailed SHAP analysis available in the visualizations below*")
                else:
                    st.info("SHAP analysis will provide detailed risk factor insights below.")

            # Display SHAP visualizations
            if shap_result:
                # Debug: Show current SHAP results count
                st.write(f"Debug: {len(st.session_state.shap_results_history)} SHAP results stored")

                # Pass collected SHAP results for global summary plot
                shap_results_for_global = st.session_state.shap_results_history if len(st.session_state.shap_results_history) >= 2 else None
                create_shap_visualization_section(shap_result, shap_results_for_global)
            else:
                st.warning("SHAP computation failed. Using fallback feature importance.")

                # Fallback: Use model's built-in feature importance
                if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                    estimator = model.named_steps['classifier']
                    if hasattr(estimator, 'feature_importances_'):
                        importances = estimator.feature_importances_
                        feature_names_fallback = input_df.columns

                        importance_df = pd.DataFrame({
                            'Feature': feature_names_fallback,
                            'Importance': importances[:len(feature_names_fallback)]
                        }).sort_values('Importance', ascending=False).head(10)

                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(importance_df['Feature'], importance_df['Importance'])
                        ax.set_xlabel('Feature Importance')
                        ax.set_title('Top 10 Most Important Features')
                        plt.tight_layout()
                        st.pyplot(fig)
    else:
        # Display placeholder content when no prediction has been made
        st.info("👈 Fill in the patient details and click 'Predict Adherence' to see results.")
        
        # Add some example information or guidance
        st.markdown("""
        ### How to use this tool:
        1. Enter all required patient information in the sidebar
        2. Click the 'Predict Adherence' button
        3. Review the prediction and recommendations
        
        ### What this prediction means:
        - **Adherent**: Patient is likely to take medication as prescribed
        - **Non-Adherent**: Patient may need additional support or interventions
        
        ### Quick Start:
        Use the "Load High-Risk Patient Profile" or "Load Low-Risk Patient Profile" buttons in the sidebar to see example predictions.
        """)

with tab2:
    st.subheader("Patient Input Summary")
    
    # Create a more readable display of patient details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Demographics")
        st.write(f"**Age:** {input_df['Age'].values[0]}")
        st.write(f"**Gender:** {input_df['Gender'].values[0]}")
        st.write(f"**BMI:** {input_df['BMI'].values[0]} ({input_df['BMI_Category'].values[0]})")
        st.write(f"**Marital Status:** {input_df['Marital_Status'].values[0]}")
        st.write(f"**Education Level:** {input_df['Education_Level'].values[0]}")
        st.write(f"**Income Level:** {input_df['Income_Level'].values[0]}")
        
        st.markdown("#### Health Status")
        st.write(f"**Primary Diagnosis:** {input_df['Primary_Diagnosis'].values[0]}")
        st.write(f"**Comorbidities:** {input_df['Comorbidities_List'].values[0]}")
        st.write(f"**Comorbidity Count:** {input_df['Comorbidity_Count'].values[0]}")
    
    with col2:
        st.markdown("#### Behavioral Factors")
        st.write(f"**App Logins Per Week:** {input_df['App_Logins_Per_Week'].values[0]}")
        st.write(f"**Missed Appointments:** {input_df['Missed_Appointments_Last_12_Months'].values[0]}")
        st.write(f"**Self-Reported Mood:** {input_df['Self_Reported_Mood'].values[0]}")
        st.write(f"**Patient Belief Profile:** {input_df['Patient_Belief_Profile'].values[0]}")
        st.write(f"**Reason for Missed Dose:** {input_df['Reason_for_Missed_Dose'].values[0]}")
        st.write(f"**Has Unhealthy Diet:** {'Yes' if input_df['Has_Unhealthy_Diet'].values[0] else 'No'}")
        st.write(f"**Is Physically Inactive:** {'Yes' if input_df['Is_Physically_Inactive'].values[0] else 'No'}")
        
        st.markdown("#### Social & Economic Factors")
        st.write(f"**Family Support Level:** {input_df['Family_Support_Level'].values[0]}")
        st.write(f"**Reports Cost Barrier:** {'Yes' if input_df['Reports_Cost_Barrier'].values[0] else 'No'}")
        st.write(f"**Experienced Stock Out:** {'Yes' if input_df['Experienced_Stock_Out'].values[0] else 'No'}")
        st.write(f"**Estate Type:** {input_df['Estate'].values[0]}")
    
    # Option to view raw data
    if st.checkbox("Show raw data"):
        st.dataframe(input_df)

with tab3:
    st.subheader("About This Tool")
    
    st.markdown("""
    ### Smart Medication Adherence Predictor
    
    This tool uses machine learning to predict whether a patient is likely to adhere to their 
    prescribed medication regimen for chronic diseases.
    
    #### Model Information
    - **Algorithm**: LGBMClassifier (Light Gradient Boosting Machine)
    - **Features**: 21 patient characteristics including demographics, health status, behavioral factors, and social determinants
    - **Target**: Binary classification (Adherent vs. Non-Adherent)
    - **Performance**: >90% accuracy on validation data
    
    #### How It Works
    The model analyzes patterns in patient data to identify factors that correlate with medication adherence.
    It has been trained on historical patient data where adherence outcomes were known.
    
    #### Intended Use
    This tool is designed to help healthcare providers identify patients who may benefit from 
    additional support or interventions to improve medication adherence. It should be used as 
    a decision support tool, not as a replacement for clinical judgment.
    
    #### Limitations
    - Predictions are based on statistical patterns and may not account for individual circumstances
    - The model should be periodically retrained with new data to maintain accuracy
    - Always combine these predictions with clinical expertise and patient communication
    
    
    """)
    # #### Data Privacy
    # This application processes all data locally in your browser. No patient information is stored or transmitted.
    
    # Add a section about medication adherence importance
    st.markdown("""
    ### Importance of Medication Adherence
    
    Medication non-adherence is associated with:
    - Poorer health outcomes
    - Increased hospitalizations
    - Higher healthcare costs
    - Disease progression
    - Reduced quality of life
    
    Studies show that 20-30% of medication prescriptions are never filled, and approximately 
    50% of medications for chronic diseases are not taken as prescribed.
    """)
    
    # Add information about the capstone project
    st.markdown("""
    ### About This Project
    
    This application was developed as part of a capstone project focused on improving medication adherence 
    for patients with chronic conditions. The project aimed to:
    
    1. Develop a predictive model for medication adherence
    2. Create a user-friendly interface for healthcare providers
    3. Generate actionable insights to guide interventions
    4. Demonstrate the potential of machine learning in improving healthcare outcomes
    
    The final mobile application would include additional features such as:
    - Patient-facing reminders and educational content
    - Integration with pharmacy systems for refill tracking
    - Secure messaging between patients and providers
    - Longitudinal adherence tracking and reporting
    """)

# Add a footer
st.markdown("""
---
<p style="text-align: center; color: #666;">
Smart Medication Adherence Predictor | Developed for Healthcare Providers | © 2025
</p>
""", unsafe_allow_html=True)

# Define patient profiles
HIGH_RISK_PROFILE = {
    'age': 65,
    'gender': 'Male',
    'height': 1.75,
    'weight': 85.0,
    'primary_diagnosis': 'Diabetes',
    'comorbidity_count': 3,
    'comorbidities_list': ['Hypertension', 'Heart Disease'],
    'has_unhealthy_diet': 'Yes',
    'is_physically_inactive': 'Yes',
    'app_logins': 1,
    'missed_appts': 5,
    'mood': 'Stressed',
    'belief_profile': 'Stops Meds When Well',
    'reason_for_missed_dose': 'Forgot',
    'family_support': 'Low',
    'cost_barrier': 'Yes',
    'experienced_stock_out': 'Yes',
    'estate': 'Low',
    'income_level': 'Low',
    'education_level': 'Primary',
    'marital_status': 'Widowed'
}

LOW_RISK_PROFILE = {
    'age': 45,
    'gender': 'Female',
    'height': 1.65,
    'weight': 60.0,
    'primary_diagnosis': 'Hypertension',
    'comorbidity_count': 1,
    'comorbidities_list': ['None'],
    'has_unhealthy_diet': 'No',
    'is_physically_inactive': 'No',
    'app_logins': 5,
    'missed_appts': 0,
    'mood': 'Good',
    'belief_profile': 'Trusts Clinical System',
    'reason_for_missed_dose': 'None',
    'family_support': 'High',
    'cost_barrier': 'No',
    'experienced_stock_out': 'No',
    'estate': 'High',
    'income_level': 'High',
    'education_level': 'University',
    'marital_status': 'Married'
}

# Handle session state for loading patient profiles
if 'load_high_risk' in st.session_state and st.session_state['load_high_risk']:
    # Load high-risk profile into session state
    for key, value in HIGH_RISK_PROFILE.items():
        st.session_state[f'profile_{key}'] = value
    st.session_state['load_high_risk'] = False
    st.rerun()

if 'load_low_risk' in st.session_state and st.session_state['load_low_risk']:
    # Load low-risk profile into session state
    for key, value in LOW_RISK_PROFILE.items():
        st.session_state[f'profile_{key}'] = value
    st.session_state['load_low_risk'] = False
    st.rerun()
