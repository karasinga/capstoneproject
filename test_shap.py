"""
Test script for SHAP functionality
"""

import pandas as pd
import numpy as np
import joblib
from shap_utils import SHAPProcessor, compute_shap_values, prepare_shap_data

def test_shap():
    try:
        # Load the model
        model = joblib.load(r'model\adherence_prediction_pipeline.pkl')
        print("Model loaded successfully")

        # Create test data
        test_data = {
            'Age': [45],
            'Comorbidity_Count': [2],
            'BMI': [28.5],
            'BMI_Category': ['Overweight'],
            'Missed_Appointments_Last_12_Months': [3],
            'App_Logins_Per_Week': [2],
            'Has_Unhealthy_Diet': [True],
            'Is_Physically_Inactive': [False],
            'Reports_Cost_Barrier': [False],
            'Experienced_Stock_Out': [False],
            'Estate': ['Medium'],
            'Income_Level': ['Medium'],
            'Education_Level': ['Secondary'],
            'Marital_Status': ['Married'],
            'Primary_Diagnosis': ['Hypertension'],
            'Comorbidities_List': ['Diabetes'],
            'Patient_Belief_Profile': ['Trusts Clinical System'],
            'Family_Support_Level': ['High'],
            'Self_Reported_Mood': ['Good'],
            'Reason_for_Missed_Dose': ['None'],
            'Gender': ['Female']
        }

        input_df = pd.DataFrame(test_data)
        print("Test data created")

        # Prepare data
        processed_input = prepare_shap_data(input_df)
        print("Data prepared for SHAP")

        # Get feature names
        feature_names = input_df.columns.tolist()

        # Test SHAP computation
        shap_result = compute_shap_values(model, processed_input, feature_names)

        if shap_result:
            print("SHAP computation successful!")
            print(f"Prediction class: {shap_result['prediction_class']}")
            print(f"Expected value: {shap_result['expected_value']:.4f}")
            print(f"Top 3 features by importance:")
            contributions = shap_result['feature_contributions']
            sorted_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            for i, (feature, value) in enumerate(sorted_features[:3]):
                print(".4f")
        else:
            print("SHAP computation failed")

    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_shap()
