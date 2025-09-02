"""
SHAP Utilities for Medication Adherence Prediction

This module provides core SHAP (SHapley Additive exPlanations) functionality
for explaining individual patient predictions in the medication adherence model.
"""

import numpy as np
import pandas as pd
import shap
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SHAPProcessor:
    """
    A class to handle SHAP computations and processing for the medication adherence model.
    """

    def __init__(self, model_pipeline, feature_names: List[str]):
        """
        Initialize the SHAP processor.

        Args:
            model_pipeline: The trained scikit-learn pipeline
            feature_names: List of feature names in the correct order
        """
        self.model_pipeline = model_pipeline
        self.feature_names = feature_names
        self.explainer = None
        self.background_data = None

    def prepare_background_data(self, background_size: int = 20) -> np.ndarray:
        """
        Prepare background data for SHAP explainer initialization.

        Args:
            background_size: Number of background samples to use

        Returns:
            Background data array for SHAP
        """
        try:
            # Simple approach: create random background data with reasonable bounds
            # This is the most robust approach that avoids pipeline complications
            n_features = len(self.feature_names) if self.feature_names else 21  # Default to 21 features

            # Create background data with reasonable ranges for each feature type
            background_data = np.random.random((background_size, n_features))

            # Scale some features to more reasonable ranges
            background_data[:, 0] = background_data[:, 0] * 70 + 18  # Age: 18-88
            background_data[:, 1] = background_data[:, 1] * 10  # Comorbidity count: 0-10
            background_data[:, 2] = background_data[:, 2] * 25 + 18  # BMI: 18-43
            background_data[:, 3] = background_data[:, 3] * 12  # Missed appointments: 0-12
            background_data[:, 4] = background_data[:, 4] * 7  # App logins: 0-7

            self.background_data = background_data
            return background_data

        except Exception as e:
            logger.error(f"Error preparing background data: {str(e)}")
            # Ultimate fallback
            self.background_data = np.random.random((10, 21))
            return self.background_data

    def initialize_explainer(self) -> bool:
        """
        Initialize the SHAP explainer with improved error handling.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if self.background_data is None:
                self.prepare_background_data()

            # Try different explainer types based on model type
            if hasattr(self.model_pipeline, 'named_steps') and 'classifier' in self.model_pipeline.named_steps:
                estimator = self.model_pipeline.named_steps['classifier']
            else:
                estimator = self.model_pipeline

            # For tree-based models, try TreeExplainer first
            if hasattr(estimator, 'predict_proba'):
                try:
                    # Try TreeExplainer first (works well with LightGBM)
                    # Create a wrapper to handle feature names issue
                    def predict_proba_wrapper(X):
                        # Ensure X is a DataFrame with proper feature names
                        if not isinstance(X, pd.DataFrame):
                            X = pd.DataFrame(X, columns=self.feature_names)
                        return estimator.predict_proba(X)

                    self.explainer = shap.TreeExplainer(estimator, self.background_data, feature_names=self.feature_names)
                    logger.info("Initialized TreeExplainer successfully")
                    return True
                except Exception as tree_error:
                    logger.warning(f"TreeExplainer failed: {str(tree_error)}")
                    # Fallback to KernelExplainer
                    try:
                        def predict_proba_wrapper(X):
                            if not isinstance(X, pd.DataFrame):
                                X = pd.DataFrame(X, columns=self.feature_names)
                            return estimator.predict_proba(X)

                        self.explainer = shap.KernelExplainer(predict_proba_wrapper, self.background_data[:50])
                        logger.info("Initialized KernelExplainer as fallback")
                        return True
                    except Exception as kernel_error:
                        logger.error(f"KernelExplainer also failed: {str(kernel_error)}")
                        return False
            else:
                # Fallback for models without predict_proba
                try:
                    def predict_wrapper(X):
                        if not isinstance(X, pd.DataFrame):
                            X = pd.DataFrame(X, columns=self.feature_names)
                        return estimator.predict(X)

                    self.explainer = shap.KernelExplainer(predict_wrapper, self.background_data[:50])
                    logger.info("Initialized KernelExplainer for regression")
                    return True
                except Exception as e:
                    logger.error(f"Failed to initialize any SHAP explainer: {str(e)}")
                    return False

        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {str(e)}")
            return False

    def compute_shap_values(self, input_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Compute SHAP values for individual prediction.

        Args:
            input_data: Input features as DataFrame

        Returns:
            Dictionary containing SHAP values and metadata
        """
        try:
            if self.explainer is None:
                if not self.initialize_explainer():
                    return None

            # Transform input data through pipeline
            processed_input = self.model_pipeline.named_steps['preprocessor'].transform(input_data)

            # Ensure processed input is numeric and handle any dtype issues
            if hasattr(processed_input, 'toarray'):  # Sparse matrix
                processed_input = processed_input.toarray()

            # Convert to float64 to ensure compatibility
            processed_input = np.array(processed_input, dtype=np.float64)

            # Handle NaN values
            if np.isnan(processed_input).any():
                processed_input = np.nan_to_num(processed_input, nan=0.0)

            # Compute SHAP values with additivity check disabled
            try:
                shap_values = self.explainer.shap_values(processed_input, check_additivity=False)
            except TypeError:
                # Older versions of SHAP don't have check_additivity parameter
                shap_values = self.explainer.shap_values(processed_input)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-class case
                shap_values_array = shap_values[1]  # Focus on positive class (Non-Adherent)
            else:
                # Binary classification case
                shap_values_array = shap_values

            # Get expected value
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

            # Create feature contributions dictionary
            feature_contributions = {}
            for i, feature_name in enumerate(self.feature_names):
                if i < len(shap_values_array[0]):
                    feature_contributions[feature_name] = float(shap_values_array[0][i])

            # Get prediction probability
            prediction_proba = self.model_pipeline.predict_proba(input_data)[0]
            prediction_class = self.model_pipeline.predict(input_data)[0]

            result = {
                'shap_values': shap_values_array[0].tolist(),
                'feature_contributions': feature_contributions,
                'expected_value': float(expected_value),
                'prediction_probability': prediction_proba.tolist(),
                'prediction_class': prediction_class,
                'feature_names': self.feature_names,
                'input_values': input_data.iloc[0].to_dict()
            }

            return result

        except Exception as e:
            logger.error(f"Error computing SHAP values: {str(e)}")
            return None

    def get_feature_importance(self, shap_result: Dict[str, Any], top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract feature importance from SHAP results.

        Args:
            shap_result: SHAP computation result
            top_n: Number of top features to return

        Returns:
            List of (feature_name, importance_score) tuples
        """
        try:
            contributions = shap_result['feature_contributions']
            sorted_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            return sorted_features[:top_n]
        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
            return []

def prepare_shap_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare input data for SHAP analysis.

    Args:
        input_df: Raw input DataFrame

    Returns:
        Processed DataFrame ready for SHAP
    """
    try:
        # Ensure data types are correct
        processed_df = input_df.copy()

        # Convert boolean columns to int
        bool_columns = ['Has_Unhealthy_Diet', 'Is_Physically_Inactive',
                       'Reports_Cost_Barrier', 'Experienced_Stock_Out']
        for col in bool_columns:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].astype(int)

        return processed_df

    except Exception as e:
        logger.error(f"Error preparing SHAP data: {str(e)}")
        return input_df

def handle_shap_errors(func):
    """
    Decorator to handle SHAP computation errors gracefully.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"SHAP computation failed: {str(e)}")
            return None
    return wrapper

@handle_shap_errors
def compute_shap_values(model_pipeline, input_data: pd.DataFrame, feature_names: List[str]) -> Optional[Dict[str, Any]]:
    """
    Convenience function to compute SHAP values with error handling.

    Args:
        model_pipeline: Trained model pipeline
        input_data: Input features
        feature_names: Feature names

    Returns:
        SHAP results dictionary or None if failed
    """
    processor = SHAPProcessor(model_pipeline, feature_names)
    return processor.compute_shap_values(input_data)

@handle_shap_errors
def get_feature_importance(shap_result: Dict[str, Any], top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Convenience function to get feature importance with error handling.

    Args:
        shap_result: SHAP computation result
        top_n: Number of top features

    Returns:
        List of feature importance tuples
    """
    processor = SHAPProcessor(None, [])
    return processor.get_feature_importance(shap_result, top_n)
