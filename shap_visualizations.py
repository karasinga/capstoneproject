"""
SHAP Visualizations for Medication Adherence Prediction

This module provides visualization functions for displaying SHAP explanations
in an intuitive and modern way for healthcare providers.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Any, Tuple
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for modern plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SHAPVisualizer:
    """
    A class to handle SHAP visualization generation for the medication adherence application.
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 5)):
        """
        Initialize the SHAP visualizer.

        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.colors = {
            'positive': '#FF6B6B',  # Red for features pushing toward non-adherence
            'negative': '#4ECDC4',  # Teal for features pushing toward adherence
            'neutral': '#95A5A6',   # Gray for neutral features
            'base': '#34495E'       # Dark blue for base value
        }

    def create_waterfall_plot(self, shap_result: Dict[str, Any], top_n: int = 8) -> plt.Figure:
        """
        Create a waterfall plot showing feature contributions to the prediction.

        Args:
            shap_result: SHAP computation results
            top_n: Number of top features to display

        Returns:
            Matplotlib figure object
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize)

            # Extract data
            contributions = shap_result['feature_contributions']
            expected_value = shap_result['expected_value']
            prediction_class = shap_result['prediction_class']

            # Sort features by absolute contribution
            sorted_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]

            # Prepare data for waterfall plot
            feature_names = [f[0] for f in sorted_features]
            shap_values = [f[1] for f in sorted_features]

            # Calculate cumulative values starting from expected value
            cumulative = expected_value
            positions = [cumulative]

            for value in shap_values:
                cumulative += value
                positions.append(cumulative)

            # Create waterfall bars
            x_positions = np.arange(len(feature_names) + 1)
            bar_values = [expected_value] + shap_values

            # Color bars based on contribution direction
            colors = [self.colors['base']]
            for value in shap_values:
                if value > 0:
                    colors.append(self.colors['positive'])
                else:
                    colors.append(self.colors['negative'])

            # Plot bars
            bars = ax.bar(x_positions, bar_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, bar_values)):
                height = bar.get_height()
                if i == 0:  # Expected value
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'Expected\n{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                else:
                    # Add only SHAP value (feature names are in x-axis labels)
                    label_text = f'{value:.3f}'
                    ax.text(bar.get_x() + bar.get_width()/2.,
                           height/2 if abs(height) > 0.05 else height + 0.01,
                           label_text, ha='center', va='center' if abs(height) > 0.05 else 'bottom',
                           fontsize=8, color='white', fontweight='bold')

            # Add feature labels
            ax.set_xticks(x_positions)
            labels = ['Expected\nValue'] + [name.replace('_', '\n') for name in feature_names]
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

            # Add horizontal line at zero
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            # Add final prediction line
            final_value = positions[-1]
            ax.axhline(y=final_value, color=self.colors['base'], linestyle='--', alpha=0.7)
            ax.text(len(feature_names), final_value + 0.01, f'Final: {final_value:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

            # Styling
            ax.set_title(f'SHAP Waterfall Plot - Prediction: {prediction_class}',
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel('SHAP Value (Contribution to Prediction)', fontsize=11)
            ax.grid(axis='y', alpha=0.3)

            # Add legend
            legend_elements = [
                plt.Rectangle((0,0),1,1, facecolor=self.colors['positive'], alpha=0.7, label='Increases Risk'),
                plt.Rectangle((0,0),1,1, facecolor=self.colors['negative'], alpha=0.7, label='Decreases Risk'),
                plt.Rectangle((0,0),1,1, facecolor=self.colors['base'], alpha=0.7, label='Expected Value')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error creating waterfall plot: {str(e)}")
            return self._create_error_plot("Waterfall Plot Error")

    def create_force_plot(self, shap_result: Dict[str, Any], top_n: int = 6) -> plt.Figure:
        """
        Create a force plot showing the overall prediction explanation.

        Args:
            shap_result: SHAP computation results
            top_n: Number of top features to display

        Returns:
            Matplotlib figure object
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 5))

            # Extract data
            contributions = shap_result['feature_contributions']
            expected_value = shap_result['expected_value']
            prediction_class = shap_result['prediction_class']

            # Sort features by absolute contribution
            sorted_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]

            # Prepare data
            feature_names = [f[0] for f in sorted_features]
            shap_values = [f[1] for f in sorted_features]

            # Create horizontal force plot
            y_pos = np.arange(len(feature_names))

            # Plot bars
            colors = [self.colors['positive'] if val > 0 else self.colors['negative'] for val in shap_values]
            bars = ax.barh(y_pos, shap_values, color=colors, alpha=0.7, height=0.6)

            # Add value labels (SHAP values only, feature names are on y-axis)
            for i, (bar, value) in enumerate(zip(bars, shap_values)):
                width = bar.get_width()
                label_text = f'{value:.3f}'

                if width > 0:
                    # Positive values (risk factors) - place to the right
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                           label_text, ha='left', va='center', fontsize=8, fontweight='bold')
                else:
                    # Negative values (protective factors) - place to the right of the bar end
                    ax.text(width - 0.01, bar.get_y() + bar.get_height()/2,
                           label_text, ha='right', va='center', fontsize=8, fontweight='bold')

            # Add expected value line
            ax.axvline(x=expected_value, color=self.colors['base'], linestyle='--',
                      linewidth=2, alpha=0.8, label=f'Expected: {expected_value:.3f}')

            # Add final prediction point
            final_value = expected_value + sum(shap_values)
            ax.scatter([final_value], [len(feature_names)-1], color='red', s=100, zorder=5,
                      label=f'Final: {final_value:.3f}')

            # Styling
            ax.set_yticks(y_pos)
            ax.set_yticklabels([name.replace('_', ' ') for name in feature_names], fontsize=10)
            ax.set_xlabel('SHAP Value', fontsize=11)
            ax.set_title(f'SHAP Force Plot - Top {top_n} Features',
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='x', alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error creating force plot: {str(e)}")
            return self._create_error_plot("Force Plot Error")

    def create_feature_bar_chart(self, shap_result: Dict[str, Any], top_n: int = 10) -> plt.Figure:
        """
        Create a bar chart showing feature importance rankings.

        Args:
            shap_result: SHAP computation results
            top_n: Number of top features to display

        Returns:
            Matplotlib figure object
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize)

            # Extract data
            contributions = shap_result['feature_contributions']

            # Sort features by absolute contribution
            sorted_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]

            # Prepare data
            feature_names = [f[0] for f in sorted_features]
            shap_values = [abs(f[1]) for f in sorted_features]  # Use absolute values for importance

            # Create horizontal bar chart
            y_pos = np.arange(len(feature_names))
            colors = [self.colors['positive'] if abs(contributions[name]) > 0.05 else self.colors['neutral']
                     for name in feature_names]

            bars = ax.barh(y_pos, shap_values, color=colors, alpha=0.7, height=0.6)

            # Add value labels (SHAP values only, feature names are on y-axis)
            for i, (bar, value, name) in enumerate(zip(bars, shap_values, feature_names)):
                width = bar.get_width()
                original_value = contributions[name]
                direction = "↑ Risk" if original_value > 0 else "↓ Protection"

                # Add main value label
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', ha='left', va='center', fontsize=8, fontweight='bold')

                # Add direction indicator on the bar
                ax.text(width - 0.02, bar.get_y() + bar.get_height()/2,
                       direction, ha='right', va='center', fontsize=7,
                       color='white', fontweight='bold', alpha=0.9)

            # Styling
            ax.set_yticks(y_pos)
            ax.set_yticklabels([name.replace('_', ' ') for name in feature_names], fontsize=10)
            ax.set_xlabel('Feature Importance (Absolute SHAP Value)', fontsize=11)
            ax.set_title(f'Top {top_n} Most Important Features',
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='x', alpha=0.3)

            # Add legend
            legend_elements = [
                plt.Rectangle((0,0),1,1, facecolor=self.colors['positive'], alpha=0.7, label='High Impact'),
                plt.Rectangle((0,0),1,1, facecolor=self.colors['neutral'], alpha=0.7, label='Moderate Impact')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error creating feature bar chart: {str(e)}")
            return self._create_error_plot("Feature Bar Chart Error")

    def create_single_prediction_plot(self, shap_result: Dict[str, Any], top_n: int = 10) -> plt.Figure:
        """
        Create a single prediction explanation plot (formerly called summary plot).

        This plot shows how individual features contribute to one specific prediction,
        with feature values displayed alongside SHAP values.

        Args:
            shap_result: SHAP computation results for a single prediction
            top_n: Number of top features to display

        Returns:
            Matplotlib figure object
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Extract data
            contributions = shap_result['feature_contributions']
            input_values = shap_result['input_values']

            # Sort features by absolute contribution
            sorted_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]

            # Prepare data for single prediction plot
            feature_names = [f[0] for f in sorted_features]
            shap_values = [f[1] for f in sorted_features]

            # Create scatter plot with feature values and SHAP values
            y_pos = np.arange(len(feature_names))

            # Plot points for each feature
            for i, (feature_name, shap_value) in enumerate(zip(feature_names, shap_values)):
                feature_value = input_values.get(feature_name, 0)

                # Color based on SHAP value direction
                color = self.colors['positive'] if shap_value > 0 else self.colors['negative']

                # Plot the point
                ax.scatter(shap_value, i, c=color, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

                # Add feature value as text
                ax.text(shap_value + 0.01 if shap_value > 0 else shap_value - 0.01,
                       i, f'{feature_value}', ha='left' if shap_value > 0 else 'right',
                       va='center', fontsize=8, fontweight='bold')

            # Add vertical line at zero
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

            # Add grid lines
            ax.grid(axis='x', alpha=0.3)

            # Styling
            ax.set_yticks(y_pos)
            ax.set_yticklabels([name.replace('_', ' ') for name in feature_names], fontsize=10)
            ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=11)
            ax.set_title('SHAP Single Prediction Plot - Feature Contributions',
                        fontsize=14, fontweight='bold', pad=20)

            # Add legend
            legend_elements = [
                plt.scatter([], [], c=self.colors['positive'], s=100, alpha=0.7, edgecolors='black',
                           linewidth=0.5, label='Increases Risk'),
                plt.scatter([], [], c=self.colors['negative'], s=100, alpha=0.7, edgecolors='black',
                           linewidth=0.5, label='Decreases Risk')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

            # Add interpretation guide
            ax.text(0.02, 0.98, '← Protective Factors    Risk Factors →',
                   transform=ax.transAxes, fontsize=10, fontweight='bold',
                   verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3',
                   facecolor='white', alpha=0.8))

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error creating single prediction plot: {str(e)}")
            return self._create_error_plot("Single Prediction Plot Error")

    def create_global_summary_plot(self, shap_results: List[Dict[str, Any]], top_n: int = 10) -> plt.Figure:
        """
        Create a SHAP summary plot following best practices using SHAP Explanation objects.

        This implementation follows SHAP best practices by:
        - Using shap.Explanation objects for proper data structure
        - Preserving original feature types (no manual encoding)
        - Handling version compatibility
        - Providing robust fallback options

        Args:
            shap_results: List of SHAP computation results from multiple predictions
            top_n: Number of top features to display

        Returns:
            Matplotlib figure object
        """
        try:
            if not shap_results or len(shap_results) < 2:
                return self._create_error_plot("Need at least 2 predictions for global summary")

            # Extract data following SHAP best practices
            feature_names = list(shap_results[0]['feature_contributions'].keys())
            n_predictions = len(shap_results)
            n_features = len(feature_names)

            # Initialize arrays for SHAP values and base values
            shap_values = np.zeros((n_predictions, n_features))
            base_values = np.zeros(n_predictions)

            # Prepare feature values (preserve original types)
            feature_values = []
            for result in shap_results:
                row_values = []
                for feature in feature_names:
                    value = result['input_values'].get(feature, None)
                    row_values.append(value)
                feature_values.append(row_values)

            # Fill SHAP values and base values
            for i, result in enumerate(shap_results):
                contributions = result['feature_contributions']
                base_values[i] = result.get('expected_value', 0.0)

                for j, feature in enumerate(feature_names):
                    shap_values[i, j] = contributions.get(feature, 0.0)

            # Convert to DataFrame to preserve data types
            features_df = pd.DataFrame(feature_values, columns=feature_names)

            # Debug logging
            logger.info(f"Creating global summary plot for {n_predictions} predictions")
            logger.info(f"Feature matrix shape: {features_df.shape}")
            logger.info(f"SHAP values shape: {shap_values.shape}")

            # Create figure
            fig = plt.figure(figsize=(14, 6))

            # Try modern SHAP approach first (v0.40+)
            try:
                # Create SHAP Explanation object (modern approach)
                explanation = shap.Explanation(
                    values=shap_values,
                    base_values=base_values,
                    data=features_df.values,  # Let SHAP handle encoding
                    feature_names=feature_names
                )

                # Create summary plot using Explanation object
                shap.summary_plot(explanation, max_display=top_n, show=False)

            except (AttributeError, TypeError) as e:
                # Fallback for older SHAP versions or when Explanation fails
                logger.warning(f"Modern SHAP approach failed: {e}. Using legacy approach.")

                try:
                    # Legacy approach - direct array usage
                    shap.summary_plot(
                        shap_values,
                        features_df.values,
                        feature_names=feature_names,
                        max_display=top_n,
                        show=False
                    )
                except Exception as legacy_error:
                    logger.error(f"Legacy SHAP approach also failed: {legacy_error}")
                    # Final fallback to custom implementation
                    return self._create_custom_summary_plot(shap_results, top_n)

            # Customize the plot for better interpretability
            ax = plt.gca()
            ax.set_title(f'SHAP Global Summary Plot - Risk vs Protective Factors\n'
                        f'({n_predictions} predictions analyzed)',
                        fontsize=16, fontweight='bold', pad=20)

            # Add clear risk/protective labels on x-axis
            ax.set_xlabel('SHAP Value (Impact on Non-Adherence Risk)', fontsize=12, fontweight='bold')

            # Add vertical line at zero to separate risk from protective
            ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8)

            # Add background colors to make regions clearer
            ax.axvspan(ax.get_xlim()[0], 0, alpha=0.1, color='green', label='Protective Factors')
            ax.axvspan(0, ax.get_xlim()[1], alpha=0.1, color='red', label='Risk Factors')

            # Add text labels for the regions
            ax.text(ax.get_xlim()[0] * 0.3, ax.get_ylim()[1] * 0.95,
                   '🛡️ PROTECTIVE\n(Lowers Risk)', fontsize=11, fontweight='bold',
                   ha='center', va='top', color='darkgreen',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

            ax.text(ax.get_xlim()[1] * 0.3, ax.get_ylim()[1] * 0.95,
                   '⚠️ RISK FACTORS\n(Increases Risk)', fontsize=11, fontweight='bold',
                   ha='center', va='top', color='darkred',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

            # Add interpretation guide (positioned to avoid overlapping feature labels)
            plt.figtext(0.98, 0.02,
                       '📊 INTERPRETATION:\n'
                       '• 🟢 LEFT = Protective (lowers risk)\n'
                       '• 🔴 RIGHT = Risk factors (increases risk)\n'
                       '• 🎨 Color = Feature value\n'
                       '• 📈 Features by impact',
                       fontsize=9, style='italic',
                       ha='right', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9),
                       multialignment='right')

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error creating global summary plot: {str(e)}")
            # Use custom implementation as final fallback
            return self._create_custom_summary_plot(shap_results, top_n)

    def _create_fallback_global_plot(self, shap_matrix, feature_names, n_predictions):
        """
        Create a fallback global plot when SHAP summary_plot fails.
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 6))

            # Calculate mean absolute SHAP values for feature ranking
            mean_abs_shap = np.abs(shap_matrix).mean(0)

            # Sort features by importance
            sorted_indices = np.argsort(mean_abs_shap)[::-1]
            top_indices = sorted_indices[:10]  # Show top 10

            # Plot horizontal bar chart
            y_pos = np.arange(len(top_indices))
            ax.barh(y_pos, mean_abs_shap[top_indices])

            # Set labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels([feature_names[i] for i in top_indices])
            ax.set_xlabel('Mean Absolute SHAP Value')
            ax.set_title(f'Global Feature Importance\n({n_predictions} predictions)')

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Fallback plot also failed: {str(e)}")
            return self._create_error_plot("Fallback Global Plot Error")

    def _create_custom_summary_plot(self, shap_results: List[Dict[str, Any]], top_n: int = 10) -> plt.Figure:
        """
        Create a custom beeswarm-style summary plot when SHAP's built-in methods fail.

        This mimics SHAP's summary plot style but handles mixed data types better.
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 6))

            # Calculate feature importance
            all_features = list(shap_results[0]['feature_contributions'].keys())
            feature_importance = {}

            for feature in all_features:
                shap_values = [abs(result['feature_contributions'].get(feature, 0))
                              for result in shap_results]
                feature_importance[feature] = np.mean(shap_values)

            # Sort and select top features
            sorted_features = sorted(feature_importance.items(),
                                   key=lambda x: x[1], reverse=True)[:top_n]
            selected_features = [f[0] for f in sorted_features]

            # Create beeswarm-style plot
            y_positions = list(range(len(selected_features)))

            for i, feature in enumerate(selected_features):
                # Get SHAP values and feature values for this feature
                shap_vals = []
                feat_vals = []

                for result in shap_results:
                    shap_val = result['feature_contributions'].get(feature, 0)
                    feat_val = result['input_values'].get(feature, 0)

                    # Normalize feature values for coloring
                    if isinstance(feat_val, str):
                        # Handle categorical features
                        if feat_val in ['High', 'Medium', 'Low']:
                            feat_val = {'Low': 0.0, 'Medium': 0.5, 'High': 1.0}[feat_val]
                        elif feat_val in ['Male', 'Female']:
                            feat_val = 1.0 if feat_val == 'Male' else 0.0
                        elif feat_val in ['Yes', 'No']:
                            feat_val = 1.0 if feat_val == 'Yes' else 0.0
                        else:
                            feat_val = hash(feat_val) % 100 / 100.0
                    else:
                        feat_val = float(feat_val) if feat_val is not None else 0.0

                    shap_vals.append(shap_val)
                    feat_vals.append(feat_val)

                # Normalize feature values to [0, 1] for coloring
                feat_vals = np.array(feat_vals)
                if feat_vals.std() > 0:
                    feat_vals = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min())

                # Add jitter to y-positions for beeswarm effect
                y_jittered = i + np.random.normal(0, 0.1, len(shap_vals))

                # Create scatter plot with color mapping
                scatter = ax.scatter(shap_vals, y_jittered, c=feat_vals,
                                   cmap='coolwarm', alpha=0.6, s=30)

            # Customize plot for better interpretability
            ax.set_yticks(y_positions)
            ax.set_yticklabels([f.replace('_', ' ') for f in selected_features])
            ax.set_xlabel('SHAP Value (Impact on Non-Adherence Risk)', fontsize=12, fontweight='bold')
            ax.set_title(f'Feature Impact Summary - Risk vs Protective Factors\n'
                        f'({len(shap_results)} predictions analyzed)',
                        fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8)

            # Add background colors to make regions clearer
            ax.axvspan(ax.get_xlim()[0], 0, alpha=0.1, color='green', label='Protective Factors')
            ax.axvspan(0, ax.get_xlim()[1], alpha=0.1, color='red', label='Risk Factors')

            # Add text labels for the regions
            ax.text(ax.get_xlim()[0] * 0.3, ax.get_ylim()[1] * 0.95,
                   '🛡️ PROTECTIVE\n(Lowers Risk)', fontsize=11, fontweight='bold',
                   ha='center', va='top', color='darkgreen',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

            ax.text(ax.get_xlim()[1] * 0.3, ax.get_ylim()[1] * 0.95,
                   '⚠️ RISK FACTORS\n(Increases Risk)', fontsize=11, fontweight='bold',
                   ha='center', va='top', color='darkred',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Feature Value\n(normalized)', fontsize=10)

            # Add interpretation guide (positioned to avoid overlapping feature labels)
            plt.figtext(0.98, 0.02,
                       '📊 INTERPRETATION:\n'
                       '• 🟢 LEFT = Protective (lowers risk)\n'
                       '• 🔴 RIGHT = Risk factors (increases risk)\n'
                       '• 🎨 Color = Feature value\n'
                       '• 📈 Features by impact',
                       fontsize=9, style='italic',
                       ha='right', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9),
                       multialignment='right')

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Custom summary plot failed: {str(e)}")
            return self._create_error_plot("Custom Summary Plot Error")

    def create_shap_summary(self, shap_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of SHAP results with key insights.

        Args:
            shap_result: SHAP computation results

        Returns:
            Dictionary containing summary statistics and insights
        """
        try:
            contributions = shap_result['feature_contributions']
            expected_value = shap_result['expected_value']
            prediction_class = shap_result['prediction_class']
            prediction_proba = shap_result['prediction_probability']

            # Calculate summary statistics
            total_contribution = sum(contributions.values())
            final_prediction = expected_value + total_contribution

            # Find top contributing features
            sorted_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

            # Identify risk and protective factors
            risk_factors = [(name, val) for name, val in sorted_features if val > 0][:3]
            protective_factors = [(name, val) for name, val in sorted_features if val < 0][:3]

            summary = {
                'prediction_class': prediction_class,
                'prediction_confidence': max(prediction_proba),
                'expected_value': expected_value,
                'final_prediction': final_prediction,
                'total_contribution': total_contribution,
                'top_risk_factors': risk_factors,
                'top_protective_factors': protective_factors,
                'feature_count': len(contributions)
            }

            return summary

        except Exception as e:
            logger.error(f"Error creating SHAP summary: {str(e)}")
            return {}

    def _create_error_plot(self, error_message: str) -> plt.Figure:
        """
        Create an error plot when visualization fails.

        Args:
            error_message: Error message to display

        Returns:
            Matplotlib figure with error message
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.text(0.5, 0.5, f"Visualization Error:\n{error_message}",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, color='red', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig

# Convenience functions for Streamlit integration

def display_waterfall_plot(shap_result: Dict[str, Any], top_n: int = 8):
    """
    Display waterfall plot in Streamlit.

    Args:
        shap_result: SHAP computation results
        top_n: Number of top features to display
    """
    try:
        visualizer = SHAPVisualizer()
        fig = visualizer.create_waterfall_plot(shap_result, top_n)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error displaying waterfall plot: {str(e)}")

def display_force_plot(shap_result: Dict[str, Any], top_n: int = 6):
    """
    Display force plot in Streamlit.

    Args:
        shap_result: SHAP computation results
        top_n: Number of top features to display
    """
    try:
        visualizer = SHAPVisualizer()
        fig = visualizer.create_force_plot(shap_result, top_n)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error displaying force plot: {str(e)}")

def display_feature_bar_chart(shap_result: Dict[str, Any], top_n: int = 10):
    """
    Display feature importance bar chart in Streamlit.

    Args:
        shap_result: SHAP computation results
        top_n: Number of top features to display
    """
    try:
        visualizer = SHAPVisualizer()
        fig = visualizer.create_feature_bar_chart(shap_result, top_n)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error displaying feature bar chart: {str(e)}")

def display_single_prediction_plot(shap_result: Dict[str, Any], top_n: int = 10):
    """
    Display single prediction explanation plot in Streamlit.

    Args:
        shap_result: SHAP computation results
        top_n: Number of top features to display
    """
    try:
        visualizer = SHAPVisualizer()
        fig = visualizer.create_single_prediction_plot(shap_result, top_n)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error displaying single prediction plot: {str(e)}")

def display_global_summary_plot(shap_results: List[Dict[str, Any]], top_n: int = 10):
    """
    Display global SHAP summary plot across multiple predictions in Streamlit.

    Args:
        shap_results: List of SHAP computation results from multiple predictions
        top_n: Number of top features to display
    """
    try:
        if not shap_results:
            st.warning("No SHAP results available for global summary plot")
            return

        visualizer = SHAPVisualizer()
        fig = visualizer.create_global_summary_plot(shap_results, top_n)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error displaying global summary plot: {str(e)}")

def display_shap_summary(shap_result: Dict[str, Any]):
    """
    Display SHAP summary with key insights in Streamlit.

    Args:
        shap_result: SHAP computation results
    """
    try:
        visualizer = SHAPVisualizer()
        summary = visualizer.create_shap_summary(shap_result)

        if summary:
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Prediction", summary['prediction_class'])
                st.metric("Confidence", f"{summary['prediction_confidence']:.1%}")

            with col2:
                st.metric("Expected Value", f"{summary['expected_value']:.3f}")
                st.metric("Final Prediction", f"{summary['final_prediction']:.3f}")

            # Display top factors
            if summary['top_risk_factors']:
                st.subheader("🔴 Top Risk Factors")
                for factor, value in summary['top_risk_factors']:
                    st.write(f"**{factor.replace('_', ' ')}**: {value:.3f}")

            if summary['top_protective_factors']:
                st.subheader("🟢 Top Protective Factors")
                for factor, value in summary['top_protective_factors']:
                    st.write(f"**{factor.replace('_', ' ')}**: {value:.3f}")

    except Exception as e:
        st.error(f"Error displaying SHAP summary: {str(e)}")

def create_shap_visualization_section(shap_result: Dict[str, Any], shap_results: Optional[List[Dict[str, Any]]] = None):
    """
    Create a complete SHAP visualization section for Streamlit.

    Args:
        shap_result: SHAP computation results for current prediction
        shap_results: Optional list of SHAP results from multiple predictions for global summary
    """
    if shap_result is None:
        st.warning("SHAP computation failed. Unable to display explanations.")
        return

    st.markdown("### 📊 SHAP Model Explanations")

    # Visualization tabs
    tab_names = ["💧 Waterfall Plot", "🎯 Force Plot", "📊 Feature Importance", "🔍 Single Prediction", "📈 Global Summary"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.markdown("**Waterfall Plot**: Shows how each feature contributes to the prediction, starting from the expected value.")
        display_waterfall_plot(shap_result)

    with tabs[1]:
        st.markdown("**Force Plot**: Displays the positive and negative forces pushing the prediction away from the expected value.")
        display_force_plot(shap_result)

    with tabs[2]:
        st.markdown("**Feature Importance**: Ranks features by their absolute contribution to the prediction.")
        display_feature_bar_chart(shap_result)

    with tabs[3]:
        st.markdown("**Single Prediction Plot**: Shows feature contributions for this specific prediction with actual feature values.")
        display_single_prediction_plot(shap_result)

    with tabs[4]:
        st.markdown("**Global Summary Plot**: Shows how features contribute across multiple predictions (requires multiple SHAP results).")
        if shap_results and len(shap_results) > 1:
            display_global_summary_plot(shap_results)
        else:
            st.info("💡 Global summary plot requires multiple predictions. Generate SHAP explanations for several patients to see this visualization.")

    # Additional insights
    with st.expander("💡 Interpretation Guide"):
        st.markdown("""
        **Understanding SHAP Values:**
        - **Positive values** (red) push the prediction toward "Non-Adherent"
        - **Negative values** (teal) push the prediction toward "Adherent"
        - **Magnitude** indicates the strength of the feature's influence
        - **Expected value** is the average prediction across all patients

        **Clinical Insights:**
        - Focus interventions on the top risk factors identified
        - Consider protective factors that may help maintain adherence
        - Use this information to personalize patient care plans
        """)
