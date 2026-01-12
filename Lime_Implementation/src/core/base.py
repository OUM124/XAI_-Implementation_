"""
Base classes for LIME explainers.
Implements the fidelity-interpretability trade-off logic.
"""

import numpy as np
from src.core.k_lasso import k_lasso_path

class LimeBase:
    """
    Abstract base class for LIME explainers.
    Handles the kernel function and the generation of linear explanations
    once the perturbed data is provided.
    """
    
    def __init__(self, kernel_width=0.75, verbose=False):
        """
        Args:
            kernel_width (float): The sigma value for the exponential kernel.
                                  Defaults to 0.75 (standard for Cosine distance).
            verbose (bool): If true, print debug info.
        """
        self.kernel_width = kernel_width
        self.verbose = verbose

        # Define the kernel function pi_x(z)
        # Formula: exp( - distance^2 / sigma^2 )
        def kernel(distance):
            return np.exp(-(distance ** 2) / (self.kernel_width ** 2))
        
        self.kernel_fn = kernel

    def _solve_explanation(self, perturbed_data, predictions, distances, num_features):
        """
        Orchestrates the K-Lasso solution.
        
        Args:
            perturbed_data (np.ndarray): The binary matrix Z' (0s and 1s).
            predictions (np.ndarray): The f(z) outputs from the black box.
            distances (np.ndarray): Pre-calculated distances D(x, z).
            num_features (int): K (complexity limit).
            
        Returns:
            dict: {
                'intercept': float,
                'local_pred': float, # Prediction of the linear model
                'domain_mapper': list of (feature_index, weight),
                'score': float # R^2 of the linear model
            }
        """
        # 1. Calculate weights using the exponential kernel
        weights = self.kernel_fn(distances)
        
        # 2. Run K-LASSO (Algorithm 1)
        selected_features, linear_weights, intercept = k_lasso_path(
            perturbed_data, 
            predictions, 
            weights, 
            num_features
        )
        
        # 3. Format the output
        # Zip the feature indices with their calculated linear weights
        # Sort them by absolute importance (magnitude of weight)
        explanation = list(zip(selected_features, linear_weights))
        explanation.sort(key=lambda x: np.abs(x[1]), reverse=True)
        
        # Calculate local prediction for the original instance (all 1s vector)
        # Note: In interpretable space, the original instance is a vector of 1s (features present)
        # local_pred = sum(weights) + intercept
        # However, strictly speaking, we must map 1s only for the selected features.
        # But since 'selected_features' implies those are the only ones used:
        if len(linear_weights) > 0:
            local_pred = intercept + np.sum(linear_weights)
        else:
            local_pred = intercept

        return {
            'intercept': intercept,
            'weights': weights, 
            'explanation': explanation,
            'local_pred': local_pred
        }