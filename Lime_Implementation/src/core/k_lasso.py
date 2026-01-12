"""
Core implementation of the K-LASSO algorithm described in Section 3.4 of the LIME paper.
"""

import numpy as np
from sklearn.linear_model import lars_path, LinearRegression

def k_lasso_path(Z, predictions, weights, K):
    """
    Solves the feature selection and weight fitting using the K-LASSO procedure.
    
    Paper Reference: Algorithm 1 (Sparse Linear Explanations using LIME)
    
    Args:
        Z (np.ndarray): The perturbed dataset in interpretable representation (binary matrix).
                        Shape: (num_samples, num_features)
        predictions (np.ndarray): The black-box model predictions for the samples Z.
                                  Shape: (num_samples,)
        weights (np.ndarray): The proximity weights (pi_x) for each sample.
                              Shape: (num_samples,)
        K (int): The maximum number of features to select (Complexity limit).

    Returns:
        tuple: (selected_feature_indices, final_coefficients, intercept)
    """
    
    # --- Step 1: Feature Selection via LARS Path ---
    # Goal: Select K features using Lasso regularization.
    # Math: Lasso minimizes ||y - Xw||^2 + alpha * ||w||_1
    # Problem: sklearn's lars_path does not accept sample_weights directly.
    # Solution: Transform the problem into OLS form by scaling X and y by sqrt(weights).
    # Proof: Sum(w * (y - Xw)^2) = || sqrt(w)y - sqrt(w)Xw ||^2
    
    weighted_Z = Z * np.sqrt(weights[:, np.newaxis])
    weighted_pred = predictions * np.sqrt(weights)

    # lars_path efficiently computes the regularization path.
    # It returns 'coefs' of shape (n_features, n_steps).
    # We set verbose=False to keep it silent.
    _, _, coefs = lars_path(weighted_Z, weighted_pred, method='lasso', verbose=False)

    selected_features = []
    
    # Iterate through the steps of the path (from high regularization to low).
    # We look for the first step where we have at least K non-zero coefficients.
    for i in range(coefs.shape[1]):
        non_zero_indices = np.nonzero(coefs[:, i])[0]
        
        if len(non_zero_indices) >= K:
            selected_features = non_zero_indices
            break
            
    # Fallback: If the path exhausted without reaching K (e.g., K > n_features), take the last step.
    if len(selected_features) == 0 and coefs.shape[1] > 0:
        selected_features = np.nonzero(coefs[:, -1])[0]

    # Truncation: If LARS added multiple features at once and we exceeded K,
    # we pick the top K by absolute weight magnitude at that step.
    if len(selected_features) > K:
        current_weights = np.abs(coefs[selected_features, i])
        top_k_indices = np.argsort(current_weights)[-K:]
        selected_features = selected_features[top_k_indices]

    # --- Step 2: Weight Refinement via Weighted Least Squares ---
    # As per Section 3.4, we re-fit a simple linear model on ONLY the selected features
    # to remove the shrinkage bias introduced by Lasso.
    
    if len(selected_features) == 0:
        return np.array([]), np.array([]), 0.0

    # Filter Z to only keep selected columns
    Z_subset = Z[:, selected_features]

    # Fit Linear Regression using the original sample weights (sklearn handles weights correctly here)
    # fit_intercept=True allows the local linear model to have a bias term.
    least_squares_model = LinearRegression(fit_intercept=True)
    least_squares_model.fit(Z_subset, predictions, sample_weight=weights)

    return selected_features, least_squares_model.coef_, least_squares_model.intercept_