"""
Unit tests for the Core K-Lasso logic.
"""
import numpy as np
import pytest
from src.core.k_lasso import k_lasso_path

def test_k_lasso_perfect_selection():
    """
    Test if K-Lasso selects the correct relevant features in a noiseless setting.
    Scenario:
    - 10 Features total.
    - Only Feature 0 and Feature 1 matter.
    - y = 5*x0 + 2*x1
    - We ask for K=2.
    """
    np.random.seed(42)
    num_samples = 100
    num_features = 10
    
    # Create random binary data (Z)
    Z = np.random.randint(0, 2, size=(num_samples, num_features))
    
    # Define ground truth relationship (No noise)
    # y = 5 * feature_0 + 2 * feature_1
    predictions = 5 * Z[:, 0] + 2 * Z[:, 1]
    
    # Weights are all 1.0 (Uniform proximity for this test)
    weights = np.ones(num_samples)
    
    K = 2
    selected_indices, coefs, intercept = k_lasso_path(Z, predictions, weights, K)
    
    # Assertions
    # 1. Did it pick exactly 2 features?
    assert len(selected_indices) == 2
    
    # 2. Did it pick Feature 0 and Feature 1?
    expected_features = {0, 1}
    assert set(selected_indices) == expected_features
    
    # 3. Are the weights accurate? (Standard OLS should solve this perfectly)
    # Since the order of coefs matches selected_indices, we assume sorted check
    # We map indices to coefs to check specifically
    coef_map = dict(zip(selected_indices, coefs))
    
    assert np.isclose(coef_map[0], 5.0, atol=0.01)
    assert np.isclose(coef_map[1], 2.0, atol=0.01)
    assert np.isclose(intercept, 0.0, atol=0.01)

def test_k_lasso_with_weights():
    """
    Test if sample weights are respected.
    We create a scenario where:
    - Feature 0 is correlated with y ONLY in high-weighted samples.
    - Feature 1 is correlated with y ONLY in low-weighted samples.
    - If weights work, Feature 0 should be picked over Feature 1.
    """
    np.random.seed(42)
    
    # Sample 1: Weight 1.0 -> y follows x0
    Z_high = np.array([[1, 0], [0, 0], [1, 0]]) 
    y_high = np.array([10, 0, 10]) # Perfect correlation with x0
    w_high = np.array([1.0, 1.0, 1.0])
    
    # Sample 2: Weight 0.001 -> y follows x1
    Z_low = np.array([[0, 1], [0, 0], [0, 1]])
    y_low = np.array([10, 0, 10]) # Perfect correlation with x1
    w_low = np.array([0.001, 0.001, 0.001])
    
    # Combine
    Z = np.vstack([Z_high, Z_low])
    y = np.concatenate([y_high, y_low])
    w = np.concatenate([w_high, w_low])
    
    # Ask for K=1. Lasso should prioritize x0 because its samples have high weight.
    K = 1
    selected_indices, coefs, _ = k_lasso_path(Z, y, w, K)
    
    assert 0 in selected_indices
    assert 1 not in selected_indices

if __name__ == "__main__":
    test_k_lasso_perfect_selection()
    test_k_lasso_with_weights()
    print("All K-Lasso tests passed.")