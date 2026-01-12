"""
Unit tests for LimeTextExplainer.
"""
import numpy as np
from src.explainers.lime_text import LimeTextExplainer

def dummy_classifier(texts):
    """
    Class 0: Negative
    Class 1: Positive
    """
    probs = []
    for t in texts:
        # If 'bad' is present, it's definitively Negative (Class 0)
        if 'bad' in t:
            p_positive = 0.0
        # If 'good' is present (and no 'bad'), it's Positive (Class 1)
        elif 'good' in t:
            p_positive = 1.0
        # Otherwise neutral
        else:
            p_positive = 0.5
    
        probs.append([1 - p_positive, p_positive])
        
    return np.array(probs)

def test_text_explanation_logic():
    # Case 1: "Bad" dominates.
    text = "good food bad service"
    explainer = LimeTextExplainer(random_state=42)
    
    # Explain Class 1 (Positive). 
    # We expect 'bad' to have a NEGATIVE weight (it kills the positive score).
    exps = explainer.explain_instance(
        text, 
        dummy_classifier, 
        labels=(1,),  # Force explaining Class 1
        num_features=2, 
        num_samples=500
    )
    
    weights = dict(exps[1]['explanation_map'])
    print(f"Weights for 'good food bad service' (Class 1): {weights}")
    
    assert 'bad' in weights
    assert weights['bad'] < 0, "Bad should negatively impact Class 1"

    # Case 2: "Good" dominates.
    text_pos = "good food service"
    exps_pos = explainer.explain_instance(
        text_pos,
        dummy_classifier,
        labels=(1,),
        num_features=2,
        num_samples=500
    )
    
    weights_pos = dict(exps_pos[1]['explanation_map'])
    print(f"Weights for 'good food service' (Class 1): {weights_pos}")
    
    assert 'good' in weights_pos
    assert weights_pos['good'] > 0, "Good should positively impact Class 1"

if __name__ == "__main__":
    test_text_explanation_logic()
    print("Text LIME test passed.")