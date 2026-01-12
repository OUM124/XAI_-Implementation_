"""
Unit tests for LimeImageExplainer.
"""
import numpy as np
import pytest
from src.explainers.lime_image import LimeImageExplainer

def dummy_red_classifier(images):
    """
    Classifies images based on how much RED is present.
    Input: (N, H, W, 3)
    Output: (N, 2) [Prob_Not_Red, Prob_Red]
    """
    preds = []
    for img in images:
        # Check image normalization (0-1 vs 0-255) to be safe
        # We assume 0-1 for this test generation
        
        # Sum red channel
        red_intensity = np.sum(img[:, :, 0])
        # Sum other channels
        other_intensity = np.sum(img[:, :, 1:])
        
        # Simple ratio
        total = red_intensity + other_intensity + 1e-5
        prob_red = red_intensity / total
        
        preds.append([1 - prob_red, prob_red])
        
    return np.array(preds)

def test_image_red_square():
    print("Running Image LIME Test (Red Square)...")
    
    # 1. Create a dummy image (100x100)
    # Left half (cols 0-50) = Red
    # Right half (cols 50-100) = Blue
    img = np.zeros((100, 100, 3), dtype=np.double)
    img[:, :50, 0] = 1.0 # Red Channel
    img[:, 50:, 2] = 1.0 # Blue Channel
    
    explainer = LimeImageExplainer(random_state=42)
    
    # Explain Class 1 (Red)
    exps = explainer.explain_instance(
        img, 
        dummy_red_classifier, 
        labels=(1,), 
        num_features=5, 
        num_samples=200,
        hide_color=0 # When hiding, make it black
    )
    
    explanation = exps[1]
    segments = explanation['segments']
    top_features = explanation['explanation'] # List of (seg_id, weight)
    
    # 2. Verify Logic
    # The top feature should correspond to a segment on the Left (Red) side.
    top_seg_id, top_weight = top_features[0]
    
    print(f"Top Segment ID: {top_seg_id} with Weight: {top_weight:.4f}")

    # Ensure the weight is positive (Red contributes to Red class)
    assert top_weight > 0
    
    # Find where this segment is located
    mask = (segments == top_seg_id)
    
    # Check spatial location
    # Get all (y, x) coordinates of this segment
    coords = np.column_stack(np.where(mask))
    avg_col_index = np.mean(coords[:, 1]) # 1 is the column (x) axis
    
    print(f"Segment Center Column: {avg_col_index:.2f} (Should be < 50)")
    
    # Left side is 0-50. Right side is 50-100.
    assert avg_col_index < 50, "LIME failed: The important segment should be on the Red (Left) side"

if __name__ == "__main__":
    test_image_red_square()
    print("Image LIME test passed.")