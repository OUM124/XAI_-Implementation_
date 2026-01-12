"""
Visualization utilities for LIME.
Reproduces the visual style of the paper.
"""
import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self):
        pass
    
    def visualize_text(self, text_explanation):
        # [Previous text code remains here...]
        GREEN = '\033[92m'
        RED = '\033[91m'
        RESET = '\033[0m'
        
        print("\n=== LIME Explanation ===")
        print(f"Target Class: {text_explanation['target_class']}")
        print(f"Local Linear Prediction: {text_explanation['local_pred']:.4f}")
        print("Features:")
        
        for word, weight in text_explanation['explanation_map']:
            color = GREEN if weight > 0 else RED
            bar_len = int(abs(weight) * 50) 
            bar = '█' * bar_len
            print(f"{color}{word:>15} | {weight:>.4f} {bar}{RESET}")

    def visualize_image_mask(self, image, explanation_dict, num_features=5, save_path=None):
        """
        Visualizes the image explanation.
        Highlights the top 'num_features' superpixels that contribute to the class.
        """
        segments = explanation_dict['segments']
        # List of (segment_id, weight)
        top_features = explanation_dict['explanation'][:num_features]
        
        # 1. Create a mask for the top features
        # We will create an image that is just the "Explanation"
        mask = np.zeros(segments.shape, dtype=bool)
        
        print(f"\nVisualizing Top {num_features} segments for Class {explanation_dict['target_class']}...")
        
        for seg_id, weight in top_features:
            if weight > 0:
                print(f"  Segment {seg_id}: Weight {weight:.4f} (Positive)")
                mask[segments == seg_id] = True
            else:
                # In this experiment, we usually only care about what *caused* the prediction (Positive)
                pass
                
        # 2. Apply the mask to the original image
        # We want to show ONLY the important parts, and gray out the rest
        # This matches Figure 11(b) in the paper.
        
        # Create a gray background
        temp_img = np.ones_like(image) * 0.5 # Gray
        
        # Copy the original image pixels ONLY where the mask is True
        # Note: image might be 0-1 or 0-255. We assume 0-1 float for plotting.
        temp_img[mask] = image[mask]
        
        # 3. Add yellow boundaries for segments
        # mark_boundaries returns a float image
        final_vis = mark_boundaries(temp_img, segments, color=(1, 1, 0), outline_color=None)
        
        # Plot
        plt.figure(figsize=(8, 8))
        plt.imshow(final_vis)
        plt.title(f"Explanation for Class {explanation_dict['target_class']}")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Explanation saved to {save_path}")
        else:
            plt.show()