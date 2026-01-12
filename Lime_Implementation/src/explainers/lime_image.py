"""
LIME Explainer for Image Data.
Implements segmentation-based perturbation.
"""

import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise_distances
from src.core.base import LimeBase
from src.utils.segmentation import SegmentationAlgorithm

class LimeImageExplainer(LimeBase):
    def __init__(self, kernel_width=0.25, verbose=False, random_state=None):
        """
        Args:
            kernel_width (float): L2 Distance width. 0.25 is standard for images 
                                  when pixel distance is small in binary space.
            random_state (int): For reproducibility.
        """
        super().__init__(kernel_width, verbose)
        self.random_state = check_random_state(random_state)

    def explain_instance(self, image, classifier_fn, labels=(1,), 
                         hide_color=None, num_features=10, num_samples=1000, 
                         segmentation_fn=None):
        """
        Generates explanation for an image.
        
        Args:
            image (np.ndarray): 3D RGB image (H, W, 3).
            classifier_fn (callable): Takes a batch of images (N, H, W, 3), returns (N, num_classes).
            labels (tuple): Class indices to explain.
            hide_color (float/None): Color to replace 'removed' segments. 
                                     If None, uses mean color of the superpixel.
                                     For 'Gray out', pass roughly 128 (if 0-255) or 0.5 (if 0-1).
            segmentation_fn (callable): Custom segmentation. If None, uses Quickshift.
        """
        
        # 1. Segment the image
        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift')
        
        segments = segmentation_fn(image)
        unique_segments = np.unique(segments)
        num_segments = len(unique_segments)
        
        if self.verbose:
            print(f"Image segmented into {num_segments} super-pixels.")
        
        # 2. Generate Synthetic Neighborhood (Perturbation)
        # Returns: data (binary matrix), perturbed_images (list of 3D arrays)
        data, perturbed_imgs = self._generate_samples(
            image, segments, num_segments, num_samples, hide_color
        )

        # 3. Get Predictions (Heavy Computation)
        predictions = classifier_fn(np.array(perturbed_imgs))

        # 4. Calculate Distances (L2 Distance for Images)
        # data[0] is the original (all 1s).
        distances = pairwise_distances(data, data[0].reshape(1, -1), metric='euclidean').ravel()

        # 5. Solve for requested labels
        explanations = {}
        for label in labels:
            class_predictions = predictions[:, label]
            
            result = self._solve_explanation(
                perturbed_data=data,
                predictions=class_predictions,
                distances=distances,
                num_features=num_features
            )
            
            # Add metadata for visualization
            result['segments'] = segments
            # We don't map to "words", we map to segment IDs (integers)
            result['explanation_map'] = result['explanation'] 
            result['target_class'] = label # <--- FIX for Visualization
            
            explanations[label] = result
            
        return explanations

    def _generate_samples(self, image, segments, num_segments, num_samples, hide_color):
        """
        Generates perturbed images by masking superpixels.
        """
        # data: Binary matrix (N x num_segments)
        # 1 = Superpixel active (visible), 0 = Superpixel inactive (hidden)
        data = self.random_state.randint(0, 2, size=(num_samples, num_segments))
        
        # First row is always the original image (all active)
        data[0, :] = 1
        
        imgs = []
        
        # Pre-calculate mean color if we aren't using a fixed hide_color
        if hide_color is None:
            # We can use the mean of the whole image for simplicity
            fudged_image = image.copy()
            fudged_image[:] = np.mean(image, axis=(0, 1))
        else:
            fudged_image = image.copy()
            fudged_image[:] = hide_color

        for row in data:
            # Fast Masking
            # Create a mask where (segments == inactive_segment_id)
            
            # Find which segments are OFF (0)
            zeros = np.where(row == 0)[0]
            
            # Create boolean mask for the whole image
            # np.isin is efficient: true if pixel belongs to a zero-segment
            mask = np.isin(segments, zeros)
            
            # Create the perturbation
            temp = image.copy()
            
            # Apply the "fudged" background to the masked areas
            temp[mask] = fudged_image[mask]
            
            imgs.append(temp)
            
        return data, imgs