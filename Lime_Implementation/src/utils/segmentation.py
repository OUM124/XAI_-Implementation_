"""
Utilities for Image Segmentation.
Wrappers around scikit-image algorithms.
"""
import numpy as np
from skimage.segmentation import quickshift, mark_boundaries

class SegmentationAlgorithm:
    """
    Wrapper for segmentation algorithms.
    """
    def __init__(self, algo_type='quickshift', **kwargs):
        """
        Args:
            algo_type: 'quickshift', 'slic', or 'felzenszwalb'
            kwargs: arguments passed to the skimage function
        """
        self.algo_type = algo_type
        # Default parameters often used in LIME
        if algo_type == 'quickshift' and not kwargs:
            self.kwargs = {'kernel_size': 4, 'max_dist': 200, 'ratio': 0.2}
        else:
            self.kwargs = kwargs

    def __call__(self, image):
        """
        Args:
            image (np.ndarray): 3D image array (H, W, Channels)
        Returns:
            np.ndarray: 2D array of segments (H, W), where each pixel value 
                        is the segment ID.
        """
        if self.algo_type == 'quickshift':
            return quickshift(image, **self.kwargs)
        else:
            raise NotImplementedError(f"Algo {self.algo_type} not implemented yet.")