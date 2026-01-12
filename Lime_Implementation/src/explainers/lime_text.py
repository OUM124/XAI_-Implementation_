"""
LIME Explainer for Text Data.
Implements the sampling and perturbation logic for text.
"""

import numpy as np
import re
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise_distances
from src.core.base import LimeBase

class LimeTextExplainer(LimeBase):
    """
    Explains text classifiers.
    """

    def __init__(self, kernel_width=25, verbose=False, random_state=None):
        """
        Args:
            kernel_width (float): Width for the exponential kernel. 
                                  Default is 25 (standard for text/cosine distance).
            verbose (bool): Print debug info.
            random_state (int): Seed for reproducibility.
        """
        super().__init__(kernel_width, verbose)
        self.random_state = check_random_state(random_state)

    def explain_instance(self, text_instance, classifier_fn, labels=(1,), num_features=10, num_samples=5000):
        """
        Generates an explanation for a specific text instance.

        Args:
            labels (tuple): Tuple of integers specifying which class indices to explain. 
                            Default is (1,) but typically we want the top predicted class.
                            If you want to explain the top prediction automatically, 
                            you'd perform a check before calling this.
        """
        
        # 1. Parse Text (Same as before)
        tokens = re.split(r'(\W+)', text_instance)
        unique_words = list(set([t for t in tokens if re.match(r'\w+', t)]))
        vocab_size = len(unique_words)
        word_to_idx = {w: i for i, w in enumerate(unique_words)}
        
        # 2. Generate Samples (Same as before)
        data, generated_sentences = self._generate_samples(
            tokens, unique_words, word_to_idx, num_samples, vocab_size
        )

        # 3. Get Predictions
        predictions = classifier_fn(generated_sentences)
        
        # 4. Calculate Distances
        distances = pairwise_distances(data, data[0].reshape(1, -1), metric='cosine').ravel()

        # 5. Solve for EACH requested label
        # The paper implies explaining one prediction at a time.
        explanations = {}
        
        for label in labels:
            # Get probabilities for this specific class
            class_predictions = predictions[:, label]
            
            # Solve
            result = self._solve_explanation(
                perturbed_data=data,
                predictions=class_predictions,
                distances=distances,
                num_features=num_features
            )
            
            # Map indices to words
            mapped_exp = []
            for feat_idx, weight in result['explanation']:
                word = unique_words[feat_idx]
                mapped_exp.append((word, weight))
            
            result['explanation_map'] = mapped_exp
            explanations[label] = result
            result['target_class'] = label

            
        return explanations
    

    def _generate_samples(self, tokens, unique_words, word_to_idx, num_samples, vocab_size):
        """
        Generates perturbed samples.
        """
        # Array to store binary vectors. 
        # Row 0 is the original instance (all 1s).
        data = np.ones((num_samples, vocab_size))
        
        # List to store the actual reconstructed strings
        sentences = []
        
        # --- Sample 0: The Original ---
        sentences.append("".join(tokens))
        
        # --- Samples 1 to N: Random Perturbations ---
        for i in range(1, num_samples):
            # Pick how many features to remove (uniform random)
            # We want to remove at least 1, and at most vocab_size words.
            # actually, standard LIME samples the number of *active* features.
            num_active = self.random_state.randint(1, vocab_size + 1)
            
            # Randomly select indices to keep active (1)
            active_indices = self.random_state.choice(vocab_size, num_active, replace=False)
            
            # Set everything to 0 first (except for the ones we picked)
            # Efficient way: create a zero vector, set active to 1
            sample_vec = np.zeros(vocab_size)
            sample_vec[active_indices] = 1
            data[i] = sample_vec
            
            # Reconstruct the sentence based on this binary vector
            # We iterate through the original tokens.
            # If a token is a word in our vocab, we check if it's active in sample_vec.
            # If inactive, we skip it (remove it). Punctuation is kept.
            temp_sentence = []
            for t in tokens:
                if t in word_to_idx:
                    # It's a word
                    idx = word_to_idx[t]
                    if sample_vec[idx] == 1:
                        temp_sentence.append(t)
                else:
                    # It's punctuation/whitespace, keep it to maintain structure
                    temp_sentence.append(t)
            
            sentences.append("".join(temp_sentence))
            
        return data, sentences