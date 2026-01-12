import numpy as np

class SubmodularPick:
    def __init__(self, explanations, num_features_to_select=10):
        """
        Args:
            explanations: List of explanation dictionaries from LIME.
                          Each dict must have 'explanation_map': [('feature_name', weight), ...]
            num_features_to_select: K (how many top features to look at per explanation)
        """
        self.explanations = explanations
        self.num_features = num_features_to_select
        
        # 1. Build the Global Matrix W and Importance Vector I
        self.W, self.feature_names, self.I = self._build_matrix()

    def _build_matrix(self):
        """
        Builds the Explanation Matrix W (n_samples x d_features).
        W[i, j] = Importance of feature j in instance i.
        """
        # Step A: Collect Global Vocabulary from the explanations
        # We only care about features that actually appeared in the top K of explanations
        global_features = set()
        for exp in self.explanations:
            # Sort by magnitude of weight
            top_k = sorted(exp['explanation_map'], key=lambda x: abs(x[1]), reverse=True)[:self.num_features]
            for feature_name, weight in top_k:
                global_features.add(feature_name)
        
        feature_list = sorted(list(global_features))
        feature_to_idx = {f: i for i, f in enumerate(feature_list)}
        
        n_samples = len(self.explanations)
        n_features = len(feature_list)
        
        # Step B: Fill the Matrix W
        W = np.zeros((n_samples, n_features))
        
        for i, exp in enumerate(self.explanations):
            top_k = sorted(exp['explanation_map'], key=lambda x: abs(x[1]), reverse=True)[:self.num_features]
            for feature_name, weight in top_k:
                if feature_name in feature_to_idx:
                    j = feature_to_idx[feature_name]
                    # The paper uses sqrt(|weight|) or just |weight| as importance.
                    # We use absolute weight.
                    W[i, j] = np.abs(weight)
        
        # Step C: Calculate Global Importance I
        # How important is feature j across ALL instances?
        # Eq: I_j = sqrt(sum(W_ij))
        I = np.sqrt(np.sum(W, axis=0))
        
        return W, feature_list, I

    def pick_instances(self, budget_B):
        """
        Algorithm 2: Greedy selection to maximize coverage.
        Returns indices of the selected instances.
        """
        n_samples, n_features = self.W.shape
        selected_indices = []
        
        # 'covered_features' is a set V in the paper
        # We keep track of the importance of features we have already "explained" to the user
        current_covered_importance = np.zeros(n_features)
        
        for _ in range(budget_B):
            best_idx = -1
            best_gain = -1
            
            # Iterate through all instances not yet selected
            for i in range(n_samples):
                if i in selected_indices:
                    continue
                
                # Calculate Marginal Gain:
                # How much DOES this instance add to our understanding?
                # Gain = Importance of (Features in i MINUS Features already covered)
                
                # Get features present in this instance
                present_mask = self.W[i, :] > 0
                
                # We want to cover features that are NOT yet fully covered.
                # Simple greedy coverage: sum(I_j) for j in instance i that isn't covered yet.
                
                # Features in 'i' that we haven't seen yet
                unseen_mask = np.logical_and(present_mask, current_covered_importance == 0)
                
                gain = np.sum(self.I[unseen_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx != -1 and best_gain > 0:
                selected_indices.append(best_idx)
                # Mark features of the selected instance as covered
                current_covered_importance = np.logical_or(current_covered_importance, self.W[best_idx, :] > 0)
            else:
                break # No more useful instances to add
                
        return selected_indices