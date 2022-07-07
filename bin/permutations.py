import torch
import numpy as np

from collections import defaultdict


def uniform_replace(unique_elements, mapping):
    mod = unique_elements.shape[0]
    shift = torch.randint_like(mapping, 1, mod)
    new_mapping = (mapping + shift) % mod
    return new_mapping


class BasePermutations:
    def __init__(self, X_num, X_cat, Y):
        self.X_num = X_num
        self.X_cat = X_cat
        self.Y = Y

    def preprocess(self):
        pass

    def permute(self):
        pass

    def gen_permutations(self, part):
        pass 


class ShufflePermutations(BasePermutations):
    def permute(self, X):
        if X is None:
            return None
        return torch.randint_like(X, X.shape[0], dtype=torch.long)

    def gen_permutations(self, part):
        X_num = self.X_num[part]
        X_cat = self.X_cat[part] if self.X_cat else None
        return self.permute(X_num), self.permute(X_cat)


class TargetShufflePermutations(BasePermutations):
    def __init__(self,  X_num, X_cat, Y, is_regression):
        super().__init__(X_num, X_cat, Y)
        self.is_regression = is_regression

    def binarize_target(self, y):
        device = y.device
        y_numpy = y.detach().cpu().numpy()
        bins = np.searchsorted(np.histogram_bin_edges(y_numpy, bins='fd'), y_numpy)
        return torch.tensor(bins, dtype=torch.long).to(device)

    def preprocess(self):
        self.unique_elements_cache = defaultdict(dict)
        self.X_target_splitted = defaultdict(dict)
        for part in ['train', 'val', 'test']: 
            Y = self.Y[part]
            if self.is_regression:
                Y = self.binarize_target(Y)

            unique_elements, mapping = torch.unique(Y, return_inverse=True)
            self.unique_elements_cache[part]['unique_elements'] = unique_elements
            self.unique_elements_cache[part]['mapping'] = mapping

            idxs = torch.arange(Y.shape[0]).to(Y.device)
            for u in unique_elements:
                self.X_target_splitted[part][u.item()] = idxs[Y == u]

    def permute(self, X, part, y_perm, unique_elements):
        if X is None:
            return None
        permutation = torch.zeros_like(X, dtype=torch.long)
        for u in unique_elements:
            cond = (y_perm == u)
            idxs_to_choose = self.X_target_splitted[part][u.item()]
            permutation[cond] = idxs_to_choose[torch.randint_like(X[cond], idxs_to_choose.shape[0], dtype=torch.long)]
        return permutation

    def gen_permutations(self, part):
        X_num = self.X_num[part]
        X_cat = self.X_cat[part] if self.X_cat else None
        cache = self.unique_elements_cache[part]
        unique_elements, mapping = cache['unique_elements'], cache['mapping']
        y_perm = uniform_replace(unique_elements, mapping)
        return self.permute(X_num, part, y_perm, unique_elements), self.permute(X_cat, part, y_perm, unique_elements)


def gen_permutations_class(name, X_num, X_cat, Y, D):
    if name == 'shuffle':
        perm_class = ShufflePermutations(X_num, X_cat, Y)
    elif name == 'target_shuffle':
        perm_class = TargetShufflePermutations(X_num, X_cat, Y, D.is_regression)
    else:
        raise ValueError("Unknown permutation type")

    perm_class.preprocess()
    return perm_class.gen_permutations
