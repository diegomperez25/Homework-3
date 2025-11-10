import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.decomposition import PCA

class Standard_PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        Z = (X - self.mean)/self.std
        self.pca.fit(Z)
        self.is_fitted = True

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        Z = (X - self.mean)/self.std
        self.pca.fit(Z)
        self.is_fitted = True
        return self.pca.transform(Z)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.is_fitted == False:
            raise NotFittedError("PCA has not yet been fitted.")
        else:
            Z = (X - self.mean)/self.std
            return self.pca.transform(Z)


def PCA_dim(X: np.ndarray, p: float) -> int:
    n_cols = X.shape[1]
    std_pca = Standard_PCA(n_components=n_cols)
    std_pca.fit(X)
    explained_cumsum_seq = np.cumsum(std_pca.pca.explained_variance_ratio_)
    d = np.searchsorted(explained_cumsum_seq, p) + 1
    return d
