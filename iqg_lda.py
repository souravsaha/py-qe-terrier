import numpy as np
from typing import Union
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from kneed import KneeLocator
from typing import Union
from iqg_learn import IdealQueryGeneration


def svd_noise_floor(S, factor=1.0):
    S = S.detach().cpu().numpy()
    noise = np.median(S[-len(S) // 3 :])
    return np.sum(S > factor * noise)


def lda_with_svd_projection(X, y, S, V, k):
    if k is not None:
        if k == 0:  # use a truncation method
            k = svd_noise_floor(S)
        else:
            k = min(k, X.shape[0])
        print(f"Truncating to {k}...")
        V_k = V[:, :k]  # (n_features, k)
    else:
        V_k = V

    # Reduced representation:
    X_reduced = X @ V_k  # (n_samples, k)

    print("Performing LDA...")
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_reduced, y)

    # Find the coefficient
    w_reduced = lda.coef_.ravel()  # (k,)

    # Project it back in the TF-iDF dimensions and normalize
    w_orig = V_k @ w_reduced  # (n_features,)
    w_orig = w_orig / np.linalg.norm(w_orig)

    # Fix direction
    scores = X @ w_orig

    if np.mean(scores[y == 1]) >= np.mean(scores[y == 0]):
        return w_orig
    return -w_orig


class IdealQueryGenerationLDA(IdealQueryGeneration):
    def train_model(self, X, y, S, V, k=None):

        print(f"--> Generating IEQ with LDA method with {k} SVD truncation")

        coef_ = lda_with_svd_projection(X, y, S, V, k)

        # sklearn-like wrapper
        class TorchLinearRegWrapper:
            def __init__(self, coef_, intercept_=0.0):
                self.coef_ = np.array([coef_])
                self.intercept_ = intercept_

            def predict(self, X):
                X = np.array(X)
                return np.dot(X, self.coef_.T) + self.intercept_

        return TorchLinearRegWrapper(coef_)
