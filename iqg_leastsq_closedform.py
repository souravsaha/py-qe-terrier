import torch
import numpy as np

from iqg_learn import IdealQueryGeneration


class IdealQueryGenerationLeastSqDirect(IdealQueryGeneration):
    def ridge_regression_cholesky_gpu(self, X, y, l2_lambda=0.1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1)

        n_features = X_tensor.shape[1]
        A = X_tensor.T @ X_tensor + l2_lambda * torch.eye(
            n_features, dtype=torch.float32, device=device
        )
        b = X_tensor.T @ y_tensor

        # Cholesky decomposition
        try:
            L = torch.linalg.cholesky(A)
            w = torch.cholesky_solve(b, L)
        except RuntimeError:
            # Fallback: use solve if Cholesky fails
            print("[WARN] Cholesky failed, falling back to torch.linalg.solve")
            w = torch.linalg.solve(A, b)

        return w.detach().cpu().numpy().flatten()

    def train_model(self, X, y, l2_lambda=0.1):
        y_scaled = np.where(y == 1, 100.0, 0.0)

        coef_ = self.ridge_regression_cholesky_gpu(X, y_scaled, X.shape[0] * l2_lambda)
        # coef_ = np.linalg.pinv(X) @ y_scaled

        # sklearn-like wrapper
        class TorchLinearRegWrapper:
            def __init__(self, coef_, intercept_=0.0):
                self.coef_ = np.array([coef_])
                self.intercept_ = intercept_

            def predict(self, X):
                X = np.array(X)
                return np.dot(X, self.coef_.T) + self.intercept_

        return TorchLinearRegWrapper(coef_)
