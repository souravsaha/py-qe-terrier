import numpy as np
from iqg_learn import IdealQueryGeneration


class IdealQueryGenerationRocchio(IdealQueryGeneration):
    def train_model(
        self,
        X,
        y,
        alpha = 2.0,
        beta = 64.0,
        gamma = 64.0,
    ):
        oracle_rocchio = beta * np.mean(X[y == 1], axis=0) - gamma * np.mean(
            X[y == 0], axis=0
        )
        class TorchLinearRegWrapper:
            def __init__(self, coef_):
                self.coef_ = np.array([coef_])

        return TorchLinearRegWrapper(oracle_rocchio)
