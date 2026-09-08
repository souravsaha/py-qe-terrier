import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

from iqg_learn import IdealQueryGeneration


class IdealQueryGenerationLeastSq(IdealQueryGeneration):
    def train_model(
        self,
        X,
        y,
        l2_lambda = 0.1,
        lr = 0.1,
        epochs = 5000,
        warmup_epochs = 1000
    ):
        print(f"--> Generating IEQ with Least Squares method with {l2_lambda} regularization constant")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reformat y: 1 -> 100, 0 -> 0
        y_scaled = np.where(np.array(y) == 1, 100.0, 0.0)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).view(-1, 1).to(device)

        model = nn.Linear(X.shape[1], 1, bias=False).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Warmup + Cosine Annealing LR Scheduler
        def get_lr(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return lr * (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing after warmup
                decay_epoch = epoch - warmup_epochs
                total_decay_epochs = epochs - warmup_epochs
                return 1e-5 + 0.5 * (lr - 1e-5) * (
                    1 + math.cos(math.pi * decay_epoch / total_decay_epochs)
                )

        for epoch in range(epochs):
            optimizer.zero_grad()

            preds = model(X_tensor)
            mse_loss = criterion(preds, y_tensor)

            # l1_loss = l2_lambda * torch.norm(model.weight, p=1)
            l2_loss = l2_lambda * torch.norm(model.weight, p=2) ** 2

            loss = mse_loss + l2_loss

            loss.backward()
            optimizer.step()

            # Update learning rate manually
            lr_now = get_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_now

            if epoch % 1000 == 0 or epoch == epochs - 1:
                print(
                    f"[Torch] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f} - LR: {lr_now:.6f}"
                )

        with torch.no_grad():
            coef_ = model.weight.detach().cpu().numpy().flatten()

        # sklearn-like wrapper
        class TorchLinearRegWrapper:
            def __init__(self, coef_, intercept_=0.0):
                self.coef_ = np.array([coef_])
                self.intercept_ = intercept_

            def predict(self, X):
                X = np.array(X)
                return np.dot(X, self.coef_.T) + self.intercept_

        return TorchLinearRegWrapper(coef_)
