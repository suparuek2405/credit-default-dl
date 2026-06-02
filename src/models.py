import torch
from pytorch_tabnet.tab_model import TabNetClassifier


def build_tabnet(params, device="cpu"):
    """
    Build TabNetClassifier with given hyperparameters.
    Best params found by Optuna on UCI Credit Card dataset:
        n_d=128, n_steps=4, gamma=1.114, n_independent=2,
        n_shared=2, momentum=0.264, lambda_sparse=1.99e-04,
        lr=3.47e-02, batch_size=1024

    Args:
        params: dict of hyperparameters
        device: "cuda" or "cpu"

    Returns:
        TabNetClassifier instance
    """
    return TabNetClassifier(
        n_d              = params["n_d"],
        n_a              = params["n_d"],
        n_steps          = params["n_steps"],
        gamma            = params["gamma"],
        n_independent    = params["n_independent"],
        n_shared         = params["n_shared"],
        momentum         = params["momentum"],
        lambda_sparse    = params["lambda_sparse"],
        clip_value       = 2,
        optimizer_fn     = torch.optim.Adam,
        optimizer_params = {"lr": params["lr"]},
        scheduler_fn     = torch.optim.lr_scheduler.StepLR,
        scheduler_params = {"step_size": 10, "gamma": 0.9},
        device_name      = device,
        verbose          = 0
    )


# Best hyperparameters found by Optuna (100 trials)
BEST_PARAMS = {
    "n_d":           128,
    "n_steps":       4,
    "gamma":         1.1142627959860096,
    "n_independent": 2,
    "n_shared":      2,
    "momentum":      0.2638934447096518,
    "lambda_sparse": 0.00019922106594145444,
    "lr":            0.034663623969772316,
    "batch_size":    1024
}
