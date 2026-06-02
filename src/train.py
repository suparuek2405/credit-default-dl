import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix


def train_tabnet(model, X_train, y_train, X_val, y_val,
                 batch_size=1024, max_epochs=200, patience=20):
    """
    Train a TabNetClassifier with early stopping on val AUC.

    Args:
        model:      TabNetClassifier instance
        X_train:    scaled training features
        y_train:    training labels (int64)
        X_val:      scaled validation features
        y_val:      validation labels (int64)
        batch_size: training batch size
        max_epochs: maximum training epochs
        patience:   early stopping patience

    Returns:
        trained model
    """
    model.fit(
        X_train            = X_train,
        y_train            = y_train,
        eval_set           = [(X_val, y_val)],
        eval_metric        = ["auc"],
        max_epochs         = max_epochs,
        patience           = patience,
        weights            = 1,
        batch_size         = batch_size,
        virtual_batch_size = batch_size // 8
    )
    return model


def evaluate_tabnet(model, X, y, threshold=None):
    """
    Evaluate TabNet on a dataset.
    If threshold is None, finds best threshold using Youden J statistic.

    Returns:
        auc, tp, fp, fn, tn, recall, threshold
    """
    probs = model.predict_proba(X)[:, 1]
    auc   = roc_auc_score(y, probs)

    if threshold is None:
        fpr, tpr, thresholds = roc_curve(y, probs)
        threshold = thresholds[np.argmax(tpr - fpr)]

    preds          = (probs >= threshold).astype(int)
    cm             = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    recall         = tp / (tp + fn)

    return auc, tp, fp, fn, tn, recall, threshold
