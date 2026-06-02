import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_split(df, test_size=0.15, val_size=0.176, random_state=42):
    """
    Split engineered dataframe into train/val/test sets.
    Fit scaler on train only to prevent data leakage.

    Returns:
        X_train_sc, X_val_sc, X_test_sc,
        y_train, y_val, y_test,
        scaler, feature_names
    """
    feature_names = df.drop("target", axis=1).columns.tolist()
    X = df.drop("target", axis=1).values.astype("float32")
    y = df["target"].values.astype("int64")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size,
        random_state=random_state, stratify=y_temp
    )

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    return (X_train_sc, X_val_sc, X_test_sc,
            y_train, y_val, y_test,
            scaler, feature_names)
