import numpy as np
import pandas as pd


def engineer_features(df):
    """
    Add 17 domain-specific financial features to raw credit data.
    Covers payment behavior, utilization ratios, trends, and risk signals.
    Clips dollar columns to 99th percentile to remove outliers.
    """
    df = df.copy()

    pay_cols     = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
    bill_cols    = ["BILL_AMT1","BILL_AMT2","BILL_AMT3",
                    "BILL_AMT4","BILL_AMT5","BILL_AMT6"]
    pay_amt_cols = ["PAY_AMT1","PAY_AMT2","PAY_AMT3",
                    "PAY_AMT4","PAY_AMT5","PAY_AMT6"]

    # Payment behavior
    df["late_pay_count"]  = (df[pay_cols] > 0).sum(axis=1)
    df["max_pay_status"]  = df[pay_cols].max(axis=1)
    df["pay_trend"]       = df["PAY_0"] - df["PAY_2"]
    df["pay_trend_long"]  = df["PAY_0"] - df["PAY_6"]

    # Bill features
    df["avg_bill"]        = df[bill_cols].mean(axis=1)
    df["avg_pay_amount"]  = df[pay_amt_cols].mean(axis=1)
    df["bill_std"]        = df[bill_cols].std(axis=1)
    df["pay_amt_std"]     = df[pay_amt_cols].std(axis=1)
    df["bill_trend"]      = df["BILL_AMT1"] - df["BILL_AMT3"]
    df["bill_trend_long"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    df["pay_amt_trend"]   = df["PAY_AMT1"]  - df["PAY_AMT3"]

    # Ratio features (clipped to prevent explosion)
    df["util_ratio"]     = (df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1)).clip(0, 5)
    df["avg_util_ratio"] = (df["avg_bill"]  / (df["LIMIT_BAL"] + 1)).clip(0, 5)
    df["pay_ratio_1"]    = (df["PAY_AMT1"]  / (df["BILL_AMT1"] + 1)).clip(0, 5)
    df["pay_ratio_2"]    = (df["PAY_AMT2"]  / (df["BILL_AMT2"] + 1)).clip(0, 5)
    df["bill_to_pay"]    = (df["avg_bill"]  / (df["avg_pay_amount"] + 1)).clip(0, 50)
    df["risk_score"]     = (df["util_ratio"] * df["late_pay_count"]).clip(0, 20)

    # Clip dollar columns to 99th percentile
    dollar_cols = bill_cols + pay_amt_cols + [
        "avg_bill", "avg_pay_amount", "bill_std",
        "pay_amt_std", "bill_trend", "bill_trend_long",
        "pay_amt_trend", "LIMIT_BAL"
    ]
    for col in dollar_cols:
        p01 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        df[col] = df[col].clip(p01, p99)

    # Safety check
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())

    return df
