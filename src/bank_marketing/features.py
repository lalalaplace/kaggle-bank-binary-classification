from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from bank_marketing.data import TARGET


def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    orig: pd.DataFrame,
    include_pair_features: bool = True,
    include_count_encoding: bool = True,
):
    """构造基础特征，并保持训练集、测试集和原始数据的编码一致。"""
    combined = pd.concat([train, test, orig], axis=0)
    numeric_cols = list(combined.select_dtypes(include=["int64", "int32"]).columns.drop(TARGET))
    categorical_cols = list(combined.select_dtypes(include="object").columns)

    discretized_cols: list[str] = []
    cardinality: dict[str, int] = {}
    for col in numeric_cols + categorical_cols:
        encoded_col = f"{col}2" if col in numeric_cols else col
        if col in numeric_cols:
            discretized_cols.append(encoded_col)
        combined[encoded_col], _ = combined[col].factorize(sort=True)
        cardinality[encoded_col] = int(combined[encoded_col].max() + 1)
        combined[col] = combined[col].astype("int32")
        combined[encoded_col] = combined[encoded_col].astype("int32")

    pair_cols: list[str] = []
    if include_pair_features:
        new_cols = {}
        for col1, col2 in combinations(categorical_cols + discretized_cols, 2):
            name = "_".join(sorted((col1, col2)))
            new_cols[name] = combined[col1] * cardinality[col2] + combined[col2]
            pair_cols.append(name)
        if new_cols:
            combined = pd.concat([combined, pd.DataFrame(new_cols, index=combined.index)], axis=1)

    count_cols: list[str] = []
    if include_count_encoding:
        for col in categorical_cols + discretized_cols + pair_cols:
            count_col = f"CE_{col}"
            counts = combined.groupby(col)[TARGET].count().astype("int32")
            combined[count_col] = combined[col].map(counts).astype("int32")
            count_cols.append(count_col)

    n_train = len(train)
    n_test = len(test)
    train_features = combined.iloc[:n_train].copy()
    test_features = combined.iloc[n_train : n_train + n_test].copy()

    features = numeric_cols + categorical_cols + discretized_cols + pair_cols + count_cols
    metadata = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "discretized_cols": discretized_cols,
        "pair_cols": pair_cols,
        "count_cols": count_cols,
        "features": features,
    }
    return train_features, test_features, metadata


def oof_target_encode(
    train_col: pd.Series,
    train_y: pd.Series,
    valid_col: pd.Series,
    test_col: pd.Series,
    n_splits: int = 5,
    dtype: str = "float32",
):
    """对单列生成无泄漏目标编码。"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_encoded = np.zeros(len(train_col), dtype=np.float32)

    for tr_idx, val_idx in kf.split(train_col):
        stats = train_y.iloc[tr_idx].groupby(train_col.iloc[tr_idx]).mean()
        oof_encoded[val_idx] = train_col.iloc[val_idx].map(stats).to_numpy()

    full_stats = train_y.groupby(train_col).mean()
    valid_encoded = valid_col.map(full_stats).to_numpy()
    test_encoded = test_col.map(full_stats).to_numpy()

    fill_value = float(np.nanmean(oof_encoded))
    if not np.isfinite(fill_value):
        fill_value = float(train_y.mean())

    return (
        np.nan_to_num(oof_encoded, nan=fill_value).astype(dtype),
        np.nan_to_num(valid_encoded, nan=fill_value).astype(dtype),
        np.nan_to_num(test_encoded, nan=fill_value).astype(dtype),
    )

