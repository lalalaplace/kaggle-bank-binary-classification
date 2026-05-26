from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "y"
ID_COL = "id"


def load_competition_data(data_dir: str | Path, sample_rows: int | None = None):
    """读取 Kaggle 训练集、测试集和 UCI 原始银行营销数据。"""
    data_path = Path(data_dir)
    train = pd.read_csv(data_path / "train.csv").set_index(ID_COL)
    test = pd.read_csv(data_path / "test.csv").set_index(ID_COL)
    orig = pd.read_csv(data_path / "bank-full.csv", delimiter=";")

    if sample_rows is not None:
        train = train.head(sample_rows).copy()
        test = test.head(max(1, sample_rows // 3)).copy()
        orig = orig.head(max(1, sample_rows // 10)).copy()

    test[TARGET] = -1
    orig[TARGET] = orig[TARGET].map({"yes": 1, "no": 0}).astype("int32")
    orig[ID_COL] = (np.arange(len(orig)) + 1_000_000).astype("int64")
    orig = orig.set_index(ID_COL)
    return train, test, orig


def load_sample_submission(data_dir: str | Path) -> pd.DataFrame:
    """读取提交模板。"""
    return pd.read_csv(Path(data_dir) / "sample_submission.csv")

