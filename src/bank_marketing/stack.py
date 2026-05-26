from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from bank_marketing.data import TARGET, load_sample_submission


def log(message: str) -> None:
    """打印带时间戳的进度信息。"""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="读取一层 OOF 和测试预测，训练二层 stacking 模型。")
    parser.add_argument("--data-dir", default="data", help="数据目录。")
    parser.add_argument("--output-dir", default="outputs", help="输出目录。")
    parser.add_argument("--model", choices=["logreg", "lgbm", "xgb", "cat"], default="logreg")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rows", type=int, default=None, help="仅用于匹配一层小样本冒烟测试。")
    parser.add_argument("--features", nargs="*", default=None, help="指定一层 run_name；默认读取 outputs/oof 下全部文件。")
    return parser.parse_args()


def discover_features(output_dir: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    names = []
    for path in sorted((output_dir / "oof").glob("oof_*.npy")):
        names.append(path.stem.replace("oof_", "", 1))
    if not names:
        raise FileNotFoundError("未找到 outputs/oof/oof_*.npy，请先训练一层模型。")
    return names


def load_meta_features(output_dir: Path, names: list[str]):
    oof_arrays = []
    pred_arrays = []
    for name in names:
        oof_path = output_dir / "oof" / f"oof_{name}.npy"
        pred_path = output_dir / "pred" / f"pred_{name}.npy"
        if not oof_path.exists() or not pred_path.exists():
            raise FileNotFoundError(f"缺少一层预测文件：{name}")
        oof_arrays.append(np.load(oof_path).reshape(-1, 1))
        pred_arrays.append(np.load(pred_path).reshape(-1, 1))
    return np.hstack(oof_arrays).astype("float32"), np.hstack(pred_arrays).astype("float32")


def make_model(model_name: str, seed: int):
    if model_name == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            learning_rate=0.03,
            n_estimators=5000,
            num_leaves=15,
            max_depth=4,
            min_child_samples=50,
            subsample=0.8,
            reg_lambda=10.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    if model_name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            learning_rate=0.03,
            n_estimators=3000,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=1.0,
            reg_lambda=10.0,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            device="cpu",
        )
    if model_name == "cat":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=3000,
            learning_rate=0.03,
            depth=4,
            random_seed=seed,
            task_type="CPU",
            verbose=False,
            od_wait=200,
            od_type="Iter",
        )
    return LogisticRegression(C=1.0, max_iter=1000, random_state=seed)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    run_name = args.run_name or f"{args.model}_stacking"
    (output_dir / "submissions").mkdir(parents=True, exist_ok=True)

    log("开始读取一层 OOF 和测试集预测")
    feature_names = discover_features(output_dir, args.features)
    X_oof, X_pred = load_meta_features(output_dir, feature_names)
    log(f"一层特征读取完成：features={feature_names}, X_oof={X_oof.shape}, X_pred={X_pred.shape}")

    train = pd.read_csv(Path(args.data_dir) / "train.csv")
    if args.sample_rows is not None:
        train = train.head(args.sample_rows).copy()
    y = train[TARGET].to_numpy(dtype=np.int32)

    if len(y) != X_oof.shape[0]:
        raise ValueError(f"OOF 行数 {X_oof.shape[0]} 与训练集行数 {len(y)} 不一致。")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_meta = np.zeros(len(y), dtype=np.float32)
    pred_meta = np.zeros(X_pred.shape[0], dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_oof, y), 1):
        log(f"开始训练 stacking Fold {fold}/{args.folds}：model={args.model}")
        model = make_model(args.model, args.seed + fold)
        model.fit(X_oof[tr_idx], y[tr_idx])
        oof_meta[va_idx] = model.predict_proba(X_oof[va_idx])[:, 1].astype(np.float32)
        pred_meta += (model.predict_proba(X_pred)[:, 1] / args.folds).astype(np.float32)
        log(f"Fold {fold} AUC: {roc_auc_score(y[va_idx], oof_meta[va_idx]):.6f}")

    log(f"{run_name} OOF AUC: {roc_auc_score(y, oof_meta):.6f}")
    np.save(output_dir / "oof" / f"oof_{run_name}.npy", oof_meta)
    np.save(output_dir / "pred" / f"pred_{run_name}.npy", pred_meta)

    sub = load_sample_submission(args.data_dir)
    if len(sub) != len(pred_meta):
        sub = sub.head(len(pred_meta)).copy()
    sub[TARGET] = pred_meta
    sub.to_csv(output_dir / "submissions" / f"{run_name}.csv", index=False)
    log(f"提交文件已保存：{output_dir / 'submissions' / f'{run_name}.csv'}")


if __name__ == "__main__":
    main()
