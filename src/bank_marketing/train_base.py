from __future__ import annotations

import argparse
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from bank_marketing.data import TARGET, load_competition_data, load_sample_submission
from bank_marketing.features import build_features, oof_target_encode


def log(message: str) -> None:
    """打印带时间戳的进度信息。"""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练一层基础模型并保存 OOF 与测试集预测。")
    parser.add_argument("--data-dir", default="data", help="数据目录。")
    parser.add_argument("--output-dir", default="outputs", help="输出目录。")
    parser.add_argument("--model", choices=["lgbm", "xgb", "cat", "rf"], default="lgbm")
    parser.add_argument("--run-name", default=None, help="输出文件名后缀，默认等于模型名。")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--n-jobs", type=int, default=-1, help="并行线程数；冒烟测试可设为 1。")
    parser.add_argument("--log-period", type=int, default=100, help="模型训练日志输出间隔。")
    parser.add_argument("--sample-rows", type=int, default=None, help="仅用于快速冒烟测试。")
    parser.add_argument("--no-pair-features", action="store_true", help="关闭组合特征。")
    parser.add_argument("--target-encode-pairs", action="store_true", help="对组合特征做无泄漏目标编码。")
    return parser.parse_args()


def make_model(model_name: str, seed: int, device: str, n_jobs: int, log_period: int):
    """按名称创建模型，重型依赖在需要时再导入。"""
    if model_name == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            learning_rate=0.03,
            n_estimators=3000,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=8.0,
            random_state=seed,
            n_jobs=n_jobs,
            device="gpu" if device == "gpu" else "cpu",
            verbosity=-1,
        )
    if model_name == "xgb":
        from xgboost import XGBClassifier

        xgb_device = "cuda" if device == "gpu" else "cpu"
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            learning_rate=0.04,
            n_estimators=3000,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=8.0,
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
            device=xgb_device,
            early_stopping_rounds=200,
        )
    if model_name == "cat":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=3000,
            learning_rate=0.05,
            depth=8,
            random_seed=seed,
            task_type="GPU" if device == "gpu" else "CPU",
            verbose=log_period,
            od_wait=200,
            od_type="Iter",
        )
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=18,
        max_features="sqrt",
        min_samples_split=100,
        min_samples_leaf=50,
        bootstrap=True,
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
        random_state=seed,
    )


def fit_predict(model_name: str, model, X_train, y_train, X_val, y_val, X_test, log_period: int):
    """训练单折模型并返回验证集和测试集概率。"""
    if model_name == "lgbm":
        import lightgbm as lgb

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(200, verbose=True), lgb.log_evaluation(log_period)],
        )
    elif model_name == "xgb":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=log_period)
    elif model_name == "cat":
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    else:
        model.fit(X_train, y_train)
    return model.predict_proba(X_val)[:, 1], model.predict_proba(X_test)[:, 1]


def main() -> None:
    args = parse_args()
    run_name = args.run_name or args.model
    output_dir = Path(args.output_dir)
    (output_dir / "oof").mkdir(parents=True, exist_ok=True)
    (output_dir / "pred").mkdir(parents=True, exist_ok=True)
    (output_dir / "submissions").mkdir(parents=True, exist_ok=True)

    log(f"开始读取数据：data_dir={args.data_dir}")
    train, test, orig = load_competition_data(args.data_dir, sample_rows=args.sample_rows)
    log(f"数据读取完成：train={train.shape}, test={test.shape}, orig={orig.shape}")

    log("开始构造特征")
    train_df, test_df, metadata = build_features(
        train,
        test,
        orig,
        include_pair_features=not args.no_pair_features,
        include_count_encoding=True,
    )
    log(f"特征构造完成：特征数={len(metadata['features'])}")

    features = metadata["features"]
    target_encode_cols = metadata["pair_cols"] if args.target_encode_pairs else []
    y = train_df[TARGET].astype("int32")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(train_df), dtype=np.float32)
    pred = np.zeros(len(test_df), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df[features], y), 1):
        log(f"开始训练 Fold {fold}/{args.folds}：model={args.model}, device={args.device}")
        X_tr = train_df.iloc[tr_idx][features].copy()
        X_va = train_df.iloc[va_idx][features].copy()
        X_te = test_df[features].copy()
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        for col in target_encode_cols:
            X_tr[col], X_va[col], X_te[col] = oof_target_encode(X_tr[col], y_tr, X_va[col], X_te[col])

        X_tr = X_tr.astype("float32")
        X_va = X_va.astype("float32")
        X_te = X_te.astype("float32")

        model = make_model(args.model, args.seed + fold, args.device, args.n_jobs, args.log_period)
        val_pred, test_pred = fit_predict(args.model, model, X_tr, y_tr, X_va, y_va, X_te, args.log_period)
        oof[va_idx] = val_pred.astype(np.float32)
        pred += (test_pred / args.folds).astype(np.float32)

        log(f"Fold {fold} AUC: {roc_auc_score(y_va, oof[va_idx]):.6f}")
        del model, X_tr, X_va, X_te
        gc.collect()

    auc = roc_auc_score(y, oof)
    log(f"{run_name} OOF AUC: {auc:.6f}")

    np.save(output_dir / "oof" / f"oof_{run_name}.npy", oof)
    np.save(output_dir / "pred" / f"pred_{run_name}.npy", pred)
    log("OOF 和测试集预测已保存")

    sub = load_sample_submission(args.data_dir)
    if len(sub) != len(pred):
        sub = sub.head(len(pred)).copy()
    sub[TARGET] = pred
    sub.to_csv(output_dir / "submissions" / f"{run_name}.csv", index=False)
    log(f"提交文件已保存：{output_dir / 'submissions' / f'{run_name}.csv'}")


if __name__ == "__main__":
    main()
