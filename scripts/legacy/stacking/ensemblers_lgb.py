from pathlib import Path

Path("outputs/oof").mkdir(parents=True, exist_ok=True)
Path("outputs/pred").mkdir(parents=True, exist_ok=True)
Path("outputs/submissions").mkdir(parents=True, exist_ok=True)
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

oof_paths = [
    r"outputs/oof/oof_cb_seed7.npy",
    r"outputs/oof/oof_cb_seed42.npy",
    r"outputs/oof/oof_cb_seed100.npy",
    r"outputs/oof/oof_cb_seed124.npy",
    r"outputs/oof/oof_cb_cb_c1.npy",
    r"outputs/oof/oof_cb_cb_c2.npy",
    r"outputs/oof/oof_cb_cb_c3.npy",
    r"outputs/oof/oof_lgb_c1_baseline.npy",
    r"outputs/oof/oof_lgb_c2_strong_reg.npy",
    r"outputs/oof/oof_lgb_c3_more_complex.npy",
    r"outputs/oof/oof_xgb_with_orig_rows.npy",
    r"outputs/oof/oof_xgb_with_orig_cols.npy",
    r"outputs/oof/oof_xgb_c1.npy",
    r"outputs/oof/oof_xgb_c2.npy",
    r"outputs/oof/oof_xgb_c3.npy",
    r"outputs/oof/oof_rf_c1.npy",

    
]
test_paths = [
    r"outputs/pred/pred_cb_seed7.npy",
    r"outputs/pred/pred_cb_seed42.npy",
    r"outputs/pred/pred_cb_seed100.npy",
    r"outputs/pred/pred_cb_seed124.npy",
    r"outputs/pred/pred_cb_cb_c1.npy",
    r"outputs/pred/pred_cb_cb_c2.npy",
    r"outputs/pred/pred_cb_cb_c3.npy",
    r"outputs/pred/pred_lgb_c1_baseline.npy",
    r"outputs/pred/pred_lgb_c2_strong_reg.npy",
    r"outputs/pred/pred_lgb_c3_more_complex.npy",
    r"outputs/pred/pred_xgb_with_orig_rows.npy",
    r"outputs/pred/pred_xgb_with_orig_cols.npy",
    r"outputs/pred/pred_xgb_c1.npy",
    r"outputs/pred/pred_xgb_c2.npy",
    r"outputs/pred/pred_xgb_c3.npy",
    r"outputs/pred/pred_rf_c1.npy",
]

oof = [np.load(p).reshape(-1, 1) for p in oof_paths]
pred = [np.load(p).reshape(-1, 1) for p in test_paths]

X_meta_oof = np.hstack(oof).astype(np.float32)  
X_meta_pred  = np.hstack(pred).astype(np.float32)  

print("X_meta_off:", X_meta_oof.shape)
print("X_meta_pred :", X_meta_pred.shape)

train = pd.read_csv("data/train.csv")
y = train["y"].values.astype(np.int32)

SEED = 42
FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

oof_meta = np.zeros(len(y), dtype=np.float32)
test_meta = np.zeros(X_meta_pred.shape[0], dtype=np.float32)

params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.03,
    n_estimators=20000,         
    num_leaves=15,               
    max_depth=4,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=1.0,       
    reg_alpha=0.0,
    reg_lambda=10.0,          
    random_state=SEED,
    n_jobs=-1
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_meta_oof, y), 1):
    X_tr, X_va = X_meta_oof[tr_idx], X_meta_oof[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=300, verbose=True),
            lgb.log_evaluation(period=200)
        ]
    )

    oof_meta[va_idx] = model.predict_proba(X_va)[:, 1].astype(np.float32)
    test_meta += (model.predict_proba(X_meta_pred)[:, 1] / FOLDS).astype(np.float32)

    auc = roc_auc_score(y_va, oof_meta[va_idx])
    print(f"Fold {fold} AUC = {auc:.6f}")

full_auc = roc_auc_score(y, oof_meta)
print(f"Meta OOF AUC = {full_auc:.6f}")

sub = pd.read_csv("data/sample_submission.csv")
sub["y"] = test_meta
sub.to_csv("outputs/submissions/LGBM_stacking.csv", index=False)

