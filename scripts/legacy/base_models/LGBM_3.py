from pathlib import Path

Path("outputs/oof").mkdir(parents=True, exist_ok=True)
Path("outputs/pred").mkdir(parents=True, exist_ok=True)
Path("outputs/submissions").mkdir(parents=True, exist_ok=True)
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# 读取三个数据集的信息
train = pd.read_csv('data/train.csv').set_index('id')

test = pd.read_csv('data/test.csv').set_index('id')
test['y'] = -1

orig = pd.read_csv('data/bank-full.csv',delimiter=";")
orig['y'] = orig.y.map({'yes':1,'no':0})
orig['id'] = (np.arange(len(orig))+1e6).astype('int')
orig = orig.set_index('id')

combine = pd.concat([train,test,orig],axis=0)
print("Shape:", combine.shape)
TARGET = 'y'

NUMS = list(combine.select_dtypes(include=['int64']).columns.drop('y'))
CATS = list(combine.select_dtypes(include='object').columns)

# 离散化分类标签，额外加上离散化连续标签
CATS1 = []
SIZES = {}
for c in NUMS + CATS:
    n = c
    if c in NUMS: 
        n = f"{c}2"
        CATS1.append(n)
    combine[n],_ = combine[c].factorize()
    SIZES[n] = combine[n].max()+1

    combine[c] = combine[c].astype('int32')
    combine[n] = combine[n].astype('int32')

print("New CATS:", CATS1 )
print("Cardinality of all CATS:", SIZES )    
print(combine.columns)

# 组合特征
pairs = combinations(CATS + CATS1, 2)
new_cols = {}
CATS2 = []

for c1, c2 in pairs:
    name = "_".join(sorted((c1, c2)))
    new_cols[name] = combine[c1] * SIZES[c2] + combine[c2]
    CATS2.append(name)               

if new_cols:
    new_df = pd.DataFrame(new_cols)         
    combine = pd.concat([combine, new_df], axis=1) 

print(f"Created {len(CATS2)} new CAT columns")

train = combine.iloc[:len(train)]
test = combine.iloc[len(train):len(train)+len(test)]
orig = combine.iloc[-len(orig):]
del combine
print("Train shape", train.shape,"Test shape", test.shape,"Original shape", orig.shape )

# LGBM算法训练 -------------------------------------------------

def oof_target_encode(train_col, train_y, valid_col, test_col, n_splits=10, dtype="float32"):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_encoded = np.zeros(len(train_col), dtype=np.float32)

    for tr_idx, val_idx in kf.split(train_col):
        tr_c = train_col.iloc[tr_idx]
        tr_y = train_y.iloc[tr_idx]
        stats = tr_y.groupby(tr_c).mean()
        oof_encoded[val_idx] = train_col.iloc[val_idx].map(stats).values

    full_stats = train_y.groupby(train_col).mean()
    valid_encoded = valid_col.map(full_stats)
    test_encoded  = test_col.map(full_stats)

    return (oof_encoded.astype(dtype), valid_encoded.astype(dtype), test_encoded.astype(dtype))

SEED = 42
FOLDS = 5

FEATURES = NUMS + CATS + CATS1 + CATS2
TARGET = "y"

LGBM_CONFIGS = [
    ("lgb_c1", dict(
        learning_rate=0.06,
        num_leaves=64,
        min_data_in_leaf=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=6.0
    )),
    ("lgb_c2", dict(
        learning_rate=0.05,
        num_leaves=48,
        min_data_in_leaf=120,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=12.0
    )),
    ("lgb_c3", dict(
        learning_rate=0.04,
        num_leaves=128,
        min_data_in_leaf=30,
        feature_fraction=0.85,
        bagging_fraction=0.75,
        bagging_freq=1,
        lambda_l2=4.0
    )),
]

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

oof_by_config = {}
pred_by_config = {}
score_by_config = {}

for cfg_name, cfg_params in LGBM_CONFIGS:
    print("=" * 70)
    print(f"LGBM CONFIG (fixed seed): {cfg_name}")
    print("=" * 70)

    oof = np.zeros(len(train), dtype=np.float32)
    pred = np.zeros(len(test), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(train)), train[TARGET]), 1
    ):
        print(f"\n--- Fold {fold}/{FOLDS} ---")

        X_train = train.iloc[train_idx][FEATURES].copy()
        X_val   = train.iloc[val_idx][FEATURES].copy()
        y_train = train.iloc[train_idx][TARGET].copy()
        y_val   = train.iloc[val_idx][TARGET].copy()
        X_test  = test[FEATURES].copy()

        # ---------- fold-wise OOF Target Encoding ----------
        for col in CATS2:
            tr_enc, va_enc, te_enc = oof_target_encode(
                X_train[col], y_train,
                X_val[col], X_test[col],
                n_splits=10, dtype="float32"
            )
            fillv = float(np.nanmean(tr_enc)) if np.isfinite(np.nanmean(tr_enc)) else float(y_train.mean())
            X_train[col] = np.nan_to_num(tr_enc, nan=fillv)
            X_val[col]   = np.nan_to_num(np.asarray(va_enc), nan=fillv)
            X_test[col]  = np.nan_to_num(np.asarray(te_enc), nan=fillv)

        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        params = {
            "objective": "binary",
            "metric": "auc",
            "seed": SEED,
            "verbosity": -1,
            "device": "gpu", 
        }
        params.update(cfg_params)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=20000,
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(300, verbose=True),
                lgb.log_evaluation(200),
            ],
        )

        oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        pred += model.predict(X_test, num_iteration=model.best_iteration) / FOLDS

        print(f"Fold {fold} AUC:",
              roc_auc_score(y_val, oof[val_idx]))

        del model, X_train, X_val, y_train, y_val, X_test
        gc.collect()

    auc = roc_auc_score(train[TARGET], oof)
    print(f"\n[{cfg_name}] OOF AUC = {auc:.6f}")

    np.save(f"outputs/oof/oof_{cfg_name}.npy", oof)
    np.save(f"outputs/pred/pred_{cfg_name}.npy", pred)

    oof_by_config[cfg_name] = oof
    pred_by_config[cfg_name] = pred
    score_by_config[cfg_name] = auc

