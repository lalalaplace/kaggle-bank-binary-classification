from pathlib import Path

Path("outputs/oof").mkdir(parents=True, exist_ok=True)
Path("outputs/pred").mkdir(parents=True, exist_ok=True)
Path("outputs/submissions").mkdir(parents=True, exist_ok=True)
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
import gc
from tqdm import tqdm

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

# CATBoost算法训练 -------------------------------------------------
def oof_target_encode(train_col, train_y, valid_col, test_col, n_splits=10, dtype='float32'):
    '''
    生成目标函数
    '''
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_encoded = np.zeros(len(train_col))

    # ---------- OOF for train ----------
    for tr_idx, val_idx in kf.split(train_col):
        tr_c = train_col.iloc[tr_idx]
        tr_y = train_y.iloc[tr_idx]

        stats = tr_y.groupby(tr_c).mean()
        oof_encoded[val_idx] = train_col.iloc[val_idx].map(stats)

    # ---------- full train stats ----------
    full_stats = train_y.groupby(train_col).mean()

    valid_encoded = valid_col.map(full_stats)
    test_encoded  = test_col.map(full_stats)

    return (oof_encoded.astype(dtype), valid_encoded.astype(dtype), test_encoded.astype(dtype))

FEATURES = NUMS+CATS+CATS1+CATS2
SEEDS = [7, 42, 100, 124]    
FOLDS = 5

# containers for mean
oof_all_seeds = []
pred_all_seeds = []
seed_scores = []

skf = StratifiedKFold(n_splits=FOLDS, random_state=42, shuffle=True)

# multi-seed loop
for seed in SEEDS:
    print("#"*25)
    print(f"### CatBoost seed = {seed} ###")
    print("#"*25)

    oof = np.zeros(len(train), dtype=np.float32)
    pred = np.zeros(len(test), dtype=np.float32)

    for idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train)), train[TARGET])):
        print(f"\n--- Fold {idx + 1}/{FOLDS} ---")

        X_train = train.iloc[train_idx][FEATURES].copy()
        X_val   = train.iloc[val_idx][FEATURES].copy()
        y_train = train.iloc[train_idx][TARGET].copy()
        y_val   = train.iloc[val_idx][TARGET].copy()
        X_test  = test[FEATURES].copy()

        # ---------- fold-wise OOF Target Encoding ----------
        for col in tqdm(CATS2, desc="OOF Target Encoding"):
            X_train[col], X_val[col], X_test[col] = oof_target_encode(
                train_col = X_train[col],
                train_y   = y_train,
                valid_col = X_val[col],
                test_col  = X_test[col],
                n_splits  = 10,
                dtype     = "float32"
            )

        # ---------- CatBoost ----------
        cat_features = [
            X_train.columns.get_loc(c)
            for c in CATS if c in X_train.columns
        ]

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=5000,
            learning_rate=0.06,
            depth=8,
            random_seed=seed,
            task_type="GPU",
            verbose=200,
            od_wait=300,
            od_type="Iter"
        )

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool   = Pool(X_val, y_val, cat_features=cat_features)
        test_pool  = Pool(X_test, cat_features=cat_features)

        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        oof[val_idx] = model.predict_proba(val_pool)[:, 1].astype(np.float32)
        pred += model.predict_proba(test_pool)[:, 1].astype(np.float32) / FOLDS

        print(f"Fold {idx + 1} AUC:",
              roc_auc_score(y_val, oof[val_idx]))

        del model, X_train, X_val, y_train, y_val, X_test
        gc.collect()

    # save per-seed
    auc = roc_auc_score(train[TARGET], oof)
    print(f"\nSeed {seed} OOF AUC = {auc:.6f}")

    np.save(f"outputs/oof/oof_cb_seed{seed}.npy", oof)
    np.save(f"outputs/pred/pred_cb_seed{seed}.npy", pred)

    oof_all_seeds.append(oof)
    pred_all_seeds.append(pred)
    seed_scores.append(auc)

# mean over seeds (for stacking)
oof_mean = np.mean(oof_all_seeds, axis=0).astype(np.float32)
pred_mean = np.mean(pred_all_seeds, axis=0).astype(np.float32)

auc_mean = roc_auc_score(train[TARGET], oof_mean)
print("\n" + "-" * 70)
print(f"Mean over {len(SEEDS)} seeds OOF AUC = {auc_mean:.6f}")
print("-" * 70)

np.save("outputs/oof/oof_cb_mean.npy", oof_mean)
np.save("outputs/pred/pred_cb_mean.npy", pred_mean)

# 保存 id，确保 stacking 对齐
np.save("outputs/train_ids.npy", train.index.values)
np.save("outputs/test_ids.npy", test.index.values)

sub = pd.read_csv("data/sample_submission.csv")
sub['y'] = pred_mean
sub.to_csv("outputs/submissions/submission.csv", index=False)


