from pathlib import Path

Path("outputs/oof").mkdir(parents=True, exist_ok=True)
Path("outputs/pred").mkdir(parents=True, exist_ok=True)
Path("outputs/submissions").mkdir(parents=True, exist_ok=True)
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# 读取三个数据集的信息
train = pd.read_csv('data/train.csv').set_index('id')

test = pd.read_csv('data/test.csv').set_index('id')
test['y'] = -1

orig = pd.read_csv('data/bank-full.csv',delimiter=";")
orig['y'] = orig.y.map({'yes':1,'no':0})
orig['id'] = (np.arange(len(orig))+1e6).astype('int')
orig = orig.set_index('id')

combine = pd.concat([train,test,orig],axis=0)

# EDA -------------------------------------------------------------
# 数据类型和缺失值情况
print("Shape:", combine.shape)
# print('\nnull and dtype', combine.info())

# 连续型特征的describe
NUMS = list(combine.select_dtypes(include=['int64']).columns.drop('y'))
# print('\n连续型特征的describe\n', combine[NUMS].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).T)

# 分类型特征的分布
CATS = list(combine.select_dtypes(include='object').columns)
# print('\n分类特征的分布')
# for col in CATS:
#     print(f'\n -----------{col}------------')
#     summary = (
#         combine.groupby(col)['y']
#         .agg(['count', 'mean'])
#         .rename(columns={'mean': 'positive_rate'})
#         .sort_values('positive_rate', ascending=False)
#     )
#     summary['proportion'] = summary['count'] / summary['count'].sum()
#     summary = summary[['count', 'proportion', 'positive_rate']]
#     print(summary)

# 特征工程 -----------------------------------------------------------
combine['balance_log'] = (np.sign(combine['balance']) * np.log1p(np.abs(combine['balance']))).astype('float32')
combine['prev_success'] = (combine['poutcome'].astype(str) == 'success').astype('int8')
combine['contacted_before'] = (combine['pdays'] != -1).astype('int8')
combine['log_pdays'] = np.where(combine['contacted_before'].values == 1,np.log1p(combine['pdays'].values),0).astype('float32')

new_nums = ['balance_log', 'log_pdays']
NUMS.extend(new_nums)
new_cats = ['prev_success', 'contacted_before']
CATS.extend(new_cats)

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

# CE，计数编码
CE = []
CC = CATS+CATS1+CATS2

print(f"Processing {len(CC)} columns... ",end="")
for i,c in enumerate(CC):
    if i%10==0: print(f"{i}, ",end="")
    tmp = combine.groupby(c).y.count()
    tmp = tmp.astype('int32')
    tmp.name = f"CE_{c}"
    CE.append( f"CE_{c}" )
    combine = combine.merge(tmp, on=c, how='left')
print()

train = combine.iloc[:len(train)]
test = combine.iloc[len(train):len(train)+len(test)]
orig = combine.iloc[-len(orig):]
del combine
print("Train shape", train.shape,"Test shape", test.shape,"Original shape", orig.shape )

# XGBoost算法训练 -------------------------------------------------
# 超参数
FEATURES = NUMS+CATS+CATS1+CATS2+CE
print(f"We have {len(FEATURES)} features.")

FOLDS = 7
SEED = 42

params = {
    "objective": "binary:logistic",  
    "eval_metric": "auc",           
    "learning_rate": 0.1,
    "max_depth": 0,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "seed": SEED,
    "device": "cuda",
    "grow_policy": "lossguide", 
    "max_leaves": 32,          
    "alpha": 2.0,
}

# 数据读取
class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, df=None, features=None, target=None, batch_size=256*1024):
        self.features = features
        self.target = target
        self.df = df
        self.it = 0 
        self.batch_size = batch_size
        self.batches = int( np.ceil( len(df) / self.batch_size ) )
        super().__init__()

    def reset(self):
        '''重置it'''
        self.it = 0

    def next(self, input_data):
        '''生成下一部分'''
        if self.it == self.batches:
            return 0 
           
        a = self.it * self.batch_size
        b = min( (self.it + 1) * self.batch_size, len(self.df) )
        dt = self.df.iloc[a:b]
        input_data(data=dt[self.features], label=dt[self.target]) 
        self.it += 1
        return 1

# 一：orig作为行。将orig作为每折的额外样本进行交叉验证---------------
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print("#"*25)  
    print(f"### Fold {fold+1} ###")
    print("#"*25)

    # 训练集加上orig数据
    Xy_train = train.iloc[train_idx][ FEATURES+['y'] ].copy()
    Xy_more = orig[ FEATURES+['y'] ]
    Xy_train = pd.concat([Xy_train,Xy_more], axis=0, ignore_index=True)

    # 验证集和测试集
    X_valid = train.iloc[val_idx][FEATURES].copy()
    y_valid = train.iloc[val_idx]['y']
    X_test = test[FEATURES].copy()

    CC = CATS1+CATS2
    # 目标函数TE，本地不好下载cuml包于是手写了函数
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
    
    for i, c in enumerate(CC):
        if i % 10 == 0: print(f"{i}, ", end="")
        Xy_train[c], X_valid[c], X_test[c] = oof_target_encode(
            Xy_train[c],
            Xy_train['y'],
            X_valid[c],
            X_test[c],
            n_splits=10
        )
    print()

    Xy_train[CATS] = Xy_train[CATS].astype("category")
    X_valid[CATS]  = X_valid[CATS].astype("category")
    X_test[CATS]   = X_test[CATS].astype("category")

    Xy_train = IterLoadForDMatrix(Xy_train, FEATURES, "y")
    dtrain = xgb.QuantileDMatrix(Xy_train, enable_categorical=True, max_bin=256)
    dval  = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, enable_categorical=True)

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=10_000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=200,
        verbose_eval=200
    )

    oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / FOLDS

np.save("outputs/oof/oof_xgb_with_orig_rows.npy", oof_preds.astype("float32"))
np.save("outputs/pred/pred_xgb_with_orig_rows.npy", test_preds.astype("float32"))

# 交叉验证分数
m = roc_auc_score(train.y, oof_preds)
print(f"XGB with Original Data as rows CV = {m}")   

# 特征重要性作图，前20个重要的特征
fig, ax = plt.subplots(figsize=(10, 5))
xgb.plot_importance(model, max_num_features=20, importance_type='gain',ax=ax)
plt.title("Top 20 Feature Importances (XGBoost)")
plt.show()

# 二：orig作为列。用orig的y给数据集一批新的特征----------------
TE_ORIG = []
CC = CATS+CATS1+CATS2

# 增加目标函数均值特征
print(f"Processing {len(CC)} columns... ",end="")
for i,c in enumerate(CC):
    if i%10==0: print(f"{i}, ",end="")
    tmp = orig.groupby(c).y.mean()
    tmp = tmp.astype('float32')
    tmp.name = f"TE_ORIG_{c}"
    TE_ORIG.append( f"TE_ORIG_{c}" )
    train = train.merge(tmp, on=c, how='left')
    test = test.merge(tmp, on=c, how='left')
print()

FEATURES += TE_ORIG
print(f"We have {len(FEATURES)} features.")

FOLDS = 7
SEED = 42

params = {
    "objective": "binary:logistic",  
    "eval_metric": "auc",           
    "learning_rate": 0.1,
    "max_depth": 0,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "seed": SEED,
    "device": "cuda",
    "grow_policy": "lossguide", 
    "max_leaves": 32,           
    "alpha": 2.0,
}

oof_preds2 = np.zeros(len(train))
test_preds2 = np.zeros(len(test))

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print("#"*25)
    print(f"### Fold {fold+1} ###")
    print("#"*25)

    Xy_train = train.iloc[train_idx][ FEATURES+['y'] ].copy()    
    X_valid = train.iloc[val_idx][FEATURES].copy()
    y_valid = train.iloc[val_idx]['y']
    X_test = test[FEATURES].copy()

    CC = CATS1+CATS2
    print(f"Target encoding {len(CC)} features... ",end="")
    for i, c in enumerate(CC):
        if i % 10 == 0: print(f"{i}, ", end="")
        Xy_train[c], X_valid[c], X_test[c] = oof_target_encode(
            Xy_train[c],
            Xy_train['y'],
            X_valid[c],
            X_test[c],
            n_splits=10
        )
    print()

    Xy_train[CATS] = Xy_train[CATS].astype('category')
    X_valid[CATS] = X_valid[CATS].astype('category')
    X_test[CATS] = X_test[CATS].astype('category')

    Xy_train = IterLoadForDMatrix(Xy_train, FEATURES, 'y')
    dtrain = xgb.QuantileDMatrix(Xy_train, enable_categorical=True, max_bin=256)
    dval   = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
    dtest  = xgb.DMatrix(X_test, enable_categorical=True)

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=10_000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=200,
        verbose_eval=200
    )

    oof_preds2[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    test_preds2 += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / FOLDS

np.save("outputs/oof/oof_xgb_with_orig_cols.npy", oof_preds2.astype("float32"))
np.save("outputs/pred/pred_xgb_with_orig_cols.npy", test_preds2.astype("float32"))

# CV分数
m = roc_auc_score(train.y, oof_preds2)
print(f"XGB with Original Data as columns CV = {m}")

# 特征重要性
fig, ax = plt.subplots(figsize=(10, 5))
xgb.plot_importance(model, max_num_features=20, importance_type='gain',ax=ax)
plt.title("Top 20 Feature Importances (XGBoost)")
plt.show()

# 两种方式结合 ---------------------------------------------------
m = roc_auc_score(train.y, oof_preds+oof_preds2)
print(f"Ensemble CV = {m}")

np.save("outputs/oof/oof_xgb_with_orig_rows.npy", oof_preds)
np.save("outputs/oof/oof_xgb_with_orig_cols.npy", oof_preds2)

# 保存文件
sub = pd.read_csv("data/sample_submission.csv")
sub['y'] = (test_preds + test_preds2)/2.
sub.to_csv("outputs/submissions/submission.csv", index=False)
print('Submission shape',sub.shape)
sub.head()

# 检查用箱线图
plt.hist(sub.y,bins=100)
plt.title('Test Preds')
plt.ylim((0,10_000))
plt.show()

