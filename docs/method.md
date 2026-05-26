# 方法说明

## 任务

本项目来自 Kaggle Playground Series S5E8，目标是预测银行客户是否会响应营销活动，评价指标为 ROC AUC。

## 数据

项目使用三类数据：

- `train.csv`：Kaggle 训练集，包含标签 `y`。
- `test.csv`：Kaggle 测试集，用于生成提交文件。
- `bank-full.csv`：UCI Bank Marketing 原始数据集（Moro et al., 2014），作为额外训练数据参与特征编码。

数据文件不提交到 GitHub。复现时需要从 Kaggle 下载比赛数据，并将文件放到 `data/`。

> **引用：** Moro, S., Cortez, P., & Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. *Decision Support Systems*, 62, 22-31. 数据集地址：<https://archive.ics.uci.edu/dataset/222/bank+marketing>

## 特征工程

数据集包含 18 个原始特征（8 个数值型 + 10 个类别型），代码自动检测列类型并构造以下特征：

### 基础编码

- **数值列**：保留原始值，同时生成 `factorize` 离散化副本（列名后缀 `2`）。
- **类别列**：使用 `factorize` 转为整数编码。

### 组合特征（默认开启，可通过 `--no-pair-features` 关闭）

- 对类别列和离散化数值列两两组合，生成交叉特征 `{col1}_{col2}`。
- 编码方式：`col1 * cardinality(col2) + col2`，保证每个组合唯一。

### 计数编码

- 对类别列、离散化列和组合特征列，计算每组在训练集中的样本数，生成计数特征 `CE_{colname}`。
- 使用全部数据（训练集 + 测试集 + 原始数据）进行编码，增强特征稳定性。

### 目标编码（可选，通过 `--target-encode-pairs` 开启）

- 仅对组合特征做 5 折无泄漏目标编码（Out-of-Fold Target Encoding）。
- 使用训练集标签的组内均值，避免数据泄漏。

最终特征矩阵以 `float32` 传入模型。

## 交叉验证

使用 `StratifiedKFold`（5 折，`shuffle=True`，`random_state=42`）进行交叉验证。每折：

1. 构造特征（如有目标编码则在折内完成，防止泄漏）
2. 训练模型，输出验证集预测概率
3. 对测试集预测概率取所有折的均值

最终输出 OOF（Out-of-Fold，即所有验证折预测的拼接）和测试集预测。

## 模型

### 一层基础模型

| 模型 | 关键参数 |
|------|----------|
| LightGBM | `learning_rate=0.03, n_estimators=3000, num_leaves=31, reg_lambda=8.0` |
| XGBoost | `learning_rate=0.04, n_estimators=3000, max_depth=6, early_stopping=200` |
| CatBoost | `learning_rate=0.05, iterations=3000, depth=8, early_stopping=200` |
| RandomForest | `n_estimators=500, max_depth=18, class_weight=balanced_subsample` |

输出：

- `outputs/oof/oof_{run_name}.npy` — 训练集 OOF 预测
- `outputs/pred/pred_{run_name}.npy` — 测试集预测
- `outputs/submissions/{run_name}.csv` — Kaggle 提交文件

### 二层 Stacking

读取多个一层模型的 OOF 和测试预测作为元特征，训练二层模型：

| 模型 | 关键参数 |
|------|----------|
| Logistic Regression | `C=1.0, max_iter=1000` |
| LightGBM | `learning_rate=0.03, n_estimators=5000, num_leaves=15, max_depth=4` |
| CatBoost | `learning_rate=0.03, iterations=3000, depth=4` |

## 历史脚本

原始比赛脚本保存在 `scripts/legacy/`：

- `base_models/`：一层模型和简单融合脚本。
- `stacking/`：二层 stacking 脚本。

这些脚本保留了当时的 Kaggle 路径和实验痕迹，不作为当前推荐运行入口。当前推荐使用 `src/bank_marketing/` 下的模块化入口。

## 命令行参数参考

### 一层模型 `train_base`

```bash
python -m bank_marketing.train_base [参数]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data-dir` | `str` | `data` | 数据目录路径 |
| `--output-dir` | `str` | `outputs` | 输出目录路径 |
| `--model` | `str` | `lgbm` | 模型选择：`lgbm` / `xgb` / `cat` / `rf` |
| `--run-name` | `str` | 同模型名 | 输出文件名后缀 |
| `--folds` | `int` | `5` | 交叉验证折数 |
| `--seed` | `int` | `42` | 随机种子（每折自动 +fold） |
| `--device` | `str` | `gpu` | 计算设备：`cpu` / `gpu` |
| `--n-jobs` | `int` | `-1` | 并行线程数（`-1` 为全部） |
| `--log-period` | `int` | `100` | 模型训练日志间隔（轮） |
| `--sample-rows` | `int` | `None` | 仅用前 N 行做冒烟测试 |
| `--no-pair-features` | `flag` | `False` | 禁用组合特征 |
| `--target-encode-pairs` | `flag` | `False` | 对组合特征做无泄漏目标编码 |

### 二层 Stacking `stack`

```bash
python -m bank_marketing.stack [参数]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data-dir` | `str` | `data` | 数据目录路径 |
| `--output-dir` | `str` | `outputs` | 输出目录路径 |
| `--model` | `str` | `logreg` | 模型选择：`logreg` / `lgbm` / `xgb` / `cat` |
| `--run-name` | `str` | `{model}_stacking` | 输出文件名后缀 |
| `--folds` | `int` | `5` | 交叉验证折数 |
| `--seed` | `int` | `42` | 随机种子 |
| `--sample-rows` | `int` | `None` | 仅用前 N 行做冒烟测试 |
| `--features` | `list` | `None` | 指定一层 run_name 列表；留空则自动读取 `outputs/oof/` 下全部文件 |
