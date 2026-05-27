# 银行客户响应营销二分类预测

## 项目简介

这是个人在 Kaggle Playground Series S5E8 银行客户响应营销二分类项目中的代码整理版。项目目标是基于客户画像、历史营销接触信息和银行营销原始数据，预测客户是否会响应营销活动。

本仓库重点整理了三部分内容：

- 特征工程：类别编码、数值离散化、组合特征、计数编码和可选目标编码。
- OOF 与验证：基于交叉验证生成无泄漏 OOF 预测，并保存测试集折均值预测。
- Stacking 融合：使用多组一层模型 OOF 和测试预测训练二层模型。

## 竞赛背景

比赛地址：<https://www.kaggle.com/competitions/playground-series-s5e8>

任务类型是二分类预测，标签为 `y`，评价指标为 ROC AUC。项目面向的是银行电话营销响应预测场景，正负样本存在不均衡，因此验证方式和 OOF 质量比单次划分更重要。

## 数据说明

本仓库不包含数据文件。复现时需要手动下载并放入 `data/`：

```text
data/train.csv
data/test.csv
data/sample_submission.csv
data/bank-full.csv
```

数据来源：

- `train.csv`：Kaggle 训练集，包含标签 `y`。
- `test.csv`：Kaggle 测试集，用于生成提交文件。
- `sample_submission.csv`：Kaggle 提交模板。
- `bank-full.csv`：UCI Bank Marketing 原始数据集（Moro et al., 2014），可从 [UCI ML Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) 下载，分隔符为分号。

数据文件、OOF 预测、测试集预测和提交结果都通过 `.gitignore` 排除，不会上传到 GitHub。

## 方法概览

整体流程分为两层：

1. 一层模型：使用 LightGBM、XGBoost、CatBoost、RandomForest 等模型分别训练，输出 OOF 预测和测试集预测。
2. 二层模型：读取一层模型的 OOF 和测试集预测作为元特征，训练 Logistic Regression、LightGBM、XGBoost 或 CatBoost stacking 模型。

推荐入口位于 `src/bank_marketing/`：

- `train_base.py`：训练一层基础模型。
- `stack.py`：训练二层 stacking 模型。
- `features.py`：集中管理特征工程。
- `data.py`：统一数据读取。

`scripts/legacy/` 保留原始比赛脚本，已改为当前仓库路径，但主要用于追溯实验过程。

## 特征工程

特征工程是本项目最核心的部分。当前整理版保留原始比赛方案中的主要思路：

- 基础编码：类别变量使用 `factorize` 转为整数编码；数值变量保留原始值。
- 数值离散化：对数值变量额外生成离散化副本，列名后缀为 `2`。
- 组合特征：对类别列和离散化数值列做两两组合，生成交叉类别特征。
- 计数编码：对类别列、离散化列和组合列统计频次，生成 `CE_{col}` 特征。
- 目标编码：可选对组合特征做折内目标编码，避免标签泄漏。

组合特征默认开启。如果需要快速验证流程，可使用 `--no-pair-features` 关闭组合特征，显著降低运行时间。

目标编码默认关闭，需要时使用：

```bash
--target-encode-pairs
```

## 建模与验证

一层模型使用 `StratifiedKFold` 交叉验证。每一折流程如下：

1. 取当前折训练集和验证集。
2. 在折内完成可选目标编码，避免验证集标签泄漏。
3. 训练模型并输出验证集概率。
4. 对测试集预测概率累加，最后取折均值。

最终输出：

```text
outputs/oof/oof_{run_name}.npy
outputs/pred/pred_{run_name}.npy
outputs/submissions/{run_name}.csv
```

OOF 是后续 stacking 的核心输入。它模拟模型在未见样本上的预测，能用于稳健估计模型效果，也能作为二层模型的训练特征。

支持的一层模型：

| 模型 | 说明 |
|---|---|
| LightGBM | 适合表格数据，支持 GPU 但需要 GPU 版安装 |
| XGBoost | 使用新版 `device=cuda` GPU 参数 |
| CatBoost | 对类别特征友好，支持 GPU |
| RandomForest | CPU 基线模型 |

## 模型融合

模型融合主要分为两种：

- 简单平均：历史脚本中保留了对同类模型预测取平均的做法。
- Stacking：当前推荐方式，使用多模型 OOF 作为二层训练特征。

二层 stacking 输入：

```text
outputs/oof/oof_lgbm_c1.npy
outputs/oof/oof_xgb_c1.npy
outputs/oof/oof_cat_c1.npy
...
```

对应测试集预测：

```text
outputs/pred/pred_lgbm_c1.npy
outputs/pred/pred_xgb_c1.npy
outputs/pred/pred_cat_c1.npy
...
```

二层模型会自动按 `run_name` 对齐 OOF 和测试预测。也可以通过 `--features` 显式指定参与融合的一层模型：

```bash
python -m bank_marketing.stack --model logreg --run-name logreg_stacking --features lgbm_c1 xgb_c1 cat_c1
```

## 实验结果

仓库不提交历史 `.npy`、提交 `.csv` 或 Kaggle 输出文件，因此 README 不直接记录不可验证的本地文件结果。复现实验后，结果以以下文件为准：

```text
outputs/oof/
outputs/pred/
outputs/submissions/
```

建议记录结果时使用下面表格：

| 阶段 | run_name | 模型 | 验证方式 | 指标 |
|---|---|---|---|---|
| 一层 | `lgbm_c1` | LightGBM | 5 折 OOF | ROC AUC |
| 一层 | `xgb_c1` | XGBoost | 5 折 OOF | ROC AUC |
| 一层 | `cat_c1` | CatBoost | 5 折 OOF | ROC AUC |
| 一层 | `rf_c1` | RandomForest | 5 折 OOF | ROC AUC |
| 二层 | `logreg_stacking` | Logistic Regression | 5 折 OOF | ROC AUC |
| 二层 | `lgbm_stacking` | LightGBM | 5 折 OOF | ROC AUC |

运行脚本会在终端打印每折 AUC 和整体 OOF AUC。正式复现实验时，应优先比较 OOF AUC，再结合 Kaggle Public / Private Leaderboard 判断是否过拟合。

## 项目结构

```text
.
├── src/bank_marketing/      # 可复现训练、特征工程和 stacking 代码
├── docs/                    # 方法说明和图片
├── scripts/legacy/          # 原始比赛脚本归档
├── LICENSE                  # MIT License
├── pyproject.toml           # Python 包配置
├── requirements.txt         # 依赖列表
└── README.md
```

## 运行方式

安装依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

快速检查：

```bash
python -m bank_marketing.train_base --model rf --run-name smoke_rf --sample-rows 1000 --folds 2 --no-pair-features --n-jobs 1
python -m bank_marketing.stack --model logreg --run-name smoke_stack --features smoke_rf --folds 2 --sample-rows 1000
```

训练一层模型：

```bash
python -m bank_marketing.train_base --model lgbm --run-name lgbm_c1 --device gpu
python -m bank_marketing.train_base --model xgb --run-name xgb_c1 --device gpu
python -m bank_marketing.train_base --model cat --run-name cat_c1 --device gpu
python -m bank_marketing.train_base --model rf --run-name rf_c1 --device cpu
```

训练二层 stacking：

```bash
python -m bank_marketing.stack --model logreg --run-name logreg_stacking
python -m bank_marketing.stack --model lgbm --run-name lgbm_stacking
```

GPU 说明：

- LightGBM 的 GPU 版本需要单独编译或安装。
- XGBoost 使用 `device=cuda`。
- CatBoost 使用 `task_type=GPU`。
- RandomForest 只支持 CPU。

如果 GPU 环境不可用，将 `--device gpu` 改为 `--device cpu`。

## 复盘与改进

本项目的主要经验：

- OOF 是表格竞赛 stacking 的关键，必须保证折内训练和验证严格隔离。
- 组合特征和计数编码对这类类别特征较多的银行营销数据很重要。
- 外部原始数据 `bank-full.csv` 可以提升特征编码稳定性，但需要注意训练集、测试集和外部数据的分布差异。
- Stacking 能融合不同模型偏差，但如果一层模型高度同质，提升会有限。

后续可改进方向：

- 增加统一实验记录文件，自动保存每次运行的 OOF AUC、参数和提交文件名。
- 增加更轻量的配置文件，避免长命令重复输入。
- 补充特征重要性、SHAP 或排列重要性分析。
- 将历史脚本中的有效实验进一步迁移到 `src/bank_marketing/` 的模块化入口。

## 许可证

本项目代码使用 MIT License。数据需遵守 Kaggle 比赛页面和 UCI Bank Marketing 数据集的使用条款。
