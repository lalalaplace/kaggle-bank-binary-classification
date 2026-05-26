# 银行客户响应营销二分类预测

## 概述

这是 Kaggle Playground Series S5E8 银行客户响应营销二分类项目的整理版。仓库目标是可复现实验：代码提交到 GitHub，数据与中间文件通过 `.gitignore` 排除，复现者按说明下载数据后运行训练和 stacking 流程。

比赛链接：https://www.kaggle.com/competitions/playground-series-s5e8

## 项目结构

```text
.
├── src/bank_marketing/      # 可复现训练与 stacking 代码
├── docs/                    # 方法说明和图片
├── scripts/legacy/          # 原始比赛脚本归档
└── README.md
```

## 数据准备

本仓库不包含 Kaggle 数据。请从比赛页面下载数据，并将文件放入 `data/`：

```text
data/train.csv
data/test.csv
data/sample_submission.csv
data/bank-full.csv
```

其中 `bank-full.csv` 是 UCI Bank Marketing 原始数据集（Moro et al., 2014），可从 [UCI ML Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) 下载 `bank-full.csv`（分隔符为分号）。

## 环境安装

建议使用 Python 3.10 或更高版本。

```bash
pip install -r requirements.txt
pip install -e .
```

### GPU 支持说明

不同模型的 GPU 后端要求不同：

- **LightGBM**：`--device gpu` 需安装 GPU 版 LightGBM（`pip install lightgbm --config-settings=cmake.define.USE_GPU=ON`）
- **XGBoost**：`--device gpu` 需要 CUDA 环境，使用 `device=cuda`
- **CatBoost**：`--device gpu` 需要 NVIDIA GPU + CUDA，自动检测
- **RandomForest**：仅支持 CPU

若 GPU 环境不可用，统一使用 `--device cpu` 即可。

## 快速检查

先运行一个小样本流程，确认路径和依赖正常：

```bash
python -m bank_marketing.train_base --model rf --run-name smoke_rf --sample-rows 1000 --folds 2 --no-pair-features --n-jobs 1
python -m bank_marketing.stack --model logreg --run-name smoke_stack --features smoke_rf --folds 2 --sample-rows 1000
```

## 正式运行

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

生成文件位置：

```text
outputs/oof/
outputs/pred/
outputs/submissions/
```

## 说明

- `data/`、`outputs/`、`.npy`、`.pkl` 和提交结果 `.csv` 已通过 `.gitignore` 排除，不会提交到仓库。
- 新代码位于 `src/bank_marketing/`，历史脚本已整理到 `scripts/legacy/`，仅用于追溯原始比赛过程。
- 原始数据、中间预测文件和历史提交结果已从仓库清除；复现时需重新下载数据并运行脚本生成。
- 默认训练入口使用 GPU，详见上方「GPU 支持说明」。
- 长任务会打印带时间戳的阶段日志；模型训练日志可通过 `--log-period` 调整。
- 更详细的方法说明见 `docs/method.md`。
