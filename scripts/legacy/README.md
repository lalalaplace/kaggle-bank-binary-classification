# 历史脚本归档

这里保存原项目中的比赛脚本，仅用于追溯当时的实验过程。

## 目录

- `base_models/`：一层基础模型和简单平均脚本。
- `stacking/`：二层 stacking 脚本。

## 注意

这些脚本已经统一改为当前仓库路径：

- 数据读取：`data/`
- OOF 输出：`outputs/oof/`
- 测试集预测输出：`outputs/pred/`
- 提交文件输出：`outputs/submissions/`

这些脚本为独立 `.py` 文件，不可作为模块导入。运行方式为在项目根目录下直接执行（如 `python scripts/legacy/base_models/rf.py`），路径依赖与 `src/` 下的模块化代码不同。

当前推荐入口：

```bash
python -m bank_marketing.train_base
python -m bank_marketing.stack
```
