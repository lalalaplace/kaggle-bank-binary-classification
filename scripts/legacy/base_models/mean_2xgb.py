from pathlib import Path

Path("outputs/oof").mkdir(parents=True, exist_ok=True)
Path("outputs/pred").mkdir(parents=True, exist_ok=True)
Path("outputs/submissions").mkdir(parents=True, exist_ok=True)
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

train = pd.read_csv('data/train.csv').set_index('id')
# =============================
# Load frozen XGB predictions
# =============================
oof_rows = np.load("outputs/oof/oof_xgb_with_orig_rows.npy")
oof_cols = np.load("outputs/oof/oof_xgb_with_orig_cols.npy")

test_rows = np.load("outputs/pred/pred_xgb_with_orig_rows.npy")
test_cols = np.load("outputs/pred/pred_xgb_with_orig_cols.npy")
test_c1 = np.load("outputs/pred/pred_xgb_c1.npy")
test_c2 = np.load("outputs/pred/pred_xgb_c2.npy")
test_c3 = np.load("outputs/pred/pred_xgb_c3.npy")

# Submission
sub = pd.read_csv("data/sample_submission.csv")
sub['y'] = (test_c1 + test_c2 + test_c3) / 3.0
sub.to_csv("outputs/submissions/xgb3.csv", index=False)

print("Submission shape:", sub.shape)
sub.head()

# Sanity check
plt.hist(sub.y, bins=100)
plt.title("Test Preds")
plt.ylim((0, 10_000))
plt.show()


