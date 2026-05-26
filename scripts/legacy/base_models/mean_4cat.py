from pathlib import Path

Path("outputs/oof").mkdir(parents=True, exist_ok=True)
Path("outputs/pred").mkdir(parents=True, exist_ok=True)
Path("outputs/submissions").mkdir(parents=True, exist_ok=True)
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# pred_cb_seed7 = np.load("outputs/pred/pred_cb_seed7.npy")
# pred_cb_seed42 = np.load("outputs/pred/pred_cb_seed42.npy")
# pred_cb_seed100 = np.load("outputs/pred/pred_cb_seed100.npy")
# pred_cb_seed124 = np.load("outputs/pred/pred_cb_seed124.npy")
# c1 = np.load("outputs/pred/pred_cb_cb_c1.npy")
# c2 = np.load("outputs/pred/pred_cb_cb_c2.npy")
# c3 = np.load("outputs/pred/pred_cb_cb_c3.npy")
rf = np.load("outputs/pred/pred_rf_c1.npy")
# Submission
sub = pd.read_csv("data/sample_submission.csv")
# sub['y'] = (c1 + c2 + c3) / 4.0
sub['y'] = (rf) / 4.0

sub.to_csv("outputs/submissions/rf.csv", index=False)

# Sanity check
plt.hist(sub.y, bins=100)
plt.title("Test Preds")
plt.ylim((0, 10_000))
plt.show()


