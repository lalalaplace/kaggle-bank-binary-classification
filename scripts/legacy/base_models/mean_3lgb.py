from pathlib import Path

Path("outputs/oof").mkdir(parents=True, exist_ok=True)
Path("outputs/pred").mkdir(parents=True, exist_ok=True)
Path("outputs/submissions").mkdir(parents=True, exist_ok=True)
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

pred_lgb_c1_baseline = np.load("outputs/pred/pred_lgb_c1_baseline.npy")
pred_lgb_c2_strong_reg = np.load("outputs/pred/pred_lgb_c2_strong_reg.npy")
pred_lgb_c3_more_complex = np.load("outputs/pred/pred_lgb_c3_more_complex.npy")

# Submission
sub = pd.read_csv("data/sample_submission.csv")
sub['y'] = (pred_lgb_c1_baseline + pred_lgb_c2_strong_reg + pred_lgb_c3_more_complex ) / 3.0
sub.to_csv("outputs/submissions/lgb3.csv", index=False)

# Sanity check
plt.hist(sub.y, bins=100)
plt.title("Test Preds")
plt.ylim((0, 10_000))
plt.show()



