from pathlib import Path

Path("outputs/oof").mkdir(parents=True, exist_ok=True)
Path("outputs/pred").mkdir(parents=True, exist_ok=True)
Path("outputs/submissions").mkdir(parents=True, exist_ok=True)
import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# 1) Reproducibility
# -----------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# 2) Simple MLP for meta features
# -----------------------------
class MetaMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims=(64, 32), dropout=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers += [nn.Linear(prev, 1)]  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)  # (B,)

# -----------------------------
# 3) Train / Eval helpers
# -----------------------------
def train_one_fold(
    X_tr, y_tr, X_va, y_va,
    seed=42,
    hidden_dims=(64, 32),
    dropout=0.2,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=1024,
    max_epochs=200,
    patience=20,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    seed_everything(seed)

    # Standardize (important for NN even if features are probabilities)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    X_tr_t = torch.from_numpy(X_tr_s.astype(np.float32))
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32))
    X_va_t = torch.from_numpy(X_va_s.astype(np.float32))
    y_va_t = torch.from_numpy(y_va.astype(np.float32))

    tr_ds = TensorDataset(X_tr_t, y_tr_t)
    va_ds = TensorDataset(X_va_t, y_va_t)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = MetaMLP(in_dim=X_tr.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    def predict_proba(loader):
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                logits = model(xb)
                prob = torch.sigmoid(logits).detach().cpu().numpy()
                preds.append(prob)
        return np.concatenate(preds, axis=0)

    for epoch in range(1, max_epochs + 1):
        # train
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # valid AUC
        va_pred = predict_proba(va_loader)
        va_auc = roc_auc_score(y_va, va_pred)

        if va_auc > best_auc + 1e-6:
            best_auc = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, best_auc, best_epoch

def predict_with_model(model, scaler, X, batch_size=4096, device="cuda" if torch.cuda.is_available() else "cpu"):
    X_s = scaler.transform(X)
    X_t = torch.from_numpy(X_s.astype(np.float32))
    ds = TensorDataset(X_t, torch.zeros(len(X_t)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            preds.append(prob)
    return np.concatenate(preds, axis=0)

# -----------------------------
# 4) Load meta features (OOF + test preds)
# -----------------------------
oof_paths = [
    r"outputs/oof/oof_cb_seed7.npy",
    r"outputs/oof/oof_cb_seed42.npy",
    r"outputs/oof/oof_cb_seed100.npy",
    r"outputs/oof/oof_cb_seed124.npy",
    r"outputs/oof/oof_cb_cb_c1.npy",
    r"outputs/oof/oof_cb_cb_c2.npy",
    r"outputs/oof/oof_cb_cb_c3.npy",
    r"outputs/oof/oof_lgb_c1_baseline.npy",
    r"outputs/oof/oof_lgb_c2_strong_reg.npy",
    r"outputs/oof/oof_lgb_c3_more_complex.npy",
    r"outputs/oof/oof_xgb_with_orig_rows.npy",
    r"outputs/oof/oof_xgb_with_orig_cols.npy",
    r"outputs/oof/oof_xgb_c1.npy",
    r"outputs/oof/oof_xgb_c2.npy",
    r"outputs/oof/oof_xgb_c3.npy",
    r"outputs/oof/oof_rf_c1.npy",

    
]
test_paths = [
    r"outputs/pred/pred_cb_seed7.npy",
    r"outputs/pred/pred_cb_seed42.npy",
    r"outputs/pred/pred_cb_seed100.npy",
    r"outputs/pred/pred_cb_seed124.npy",
    r"outputs/pred/pred_cb_cb_c1.npy",
    r"outputs/pred/pred_cb_cb_c2.npy",
    r"outputs/pred/pred_cb_cb_c3.npy",
    r"outputs/pred/pred_lgb_c1_baseline.npy",
    r"outputs/pred/pred_lgb_c2_strong_reg.npy",
    r"outputs/pred/pred_lgb_c3_more_complex.npy",
    r"outputs/pred/pred_xgb_with_orig_rows.npy",
    r"outputs/pred/pred_xgb_with_orig_cols.npy",
    r"outputs/pred/pred_xgb_c1.npy",
    r"outputs/pred/pred_xgb_c2.npy",
    r"outputs/pred/pred_xgb_c3.npy",
    r"outputs/pred/pred_rf_c1.npy",
]

oof = [np.load(p).reshape(-1, 1) for p in oof_paths]
pred = [np.load(p).reshape(-1, 1) for p in test_paths]

X_meta_oof = np.hstack(oof).astype(np.float32)
X_meta_pred = np.hstack(pred).astype(np.float32)

print("X_meta_oof :", X_meta_oof.shape)
print("X_meta_pred:", X_meta_pred.shape)

train_df = pd.read_csv("data/train.csv")
y = train_df["y"].values.astype(np.int32)

# -----------------------------
# 5) KFold NN stacking
# -----------------------------
SEED = 42
FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

oof_meta = np.zeros(len(y), dtype=np.float32)
test_meta = np.zeros(X_meta_pred.shape[0], dtype=np.float32)

nn_params = dict(
    hidden_dims=(64, 32),
    dropout=0.25,
    lr=1e-3,
    weight_decay=5e-4,
    batch_size=1024,
    max_epochs=300,
    patience=25,
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_meta_oof, y), 1):
    X_tr, X_va = X_meta_oof[tr_idx], X_meta_oof[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    model, scaler, best_auc, best_epoch = train_one_fold(
        X_tr, y_tr, X_va, y_va,
        seed=SEED + fold,
        **nn_params
    )

    va_pred = predict_with_model(model, scaler, X_va)
    oof_meta[va_idx] = va_pred.astype(np.float32)

    test_pred = predict_with_model(model, scaler, X_meta_pred)
    test_meta += test_pred.astype(np.float32) / FOLDS

    auc = roc_auc_score(y_va, va_pred)
    print(f"Fold {fold} AUC = {auc:.6f} | best_epoch={best_epoch} | best_auc_tracked={best_auc:.6f}")

full_auc = roc_auc_score(y, oof_meta)
print(f"Meta OOF AUC = {full_auc:.6f}")

sub = pd.read_csv("data/sample_submission.csv")
sub["y"] = test_meta
sub.to_csv("outputs/submissions/nn_stacking.csv", index=False)
print("Saved: nn_stacking.csv")


