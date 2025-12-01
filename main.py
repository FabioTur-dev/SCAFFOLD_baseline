#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, random, numpy as np, scipy.io as sio, torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn.functional as F


# ==============================================================
# CONFIG
# ==============================================================

NUM_CLIENTS = 10
DIR_ALPHAS = [0.05, 0.5]
NUM_ROUNDS = 30
LOCAL_EPOCHS = 1
BATCH = 128

LR_INIT = 0.01
LR_DECAY_ROUND = 15
LR_DECAY = 0.003

BETA = 0.01
DAMPING = 0.1
GRAD_CLIP = 5.0
SEED = 42


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def log(x):
    print(x, flush=True)


# ==============================================================
# GPU AUGMENTATION — ULTRA FAST
# ==============================================================

def gpu_augment(x):
    # x shape: [B,3,160,160]
    # Random horizontal flip
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[3])

    # Random crop 160→ slightly jittered 148-160 window
    crop = int(torch.randint(148, 161, (1,)))
    if crop != 160:
        off = torch.randint(0, 160-crop+1, (2,))
        x = x[:, :, off[0]:off[0]+crop, off[1]:off[1]+crop]
        x = F.interpolate(x, size=160, mode='bilinear')

    return x


# ==============================================================
# LOAD SVHN — RAW FORMAT -> PREPROCESS -> TENSOR 160×160
# ==============================================================

def load_svhn_preprocessed(path, device):
    """Load SVHN mat file and produce [N,3,160,160] float tensor normalized."""
    mat = sio.loadmat(path)
    X = mat["X"]                            # (32,32,3,N)
    y = mat["y"].squeeze()
    y[y == 10] = 0
    y = y.astype(np.int64)

    # Move to (N,3,32,32)
    X = np.transpose(X, (3, 2, 0, 1))       # (N,3,32,32)
    X = torch.tensor(X, dtype=torch.float32, device=device) / 255.

    # Resize ONCE to 160x160 (GPU)
    X = F.interpolate(X, size=160, mode="bilinear")

    # Normalize
    mean = torch.tensor([0.5,0.5,0.5], device=device).view(1,3,1,1)
    std  = torch.tensor([0.5,0.5,0.5], device=device).view(1,3,1,1)
    X = (X - mean) / std

    return X, torch.tensor(y, device=device)


# ==============================================================
# DIRICHLET SPLIT (prevents empty clients)
# ==============================================================

def dirichlet_split(labels, n_clients, alpha):
    labels = labels.cpu().numpy()
    per = [[] for _ in range(n_clients)]
    classes = np.unique(labels)

    for c in classes:
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        p = np.random.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(p) * len(idx)).astype(int)
        chunks = np.split(idx, cuts[:-1])
        for cid in range(n_clients):
            per[cid].extend(chunks[cid])

    for cid in range(n_clients):
        if len(per[cid]) == 0:
            per[cid].append(np.random.randint(0, len(labels)))
        random.shuffle(per[cid])

    return per


# ==============================================================
# RESNET18 — compatible with all torchvision
# ==============================================================

class ResNet18Pre(nn.Module):
    def __init__(self, nc):
        super().__init__()
        try:
            self.m = models.resnet18(weights="IMAGENET1K_V1")
        except:
            try:
                self.m = models.resnet18(pretrained=True)
            except:
                self.m = models.resnet18()

        in_f = self.m.fc.in_features
        self.m.fc = nn.Linear(in_f, nc)

        for p in self.m.parameters():
            p.requires_grad = False

        for name, p in self.m.named_parameters():
            if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
                p.requires_grad = True

    def forward(self, x):
        return self.m(x)


# ==============================================================
# LOCAL CLIENT TRAINING (ULTRA FAST)
# ==============================================================

def run_client(model, idxs, X, y, c_global, c_local, lr):

    # subset selection — but WITHOUT DataLoader
    xb = X[idxs]        # [k,3,160,160]
    yb = y[idxs]        # [k]

    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]

    old_params = [p.detach().clone() for p in trainable]

    opt = optim.SGD(trainable, lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    # 1 epoch only, but manually batched
    for start in range(0, len(idxs), BATCH):
        xb_batch = xb[start:start+BATCH]
        yb_batch = yb[start:start+BATCH]

        # GPU augment
        xb_batch = gpu_augment(xb_batch)

        opt.zero_grad()
        out = model(xb_batch)
        loss = loss_fn(out, yb_batch)
        loss.backward()

        for i, p in enumerate(trainable):
            p.grad += DAMPING * (c_global[i] - c_local[i])

        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        opt.step()

    # compute deltas
    new_params = [p.detach().clone() for p in trainable]
    delta_c = []

    E = max(1, len(idxs) // BATCH)

    for i in range(len(trainable)):
        diff = new_params[i] - old_params[i]
        dc = BETA * (diff / E)
        delta_c.append(dc)
        c_local[i] += dc

    return new_params, delta_c, c_local


# ==============================================================
# EVAL — ultra fast
# ==============================================================

def evaluate(model, X, y):
    model.eval()
    correct = 0
    total = len(y)

    with torch.no_grad():
        for i in range(0, total, 256):
            xb = X[i:i+256]
            out = model(xb)
            pred = out.argmax(1)
            correct += (pred == y[i:i+256]).sum().item()

    return correct / total


# ==============================================================
# FEDERATED
# ==============================================================

def federated_run():

    device = "cuda:0"

    log("Preprocessing SVHN on GPU…")
    X_train, y_train = load_svhn_preprocessed("./data/train_32x32.mat", device)
    X_test,  y_test  = load_svhn_preprocessed("./data/test_32x32.mat", device)

    for alpha in DIR_ALPHAS:

        log(f"\n==== SVHN | α={alpha} ====\n")

        splits = dirichlet_split(y_train, NUM_CLIENTS, alpha)

        global_model = ResNet18Pre(10).to(device)
        trainable = [p for p in global_model.parameters() if p.requires_grad]

        c_global = [torch.zeros_like(p) for p in trainable]
        c_locals = [[torch.zeros_like(p) for p in trainable] for _ in range(NUM_CLIENTS)]

        for rnd in range(1, NUM_ROUNDS+1):
            lr = LR_INIT if rnd <= LR_DECAY_ROUND else LR_DECAY
            global_start = [p.detach().clone() for p in trainable]

            new_params_all = []
            delta_c_all = []

            for cid in range(NUM_CLIENTS):

                local_model = ResNet18Pre(10).to(device)

                with torch.no_grad():
                    j = 0
                    for p in local_model.parameters():
                        if p.requires_grad:
                            p.copy_(global_start[j])
                            j += 1

                new_params, dc, new_cl = run_client(
                    local_model,
                    splits[cid],
                    X_train,
                    y_train,
                    c_global,
                    c_locals[cid],
                    lr
                )

                c_locals[cid] = new_cl
                new_params_all.append(new_params)
                delta_c_all.append(dc)

            # aggregate
            avg_params = []
            for i in range(len(trainable)):
                stack = torch.stack([cp[i] for cp in new_params_all])
                avg_params.append(stack.mean(0))

            with torch.no_grad():
                j = 0
                for p in global_model.parameters():
                    if p.requires_grad:
                        p.copy_(avg_params[j])
                        j += 1

            # update global c
            for i in range(len(c_global)):
                c_global[i] = torch.stack([dc[i] for dc in delta_c_all]).mean(0)

            # eval
            acc = evaluate(global_model, X_test, y_test)
            log(f"[ROUND {rnd}] ACC = {acc*100:.2f}%")


# ==============================================================
# MAIN
# ==============================================================

def main():
    set_seed(SEED)
    federated_run()


if __name__ == "__main__":
    main()













