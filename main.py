#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, random, numpy as np, scipy.io as sio, torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models
import torch.nn.functional as F


# ======================================================
# CONFIG
# ======================================================

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

CHUNK = 2048  # preprocessing chunk size (safe for 4090)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def log(x):
    print(x, flush=True)


# ======================================================
# GPU AUGMENT
# ======================================================

def gpu_augment(x):
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[3])
    return x


# ======================================================
# SAFE SVHN PREPROCESS — NO OOM
# ======================================================

def load_svhn_preprocessed(path, device):
    mat = sio.loadmat(path)
    X = mat["X"]               # (32,32,3,N)
    y = mat["y"].squeeze()
    y[y == 10] = 0
    y = y.astype(np.int64)

    # Convert to (N,3,32,32)
    X = np.transpose(X, (3, 2, 0, 1))
    N = X.shape[0]

    # Allocate final tensor on GPU
    X_final = torch.empty((N,3,160,160), dtype=torch.float32, device=device)

    mean = torch.tensor([0.5,0.5,0.5], device=device).view(1,3,1,1)
    std  = torch.tensor([0.5,0.5,0.5], device=device).view(1,3,1,1)

    # Process in chunks to avoid OOM
    for start in range(0, N, CHUNK):
        end = min(start+CHUNK, N)
        batch = torch.tensor(X[start:end], dtype=torch.float32, device=device) / 255.

        # Resize each chunk
        batch = F.interpolate(batch, size=160, mode="bilinear", align_corners=False)

        # Normalize
        batch = (batch - mean) / std

        X_final[start:end] = batch

    return X_final, torch.tensor(y, device=device)


# ======================================================
# DIRICHLET SPLIT
# ======================================================

def dirichlet_split(labels, n_clients, alpha):
    labels_np = labels.cpu().numpy()
    per = [[] for _ in range(n_clients)]
    classes = np.unique(labels_np)

    for c in classes:
        idx = np.where(labels_np == c)[0]
        np.random.shuffle(idx)
        p = np.random.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(p)*len(idx)).astype(int)
        chunks = np.split(idx, cuts[:-1])
        for cid in range(n_clients):
            per[cid].extend(chunks[cid])

    # ensure no empty client
    for cid in range(n_clients):
        if len(per[cid]) == 0:
            per[cid].append(np.random.randint(0, len(labels_np)))

    return per


# ======================================================
# RESNET18 (train only layer3/4/fc)
# ======================================================

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


# ======================================================
# LOCAL TRAINING (ULTRA FAST)
# ======================================================

def run_client(model, idxs, X, y, c_global, c_local, lr):

    model.train()
    Xb = X[idxs]
    yb = y[idxs]

    train_params = [p for p in model.parameters() if p.requires_grad]
    old_params = [p.detach().clone() for p in train_params]

    opt = optim.SGD(train_params, lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    for start in range(0, len(idxs), BATCH):
        xb = Xb[start:start+BATCH]
        ybb = yb[start:start+BATCH]

        xb = gpu_augment(xb)

        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, ybb)
        loss.backward()

        for i, p in enumerate(train_params):
            p.grad += DAMPING * (c_global[i] - c_local[i])

        torch.nn.utils.clip_grad_norm_(train_params, GRAD_CLIP)
        opt.step()

    # compute deltas
    new_params = [p.detach().clone() for p in train_params]
    delta_c = []

    E = max(1, len(idxs)//BATCH)

    for i in range(len(train_params)):
        diff = new_params[i] - old_params[i]
        dc = BETA * (diff / E)
        delta_c.append(dc)
        c_local[i] += dc

    return new_params, delta_c, c_local


# ======================================================
# EVAL
# ======================================================

def evaluate(model, X, y):
    model.eval()
    tot = len(y)
    correct = 0

    with torch.no_grad():
        for start in range(0, tot, 256):
            xb = X[start:start+256]
            out = model(xb)
            pred = out.argmax(1)
            correct += (pred == y[start:start+256]).sum().item()

    return correct / tot


# ======================================================
# FEDERATED RUN
# ======================================================

def federated_run():

    device = "cuda:0"

    log("Preprocessing SVHN in CHUNKS (GPU)…")
    X_train, y_train = load_svhn_preprocessed("./data/train_32x32.mat", device)
    X_test,  y_test  = load_svhn_preprocessed("./data/test_32x32.mat", device)

    for alpha in DIR_ALPHAS:
        log(f"\n==== SVHN | α={alpha} ====\n")

        splits = dirichlet_split(y_train, NUM_CLIENTS, alpha)

        global_model = ResNet18Pre(10).to(device)
        train_params = [p for p in global_model.parameters() if p.requires_grad]

        c_global = [torch.zeros_like(p) for p in train_params]
        c_locals = [[torch.zeros_like(p) for p in train_params] for _ in range(NUM_CLIENTS)]

        for rnd in range(1, NUM_ROUNDS+1):

            lr = LR_INIT if rnd <= LR_DECAY_ROUND else LR_DECAY
            gs = [p.detach().clone() for p in train_params]

            new_params_all = []
            delta_c_all = []

            for cid in range(NUM_CLIENTS):
                local_model = ResNet18Pre(10).to(device)

                with torch.no_grad():
                    j = 0
                    for p in local_model.parameters():
                        if p.requires_grad:
                            p.copy_(gs[j])
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

            # aggregate parameters
            avg_params = []
            for i in range(len(train_params)):
                stack = torch.stack([cp[i] for cp in new_params_all])
                avg_params.append(stack.mean(0))

            with torch.no_grad():
                j=0
                for p in global_model.parameters():
                    if p.requires_grad:
                        p.copy_(avg_params[j])
                        j+=1

            # update c_global
            for i in range(len(train_params)):
                c_global[i] = torch.stack([dc[i] for dc in delta_c_all]).mean(0)

            # eval
            acc = evaluate(global_model, X_test, y_test)
            log(f"[ROUND {rnd}] ACC = {acc*100:.2f}%")



# ======================================================
# MAIN
# ======================================================

def main():
    set_seed(SEED)
    federated_run()


if __name__ == "__main__":
    main()













