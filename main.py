#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, datasets

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

IMG_SIZE = 160   # target resolution


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def log(x):
    print(x, flush=True)


# ======================================================
# GPU AUGMENT (leggero)
# ======================================================

def gpu_augment(x):
    # x: [B,3,H,W]
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[3])  # flip orizzontale
    return x


# ======================================================
# CIFAR-100 loader (CPU), niente cv2, niente PIL runtime
# ======================================================

def load_cifar100_raw(root="./data"):
    """
    Carica CIFAR-100 e restituisce:
    - X_train, X_test: tensor CPU uint8 [N, 3, 32, 32]
    - y_train, y_test: tensor CPU long [N]
    """
    tr = datasets.CIFAR100(root, train=True, download=True)
    te = datasets.CIFAR100(root, train=False, download=True)

    # tr.data: [N,32,32,3] uint8
    Xtr = torch.from_numpy(tr.data).permute(0, 3, 1, 2).to(torch.uint8)
    ytr = torch.tensor(tr.targets, dtype=torch.long)

    Xte = torch.from_numpy(te.data).permute(0, 3, 1, 2).to(torch.uint8)
    yte = torch.tensor(te.targets, dtype=torch.long)

    return Xtr, ytr, Xte, yte


# ======================================================
# DIRICHLET SPLIT
# ======================================================

def dirichlet_split(labels, n_clients, alpha):
    """
    labels: torch.LongTensor CPU [N]
    restituisce: lista di liste di indici per client
    """
    labels_np = labels.numpy()
    per = [[] for _ in range(n_clients)]
    classes = np.unique(labels_np)

    for c in classes:
        idx = np.where(labels_np == c)[0]
        np.random.shuffle(idx)
        p = np.random.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(p) * len(idx)).astype(int)
        chunks = np.split(idx, cuts[:-1])
        for cid in range(n_clients):
            per[cid].extend(chunks[cid])

    # evita client vuoti
    for cid in range(n_clients):
        if len(per[cid]) == 0:
            per[cid].append(np.random.randint(0, len(labels_np)))

    # shuffle finale per client
    for cid in range(n_clients):
        random.shuffle(per[cid])

    return per


# ======================================================
# RESNET18 – solo layer3, layer4, fc allenabili
# ======================================================

class ResNet18Pre(nn.Module):
    def __init__(self, nc):
        super().__init__()

        # compatibile con versioni vecchie di torchvision
        try:
            self.m = models.resnet18(weights="IMAGENET1K_V1")
        except Exception:
            try:
                self.m = models.resnet18(pretrained=True)
            except Exception:
                self.m = models.resnet18()

        in_f = self.m.fc.in_features
        self.m.fc = nn.Linear(in_f, nc)

        # freeza tutto
        for p in self.m.parameters():
            p.requires_grad = False

        # sblocca solo layer3, layer4, fc
        for name, p in self.m.named_parameters():
            if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
                p.requires_grad = True

    def forward(self, x):
        return self.m(x)


# ======================================================
# LOCAL TRAINING – SCAFFOLD + resize on-the-fly
# ======================================================

def run_client(model, idxs, X_cpu_u8, y_cpu, c_global, c_local, lr, device):
    """
    model: ResNet18Pre su GPU
    idxs: lista di indici di questo client
    X_cpu_u8: tensor CPU uint8 [N,3,32,32]
    y_cpu: tensor CPU long [N]
    c_global, c_local: liste di tensori (stesso device della model)
    """
    model.train()
    train_params = [p for p in model.parameters() if p.requires_grad]

    # copia parametri di partenza
    old_params = [p.detach().clone() for p in train_params]

    opt = optim.SGD(train_params, lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    # mean/std su device una sola volta (qui usiamo 0.5/0.5 come nel codice originale)
    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

    for _ in range(LOCAL_EPOCHS):
        for start in range(0, len(idxs), BATCH):
            batch_idx = idxs[start:start + BATCH]

            xb_u8 = X_cpu_u8[batch_idx]          # [b,3,32,32] CPU uint8
            yb = y_cpu[batch_idx].to(device)     # [b] GPU long

            xb = xb_u8.to(device=device, dtype=torch.float32) / 255.0
            # resize a 160x160 solo per questo batch
            xb = F.interpolate(xb, size=IMG_SIZE, mode="bilinear", align_corners=False)
            xb = (xb - mean) / std

            xb = gpu_augment(xb)

            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()

            # SCAFFOLD correction
            for i, p in enumerate(train_params):
                p.grad += DAMPING * (c_global[i] - c_local[i])

            torch.nn.utils.clip_grad_norm_(train_params, GRAD_CLIP)
            opt.step()

    # compute deltas
    new_params = [p.detach().clone() for p in train_params]
    delta_c = []
    E = max(1, len(idxs) // BATCH)

    for i in range(len(train_params)):
        diff = new_params[i] - old_params[i]
        dc = BETA * (diff / E)
        delta_c.append(dc)
        c_local[i] += dc

    return new_params, delta_c, c_local


# ======================================================
# EVAL – resize on-the-fly
# ======================================================

def evaluate(model, X_cpu_u8, y_cpu, device):
    model.eval()
    correct = 0
    tot = len(y_cpu)

    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for start in range(0, tot, 256):
            xb_u8 = X_cpu_u8[start:start + 256]
            yb = y_cpu[start:start + 256].to(device)

            xb = xb_u8.to(device=device, dtype=torch.float32) / 255.0
            xb = F.interpolate(xb, size=IMG_SIZE, mode="bilinear", align_corners=False)
            xb = (xb - mean) / std

            out = model(xb)
            pred = out.argmax(1)
            correct += (pred == yb).sum().item()

    return correct / tot


# ======================================================
# FEDERATED LOOP
# ======================================================

def federated_run():
    device = "cuda:0"

    log("Carico CIFAR-100 (CPU, 32x32)...")
    X_train, y_train, X_test, y_test = load_cifar100_raw("./data")

    for alpha in DIR_ALPHAS:
        log(f"\n==== CIFAR-100 | α={alpha} ====\n")

        splits = dirichlet_split(y_train, NUM_CLIENTS, alpha)

        global_model = ResNet18Pre(100).to(device)
        train_params = [p for p in global_model.parameters() if p.requires_grad]

        # SCAFFOLD c-variates
        c_global = [torch.zeros_like(p) for p in train_params]
        c_locals = [[torch.zeros_like(p) for p in train_params] for _ in range(NUM_CLIENTS)]

        for rnd in range(1, NUM_ROUNDS + 1):
            lr = LR_INIT if rnd <= LR_DECAY_ROUND else LR_DECAY
            gs = [p.detach().clone() for p in train_params]

            new_params_all = []
            delta_c_all = []

            for cid in range(NUM_CLIENTS):
                local_model = ResNet18Pre(100).to(device)

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
                    lr,
                    device
                )

                c_locals[cid] = new_cl
                new_params_all.append(new_params)
                delta_c_all.append(dc)

            # aggregate params
            avg_params = []
            for i in range(len(train_params)):
                stack = torch.stack([cp[i] for cp in new_params_all])
                avg_params.append(stack.mean(0))

            with torch.no_grad():
                j = 0
                for p in global_model.parameters():
                    if p.requires_grad:
                        p.copy_(avg_params[j])
                        j += 1

            # aggregate c_global
            for i in range(len(train_params)):
                c_global[i] = torch.stack([dc[i] for dc in delta_c_all]).mean(0)

            acc = evaluate(global_model, X_test, y_test, device)
            log(f"[ROUND {rnd}] ACC = {acc*100:.2f}%")

# ======================================================
# MAIN
# ======================================================

def main():
    set_seed(SEED)
    federated_run()


if __name__ == "__main__":
    main()














