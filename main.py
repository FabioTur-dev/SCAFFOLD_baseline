#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, random, numpy as np, torch, time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image
import scipy.io as sio


# ======================================================================
# CONFIG
# ======================================================================

NUM_CLIENTS = 10
DIR_ALPHAS = [0.05, 0.5]
NUM_ROUNDS = 50
LOCAL_EPOCHS = 1
BATCH = 128

LR_INIT = 0.004
LR_DECAY_ROUND = 20
LR_DECAY = 0.0015

BETA = 0.01
DAMPING = 0.1
GRAD_CLIP = 5.0
SEED = 42


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def log(x):
    print(x, flush=True)


# ======================================================================
# REAL SVHN LOADER (WORKS 100% FOR .MAT FORMAT)
# ======================================================================

def load_svhn_split(split):
    path = "./data/{}_32x32.mat".format("train" if split=="train" else "test")

    mat = sio.loadmat(path)

    # mat["X"] shape = (32, 32, 3, N)
    X = mat["X"]
    y = mat["y"].squeeze()

    # Fix labels: 10 -> 0
    y[y == 10] = 0

    # Convert to (N, 32,32,3)
    X = np.transpose(X, (3, 0, 1, 2))

    return X, y



# ======================================================================
# DATASET WRAPPER 160×160
# ======================================================================

class RawDataset(Dataset):
    def __init__(self, data, labels, indices, augment=False):
        self.data = data
        self.labels = labels
        self.indices = indices

        if augment:
            self.T = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(160),
                transforms.ToTensor(),
                transforms.Normalize((0.485,0.456,0.406),
                                     (0.229,0.224,0.225)),
                transforms.RandomErasing(p=0.25)
            ])
        else:
            self.T = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(160),
                transforms.ToTensor(),
                transforms.Normalize((0.485,0.456,0.406),
                                     (0.229,0.224,0.225)),
            ])

    def __getitem__(self, i):
        idx = self.indices[i]
        img = self.data[idx]      # shape (32,32,3)
        img = self.T(img)
        return img, self.labels[idx]

    def __len__(self):
        return len(self.indices)



# ======================================================================
# DIRICHLET SPLIT
# ======================================================================

def dirichlet_split(labels, n_clients, alpha):
    labels = np.array(labels)
    per = [[] for _ in range(n_clients)]
    classes = np.unique(labels)

    for c in classes:
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        p = np.random.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(p) * len(idx)).astype(int)
        chunks = np.split(idx, cuts[:-1])
        for i in range(n_clients):
            per[i].extend(chunks[i])

    for cl in per:
        random.shuffle(cl)
    return per



# ======================================================================
# RESNET18 (solo layer3,4,fc allenabili)
# ======================================================================

# ======================================================================
# RESNET18 (solo layer3,4,fc allenabili) — COMPATIBILE CON OGNI TORCHVISION
# ======================================================================

class ResNet18Pre(nn.Module):
    def __init__(self, nc):
        super().__init__()

        # Compatibile con TUTTE le versioni di torchvision
        try:
            self.m = models.resnet18(weights='IMAGENET1K_V1')
        except:
            try:
                self.m = models.resnet18(pretrained=True)
            except:
                self.m = models.resnet18()

        # Replace FC
        in_f = self.m.fc.in_features
        self.m.fc = nn.Linear(in_f, nc)

        # Freeze everything
        for p in self.m.parameters():
            p.requires_grad = False

        # Unfreeze only layer3, layer4, fc
        for name, p in self.m.named_parameters():
            if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
                p.requires_grad = True

    def forward(self, x):
        return self.m(x)




# ======================================================================
# LOCAL TRAINING (SCAFFOLD)
# ======================================================================

def run_client(model, train_idx, data, labels, c_global, c_local, lr, device):

    ds = RawDataset(data, labels, train_idx, augment=True)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    old_params = [p.detach().clone().cpu() for p in trainable]

    opt = optim.SGD(trainable, lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    E = len(loader)

    for _ in range(LOCAL_EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()

            for i, p in enumerate(trainable):
                p.grad += 0.1 * (c_global[i].to(device) - c_local[i].to(device))

            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            opt.step()

    new_params = [p.detach().clone().cpu() for p in trainable]
    delta_c = []

    for i in range(len(trainable)):
        diff = new_params[i] - old_params[i]
        dc = 0.01 * (diff / max(E, 1))
        delta_c.append(dc)
        c_local[i] += dc

    return new_params, delta_c, c_local



# ======================================================================
# GLOBAL EVAL
# ======================================================================

def evaluate(model, loader, device):
    model.eval()
    tot = 0
    corr = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            corr += (pred == y).sum().item()
            tot += y.size(0)

    return corr/tot



# ======================================================================
# FEDERATED RUN
# ======================================================================

def federated_run():

    device = "cuda:0"

    # LOAD SVHN
    train_X, train_y = load_svhn_split("train")
    test_X, test_y = load_svhn_split("test")

    train_y = train_y.astype(np.int64)
    test_y = test_y.astype(np.int64)

    labels_np = train_y
    data_np = train_X

    # test loader
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(160),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),
                             (0.229,0.224,0.225))
    ])

    class TestWrapper(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            img = self.X[i]    # (32,32,3)
            img = transform_test(img)
            return img, self.y[i]

    testloader = DataLoader(TestWrapper(test_X, test_y),
                            batch_size=256, shuffle=False)

    for alpha in DIR_ALPHAS:

        log(f"\n==== SVHN | α={alpha} ====\n")

        splits = dirichlet_split(labels_np, NUM_CLIENTS, alpha)

        global_model = ResNet18Pre(10).to(device)
        trainable = [p for p in global_model.parameters() if p.requires_grad]

        c_global = [torch.zeros_like(p).cpu() for p in trainable]
        c_locals = [[torch.zeros_like(p).cpu() for p in trainable] for _ in range(NUM_CLIENTS)]

        for rnd in range(1, NUM_ROUNDS + 1):

            lr = LR_INIT if rnd <= LR_DECAY_ROUND else LR_DECAY

            global_start = [p.detach().clone().cpu() for p in trainable]

            new_params_all = []
            delta_c_all = []

            for cid in range(NUM_CLIENTS):

                local_model = ResNet18Pre(10).to(device)

                with torch.no_grad():
                    idx = 0
                    for p in local_model.parameters():
                        if p.requires_grad:
                            p.copy_(global_start[idx].to(device))
                            idx += 1

                new_params, dc, new_c_local = run_client(
                    local_model,
                    splits[cid],
                    data_np,
                    labels_np,
                    c_global,
                    c_locals[cid],
                    lr,
                    device
                )

                c_locals[cid] = new_c_local
                new_params_all.append(new_params)
                delta_c_all.append(dc)

            avg_params = []
            for i in range(len(trainable)):
                stacked = torch.stack([cp[i] for cp in new_params_all], dim=0)
                avg_params.append(stacked.mean(0))

            with torch.no_grad():
                idx=0
                for p in global_model.parameters():
                    if p.requires_grad:
                        p.copy_(avg_params[idx].to(device))
                        idx+=1

            for i in range(len(c_global)):
                c_global[i] = torch.stack([dc[i] for dc in delta_c_all]).mean(0)

            acc = evaluate(global_model, testloader, device)
            log(f"[ROUND {rnd}] ACC={acc*100:.2f}%")


# ======================================================================
# MAIN
# ======================================================================

def main():
    set_seed(SEED)
    federated_run()


if __name__ == "__main__":
    main()













