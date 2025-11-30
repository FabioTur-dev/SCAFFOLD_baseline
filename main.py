#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import to_pil_image


# ==============================================================
# CONFIG (SEQUENTIAL VERSION)
# ==============================================================

NUM_CLIENTS = 10
DIR_ALPHAS = [0.5]
NUM_ROUNDS = 50
LOCAL_EPOCHS = 2
BATCH = 128

LR_INIT = 0.004
LR_DECAY = 0.0015
DECAY_ROUND = 25

BETA = 0.01
DAMPING = 0.1
GRAD_CLIP = 5.0
SEED = 42


# ==============================================================

def log(msg):
    print(msg, flush=True)


# ==============================================================

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ==============================================================

class RawDataset(Dataset):
    def __init__(self, data, labels, indices, augment=False):
        self.data = data
        self.labels = labels
        self.indices = indices

        if augment:
            self.T = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(160),
                transforms.ToTensor(),
                transforms.Normalize((0.485,0.456,0.406),
                                     (0.229,0.224,0.225)),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
            ])
        else:
            self.T = transforms.Compose([
                transforms.Resize(160),
                transforms.ToTensor(),
                transforms.Normalize((0.485,0.456,0.406),
                                     (0.229,0.224,0.225)),
            ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img = to_pil_image(self.data[self.indices[i]])
        img = self.T(img)
        return img, self.labels[self.indices[i]]


# ==============================================================

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


# ==============================================================

class ResNet18Pre(nn.Module):
    def __init__(self, nc):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights
            self.m = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except:
            self.m = models.resnet18(pretrained=True)

        in_f = self.m.fc.in_features
        self.m.fc = nn.Linear(in_f, nc)

        # Solo layer3/4 + fc sbloccati
        for name, p in self.m.named_parameters():
            if "layer3" in name or "layer4" in name or "fc" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

    def forward(self, x):
        return self.m(x)


# ==============================================================

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1)
            correct += (pred == y.to(device)).sum().item()
            total += y.size(0)
    return correct / total


# ==============================================================

def client_update_single(model, c_global, c_local, train_idx, data, labels, lr, device):

    ds = RawDataset(data, labels, train_idx, augment=True)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True,
                        num_workers=2, pin_memory=True)

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
                p.grad += DAMPING * (c_global[i].to(device) - c_local[i].to(device))

            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            opt.step()

    new_params = [p.detach().clone().cpu() for p in trainable]
    delta_c = []

    for i in range(len(trainable)):
        diff = new_params[i] - old_params[i]
        dc = BETA * (diff / max(E,1))
        delta_c.append(dc)
        c_local[i] += dc

    return new_params, c_local, delta_c


# ==============================================================

def federated_run(ds_name):

    device = "cuda:0"

    # ----- Load dataset -----
    if ds_name == "CIFAR10":
        tr = datasets.CIFAR10("./data", train=True, download=True)
        data = torch.tensor(tr.data).permute(0,3,1,2)
        labels = torch.tensor(tr.targets)
        nc = 10
    else:
        raise NotImplementedError()

    transform_test = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),
                             (0.229,0.224,0.225)),
    ])

    te = datasets.CIFAR10("./data", train=False, download=True,
                          transform=transform_test)

    testloader = DataLoader(te, batch_size=256, shuffle=False)

    labels_np = labels.numpy()

    for alpha in DIR_ALPHAS:
        log(f"\n==== DATASET={ds_name} | α={alpha} ====\n")

        splits = dirichlet_split(labels_np, NUM_CLIENTS, alpha)

        global_model = ResNet18Pre(nc).to(device)
        trainable = [p for p in global_model.parameters() if p.requires_grad]

        c_global = [torch.zeros_like(p).cpu() for p in trainable]
        c_local = [
            [torch.zeros_like(p).cpu() for p in trainable]
            for _ in range(NUM_CLIENTS)
        ]

        # FEDERATED ROUNDS (NO PARALLELISM)
        for rnd in range(1, NUM_ROUNDS + 1):

            lr = LR_INIT if rnd <= DECAY_ROUND else LR_DECAY

            new_accum = [torch.zeros_like(p).cpu() for p in trainable]
            all_delta_c = [[] for _ in trainable]

            # ---- EACH CLIENT SEQUENTIALLY ----
            for cid in range(NUM_CLIENTS):

                new_params, new_c_local, delta_c = client_update_single(
                    model=global_model,
                    c_global=c_global,
                    c_local=c_local[cid],
                    train_idx=splits[cid],
                    data=data,
                    labels=labels,
                    lr=lr,
                    device=device,
                )

                c_local[cid] = new_c_local

                for i,p in enumerate(new_params):
                    new_accum[i] += p
                    all_delta_c[i].append(delta_c[i])

            # ---- AGGREGATION ----
            avg_params = [x/NUM_CLIENTS for x in new_accum]

            with torch.no_grad():
                pi=0
                for p in global_model.parameters():
                    if p.requires_grad:
                        p.copy_(avg_params[pi].to(device))
                        pi+=1

            for i in range(len(c_global)):
                c_global[i] = sum(all_delta_c[i]) / NUM_CLIENTS

            acc = evaluate(global_model, testloader, device)
            log(f"[ROUND {rnd}] ACC = {acc*100:.2f}%")


# ==============================================================

def main():
    set_seed(SEED)
    federated_run("CIFAR10")


if __name__ == "__main__":
    main()












