#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCAFFOLD (CIFAR-10) — fully comparable with FedAvg/FedProx
Includes:
- weighted aggregation
- correct BN buffer aggregation
- identical training hyperparameters
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# ======================================================
# CONFIG (IDENTICI)
# ======================================================
NUM_CLIENTS = 10
ALPHAS = [0.5, 0.1, 0.05]
LOCAL_EPOCHS = 1
BATCH = 256
ROUNDS = 50
LR = 0.001
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# SEED
# ======================================================
def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

# ======================================================
# MODEL — CIFAR-10
# ======================================================
class ResNet18_C10(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.model(x)

# ======================================================
# PARAM HELPERS
# ======================================================
def get_param_dict(model):
    return {name: p for name, p in model.named_parameters()}

def get_param_tensor_dict(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}

# ======================================================
# AMP
# ======================================================
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# ======================================================
# LOCAL TRAIN — SCAFFOLD
# ======================================================
def local_train_scaffold(local_model, global_model, c_local, c_global, loader):
    local_model.train()
    opt = optim.SGD(local_model.parameters(), lr=LR, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    global_params = get_param_tensor_dict(global_model)

    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss = loss_fn(local_model(x), y)

            scaler.scale(loss).backward()

            # SCAFFOLD correction
            with torch.no_grad():
                for name, p in local_model.named_parameters():
                    if p.grad is not None:
                        p.grad += (c_global[name] - c_local[name])

            scaler.step(opt)
            scaler.update()

    # Δw
    delta_w = {}
    with torch.no_grad():
        for name, p in local_model.named_parameters():
            delta_w[name] = p.detach() - global_params[name]

    # update c_local
    T = LOCAL_EPOCHS * len(loader)
    c_local_new = {}
    with torch.no_grad():
        for k in c_local:
            c_local_new[k] = (
                c_local[k]
                - c_global[k]
                + (1.0 / (T * LR)) * delta_w[k]
            )

    return c_local_new, delta_w

# ======================================================
# AGGREGATION — PESATA + BN BUFFERS (FAIR)
# ======================================================
def scaffold_aggregate_weighted(
    global_model,
    local_models,
    deltas,
    c_global,
    c_locals_new,
    client_sizes
):
    total = float(sum(client_sizes))

    with torch.no_grad():

        # 1) Update trainable parameters
        for name, p in global_model.named_parameters():
            if p.dtype.is_floating_point:
                p.add_(sum(
                    (client_sizes[i] / total) * deltas[i][name]
                    for i in range(len(deltas))
                ))

        # 2) Update global control variate
        for k in c_global:
            c_global[k] = sum(
                (client_sizes[i] / total) * c_locals_new[i][k]
                for i in range(len(c_locals_new))
            )

        # 3) Aggregate BN buffers
        global_buffers = dict(global_model.named_buffers())
        local_buffers_list = [dict(m.named_buffers()) for m in local_models]

        for bname, gb in global_buffers.items():
            if gb.dtype.is_floating_point:
                gb.copy_(sum(
                    (client_sizes[i] / total) *
                    local_buffers_list[i][bname].to(gb.device)
                    for i in range(len(local_models))
                ))
            else:
                gb.copy_(torch.max(torch.stack([
                    local_buffers_list[i][bname].to(gb.device)
                    for i in range(len(local_models))
                ])))

# ======================================================
# EVALUATION
# ======================================================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100 * correct / total

# ======================================================
# DIRICHLET SPLIT
# ======================================================
def dirichlet_split(labels, num_clients, alpha):
    labels = np.array(labels)
    num_classes = len(np.unique(labels))

    while True:
        client_indices = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            idx = np.where(labels == c)[0]
            np.random.shuffle(idx)

            props = np.random.dirichlet([alpha] * num_clients)
            props = (props * len(idx)).astype(int)

            while props.sum() < len(idx):
                props[np.argmax(props)] += 1

            start = 0
            for i in range(num_clients):
                end = start + props[i]
                client_indices[i].extend(idx[start:end])
                start = end

        if all(len(ci) > 0 for ci in client_indices):
            return client_indices

# ======================================================
# MAIN
# ======================================================
def main():
    seed_everything(SEED)

    transform = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    train_raw = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test_raw  = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    testloader = DataLoader(
        test_raw, batch_size=BATCH, shuffle=False,
        num_workers=2, pin_memory=True
    )

    labels_train = train_raw.targets

    for ALPHA in ALPHAS:
        print("\n============================")
        print(f"=== Dirichlet alpha = {ALPHA} | SCAFFOLD (CIFAR-10 FAIR) ===")
        print("============================")

        global_model = ResNet18_C10().to(DEVICE)

        params = get_param_dict(global_model)
        c_global = {k: torch.zeros_like(v, device=DEVICE) for k, v in params.items()}
        c_locals = [
            {k: torch.zeros_like(v, device=DEVICE) for k, v in params.items()}
            for _ in range(NUM_CLIENTS)
        ]

        client_indices = dirichlet_split(labels_train, NUM_CLIENTS, ALPHA)
        client_sizes = [len(idx) for idx in client_indices]

        for rnd in range(1, ROUNDS + 1):

            deltas = []
            new_c_locals = []
            local_models = []

            for cid in range(NUM_CLIENTS):
                subset = Subset(train_raw, client_indices[cid])
                loader = DataLoader(
                    subset, batch_size=BATCH, shuffle=True,
                    num_workers=2, pin_memory=True
                )

                local_model = ResNet18_C10().to(DEVICE)
                local_model.load_state_dict(global_model.state_dict(), strict=True)

                c_new, delta = local_train_scaffold(
                    local_model, global_model,
                    c_locals[cid], c_global, loader
                )

                new_c_locals.append(c_new)
                deltas.append(delta)
                local_models.append(local_model)

            scaffold_aggregate_weighted(
                global_model,
                local_models,
                deltas,
                c_global,
                new_c_locals,
                client_sizes
            )

            c_locals = new_c_locals

            acc = evaluate(global_model, testloader)
            print(f"[ALPHA {ALPHA}][ROUND {rnd}] ACC = {acc:.2f}%")

if __name__ == "__main__":
    main()
