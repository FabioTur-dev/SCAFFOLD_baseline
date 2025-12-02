#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
import random
import copy
import kornia as K

# ======================================================
# CONFIG
# ======================================================
NUM_CLIENTS = 10
ALPHAS = [0.05, 0.5, 0.1]     # due dirichlet da testare in sequenza
LOCAL_EPOCHS = 1
BATCH = 512               # EXTREME batch size
ROUNDS = 25
LR = 0.001
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ======================================================
# LOAD + PRELOAD CIFAR-100 IN RAM
# ======================================================
def preload_dataset_to_ram():
    cpu_tf = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor()
    ])

    train_raw = datasets.CIFAR100("./data", train=True, download=True, transform=cpu_tf)
    test_raw  = datasets.CIFAR100("./data", train=False, download=True, transform=cpu_tf)

    # Stack in RAM (circa ~700–900MB)
    X_train = torch.stack([train_raw[i][0] for i in range(len(train_raw))])
    y_train = torch.tensor([train_raw[i][1] for i in range(len(train_raw))], dtype=torch.long)

    X_test = torch.stack([test_raw[i][0] for i in range(len(test_raw))])
    y_test = torch.tensor([test_raw[i][1] for i in range(len(test_raw))], dtype=torch.long)

    return X_train, y_train, X_test, y_test


# ======================================================
# GPU TRANSFORMS (KORNIA)
# ======================================================
gpu_normalize = K.enhance.Normalize(
    mean=torch.tensor([0.485, 0.456, 0.406], device=DEVICE),
    std=torch.tensor([0.229, 0.224, 0.225], device=DEVICE)
)


# ======================================================
# DIRICHLET SPLIT (100 classi)
# ======================================================
def dirichlet_split(y, num_clients, alpha):
    labels = y.numpy()
    num_classes = 100
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

    return client_indices


# ======================================================
# MODEL FACTORY — COMPILE + CHANNELS_LAST (100 CLASSI)
# ======================================================
def build_resnet18():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 100)   # CIFAR-100: 100 classi

    # Boost prestazioni
    model.to(memory_format=torch.channels_last)
    model = torch.compile(model, mode="max-autotune")

    return model.to(DEVICE)


# ======================================================
# LOCAL TRAIN (RAM-BASED + GPU transforms)
# ======================================================
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

def local_train(local_model, X, y):
    local_model.train()
    opt = optim.SGD(local_model.parameters(), lr=LR, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    N = len(X)
    indices = torch.randperm(N)

    for _ in range(LOCAL_EPOCHS):
        for i in range(0, N, BATCH):
            batch_idx = indices[i:i+BATCH]

            xb = X[batch_idx].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
            yb = y[batch_idx].to(DEVICE, non_blocking=True)

            xb = gpu_normalize(xb)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                loss = loss_fn(local_model(xb), yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    return local_model.state_dict()


# ======================================================
# FEDAVG (GPU)
# ======================================================
def fedavg(states):
    states_float = [
        {k: v.float() if v.dtype in (torch.long, torch.int64) else v
         for k, v in sd.items()}
        for sd in states
    ]

    avg = {}
    with torch.no_grad():
        for k in states_float[0].keys():
            stacked = torch.stack([sd[k] for sd in states_float], dim=0).to(DEVICE)
            avg[k] = stacked.mean(dim=0)

    return avg


# ======================================================
# EVALUATION (GPU)
# ======================================================
def evaluate(model, X_test, y_test):
    model.eval()
    correct = total = 0

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        for i in range(0, len(X_test), 512):
            xb = X_test[i:i+512].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
            yb = y_test[i:i+512].to(DEVICE, non_blocking=True)

            xb = gpu_normalize(xb)
            pred = model(xb).argmax(1)

            correct += (pred == yb).sum().item()
            total += len(yb)

    return 100 * correct / total


# ======================================================
# MAIN EXTREME PIPELINE
# ======================================================
def main():
    seed_everything(SEED)

    print("\n=========== EXTREME VERSION CIFAR-100 ===========")
    print("Preloading entire CIFAR-100 into RAM...")
    print("GPU transforms + torch.compile + channels_last")
    print("=================================================\n")

    # LOAD DATA IN RAM
    X_train, y_train, X_test, y_test = preload_dataset_to_ram()

    for ALPHA in ALPHAS:

        print(f"\n===== ALPHA = {ALPHA} =====\n")

        # Dirichlet split (100 classi)
        client_indices = dirichlet_split(y_train, NUM_CLIENTS, ALPHA)

        # Modello globale
        global_model = build_resnet18()

        # Pre-carico tensori client-side una sola volta
        client_data = []
        for cid in range(NUM_CLIENTS):
            idx = client_indices[cid]
            client_data.append((X_train[idx], y_train[idx]))

        # ROUND TRAINING
        for rnd in range(1, ROUNDS + 1):

            local_states = []

            for cid in range(NUM_CLIENTS):
                local_model = copy.deepcopy(global_model)
                Xc, yc = client_data[cid]
                st = local_train(local_model, Xc, yc)
                local_states.append(st)

            global_model.load_state_dict(fedavg(local_states))

            acc = evaluate(global_model, X_test, y_test)
            print(f"[ALPHA {ALPHA}] ROUND {rnd:02d} → ACC = {acc:.2f}%")

            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
