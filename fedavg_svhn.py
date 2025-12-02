#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import random

# ======================================================
# CONFIG
# ======================================================
NUM_CLIENTS = 10
ALPHAS = [0.5, 0.1, 0.05]      # più alpha
LOCAL_EPOCHS = 1
BATCH = 256
ROUNDS = 25
LR = 0.001
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ======================================================
# FAST RAM DATASET
# ======================================================
class RAMDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def main():

    seed_everything(SEED)

    # ======================================================
    # TRANSFORMS (applied ONCE only)
    # ======================================================
    preprocess = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ======================================================
    # LOAD RAW SVHN
    # ======================================================
    raw_train = datasets.SVHN(root="./data", split='train',
                              download=True, transform=None)
    raw_test  = datasets.SVHN(root="./data", split='test',
                              download=True, transform=None)

    # ======================================================
    # PREPROCESS SVHN INTO RAM  (🔥 key difference)
    # ======================================================
    print("\n=== Preprocessing SVHN into RAM (train) ===")
    X_train, Y_train = [], []
    for img, label in raw_train:
        X_train.append(preprocess(img))
        Y_train.append(label)

    X_train = torch.stack(X_train, dim=0)
    Y_train = torch.tensor(Y_train, dtype=torch.long)

    print("Train RAM ready:", X_train.shape)

    print("\n=== Preprocessing SVHN into RAM (test) ===")
    X_test, Y_test = [], []
    for img, label in raw_test:
        X_test.append(preprocess(img))
        Y_test.append(label)

    X_test = torch.stack(X_test, dim=0)
    Y_test = torch.tensor(Y_test, dtype=torch.long)

    print("Test RAM ready:", X_test.shape)

    # RAM datasets
    trainRAM = RAMDataset(X_train, Y_train)
    testRAM  = RAMDataset(X_test,  Y_test)

    testloader = DataLoader(
        testRAM, batch_size=256, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # ======================================================
    # DIRICHLET SPLIT
    # ======================================================
    def dirichlet_split(Y, num_clients, alpha):
        labels = np.array(Y)
        num_classes = 10
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
    # MODEL FACTORY
    # ======================================================
    def build_resnet18():
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 10)
        return model

    # ======================================================
    # LOCAL TRAIN
    # ======================================================
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    def local_train(local_model, loader):
        local_model.train()
        opt = optim.SGD(local_model.parameters(), lr=LR, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(LOCAL_EPOCHS):
            for x, y in loader:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    loss = loss_fn(local_model(x), y)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

        return local_model.state_dict()

    # ======================================================
    # FEDAVG
    # ======================================================
    def fedavg(states):
        avg = {}
        with torch.no_grad():
            for k in states[0].keys():
                stacked = torch.stack([s[k].float() for s in states], dim=0).to(DEVICE)
                avg[k] = stacked.mean(dim=0)
        return avg

    # ======================================================
    # EVALUATION
    # ======================================================
    def evaluate(model):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            for x, y in testloader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return 100 * correct / total

    # ======================================================
    # MAIN FEDAVG LOOP
    # ======================================================
    for ALPHA in ALPHAS:
        print(f"\n============================")
        print(f"=== Dirichlet alpha = {ALPHA} ===")
        print(f"============================\n")

        client_splits = dirichlet_split(Y_train, NUM_CLIENTS, ALPHA)
        global_model = build_resnet18().to(DEVICE)

        for rnd in range(1, ROUNDS + 1):
            local_states = []

            for cid in range(NUM_CLIENTS):

                subset = Subset(trainRAM, client_splits[cid])
                loader = DataLoader(
                    subset,
                    batch_size=BATCH,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True
                )

                local_model = build_resnet18().to(DEVICE)
                local_model.load_state_dict(global_model.state_dict())

                st = local_train(local_model, loader)
                local_states.append(st)

            new_state = fedavg(local_states)
            global_model.load_state_dict(new_state)

            acc = evaluate(global_model)
            print(f"[ALPHA {ALPHA}][ROUND {rnd}] ACC = {acc:.2f}%")



if __name__ == "__main__":
    main()
