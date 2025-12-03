#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import random

# ======================================================
# CONFIG
# ======================================================
NUM_CLIENTS = 10
ALPHAS = [0.05, 0.1, 0.5]
LOCAL_EPOCHS = 1
BATCH = 256
ROUNDS = 100
LR = 0.001
SEED = 42

# Valore "neutro" tipico in letteratura FedProx (usato spessissimo)
FEDPROX_MU = 0.01

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================
# UTILITIES
# ======================================================
def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ======================================================
# MODEL — ResNet18 PRETRAINED (TESTINA CIFAR-10)
# ======================================================
class ResNet18_C10_FULL(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10)  # 10 classi CIFAR-10

    def forward(self, x):
        return self.model(x)


# ======================================================
# LOCAL TRAIN (FedProx)
# ======================================================
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

def local_train_fedprox(local_model, global_model, loader, mu=FEDPROX_MU):
    local_model.train()
    opt = optim.SGD(local_model.parameters(), lr=LR, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    # Congela i pesi globali per il termine di prossimità
    global_params = {k: v.detach().clone() for k, v in global_model.state_dict().items()}

    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                preds = local_model(x)
                loss = loss_fn(preds, y)

                # ----- FedProx proximal term -----
                prox = 0.0
                for (name, param) in local_model.named_parameters():
                    prox += ((param - global_params[name].to(DEVICE)) ** 2).sum()
                loss = loss + (mu / 2.0) * prox

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    return local_model.state_dict()


# ======================================================
# FEDAVG (server aggregation, unchanged)
# ======================================================
def fedavg(states):
    avg = {}
    with torch.no_grad():
        for k in states[0]:
            tensors = [s[k] for s in states]

            if tensors[0].dtype in [torch.float16, torch.float32, torch.float64]:
                stacked = torch.stack(tensors, dim=0).to(DEVICE)
                avg[k] = stacked.mean(dim=0)
            else:
                avg[k] = tensors[0].clone().to(DEVICE)

    return avg


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
# MAIN
# ======================================================
def main():
    seed_everything(SEED)

    transform = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),  # CIFAR-10 mean/std
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    train_raw = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test_raw  = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    testloader = DataLoader(test_raw, batch_size=BATCH, shuffle=False,
                            num_workers=2, pin_memory=True)

    labels_train = train_raw.targets

    for ALPHA in ALPHAS:
        print("\n============================")
        print(f"=== Dirichlet alpha = {ALPHA} (FedProx μ={FEDPROX_MU}) ===")
        print("============================")

        global_model = ResNet18_C10_FULL().to(DEVICE)

        client_indices = dirichlet_split(labels_train, NUM_CLIENTS, ALPHA)

        for rnd in range(1, ROUNDS + 1):
            local_states = []

            for cid in range(NUM_CLIENTS):
                subset = Subset(train_raw, client_indices[cid])
                loader = DataLoader(subset, batch_size=BATCH, shuffle=True,
                                    num_workers=2, pin_memory=True)

                local_model = ResNet18_C10_FULL().to(DEVICE)
                local_model.load_state_dict(global_model.state_dict(), strict=True)

                state = local_train_fedprox(local_model, global_model, loader)
                local_states.append(state)

            new_state = fedavg(local_states)
            global_model.load_state_dict(new_state)

            acc = evaluate(global_model, testloader)
            print(f"[ALPHA {ALPHA}][ROUND {rnd}] ACC = {acc:.2f}%")


# ======================================================
# ENTRYPOINT
# ======================================================
if __name__ == "__main__":
    main()
