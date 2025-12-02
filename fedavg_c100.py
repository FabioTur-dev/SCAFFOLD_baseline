#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import copy

# ======================================================
# CONFIG
# ======================================================
NUM_CLIENTS = 10
ALPHAS = [0.05, 0.1]    # due alpha in sequenza
LOCAL_EPOCHS = 1
BATCH = 256
ROUNDS = 25
LR = 0.001
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():

    seed_everything(SEED)

    print("\n=============================================")
    print("       ULTRA-FAST FedAvg (Linux build)")
    print("=============================================\n")

    # ======================================================
    # FAST TRANSFORMS (preprocess una sola volta)
    # ======================================================
    transform = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # LOAD DATASET UNA SOLA VOLTA
    trainset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    testset  = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    # Testloader veloce
    testloader = DataLoader(
        testset, batch_size=512, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # ======================================================
    # DIRICHLET SPLIT VELOCE
    # ======================================================
    def dirichlet_split(dataset, num_clients, alpha):
        labels = np.array(dataset.targets)
        num_classes = 10
        out = [[] for _ in range(num_clients)]

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
                out[i].extend(idx[start:end])
                start = end
        return out

    # ======================================================
    # MODEL FACTORY (SUPER-OPTIMIZED)
    # ======================================================
    def build_resnet18():
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 10)

        # Boost GPU performance
        model.to(memory_format=torch.channels_last)
        model = torch.compile(model)

        return model.to(DEVICE)

    # ======================================================
    # LOCAL TRAIN con AMP + channels_last
    # ======================================================
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    def local_train(local_model, loader):
        local_model.train()
        opt = optim.SGD(local_model.parameters(), lr=LR, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(LOCAL_EPOCHS):
            for x, y in loader:
                x = x.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
                y = y.to(DEVICE, non_blocking=True)

                opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    loss = loss_fn(local_model(x), y)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

        return local_model.state_dict()

    # ======================================================
    # FedAvg FULL GPU
    # ======================================================
    def fedavg(states):
        states_float = [
            {k: v.float() if v.dtype in (torch.int64, torch.long) else v
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
    # Evaluation veloce
    # ======================================================
    def evaluate(model):
        model.eval()
        correct = total = 0

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            for x, y in testloader:
                x = x.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
                y = y.to(DEVICE, non_blocking=True)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        return 100 * correct / total

    # ======================================================
    # LOOP SU ALPHA
    # ======================================================
    for ALPHA in ALPHAS:

        print(f"\n===== ALPHA = {ALPHA} =====\n")

        client_indices = dirichlet_split(trainset, NUM_CLIENTS, ALPHA)

        # Modello globale
        global_model = build_resnet18()

        # Client loaders persistenti
        client_loaders = []
        for cid in range(NUM_CLIENTS):
            subset = Subset(trainset, client_indices[cid])
            loader = DataLoader(
                subset,
                batch_size=BATCH,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )
            client_loaders.append(loader)

        # ROUNDS
        for rnd in range(1, ROUNDS + 1):
            local_states = []

            for cid in range(NUM_CLIENTS):

                # Clonazione super veloce (NO ricreazione resnet)
                local_model = copy.deepcopy(global_model)

                st = local_train(local_model, client_loaders[cid])
                local_states.append(st)

            new_state = fedavg(local_states)
            global_model.load_state_dict(new_state)

            acc = evaluate(global_model)
            print(f"[ALPHA {ALPHA}] ROUND {rnd:02d} → ACC = {acc:.2f}%")

            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
