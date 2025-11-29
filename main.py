#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MULTI-GPU FEDERATED SCAFFOLD — SAFE VERSION
-------------------------------------------
- Soluzione A: ResNet18 completamente addestrabile (NO FREEZE)
- 1 worker permanente per GPU (nessun leak di file descriptors)
- Ogni worker allena 1 client alla volta (stabile)
- Compatibile con torchvision vecchio (pretrained=True)
- Niente mismatch tra tensor dims
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from queue import Empty

# ============================================================
# CONFIG
# ============================================================

NUM_CLIENTS = 10
DIRICHLET_ALPHAS = [0.5, 0.1, 0.05]
NUM_ROUNDS = 100
LOCAL_EPOCHS = 2
BATCH_SIZE = 64
LR_INIT = 0.01
LR_DECAY_ROUND = 50
BETA = 0.01
SEED = 42

torch.multiprocessing.set_sharing_strategy("file_system")


# ============================================================
# UTILS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dirichlet_split(labels, n_clients, alpha):
    labels = np.array(labels)
    classes = np.unique(labels)
    per_client = [[] for _ in range(n_clients)]

    for c in classes:
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        p = np.random.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(p) * len(idx)).astype(int)
        chunks = np.split(idx, cuts[:-1])
        for i in range(n_clients):
            per_client[i].extend(chunks[i])

    for c in per_client:
        random.shuffle(c)

    return per_client


# ============================================================
# MODEL
# ============================================================

class ResNet18_Pretrained(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # torchvision 0.9 fallback
        try:
            from torchvision.models import ResNet18_Weights
            self.model = models.resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            self.model = models.resnet18(pretrained=True)

        # override last layer
        in_f = self.model.fc.in_features
        self.model.fc = nn.Linear(in_f, num_classes)

        # SOLUZIONE A: ALL parameters trainable
        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)


def get_trainable_params(model):
    return [p.detach().cpu().clone() for p in model.parameters()
            if p.requires_grad]


def set_trainable_params(model, params):
    idx = 0
    for p in model.parameters():
        if p.requires_grad:
            p.data = params[idx].clone().to(p.device)
            idx += 1


def zero_like_trainable(model):
    return [torch.zeros_like(p).cpu() for p in model.parameters()
            if p.requires_grad]


# ============================================================
# CLIENT UPDATE
# ============================================================

def client_update(gpu_id, client_id, train_idx, trainset, global_params,
                  c_local, c_global, lr, num_classes, result_queue):

    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    # Build model
    model = ResNet18_Pretrained(num_classes).to(device)

    # Load global params
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p, new in zip(trainable, global_params):
        p.data = new.clone().to(device)

    loader = DataLoader(
        Subset(trainset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    old_params = [p.detach().clone().cpu() for p in trainable]

    opt = optim.SGD(trainable, lr=lr, momentum=0.9, weight_decay=5e-4)
    E = len(loader)

    # TRAIN
    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()

            # SCAFFOLD correction
            for i, p in enumerate(trainable):
                p.grad += (c_global[i].to(device) -
                           c_local[i].to(device))

            opt.step()

    # NEW PARAMS
    new_params = [p.detach().cpu().clone() for p in trainable]

    # UPDATE c_local
    delta_c = []
    for i in range(len(new_params)):
        diff = new_params[i] - old_params[i]
        dc = BETA * (diff / max(E, 1))
        delta_c.append(dc)
        c_local[i] += dc

    result_queue.put((client_id, new_params, c_local, delta_c))


# ============================================================
# WORKER (1 per GPU)
# ============================================================

def worker_loop(gpu_id, task_queue, result_queue, trainset, num_classes):

    while True:
        job = task_queue.get()

        if job[0] == "STOP":
            return

        _, cid, train_idx, global_params, c_local, c_global, lr = job

        client_update(
            gpu_id, cid, train_idx, trainset,
            global_params, c_local, c_global,
            lr, num_classes,
            result_queue
        )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


# ============================================================
# FEDERATED SERVER
# ============================================================

def run_federated(dataset_name, gpus):

    print(f"\n========== DATASET: {dataset_name} ==========\n")

    # ------------------------------------
    # LOAD DATASET
    # ------------------------------------
    if dataset_name == "CIFAR10":
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=16),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        trainset = datasets.CIFAR10("./data", train=True, download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10("./data", train=False, download=True,
                                   transform=transform_test)
        labels = trainset.targets

    elif dataset_name == "CIFAR100":
        num_classes = 100
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=16),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        trainset = datasets.CIFAR100("./data", train=True, download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100("./data", train=False, download=True,
                                   transform=transform_test)
        labels = trainset.targets

    elif dataset_name == "SVHN":
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        trainset = datasets.SVHN("./data", split="train", download=True,
                                 transform=transform_train)
        testset = datasets.SVHN("./data", split="test", download=True,
                                transform=transform_test)
        labels = trainset.labels

    testloader = DataLoader(testset, batch_size=256, shuffle=False)

    # ====================================================
    # LOOP ON ALPHAS
    # ====================================================

    for alpha in DIRICHLET_ALPHAS:

        print(f"\n==== α = {alpha} ====\n")

        splits = dirichlet_split(labels, NUM_CLIENTS, alpha)

        device0 = "cuda:0"
        global_model = ResNet18_Pretrained(num_classes).to(device0)
        global_params = get_trainable_params(global_model)
        c_global = zero_like_trainable(global_model)
        c_local = [zero_like_trainable(global_model) for _ in range(NUM_CLIENTS)]

        # ------------------------------------------------
        # WORKERS (PERMANENT)
        # ------------------------------------------------

        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue(maxsize=gpus)
        result_queue = ctx.Queue(maxsize=gpus)

        workers = []
        for gpu_id in range(gpus):
            p = ctx.Process(
                target=worker_loop,
                args=(gpu_id, task_queue, result_queue,
                      trainset, num_classes)
            )
            p.start()
            workers.append(p)

        # ------------------------------------------------
        # FEDERATED ROUNDS
        # ------------------------------------------------

        for rnd in range(1, NUM_ROUNDS + 1):

            lr = LR_INIT if rnd <= LR_DECAY_ROUND else LR_INIT * 0.1
            print(f"\n--- ROUND {rnd}/{NUM_ROUNDS} ---")

            next_client = 0
            finished = 0

            # Multi-GPU streaming
            while finished < NUM_CLIENTS:

                # dispatch until GPUs full
                while next_client < NUM_CLIENTS and not task_queue.full():
                    cid = next_client
                    next_client += 1

                    task_queue.put((
                        "RUN",
                        cid,
                        splits[cid],
                        global_params,
                        c_local[cid],
                        c_global,
                        lr
                    ))

                # receive one result
                cid, new_params, new_c_local, delta_c = result_queue.get()
                c_local[cid] = new_c_local

                if finished == 0:
                    accum = [torch.zeros_like(new_params[i])
                             for i in range(len(new_params))]

                for i in range(len(accum)):
                    accum[i] += new_params[i]

                finished += 1

            global_params = [p / NUM_CLIENTS for p in accum]
            set_trainable_params(global_model, global_params)

            # Evaluate
            acc = evaluate(global_model, testloader, device0)
            print(f"[ROUND {rnd}] ACC = {acc*100:.2f}%")

        # Stop workers
        for _ in range(gpus):
            task_queue.put(("STOP",))
        for p in workers:
            p.join()

        print(f"\n✓ DONE dataset={dataset_name}, α={alpha}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    set_seed(SEED)

    for ds in ["CIFAR10", "CIFAR100", "SVHN"]:
        run_federated(ds, gpus=args.gpus)


if __name__ == "__main__":
    main()

