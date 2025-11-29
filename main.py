#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import numpy as np
import sys
import io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


# ============================================================
# LOGGING
# ============================================================

DEBUG = True

def log_debug(msg):
    if DEBUG:
        print(msg, file=sys.stderr, flush=True)

def log_accuracy(msg):
    print(msg, file=sys.stdout, flush=True)


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
    per_client = [[] for _ in range(n_clients)]
    classes = np.unique(labels)

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
        try:
            from torchvision.models import ResNet18_Weights
            self.model = models.resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1
            )
        except Exception:
            self.model = models.resnet18(pretrained=True)

        in_f = self.model.fc.in_features
        self.model.fc = nn.Linear(in_f, num_classes)

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)


# ============================================================
# SERIALIZATION HELPERS
# ============================================================

def tensorlist_to_bytes(params):
    """Serialize list of CPU tensors into bytes."""
    buffer = io.BytesIO()
    torch.save(params, buffer)
    return buffer.getvalue()


def bytes_to_tensorlist(b):
    """Deserialize bytes back into list of tensors."""
    buffer = io.BytesIO(b)
    params = torch.load(buffer)
    return params


# ============================================================
# CLIENT UPDATE
# ============================================================

def client_update(model, device, train_idx, trainset,
                  c_local, c_global, lr):

    trainable = [p for p in model.parameters() if p.requires_grad]
    old_params = [p.detach().clone().cpu() for p in trainable]

    loader = DataLoader(
        Subset(trainset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    opt = optim.SGD(trainable, lr=lr, momentum=0.9, weight_decay=5e-4)
    E = len(loader)

    for ep in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            for i, p in enumerate(trainable):
                p.grad += (c_global[i].to(device) - c_local[i].to(device))
            opt.step()

    new_params = [p.detach().clone().cpu() for p in trainable]

    delta_c = []
    for i in range(len(new_params)):
        diff = new_params[i] - old_params[i]
        dc = BETA * (diff / max(E, 1))
        delta_c.append(dc)
        c_local[i] += dc

    return new_params, c_local, delta_c


# ============================================================
# WORKER LOOP (spawned)
# ============================================================

def worker_loop(gpu_id, pipe, trainset, num_classes):

    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    log_debug(f"[WORKER {gpu_id}] Started on {device}")

    model = ResNet18_Pretrained(num_classes).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]

    while True:
        msg = pipe.recv()

        if msg[0] == "STOP":
            log_debug(f"[WORKER {gpu_id}] Stopping.")
            return

        if msg[0] == "SET_GLOBAL":
            # receive whole model as bytes
            byte_blob = pipe.recv()
            param_list = bytes_to_tensorlist(byte_blob)
            for p, newp in zip(trainable, param_list):
                p.data.copy_(newp.to(device))
            pipe.send("OK_SET")
            continue

        if msg[0] == "RUN":
            _, cid, train_idx, c_local, c_global, lr = msg

            new_params, new_c_local, delta_c = client_update(
                model, device, train_idx,
                trainset,
                c_local,
                c_global,
                lr
            )

            pipe.send((cid, new_params, new_c_local, delta_c))


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(name):

    if name == "CIFAR10":
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
        train = datasets.CIFAR10("./data", train=True, download=True,
                                 transform=transform_train)
        test = datasets.CIFAR10("./data", train=False, download=True,
                                transform=transform_test)
        labels = train.targets

    elif name == "CIFAR100":
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
        train = datasets.CIFAR100("./data", train=True, download=True,
                                  transform=transform_train)
        test = datasets.CIFAR100("./data", train=False, download=True,
                                 transform=transform_test)
        labels = train.targets

    else:
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        train = datasets.SVHN("./data", split="train", download=True,
                              transform=transform_train)
        test = datasets.SVHN("./data", split="test", download=True,
                             transform=transform_test)
        labels = train.labels

    return train, test, labels, num_classes


# ============================================================
# EVAL
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
# FEDERATED
# ============================================================

def federated_run(dataset_name, gpus):

    print(f"\n========== DATASET: {dataset_name} ==========\n", flush=True)

    trainset, testset, labels, num_classes = load_dataset(dataset_name)
    testloader = DataLoader(testset, batch_size=256, shuffle=False)

    for alpha in DIRICHLET_ALPHAS:

        print(f"\n==== α = {alpha} ====\n", flush=True)

        splits = dirichlet_split(labels, NUM_CLIENTS, alpha)

        device0 = "cuda:0"
        global_model = ResNet18_Pretrained(num_classes).to(device0)
        trainable_params = [p for p in global_model.parameters() if p.requires_grad]

        c_global = [torch.zeros_like(p).cpu() for p in trainable_params]
        c_local = [[torch.zeros_like(p).cpu() for p in trainable_params]
                   for _ in range(NUM_CLIENTS)]

        # =====================================================
        # SPAWN WORKERS + PIPE
        # =====================================================
        ctx = mp.get_context("spawn")
        workers = []
        pipes = []

        for gpu in range(gpus):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=worker_loop,
                args=(gpu, child_conn, trainset, num_classes)
            )
            p.start()
            workers.append(p)
            pipes.append(parent_conn)

        # =====================================================
        # RUN ROUNDS
        # =====================================================
        for rnd in range(1, NUM_ROUNDS + 1):

            lr = LR_INIT if rnd <= LR_DECAY_ROUND else LR_INIT * 0.1
            print(f"--- ROUND {rnd}/{NUM_ROUNDS} ---", flush=True)

            # -------------------------------------------------
            # BROADCAST GLOBAL MODEL (FULL)
            # -------------------------------------------------
            param_list = [p.detach().cpu() for p in trainable_params]
            byte_blob = tensorlist_to_bytes(param_list)

            for pipe in pipes:
                pipe.send(("SET_GLOBAL",))
                pipe.send(byte_blob)

            for pipe in pipes:
                ok = pipe.recv()   # wait confirmation

            # -------------------------------------------------
            # CLIENT JOBS
            # -------------------------------------------------
            # round robin assignment
            for cid in range(NUM_CLIENTS):
                gpu = cid % gpus
                pipes[gpu].send((
                    "RUN",
                    cid,
                    splits[cid],
                    c_local[cid],
                    c_global,
                    lr
                ))

            # -------------------------------------------------
            # COLLECT RESULTS
            # -------------------------------------------------
            new_params_accum = None

            for _ in range(NUM_CLIENTS):
                # receive from ANY pipe non-blocking round robin
                got = False
                while not got:
                    for pipe in pipes:
                        if pipe.poll():
                            cid, new_params, new_c_local, delta_c = pipe.recv()
                            got = True
                            break

                c_local[cid] = new_c_local

                if new_params_accum is None:
                    new_params_accum = [torch.zeros_like(p) for p in new_params]

                for i in range(len(new_params)):
                    new_params_accum[i] += new_params[i]

                for i in range(len(c_global)):
                    c_global[i] += delta_c[i] / NUM_CLIENTS

            # -------------------------------------------------
            # UPDATE GLOBAL MODEL
            # -------------------------------------------------
            avg_params = [p / NUM_CLIENTS for p in new_params_accum]

            idx = 0
            with torch.no_grad():
                for p in trainable_params:
                    p.copy_(avg_params[idx].to(device0))
                    idx += 1

            # -------------------------------------------------
            # EVAL
            # -------------------------------------------------
            acc = evaluate(global_model, testloader, device0)
            log_accuracy(f"[ROUND {rnd}] ACC = {acc*100:.2f}%")

        # =====================================================
        # SHUTDOWN WORKERS
        # =====================================================
        for pipe in pipes:
            pipe.send(("STOP",))
        for p in workers:
            p.join()


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)

    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    for ds in ["CIFAR10", "CIFAR100", "SVHN"]:
        federated_run(ds, args.gpus)


if __name__ == "__main__":
    main()



