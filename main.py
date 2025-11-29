#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import random
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# ======================================================================
# LOGGING
# ======================================================================

DEBUG = True

def logd(msg: str):
    if DEBUG:
        print(msg, file=sys.stderr, flush=True)

def loga(msg: str):
    print(msg, file=sys.stdout, flush=True)


# ======================================================================
# CONFIG
# ======================================================================

NUM_CLIENTS = 10
DIR_ALPHAS = [0.5, 0.1, 0.05]
NUM_ROUNDS = 100
LOCAL_EPOCHS = 2
BATCH = 64
LR0 = 0.01
LR_DECAY = 50
BETA = 0.01
SEED = 42


# ======================================================================
# UTILS
# ======================================================================

def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


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
# MODEL
# ======================================================================

class ResNet18Pre(nn.Module):
    def __init__(self, nc: int):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights
            self.m = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            self.m = models.resnet18(pretrained=True)

        in_f = self.m.fc.in_features
        self.m.fc = nn.Linear(in_f, nc)
        for p in self.m.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.m(x)


# ======================================================================
# DATA PREPROCESSING (TRAIN SET → TENSOR CACHE)
# ======================================================================

def preprocess_dataset(ds_name: str):
    """
    Preprocess train set ONCE per dataset:
    save tensors in cached/{ds_name}_train_data.pt and _labels.pt.
    """
    os.makedirs("cached", exist_ok=True)
    data_file = f"cached/{ds_name}_train_data.pt"
    label_file = f"cached/{ds_name}_train_labels.pt"

    if os.path.exists(data_file) and os.path.exists(label_file):
        logd(f"[SERVER] Cached tensors for {ds_name} found. Skipping preprocess.")
        X = torch.load(data_file)
        y = torch.load(label_file)
        return X, y

    logd(f"[SERVER] Preprocessing dataset {ds_name} ONCE...")

    if ds_name == "CIFAR10":
        T = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        ds = datasets.CIFAR10("./data", train=True, download=True, transform=T)
        X = torch.stack([ds[i][0] for i in range(len(ds))])
        y = torch.tensor(ds.targets)

    elif ds_name == "CIFAR100":
        T = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        ds = datasets.CIFAR100("./data", train=True, download=True, transform=T)
        X = torch.stack([ds[i][0] for i in range(len(ds))])
        y = torch.tensor(ds.targets)

    else:  # SVHN
        from torchvision.datasets import SVHN
        T = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        ds = SVHN("./data", split="train", download=True, transform=T)
        X = torch.stack([ds[i][0] for i in range(len(ds))])
        y = torch.tensor(ds.labels)

    torch.save(X, data_file)
    torch.save(y, label_file)

    logd(f"[SERVER] Preprocess {ds_name} completed.")
    return X, y


# ======================================================================
# EVAL
# ======================================================================

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


# ======================================================================
# PERSISTENT WORKER (1 PER GPU)
# ======================================================================

def persistent_worker(gpu: int, dataset_name: str):
    """
    Worker persistente: gira finché non riceve "STOP",
    leggendo i job da jobs/worker_{gpu}.queue
    """
    device = f"cuda:{gpu}"
    logd(f"[WORKER GPU {gpu}] Started. Device={device}, dataset={dataset_name}")

    # Load cached dataset ONCE
    X = torch.load(f"cached/{dataset_name}_train_data.pt")
    y = torch.load(f"cached/{dataset_name}_train_labels.pt")

    queue_file = f"jobs/worker_{gpu}.queue"

    while True:
        # Wait for job file
        while not os.path.exists(queue_file):
            time.sleep(0.1)

        with open(queue_file, "r") as f:
            job = f.read().strip()

        # Clear queue file ASAP
        os.remove(queue_file)

        if job == "STOP":
            logd(f"[WORKER GPU {gpu}] STOP received. Exiting.")
            return

        # job format:
        # cid,nc,global_ckpt,state_c_path,idx_path,lr,out_path
        cid_str, nc_str, gckpt, statec, idx_path, lr_str, out_path = job.split(",")
        cid = int(cid_str)
        nc = int(nc_str)
        lr = float(lr_str)

        logd(f"[WORKER GPU {gpu}] Starting job cid={cid}, lr={lr}")

        # Load model
        model = ResNet18Pre(nc).to(device)
        model.load_state_dict(torch.load(gckpt, map_location="cpu"))
        trainable = [p for p in model.parameters() if p.requires_grad]

        # Load control variates
        s = torch.load(statec, map_location="cpu")
        c_global = s["c_global"]           # list of tensors
        c_local = s["c_local"][cid]        # list of tensors (this client)

        # Load indices & build client dataset
        idx = torch.load(idx_path)
        Xc = X[idx]
        yc = y[idx]

        loader = torch.utils.data.DataLoader(
            list(zip(Xc, yc)),
            batch_size=BATCH,
            shuffle=True
        )

        old_params = [p.detach().clone().cpu() for p in trainable]
        opt = optim.SGD(trainable, lr=lr, momentum=0.9, weight_decay=5e-4)

        E = len(loader)
        for _ in range(LOCAL_EPOCHS):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                out = model(xb)
                loss = nn.CrossEntropyLoss()(out, yb)
                loss.backward()
                # SCAFFOLD correction
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

        torch.save({
            "cid": cid,
            "new_params": new_params,
            "new_c_local": c_local,
            "delta_c": delta_c,
        }, out_path)

        logd(f"[WORKER GPU {gpu}] Finished job cid={cid}")


# ======================================================================
# FEDERATED SERVER LOOP
# ======================================================================

def federated_run(ds_name: str, gpus: int):
    print(f"\n========== DATASET: {ds_name} ==========\n", flush=True)

    # Preprocess train set once
    _, y = preprocess_dataset(ds_name)

    # Load test set for evaluation
    if ds_name == "CIFAR10":
        nc = 10
        Tte = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
        te = datasets.CIFAR10("./data", train=False, download=True, transform=Tte)
    elif ds_name == "CIFAR100":
        nc = 100
        Tte = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
        te = datasets.CIFAR100("./data", train=False, download=True, transform=Tte)
    else:
        from torchvision.datasets import SVHN
        nc = 10
        Tte = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
        te = SVHN("./data", split="test", download=True, transform=Tte)

    testloader = DataLoader(te, batch_size=256, shuffle=False)

    # Prepare directories
    os.makedirs("global_ckpt", exist_ok=True)
    os.makedirs("client_updates", exist_ok=True)
    os.makedirs("jobs", exist_ok=True)

    # Clean jobs directory
    for f in os.listdir("jobs"):
        os.remove(os.path.join("jobs", f))

    # Spawn persistent workers for this dataset
    workers = []
    for gpu in range(gpus):
        cmd = [
            sys.executable,
            "main.py",
            "--persistent-worker",
            "--gpu", str(gpu),
            "--dataset", ds_name,
        ]
        p = subprocess.Popen(cmd)
        workers.append(p)

    labels_np = y.numpy()

    for alpha in DIR_ALPHAS:

        print(f"\n==== α = {alpha} ====\n", flush=True)

        splits = dirichlet_split(labels_np, NUM_CLIENTS, alpha)

        # Initialize global model for this alpha
        device0 = "cuda:0"
        global_model = ResNet18Pre(nc).to(device0)
        trainable = [p for p in global_model.parameters() if p.requires_grad]

        # Init control variates
        c_global = [torch.zeros_like(p).cpu() for p in trainable]
        c_local = [
            [torch.zeros_like(p).cpu() for p in trainable]
            for _ in range(NUM_CLIENTS)
        ]

        # Save initial global model
        ckpt_path_prev = f"global_ckpt/{ds_name}_alpha{alpha}_round0.pth"
        torch.save(global_model.state_dict(), ckpt_path_prev)

        # ROUNDS
        for rnd in range(1, NUM_ROUNDS + 1):

            lr = LR0 if rnd <= LR_DECAY else LR0 * 0.1
            print(f"--- ROUND {rnd}/{NUM_ROUNDS} ---", flush=True)

            # Save control variates for this round
            state_c_path = f"global_ckpt/{ds_name}_alpha{alpha}_state_c_r{rnd}.pth"
            torch.save({"c_local": c_local, "c_global": c_global}, state_c_path)

            # Save train indices (one per client, static over rounds)
            idx_paths = []
            for cid in range(NUM_CLIENTS):
                idx_path = f"global_ckpt/{ds_name}_alpha{alpha}_cid{cid}_idx.pth"
                if not os.path.exists(idx_path):
                    torch.save(splits[cid], idx_path)
                idx_paths.append(idx_path)

            # Use last global checkpoint as base
            ckpt_path = ckpt_path_prev

            # Prepare output paths for this round
            out_paths = {
                cid: f"client_updates/{ds_name}_alpha{alpha}_cid{cid}_r{rnd}.pth"
                for cid in range(NUM_CLIENTS)
            }

            # ---- JOB SCHEDULING over persistent workers ----

            pending = list(range(NUM_CLIENTS))
            completed = set()

            while len(completed) < NUM_CLIENTS:
                # Assign jobs to free workers
                for gpu in range(gpus):
                    qfile = f"jobs/worker_{gpu}.queue"
                    if not pending:
                        break
                    if not os.path.exists(qfile):
                        cid = pending.pop(0)
                        job_line = ",".join([
                            str(cid),
                            str(nc),
                            ckpt_path,
                            state_c_path,
                            idx_paths[cid],
                            str(lr),
                            out_paths[cid],
                        ])
                        with open(qfile, "w") as f:
                            f.write(job_line)

                # Check which jobs finished
                for cid in range(NUM_CLIENTS):
                    if cid in completed:
                        continue
                    if os.path.exists(out_paths[cid]):
                        completed.add(cid)

                if len(completed) < NUM_CLIENTS:
                    time.sleep(0.2)

            # ---- All client updates are ready ----

            updates = [
                torch.load(out_paths[cid], map_location="cpu")
                for cid in range(NUM_CLIENTS)
            ]

            # Aggregate params
            new_params_acc = None
            for upd in updates:
                if new_params_acc is None:
                    new_params_acc = [
                        torch.zeros_like(p) for p in upd["new_params"]
                    ]
                for i, p in enumerate(upd["new_params"]):
                    new_params_acc[i] += p

            avg_params = [p / NUM_CLIENTS for p in new_params_acc]

            # Load into global model
            idx_param = 0
            with torch.no_grad():
                for p in global_model.parameters():
                    if p.requires_grad:
                        p.copy_(avg_params[idx_param].to(device0))
                        idx_param += 1

            # Update c_global
            for i in range(len(c_global)):
                c_global[i] = sum(
                    upd["delta_c"][i] for upd in updates
                ) / NUM_CLIENTS

            # Update c_local
            for upd in updates:
                cid = upd["cid"]
                c_local[cid] = upd["new_c_local"]

            # Save new global model
            ckpt_path_prev = f"global_ckpt/{ds_name}_alpha{alpha}_round{rnd}.pth"
            torch.save(global_model.state_dict(), ckpt_path_prev)

            # Eval
            acc = evaluate(global_model, testloader, device0)
            loga(f"[ROUND {rnd}] ACC = {acc*100:.2f}%")

    # After all α, send STOP to workers
    for gpu in range(gpus):
        qfile = f"jobs/worker_{gpu}.queue"
        with open(qfile, "w") as f:
            f.write("STOP")

    for p in workers:
        p.wait()

    logd(f"[SERVER] Finished dataset {ds_name}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persistent-worker", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--gpus", type=int, default=1)

    args = parser.parse_args()

    # Worker mode
    if args.persistent_worker:
        if args.dataset is None:
            raise ValueError("Worker mode requires --dataset")
        set_seed(SEED)
        persistent_worker(args.gpu, args.dataset)
        return

    # Server mode
    set_seed(SEED)

    for ds in ["CIFAR10", "CIFAR100", "SVHN"]:
        federated_run(ds, args.gpus)


if __name__ == "__main__":
    main()




