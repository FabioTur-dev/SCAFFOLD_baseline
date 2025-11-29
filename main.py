#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


# ======================================================================
# LOGGING (DEBUG→stderr, ACCURACY→stdout)
# ======================================================================

DEBUG = True

def logd(msg):
    if DEBUG:
        print(msg, file=sys.stderr, flush=True)

def loga(msg):
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
# SEED
# ======================================================================

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


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
    for c in per:
        random.shuffle(c)
    return per


# ======================================================================
# MODEL DEF
# ======================================================================

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
        for p in self.m.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.m(x)


# ======================================================================
# DATASET
# ======================================================================

def load_dataset(name):
    if name == "CIFAR10":
        nc = 10
        Ttr = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=16),
            transforms.ToTensor(),
        ])
        Tte = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        tr = datasets.CIFAR10("./data", train=True, download=True, transform=Ttr)
        te = datasets.CIFAR10("./data", train=False, download=True, transform=Tte)
        labels = tr.targets

    elif name == "CIFAR100":
        nc = 100
        Ttr = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=16),
            transforms.ToTensor(),
        ])
        Tte = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        tr = datasets.CIFAR100("./data", train=True, download=True, transform=Ttr)
        te = datasets.CIFAR100("./data", train=False, download=True, transform=Tte)
        labels = tr.targets

    else:  # SVHN
        nc = 10
        Ttr = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        Tte = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        tr = datasets.SVHN("./data", split="train", download=True, transform=Ttr)
        te = datasets.SVHN("./data", split="test", download=True, transform=Tte)
        labels = tr.labels

    return tr, te, labels, nc


# ======================================================================
# CLIENT UPDATE (called ONLY when --worker)
# ======================================================================

def client_update_worker(args):
    """
    This is executed when main.py is called with --worker.
    """

    device = f"cuda:{args.gpu}"
    logd(f"[WORKER] Running client {args.cid} on {device}")

    # Load dataset
    tr, _, labels, nc = load_dataset(args.dataset)

    # Load model
    model = ResNet18Pre(nc).to(device)

    # Load global weights
    global_sd = torch.load(args.global_ckpt, map_location="cpu")
    model.load_state_dict(global_sd)

    trainable = [p for p in model.parameters() if p.requires_grad]

    # Load c_local, c_global
    state = torch.load(args.state_c, map_location="cpu")
    c_local = state["c_local"]
    c_global = state["c_global"]

    # Indices of this client
    train_idx = torch.load(args.train_idx)

    # Client update
    old_params = [p.detach().clone().cpu() for p in trainable]

    loader = torch.utils.data.DataLoader(
        Subset(tr, train_idx),
        batch_size=BATCH,
        shuffle=True
    )

    lr = args.lr
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

    # Save results
    torch.save({
        "cid": args.cid,
        "new_params": new_params,
        "new_c_local": c_local,
        "delta_c": delta_c,
    }, args.output)

    logd(f"[WORKER] Client {args.cid} finished")
    return


# ======================================================================
# MAIN FEDERATED LOOP
# ======================================================================

def federated_run(dataset_name, gpus):

    print(f"\n========== DATASET: {dataset_name} ==========\n", flush=True)

    tr, te, labels, nc = load_dataset(dataset_name)
    testloader = DataLoader(te, batch_size=256, shuffle=False)

    splits = {}

    for alpha in DIR_ALPHAS:

        print(f"\n==== α = {alpha} ====\n", flush=True)

        splits = dirichlet_split(labels, NUM_CLIENTS, alpha)

        # Prepare directories
        os.makedirs("global_ckpt", exist_ok=True)
        os.makedirs("client_updates", exist_ok=True)

        # Initialize global model
        device0 = "cuda:0"
        global_model = ResNet18Pre(nc).to(device0)

        ckpt_path = "global_ckpt/global_round_0.pth"
        torch.save(global_model.state_dict(), ckpt_path)

        # Scaffold buffers
        trainable = [p for p in global_model.parameters() if p.requires_grad]
        c_global = [torch.zeros_like(p).cpu() for p in trainable]
        c_local = [[torch.zeros_like(p).cpu() for p in trainable]
                   for _ in range(NUM_CLIENTS)]

        # Round loop
        for rnd in range(1, NUM_ROUNDS + 1):

            lr = LR0 if rnd <= LR_DECAY else LR0 * 0.1
            print(f"--- ROUND {rnd}/{NUM_ROUNDS} ---", flush=True)

            # Broadcast c_local / c_global for all clients
            state_c_path = "global_ckpt/state_c.pth"
            torch.save({"c_local": c_local, "c_global": c_global}, state_c_path)

            # Broadcast train indices
            idx_paths = []
            for cid in range(NUM_CLIENTS):
                p = f"global_ckpt/train_idx_{cid}.pth"
                torch.save(splits[cid], p)
                idx_paths.append(p)

            # Re-save global model (fresh)
            ckpt_path = f"global_ckpt/global_round_{rnd-1}.pth"
            torch.save(global_model.state_dict(), ckpt_path)

            # ---------------------------------------------
            # Launch clients as subprocesses
            # ---------------------------------------------
            procs = []
            out_paths = []
            for cid in range(NUM_CLIENTS):
                gpu = cid % gpus
                outp = f"client_updates/cid_{cid}_r{rnd}.pth"
                out_paths.append(outp)

                cmd = [
                    sys.executable,
                    "main.py",
                    "--worker",
                    "--dataset", dataset_name,
                    "--cid", str(cid),
                    "--gpu", str(gpu),
                    "--global_ckpt", ckpt_path,
                    "--state_c", state_c_path,
                    "--train_idx", idx_paths[cid],
                    "--lr", str(lr),
                    "--output", outp
                ]

                procs.append(subprocess.Popen(cmd))

            # Wait for all
            for p in procs:
                p.wait()

            # ---------------------------------------------
            # Aggregate
            # ---------------------------------------------
            updates = []
            for cid in range(NUM_CLIENTS):
                upd = torch.load(out_paths[cid], map_location="cpu")
                updates.append(upd)

            # Aggregate parameters
            trainable = [p for p in global_model.parameters() if p.requires_grad]

            new_params_acc = None
            for upd in updates:
                if new_params_acc is None:
                    new_params_acc = [
                        torch.zeros_like(p) for p in upd["new_params"]
                    ]
                for i, p in enumerate(upd["new_params"]):
                    new_params_acc[i] += p

            avg_params = [x / NUM_CLIENTS for x in new_params_acc]

            # Load averaged weights
            idx = 0
            with torch.no_grad():
                for p in global_model.parameters():
                    if p.requires_grad:
                        p.copy_(avg_params[idx].to(device0))
                        idx += 1

            # Update c_global
            for i in range(len(c_global)):
                c_global[i] = torch.zeros_like(c_global[i])
                for upd in updates:
                    c_global[i] += upd["delta_c"][i] / NUM_CLIENTS

            # Update c_local
            for upd in updates:
                cid = upd["cid"]
                c_local[cid] = upd["new_c_local"]

            # Save new global model
            ckpt_path = f"global_ckpt/global_round_{rnd}.pth"
            torch.save(global_model.state_dict(), ckpt_path)

            # Eval
            acc = evaluate(global_model, testloader, device0)
            loga(f"[ROUND {rnd}] ACC = {acc*100:.2f}%")

    return


# ======================================================================
# EVAL
# ======================================================================

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


# ======================================================================
# MAIN ENTRY
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--cid", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--global_ckpt", type=str)
    parser.add_argument("--state_c", type=str)
    parser.add_argument("--train_idx", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--gpus", type=int, default=1)

    args = parser.parse_args()

    # WORKER MODE
    if args.worker:
        return client_update_worker(args)

    # MAIN (SERVER) MODE
    set_seed(SEED)

    for ds in ["CIFAR10", "CIFAR100", "SVHN"]:
        federated_run(ds, args.gpus)


if __name__ == "__main__":
    main()



