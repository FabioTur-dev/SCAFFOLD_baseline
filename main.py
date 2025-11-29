#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models



# ======================================================================
# LOGGING
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
# SET SEED
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

    for cl in per:
        random.shuffle(cl)

    return per



# ======================================================================
# MODEL
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
# WORKER MODE (FAST, NO HUGE CACHE)
# ======================================================================

def client_update_worker(args):

    device = f"cuda:{args.gpu}"
    logd(f"[WORKER {args.cid}] Starting on {device}")

    # --- Dataset (no caching) ---
    T = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    if args.dataset == "CIFAR10":
        ds = datasets.CIFAR10("./data", train=True, download=True, transform=T)
    elif args.dataset == "CIFAR100":
        ds = datasets.CIFAR100("./data", train=True, download=True, transform=T)
    else:
        from torchvision.datasets import SVHN
        ds = SVHN("./data", split="train", download=True, transform=T)

    idx = torch.load(args.train_idx)
    subset = Subset(ds, idx)

    loader = DataLoader(
        subset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # --- Model ---
    nc = args.num_classes
    model = ResNet18Pre(nc).to(device)

    state_dict = torch.load(args.global_ckpt, map_location="cpu")
    model.load_state_dict(state_dict)

    trainable = [p for p in model.parameters() if p.requires_grad]

    # Control variates
    c_state = torch.load(args.state_c, map_location="cpu")
    c_global = c_state["c_global"]
    c_local = c_state["c_local"][args.cid]

    # Training
    old_params = [p.detach().clone().cpu() for p in trainable]

    lr = args.lr
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

    # Save updates
    new_params = [p.detach().clone().cpu() for p in trainable]

    delta_c = []
    for i in range(len(new_params)):
        diff = new_params[i] - old_params[i]
        dc = BETA * (diff / max(E, 1))
        delta_c.append(dc)
        c_local[i] += dc

    torch.save({
        "cid": args.cid,
        "new_params": new_params,
        "new_c_local": c_local,
        "delta_c": delta_c,
    }, args.output)

    logd(f"[WORKER {args.cid}] Done")
    return



# ======================================================================
# EVALUATE
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
# FEDERATED LOOP
# ======================================================================

def federated_run(ds_name, gpus):

    print(f"\n========== DATASET: {ds_name} ==========\n", flush=True)

    # -----------------------------
    # TEST SET (unchanged)
    # -----------------------------
    if ds_name == "CIFAR10":
        nc = 10
        T = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
        te = datasets.CIFAR10("./data", train=False, download=True, transform=T)

    elif ds_name == "CIFAR100":
        nc = 100
        T = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
        te = datasets.CIFAR100("./data", train=False, download=True, transform=T)

    else:
        nc = 10
        from torchvision.datasets import SVHN
        T = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
        te = SVHN("./data", split="test", download=True, transform=T)

    testloader = DataLoader(te, batch_size=256, shuffle=False)



    # -----------------------------------------
    # Load training labels once (small, fast)
    # -----------------------------------------
    if ds_name == "CIFAR10":
        tmp = datasets.CIFAR10("./data", train=True, download=True)
        labels = np.array(tmp.targets)

    elif ds_name == "CIFAR100":
        tmp = datasets.CIFAR100("./data", train=True, download=True)
        labels = np.array(tmp.targets)

    else:
        from torchvision.datasets import SVHN
        tmp = SVHN("./data", split="train", download=True)
        labels = np.array(tmp.labels)



    for alpha in DIR_ALPHAS:

        print(f"\n==== α = {alpha} ====\n", flush=True)

        splits = dirichlet_split(labels, NUM_CLIENTS, alpha)

        # GLOBAL MODEL
        device0 = "cuda:0"
        global_model = ResNet18Pre(nc).to(device0)
        trainable = [p for p in global_model.parameters() if p.requires_grad]

        # Initial save
        os.makedirs("global_ckpt", exist_ok=True)
        ckpt_path = "global_ckpt/global_round_0.pth"
        torch.save(global_model.state_dict(), ckpt_path)

        # Control variates
        c_global = [torch.zeros_like(p).cpu() for p in trainable]
        c_local = [[torch.zeros_like(p).cpu() for p in trainable]
                   for _ in range(NUM_CLIENTS)]

        # --------------------------
        # ROUNDS
        # --------------------------
        for rnd in range(1, NUM_ROUNDS + 1):

            lr = LR0 if rnd <= LR_DECAY else LR0 * 0.1
            print(f"--- ROUND {rnd}/{NUM_ROUNDS} ---", flush=True)

            # Save control variates
            state_c_path = "global_ckpt/state_c.pth"
            torch.save({"c_local": c_local, "c_global": c_global}, state_c_path)

            # Save train index files
            idx_paths = []
            for cid in range(NUM_CLIENTS):
                p = f"global_ckpt/train_idx_{cid}.pth"
                torch.save(splits[cid], p)
                idx_paths.append(p)

            # Save global model
            ckpt_path = f"global_ckpt/global_round_{rnd-1}.pth"
            torch.save(global_model.state_dict(), ckpt_path)

            # ---------------------------
            # Launch worker subprocesses
            # ---------------------------
            os.makedirs("client_updates", exist_ok=True)
            procs = []
            out_paths = []

            for cid in range(NUM_CLIENTS):
                gpu = cid % gpus
                outp = f"client_updates/cid_{cid}_r{rnd}.pth"
                out_paths.append(outp)

                cmd = [
                    sys.executable, "main.py",
                    "--worker",
                    "--dataset", ds_name,
                    "--num_classes", str(nc),
                    "--cid", str(cid),
                    "--gpu", str(gpu),
                    "--global_ckpt", ckpt_path,
                    "--state_c", state_c_path,
                    "--train_idx", idx_paths[cid],
                    "--lr", str(lr),
                    "--output", outp,
                    "--gpus", str(gpus)
                ]

                procs.append(subprocess.Popen(cmd))

            for p in procs:
                p.wait()

            # ---------------------------
            # Aggregate
            # ---------------------------
            updates = [torch.load(out_paths[c], map_location="cpu")
                       for c in range(NUM_CLIENTS)]

            new_acc = None
            for u in updates:
                if new_acc is None:
                    new_acc = [torch.zeros_like(p) for p in u["new_params"]]
                for i, p in enumerate(u["new_params"]):
                    new_acc[i] += p

            avg_params = [p / NUM_CLIENTS for p in new_acc]

            idx = 0
            with torch.no_grad():
                for p in global_model.parameters():
                    if p.requires_grad:
                        p.copy_(avg_params[idx].to(device0))
                        idx += 1

            # Update control variates
            for i in range(len(c_global)):
                c_global[i] = sum(u["delta_c"][i] for u in updates) / NUM_CLIENTS

            for u in updates:
                c_local[u["cid"]] = u["new_c_local"]

            # Save new model
            ckpt_path = f"global_ckpt/global_round_{rnd}.pth"
            torch.save(global_model.state_dict(), ckpt_path)

            # Eval
            acc = evaluate(global_model, testloader, device0)
            loga(f"[ROUND {rnd}] ACC = {acc*100:.2f}%")


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--cid", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--global_ckpt", type=str)
    parser.add_argument("--state_c", type=str)
    parser.add_argument("--train_idx", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--gpus", type=int, default=1)

    args = parser.parse_args()

    if args.worker:
        return client_update_worker(args)

    set_seed(SEED)

    for ds in ["CIFAR10", "CIFAR100", "SVHN"]:
        federated_run(ds, args.gpus)


if __name__ == "__main__":
    main()




