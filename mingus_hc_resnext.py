"""
MingusHC + Adam Gating (MNIST Mixture-of-Experts)
- ResNeXt CNN gating network (sees MNIST as images)
- All data lives on GPU at all times, except final plotting
- Uses pre-trained MingusHC experts saved on disk
"""
from __future__ import annotations
import argparse, itertools, random, math, os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


# ------------------ utils ------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


@dataclass
class HCConfig:
    n_trials: int = 100000
    log_every: int = 500


# ------------------ expert ------------------
class MingusHCExpert(nn.Module):
    def __init__(self, n_features: int, n_classes: int, device: torch.device):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(n_features, n_classes, device=device), requires_grad=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.W


def load_experts(save_dir: str, device: torch.device, n_features: int, n_classes: int):
    experts = []
    if not os.path.exists(save_dir):
        raise RuntimeError(f"No experts found in {save_dir}")
    files = sorted([f for f in os.listdir(save_dir) if f.startswith("expert_")])
    for f in files:
        exp = MingusHCExpert(n_features, n_classes, device)
        exp.load_state_dict(torch.load(os.path.join(save_dir, f), map_location=device))
        experts.append(exp)
    print(f"Loaded {len(experts)} experts from {save_dir}")
    return experts


# ------------------ ResNeXt CNN gate ------------------
class ResNeXtBottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, groups=8, width_per_group=16, downsample=None):
        super().__init__()
        width = int(planes * (width_per_group / 16.0))
        width = max(width, width_per_group)
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample; self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNeXtGating(nn.Module):
    def __init__(self, n_experts: int, layers=(2,2,2), groups=8, width_per_group=16):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(ResNeXtBottleneck, 64, layers[0], stride=1,
                                       groups=groups, width_per_group=width_per_group)
        self.layer2 = self._make_layer(ResNeXtBottleneck, 128, layers[1], stride=2,
                                       groups=groups, width_per_group=width_per_group)
        self.layer3 = self._make_layer(ResNeXtBottleneck, 256, layers[2], stride=2,
                                       groups=groups, width_per_group=width_per_group)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * ResNeXtBottleneck.expansion, n_experts)

    def _make_layer(self, block, planes, blocks, stride, groups, width_per_group):
        downsample = None
        outplanes = planes * block.expansion
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes))
        layers = [block(self.inplanes, planes, stride, groups, width_per_group, downsample)]
        self.inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups, width_per_group=width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:  # flat -> image
            x = x.view(-1,1,28,28)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        return self.fc(x)


# ------------------ data ------------------
def load_mnist(device):
    mnist = fetch_openml('mnist_784', cache=True)
    X = torch.tensor(mnist.data.values, dtype=torch.float32, device=device) / 255.0
    y = torch.tensor(mnist.target.astype('int64').values, dtype=torch.long, device=device)
    return (X[:60000], y[:60000]), (X[60000:], y[60000:])


# ------------------ mixture ------------------
def mixture_logits(expert_logits, gate_logits):
    gate_probs = F.softmax(gate_logits, dim=1)
    probs_stack = torch.stack([F.softmax(L, dim=1) for L in expert_logits], dim=0)
    return (gate_probs.t().unsqueeze(-1) * probs_stack).sum(dim=0)

def train_gating(experts, Xtr, ytr, Xval, yval, device, *, epochs, batch_size, lr, gate_cfg):
    gate = ResNeXtGating(**gate_cfg).to(device)
    opt = torch.optim.Adam(gate.parameters(), lr=lr)
    def run_epoch(X,y,train):
        gate.train(train)
        idx = torch.randperm(len(X), device=device)
        total_acc = 0
        for st in range(0,len(X),batch_size):
            sel = idx[st:st+batch_size]; xb,yb=X[sel],y[sel]
            with torch.no_grad(): exp_logits=[e(xb) for e in experts]
            gate_logits = gate(xb)
            mix_probs = mixture_logits(exp_logits, gate_logits)
            loss = F.nll_loss(torch.log(mix_probs+1e-12), yb)
            if train: opt.zero_grad(); loss.backward(); opt.step()
            total_acc += accuracy(mix_probs,yb)*len(yb)
        return total_acc/len(X)
    best_val, best_state=0,None
    for ep in range(epochs):
        _=run_epoch(Xtr,ytr,True); v=run_epoch(Xval,yval,False)
        if v>best_val: best_val=v; best_state={k:v.detach().clone() for k,v in gate.state_dict().items()}
    if best_state: gate.load_state_dict(best_state)
    return gate,best_val

def evaluate(experts, gate, X,y):
    exp_logits=[e(X) for e in experts]; gate_logits=gate(X)
    mix_probs=mixture_logits(exp_logits,gate_logits)
    return accuracy(mix_probs,y)


# ------------------ search ------------------
def grid_search(X,y,device):
    param_grid={
        'epochs':[10],
        'batch_size':[512],
        'lr':[1e-3],
        'gate_cfg':[{'n_experts':0,'layers':(2,2,1),'groups':4,'width_per_group':8}]
    }
    best_val=0; best_cfg=None
    N=len(X); val_size=int(0.1*N)
    perm=torch.randperm(N,device=device)
    Xtr,ytr=X[perm[val_size:]],y[perm[val_size:]]
    Xval,yval=X[perm[:val_size]],y[perm[:val_size]]
    for cfg in [dict(zip(param_grid,vals)) for vals in itertools.product(*param_grid.values())]:
        n_features, n_classes = Xtr.size(1), int(ytr.max().item() + 1)
        experts = load_experts("experts", device, n_features, n_classes)
        cfg['gate_cfg']['n_experts'] = len(experts)
        gate,val = train_gating(
            experts, Xtr,ytr,Xval,yval,device,
            epochs=cfg['epochs'], batch_size=cfg['batch_size'], lr=cfg['lr'],
            gate_cfg=cfg['gate_cfg']
        )
        print("Val acc",val)
        if val>best_val: best_val=val; best_cfg=cfg
    print("Best",best_cfg,"val=",best_val)
    return best_cfg

def cross_validate(cfg,X,y,device,folds=5):
    N=len(X); idx=torch.randperm(N,device=device); fs=N//folds; accs=[]
    for f in range(folds):
        val_idx=idx[f*fs:(f+1)*fs]; train_idx=torch.cat([idx[:f*fs],idx[(f+1)*fs:]])
        Xtr,ytr=X[train_idx],y[train_idx]; Xval,yval=X[val_idx],y[val_idx]
        n_features, n_classes = Xtr.size(1), int(ytr.max().item() + 1)
        experts = load_experts("experts", device, n_features, n_classes)
        cfg['gate_cfg']['n_experts'] = len(experts)
        gate,val=train_gating(experts,Xtr,ytr,Xval,yval,device,
                              epochs=cfg['epochs'],batch_size=cfg['batch_size'],
                              lr=cfg['lr'],gate_cfg=cfg['gate_cfg'])
        accs.append(val); print(f"Fold {f+1}: {val:.4f}")
    return float(np.mean(accs)),float(np.std(accs))


# ------------------ main ------------------
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    set_seed(args.seed)
    device=torch.device('cuda')
    (Xtrval,ytrval),(Xte,yte)=load_mnist(device)
    best_cfg=grid_search(Xtrval,ytrval,device)
    mean,std=cross_validate(best_cfg,Xtrval,ytrval,device)
    print(f"CV mean={mean:.4f} std={std:.4f}")
    n_features, n_classes = Xtrval.size(1), int(ytrval.max().item() + 1)
    experts = load_experts("experts", device, n_features, n_classes)
    best_cfg['gate_cfg']['n_experts'] = len(experts)
    gate,_=train_gating(experts,Xtrval,ytrval,Xtrval,ytrval,device,
                        epochs=best_cfg['epochs'],batch_size=best_cfg['batch_size'],
                        lr=best_cfg['lr'],gate_cfg=best_cfg['gate_cfg'])
    test_acc=evaluate(experts,gate,Xte,yte); print("Test acc",test_acc)
    # Plot final results (CPU only here)
    accs=[mean,std,test_acc]
    labs=['CV mean','CV std','Test']
    plt.bar(labs,accs); plt.title("Performance"); plt.ylabel("Accuracy")
    plt.savefig("results.png")

if __name__=='__main__': main()
