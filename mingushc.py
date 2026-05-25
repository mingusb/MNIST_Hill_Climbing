from math import log
import pickle
import time
import os

from torch import Tensor, argmax, zeros, rand, eq, sum

import torch
torch.set_printoptions(sci_mode=False, precision=30)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from matplotlib.ticker import MaxNLocator
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', cache=True)
X = Tensor(mnist.data.values).to_sparse()
y = Tensor(mnist.target.astype("int64"))

n_trials = 10000

W        = zeros(size=(784, 10))
U        = rand(size=(784, 10))
results  = Tensor(3000, 2).long()
attempts = Tensor([0]).long()
trial    = Tensor([0]).long()
best     = Tensor([0]).long()
correct  = Tensor([0]).long()
i        = Tensor([1]).long()

start = time.time()

for i in range(n_trials):
    correct = sum(eq(argmax(X @ (W + U.uniform_()), 1), y))

    if correct > best:
        W += U

        best = correct

        results[trial,0] = attempts
        results[trial,1] = best

        trial += 1

        attempts = i

        print(attempts, trial, best, best / 70000, time.time() - start)

with open("results.pickle", "wb") as f:
    pickle.dump(results, f)

with open("w.pickle", "wb") as f:
    pickle.dump(W, f)

trials, attempts, accuracies = [], [], []

for t in range(trial):

    attempt = results[t,0].item()
    accuracy = results[t,1].item() / 70000

    if attempt == 0:
        attempts.append(0)
    else:
        attempts.append(log(attempt))

    trials.append(t)

    accuracies.append(accuracy)

fig, ax1 = plt.subplots()
ax2 = ax1.twiny()

ax1.tick_params(labelright=True, labelleft=True)
ax1.yaxis.tick_right()
ax1.yaxis.set_ticks_position('both')

ax1.yaxis.set_major_locator(MaxNLocator(10))
ax1.xaxis.set_major_locator(MaxNLocator(10))

plt.title("Training performance on MNIST (70000 instances)")

plt.ylim(0,1.0)

ax1.plot(attempts, accuracies, color='blue')
ax1.set_ylabel(f"Accuracy (best = {str(best.item()/70000)[:5]})")
ax1.set_xlabel(f"log(log(attempt on which learning occurred)) (total = {attempts.item()}, log({attempts.item()}) = {str(log(attempts.item()))[:5]})", color="blue")

ax2.plot(trials, accuracies, color="green")
ax2.set_xlabel(f"trial (number of times learning occurred) (total = {trial.item()})", color="green")

plt.grid(axis='both', which='both')
ax1.grid(axis='both', which='both')

fig.savefig("hc.svg")

plt.show()
