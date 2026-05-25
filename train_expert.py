# train_expert.py
import torch, os, argparse, random
from mingus_hc_resnext import MingusHCExpert, HCConfig, load_mnist, set_seed, accuracy

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None,
                   help="Optional random seed. If not set, a random seed is chosen.")
    p.add_argument("--save_dir", type=str, default="experts")
    p.add_argument("--max_trials", type=int, default=200000)
    p.add_argument("--patience", type=int, default=10000,
                   help="Stop if no improvement after this many trials")
    args = p.parse_args()

    # Pick random seed if none provided
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
    print(f"Using seed {args.seed}")
    set_seed(args.seed)

    device = torch.device("cuda")

    # Load full train/val set
    (X, y), _ = load_mnist(device)
    N = len(X)
    val_size = int(0.1 * N)
    perm = torch.randperm(N, device=device)
    Xtr, ytr = X[perm[val_size:]], y[perm[val_size:]]
    Xval, yval = X[perm[:val_size]], y[perm[:val_size]]

    # Build expert
    n_features, n_classes = X.size(1), int(y.max().item() + 1)
    expert = MingusHCExpert(n_features, n_classes, device)

    # Train with convergence criteria
    best_val, best_W = 0.0, None
    no_improve = 0
    for t in range(1, args.max_trials + 1):
        # one hill-climb step
        U = torch.rand_like(expert.W).uniform_()
        correct = int((((Xtr @ (expert.W + U)).argmax(1)) == ytr).sum().item())
        if correct > int(((Xtr @ expert.W).argmax(1) == ytr).sum().item()):
            expert.W.add_(U)

        # check val accuracy every 1000 steps
        if t % 1000 == 0:
            val_acc = accuracy(expert(Xval), yval)
            if val_acc > best_val:
                best_val = val_acc
                best_W = expert.W.clone()
                no_improve = 0
            else:
                no_improve += 1000
            print(f"Trial {t}: val_acc={val_acc:.4f} best={best_val:.4f}")
            if no_improve >= args.patience:
                print("Converged: stopping early")
                break

    # restore best weights
    if best_W is not None:
        expert.W.copy_(best_W)

    # Save
    os.makedirs(args.save_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(args.save_dir) if f.startswith("expert_")])
    next_id = len(files) + 1
    save_path = os.path.join(
        args.save_dir,
        f"expert_{next_id}_val{best_val:.4f}_seed{args.seed}.pt"
    )
    torch.save(expert.state_dict(), save_path)
    print(f"Saved expert to {save_path}")

if __name__ == "__main__":
    main()
