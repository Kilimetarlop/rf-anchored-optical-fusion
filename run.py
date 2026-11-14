"""One cross-session run of RF-anchored optical feature fusion (paper Algorithm 1): enrolment on one session, naming and decisions on
the other.

  python run.py --windows data/windows.csv \
      --tokens data/rf_tokens.npz,data/appearance_tokens.npz,data/trajectory_tokens.npz --train D00 --test D01 --seed 0 \
      [--audit bg=data/appearance_bg.npz,tp0=data/appearance_tp0.npz,tp1=...,tp2=...] [--out run.json]

Prints the RF anchor, the naming rounds and the fused decision, and writes the readouts as JSON. Deployment labels are read only to
score these readouts.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from audit import follow_target, swap_tokens
from data import TOKENS, TokenData, split
from enrol import enrol
from model import TokenFusion
from naming import decide, name_in_session

try:
    import torch_musa  # noqa: F401  (MTT S4000)
    DEV = "musa"
except ImportError:
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)


def per_record(x, rec):
    """Value at the first window of every record (record-level arrays in record order)."""
    _, first = np.unique(rec.cpu().numpy(), return_index=True)
    return x[torch.as_tensor(first, device=x.device)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True, help="windows.csv of the crop package")
    ap.add_argument("--tokens", required=True, help="comma-separated token files (.npz)")
    ap.add_argument("--names", default=",".join(TOKENS), help="token sources to use")
    ap.add_argument("--train", default="D00"); ap.add_argument("--test", default="D01"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit", default="", help="tag=path,... optical tokens swapped in after naming (bg, tp0, tp1, tp2)")
    ap.add_argument("--out", default="")
    a = ap.parse_args(); t0 = time.time()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    w = pd.read_csv(a.windows)
    tr, te = split(w, a.train, a.test, DEV)
    X = TokenData(w, a.tokens.split(","), tr, DEV, tokens=a.names.split(","))
    net = TokenFusion(X.tokens, X.dims).to(DEV)
    print(f"{a.train}->{a.test} seed {a.seed}: {len(tr)} enrolment / {len(te)} deployment windows, tokens {X.tokens}, "
          f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f} M parameters, device {DEV}", flush=True)
    enrol(net, X, X.y, tr, te)                                                     # Step 2
    nets, fold_of, ex = name_in_session(net, X, X.y, tr, te, a.seed,
                                        log=lambda s: print(s, flush=True))      # Steps 3-5
    rec, true = X.rec[te], X.y[te]
    t_rec = per_record(true, rec)
    ex["fused"] = float((per_record(decide(nets, fold_of, X, te, "full"), rec) == t_rec).float().mean())   # Step 6
    for v in ("rf", "vis"): ex[f"final_{v}"] = float((per_record(decide(nets, fold_of, X, te, v), rec) == t_rec).float().mean())
    for item in filter(None, a.audit.split(",")):                                  # counterfactual target audit (after naming)
        tag, path = item.split("=")
        with swap_tokens(X, te, path):
            for v in ("full", "vis"):
                pred = per_record(decide(nets, fold_of, X, te, v), rec)
                if tag in ("tp1", "tp2"): ex[f"{tag}_{v}_F_tgt"] = follow_target(pred, t_rec, int(tag[2:]))
                ex[f"{tag}_{v}_acc"] = float((pred == t_rec).float().mean())
    ex.update(train=a.train, test=a.test, seed=a.seed, tokens=X.tokens, seconds=round(time.time() - t0, 1))
    print(json.dumps(ex), flush=True)
    if a.out: json.dump(ex, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
