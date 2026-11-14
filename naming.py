"""In-session naming and inference (paper Section IV-E, Eqs. (13)-(16), Algorithm 1 Steps 3-6, Fig. 4).

The RF anchor gives every deployment record a balanced soft name; cross-fitted fold networks learn the names through the optical
tokens; the held-out RF and optical scores rename the records; after R rounds the fold networks decide on the full view.
The hard names that are reported are the argmax of the soft names.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def view_logits(net, X, idx, view):
    """Window logits under a view: 'full' (all tokens), 'rf' (optical tokens masked) or 'vis' (RF tokens masked)."""
    net.eval(); out = []
    for j in range(0, len(idx), 1024):
        xs, av = X(idx[j:j + 1024])
        if view == "rf": av = av & ~X.isvis[None]
        elif view == "vis": av = av & X.isvis[None]
        out.append(net((xs, av)).float())
    return torch.cat(out)


def record_mean(lp, rec):
    """Record score: the mean window log-posterior of each record (Eqs. (7), (13), (16)), broadcast back to the record's windows."""
    u, inv = torch.unique(rec, return_inverse=True)
    s = torch.zeros(len(u), lp.shape[1], device=lp.device).index_add_(0, inv, lp)
    return (s / torch.bincount(inv, minlength=len(u))[:, None].float())[inv]


def balanced_names(lpw, rec, it=50):
    """Eq. (14): fifty Sinkhorn iterations turn the record scores S into soft names Q; every row sums to 1 and every identity
    receives |I_d| / N records. Input and output are broadcast to the windows."""
    _, first = np.unique(rec.cpu().numpy(), return_index=True); inv = torch.unique(rec, return_inverse=True)[1]
    P = torch.exp(lpw[torch.as_tensor(first, device=lpw.device)] - lpw.max())
    for _ in range(it):
        P = P / P.sum(0, keepdim=True) / P.shape[1]; P = P / P.sum(1, keepdim=True)
    return (P / P.sum(1, keepdim=True))[inv]


@torch.no_grad()
def fold_logprobs(nets, fold_of, X, idx, view):
    """Window log-posteriors of the deployment windows idx, each window scored by the network of its own record's fold, which never
    trained on that record's name (cross-fitting)."""
    lp = torch.zeros(len(idx), 5, device=idx.device)
    for k, m in enumerate(nets):
        s = fold_of[idx] == k
        lp[s] = F.log_softmax(view_logits(m, X, idx[s], view), 1)
    return lp


def name_in_session(net, X, y, tr, te, seed, rounds=5, folds=5, ep=20, lr=1e-3, lr_mult=0.3, wd=0.05, batch=256, rf_mask=0.5,
                    w=1.0, log=print):
    """Algorithm 1, Steps 3-5. net is psi_0 after enrolment; it is never changed, and every fold network of every round restarts
    from it. Labels y are read on enrolment windows; the deployment labels y[te] are read only for the printed readouts.
    Returns the last-round fold networks, the fold of every window and the readouts."""
    dev = te.device; rec = X.rec[te]; true = y[te]
    racc = lambda lp: float((record_mean(lp, rec).argmax(1) == true).float().mean())          # readout only
    lp_rf = record_mean(F.log_softmax(view_logits(net, X, te, "rf"), 1), rec)                 # RF anchor, Eq. (13)
    Q = torch.zeros(len(y), 5, device=dev); Q[te] = balanced_names(lp_rf, rec)               # names Q^(0), Step 3
    names = Q[te].argmax(1)
    ex = {"rf_anchor": racc(lp_rf), "names_r0": float((names == true).float().mean())}
    urec = torch.unique(rec); g = torch.Generator().manual_seed(seed)
    perm = urec[torch.randperm(len(urec), generator=g).to(dev)]                               # random record folds F_1..F_M
    fold_rec = {int(r): k % folds for k, r in enumerate(perm.tolist())}
    fold_of = torch.full((len(X.rec),), -1, dtype=torch.long, device=dev)
    fold_of[te] = torch.tensor([fold_rec[int(r)] for r in rec.tolist()], device=dev)
    hasvis = X.av[:, 2:].any(1)                                                               # a camera recorded the window
    steps = ep * ((len(tr) + batch - 1) // batch)                                              # twenty enrolment passes
    for rnd in range(1, rounds + 1):
        nets = []
        for k in range(folds):                                                               # Step 4
            m = copy.deepcopy(net)                                                           # restart from psi_0
            opt = torch.optim.AdamW(m.parameters(), lr * lr_mult, weight_decay=wd)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
            tt = te[fold_of[te] != k]                                                        # names outside fold k only
            for _ in range(steps):
                m.train()
                b = tr[torch.randint(0, len(tr), (batch,), device=dev)]
                bt = tt[torch.randint(0, len(tt), (batch,), device=dev)]
                xs, av = X(b, train=True); av = av & ~X.isvis[None]                          # enrolment windows: RF view, labels
                loss = F.cross_entropy(m((xs, av)), y[b], label_smoothing=0.1)              # L_win
                xs, av = X(bt, train=True)                                                    # deployment windows: names
                u = torch.rand(len(bt), 1, device=dev)
                av = torch.where((u < rf_mask) & hasvis[bt][:, None], av & X.isvis[None], av)  # RF-token masking -> optical view
                kk = av.any(1); o = m(([x[kk] for x in xs], av[kk]))
                loss = loss + w * -(Q[bt][kk] * F.log_softmax(o, 1)).sum(1).mean()              # Eq. (15) with soft names
                opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); sch.step()
            nets.append(m.eval())
        lps = {v: fold_logprobs(nets, fold_of, X, te, v) for v in ("full", "rf", "vis")}      # held-out scores
        for v, lp in lps.items(): ex[f"r{rnd}_{v}"] = racc(lp)
        Q[te] = balanced_names(record_mean(lps["rf"] + lps["vis"], rec), rec)                # Eq. (16), then Eq. (14): Step 5
        new = Q[te].argmax(1)
        ex[f"r{rnd}_names"] = float((new == true).float().mean())
        ex[f"r{rnd}_renamed"] = int(torch.unique(rec[new != names]).numel()); names = new
        log(f"round {rnd}: full {ex[f'r{rnd}_full']:.3f} rf {ex[f'r{rnd}_rf']:.3f} optical {ex[f'r{rnd}_vis']:.3f} "
            f"names {ex[f'r{rnd}_names']:.3f} renamed {ex[f'r{rnd}_renamed']}")
    return nets, fold_of, ex


def decide(nets, fold_of, X, te, view="full"):
    """Step 6 and Eq. (7): record decisions of the last-round fold networks; view 'full' gives the fused decision."""
    return record_mean(fold_logprobs(nets, fold_of, X, te, view), X.rec[te]).argmax(1)
