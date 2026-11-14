"""Enrolment training (paper Section IV-D, Eqs. (10)-(12); Algorithm 1, Step 2).

The network psi_0 learns the enrolment labels on the RF view only. The loss adds a record (bag) term and a session-adversarial term
that sees unlabelled deployment windows. The RF view of psi_0 then gives the RF anchor (Eq. (13), naming.py).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GradReverse


def bag_loss(out, yb, bag):
    """Eq. (11): cross-entropy of the record decision (mean window log-softmax) over bags of `bag` windows of one record."""
    lp = F.log_softmax(out, 1).view(-1, bag, out.shape[1]).mean(1)
    return F.cross_entropy(lp, yb.view(-1, bag)[:, 0])


def bag_batches(X, tr, batch, bag):
    """One pass of batches; every batch holds batch // bag enrolment records with `bag` random windows each (16 x 16 = 256)."""
    dev = tr.device; trec = torch.unique(X.rec[tr]); R = batch // bag
    reps = (len(tr) // batch) * R // len(trec) + 1
    order = torch.cat([trec[torch.randperm(len(trec), device=dev)] for _ in range(reps)])
    return [X.recwin[order[i:i + R]][:, torch.randint(0, X.recwin.shape[1], (bag,), device=dev)].reshape(-1)
            for i in range(0, (len(tr) // batch) * R, R)]


def enrol(net, X, y, tr, te, epochs=60, batch=256, lr=1e-3, wd=0.05, bag=16, dann=1.0):
    """Trains psi_0 in place. Labels y are read on enrolment windows tr only; deployment windows te enter without labels."""
    dev = tr.device
    steps_ep = (len(tr) + batch - 1) // batch; total = epochs * steps_ep
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=total, pct_start=0.05)
    X.mask_vis = True             # optical tokens masked throughout: the enrolment session's optical identity does not carry over
    step = 0
    while step < total:
        net.train()
        for b in bag_batches(X, tr, batch, bag):
            if step >= total: break
            bt = te[torch.randint(0, len(te), (len(b),), device=dev)]                    # unlabelled deployment windows
            out, zs = net(X(b, train=True), emb=True)
            _, zt = net(X(bt, train=True), emb=True)
            lam = 2 / (1 + np.exp(-10 * step / total)) - 1                                 # GRL weight rises from 0 to 1
            loss = F.cross_entropy(out, y[b], label_smoothing=0.1) + bag_loss(out, y[b], bag)          # L_win + L_bag
            dl = net.dom(GradReverse.apply(torch.cat([zs, zt]), dann * lam)).squeeze(1)
            loss = loss + F.binary_cross_entropy_with_logits(dl, torch.cat([torch.zeros(len(b)), torch.ones(len(b))]).to(dev))  # L_adv
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
    X.mask_vis = False
    return net
