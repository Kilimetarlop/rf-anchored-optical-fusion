"""Token inputs (paper Section IV-B, standardisation of Eq. (8); sensor dropout of Section IV-C).

windows.csv has one row per window: record `tuple`, session `D`, identity `T`, window index `k`, and one 0/1 column per sensor
(Dev02, Dev01, Cam01, Cam02, Inf01, Inf02) saying whether the sensor recorded the window. Each token file (.npz) holds one array per
token source, shape (n_windows, d), and a boolean `<token>_ok` (the token could be extracted).
"""
import numpy as np
import torch

IDS = ["T0001", "T0010", "T0011", "T0100", "T0101"]          # T0001 and T0010 are both DJI Air 2S
SENSORS = ["Dev02", "Dev01", "Cam01", "Cam02", "Inf01", "Inf02"]
SENSOR_OF = {"fp02": 0, "eps": 0, "fp01": 1, "vs01": 2, "tc01": 2, "vs02": 3, "tc02": 3, "vi01": 4, "ti01": 4, "vi02": 5, "ti02": 5}
TOKENS = ["fp02", "fp01", "eps", "vs01", "vs02", "vi01", "vi02", "tc01", "tc02", "ti01", "ti02"]   # final configuration


def split(w, train, test, dev):
    """Cross-session protocol: all windows of session `train` enrol, all windows of session `test` are deployed."""
    tr, te = (w.D == train).values, (w.D == test).values
    assert not set(w.tuple[tr]) & set(w.tuple[te]), "enrolment and deployment records overlap"
    return torch.tensor(np.where(tr)[0], device=dev), torch.tensor(np.where(te)[0], device=dev)


class TokenData:
    """Holds every window's tokens on the device; X(idx) returns the standardised tokens and their availability."""

    def __init__(self, w, token_files, train_idx, dev, tokens=TOKENS, md=0.2, vis_drop=0.5):
        self.w = w
        self.av = torch.tensor(w[SENSORS].values.astype(bool), device=dev)                    # (N, 6) sensor recorded the window
        self.y = torch.tensor(w["T"].map({t: i for i, t in enumerate(IDS)}).values, device=dev)
        rec = w.tuple.astype("category").cat.codes.values.astype(np.int64)
        self.rec = torch.tensor(rec, device=dev)                                                 # record id of every window
        self.recwin = torch.tensor(np.stack([np.where(rec == r)[0] for r in range(rec.max() + 1)]), device=dev)   # (records, K)
        aux = {}
        for p in token_files: aux.update(dict(np.load(p)))
        self.tokens, self.md, self.vis_drop = list(tokens), md, vis_drop
        self.aux = {k: torch.from_numpy(aux[k]).to(dev) for t in self.tokens for k in (t, f"{t}_ok")}
        self.dims = {t: aux[t].reshape(len(w), -1).shape[1] for t in self.tokens}
        self.tsens = torch.tensor([SENSOR_OF[t] for t in self.tokens], device=dev)              # sensor of every token source
        self.isvis = self.tsens >= 2                                                             # optical token sources
        self.ok = torch.stack([self.aux[f"{t}_ok"] for t in self.tokens], 1)                    # (N, T) token extracted
        self.mu, self.sd = {}, {}
        for t in self.tokens:                       # mean and standard deviation over the enrolment windows only (Eq. (8))
            x = self.aux[t].float().reshape(len(w), -1)
            m = torch.zeros_like(self.aux[f"{t}_ok"]); m[train_idx] = True; m &= self.aux[f"{t}_ok"]
            self.mu[t], self.sd[t] = x[m].mean(0), x[m].std(0).clamp_min(1e-6)
        self.mask_vis = False                       # set during enrolment training: RF view only

    def __call__(self, idx, train=False):
        """Standardised tokens (list of (B, d) tensors) and availability a (B, T) of the windows idx. A token is available when its
        sensor recorded the window and the token could be extracted. Training drops every sensor with probability md and all
        cameras together with probability vis_drop, but keeps the window as it was if no token would be left."""
        B = len(idx); sens = self.av[idx].clone()
        if train:
            keep = sens & (torch.rand(sens.shape, device=sens.device) >= self.md)
            if self.vis_drop > 0: keep[:, 2:] &= torch.rand(B, 1, device=sens.device) >= self.vis_drop
            sens = torch.where((keep[:, self.tsens] & self.ok[idx]).any(1, keepdim=True), keep, sens)
        av = sens[:, self.tsens] & self.ok[idx]
        xs = [(self.aux[t][idx].float().reshape(B, -1) - self.mu[t]) / self.sd[t] for t in self.tokens]
        if self.mask_vis: av = av & ~self.isvis[None]
        return xs, av
