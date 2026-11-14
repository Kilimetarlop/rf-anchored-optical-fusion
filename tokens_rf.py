"""RF tokens (paper Section IV-B, "RF tokens"): the 13-d blind OFDM fingerprint of each receiver and the oscillator error eps.

The functions reproduce the computation of the experiment code that built the paper's RF tokens from the raw DroneMMset IQ
recordings; batch drivers, caches and the feature columns that the tokens do not use are left out.

Numerology: the 40 MS/s recording is resampled by 48/125 to 15.36 MS/s, where one OFDM symbol is N = 1024 samples plus a 72-sample
cyclic prefix and 600 sub-carriers (-300..300 without DC) are used; the downlink reference symbol repeats every 0.5 ms (7680 samples).
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, resample_poly

FS, SEG = 40e6, 1 << 20                                  # recording rate; one fingerprint segment = 2^20 samples (26 ms)
N, LCP = 1024, 72                                        # symbol length and cyclic prefix at 15.36 MS/s
KU = np.r_[np.arange(-300, 0), np.arange(1, 301)]        # used sub-carriers, DC excluded
PER, FSR = 7680, 15.36e6                                 # reference-symbol period (samples) and rate for the clock


def read_iq(path, dev, frac, n):
    """n complex samples starting at relative position frac of a raw file (Dev02: float32 I/Q pairs, Dev01: int16 I/Q pairs)."""
    dt = np.dtype("<i2" if dev == "Dev01" else "<f4"); item = dt.itemsize * 2
    tot = os.path.getsize(path) // item; off = int(frac * (tot - n))
    with open(path, "rb") as f:
        f.seek(off * item); a = np.frombuffer(f.read(n * item), dtype=dt).astype(np.float32)
    return (a[0::2] + 1j * a[1::2]).astype(np.complex64)


def ofdm_symbols(y, distance):
    """Blind symbol detection: the cyclic-prefix correlation Gamma(n), the sum over the prefix of y[n] conj(y[n + N]), peaks at every
    symbol start. Each symbol is cut 4 samples before the end of its prefix and transformed by an N-point FFT.
    Returns the symbol spectra S (n_sym, N), the prefix phase of every symbol and the symbol starts.
    distance: minimum peak spacing, 1000 for the fingerprint and int(0.95 N) = 972 for the clock (as used for the paper)."""
    c = y[:-N] * np.conj(y[N:]); cs = np.convolve(c, np.ones(LCP), "valid"); m = np.abs(cs)
    pk, _ = find_peaks(m, distance=distance, height=4 * np.median(m))
    pk = pk[pk + LCP - 4 + N <= len(y)]
    S = np.array([np.fft.fft(y[p + LCP - 4:p + LCP - 4 + N]) for p in pk])
    return S, np.angle(cs[pk]), pk


# ------------------------------------------------------------------ fingerprint (13-d per segment)
def fingerprint13(S):
    """Token columns in order: papr_q0.9, papr_q0.99, papr_q0.999, iq_rho0 .. iq_rho9.
    PAPR: instantaneous power of the time-domain symbols over each symbol's mean power; its 0.9, 0.99 and 0.999 quantiles in dB
          trace power-amplifier clipping.
    iq_rho: |E[X_k X_-k]| / sqrt(E|X_k|^2 E|X_-k|^2) between the mirror sub-carriers k and -k, averaged over 10 sub-bands of
          k = 1..300; it traces IQ imbalance.
    Both are ratios of signal quantities, so the channel gain cancels."""
    idx = lambda k: np.mod(k, N)
    P = np.abs(S) ** 2
    t = np.fft.ifft(S, axis=1); pp = np.abs(t) ** 2 / (np.abs(t) ** 2).mean(1, keepdims=True)
    papr = [10 * np.log10(np.quantile(pp, q)) for q in (0.9, 0.99, 0.999)]
    rho = []
    for ks in np.array_split(np.arange(1, 301), 10):
        num = (S[:, idx(ks)] * S[:, idx(-ks)]).mean(0)
        den = np.sqrt(P[:, idx(ks)].mean(0) * P[:, idx(-ks)].mean(0))
        rho.append(np.abs(num / den).mean())
    return np.array(papr + rho, np.float32)


def dev01_carrier(path, c0):
    """Carrier of a Dev01 record. Band edges of the smoothed PSD of 4 segments within +-6 MHz of the start value c0 give a centre;
    the prefix phase then gives the residual modulo the 15 kHz sub-carrier spacing.
    c0 is the record's own band centre: the median over its windows of the centre of the strongest 10 MHz (windows.csv column
    Dev01_fc_mhz), so no label is used; a start value taken per identity and session would use labels."""
    NFFT, SC = 1 << 16, 15e3
    P = np.zeros(NFFT)
    for k in range(4):
        x = read_iq(path, "Dev01", (k + 0.5) / 4, SEG).reshape(-1, NFFT)
        P += (np.abs(np.fft.fftshift(np.fft.fft(x * np.hanning(NFFT), axis=1), axes=1)) ** 2).sum(0)
    f = (np.arange(NFFT) - NFFT // 2) * FS / NFFT; p = 10 * np.log10(np.convolve(P, np.ones(9) / 9, "same") + 1e-12)
    floor = np.percentile(p, 10); inb = np.percentile(p[np.abs(f - c0) < 3e6], 50); core = np.abs(f - c0) < 6e6
    above = np.where(core & (p > (floor + inb) / 2))[0]; centre = (f[above.min()] + f[above.max()]) / 2
    x = read_iq(path, "Dev01", 0.5, SEG) * np.exp(-2j * np.pi * centre * np.arange(SEG) / FS)
    y = resample_poly(x, 48, 125); c = y[:-N] * np.conj(y[N:]); m = np.convolve(c, np.ones(LCP), "valid")
    top = np.argsort(np.abs(m))[-2000:]; res = -np.angle(m[top].sum()) / (2 * np.pi) * SC
    return centre + res


def segment_fingerprints(path, dev, fc, nseg=32):
    """Fingerprints of nseg segments at relative positions (k + 0.5) / nseg of one record, after moving the carrier fc to 0 Hz
    and resampling to 15.36 MS/s; segments with fewer than 50 symbols are skipped.
    fc: Dev02, the prefix-phase carrier offset unwrapped onto the band-edge centre of the record (estimated upstream, not
        included here); Dev01, dev01_carrier(path, c0)."""
    out = []
    for k in range(nseg):
        x = read_iq(path, dev, (k + 0.5) / nseg, SEG) * np.exp(-2j * np.pi * fc * np.arange(SEG) / FS).astype(np.complex64)
        S, _, _ = ofdm_symbols(resample_poly(x, 48, 125).astype(np.complex64), distance=1000)
        if len(S) >= 50: out.append(fingerprint13(S))
    return np.array(out, np.float32)


def window_tokens(seg_fp, K=64):
    """Window k of a record takes segment k mod n of the record's n usable segments; no usable segment -> availability 0."""
    if len(seg_fp) == 0: return np.zeros((K, 13), np.float32), np.zeros(K, bool)
    return np.stack([seg_fp[k % len(seg_fp)] for k in range(K)]), np.ones(K, bool)


# ------------------------------------------------------------------ oscillator error eps (Dev02, record level)
def diffspec(S):
    """Differential spectrum Y_k conj(Y_k+1): removes the common phase and, to first order, the timing offset, so that identical
    reference symbols correlate near 1 and data symbols near 1/sqrt(K)."""
    Y = S[:, np.mod(KU, N)]; D = Y[:, :-1] * np.conj(Y[:, 1:])
    return D / np.linalg.norm(D, axis=1, keepdims=True)


def symbol_classes(S, thr=0.5, min_size=8):
    """Groups of symbols with identical content (|correlation of differential spectra| > thr), largest first."""
    D = diffspec(S); C = np.abs(D @ D.conj().T); np.fill_diagonal(C, 0); used = np.zeros(len(D), bool); out = []
    while True:
        cnt = ((C > thr) & ~used[None, :]).sum(1); cnt[used] = -1; i = int(np.argmax(cnt))
        if cnt[i] + 1 < min_size: break
        g = np.r_[i, np.where((C[i] > thr) & ~used)[0]]; used[g] = True; out.append(g)
    return out


def align(Y, ref, pad=16):
    """Per copy: delay (phase ramp over the sub-carriers, coarse by a zero-padded FFT, then fine) and common phase relative to ref."""
    Kn = Y.shape[1]; L = pad * Kn; k = KU.astype(float)
    X = np.fft.fft(Y * np.conj(ref)[None, :], n=L, axis=1)
    j = np.argmax(np.abs(X), 1); slope = np.angle(np.exp(2j * np.pi * j / L))
    Yc = Y * np.exp(-1j * slope[:, None] * k[None, :])
    Dk = (Yc[:, 1:] * np.conj(Yc[:, :-1])) * np.conj(ref[1:] * np.conj(ref[:-1]))[None, :]
    dk = np.diff(k); fine = np.angle((Dk * (dk == 1)[None, :]).sum(1))          # skip the DC gap
    Yc *= np.exp(-1j * fine[:, None] * k[None, :]); slope = slope + fine
    ph = np.angle((Yc * np.conj(ref)[None, :]).sum(1)); Yc *= np.exp(-1j * ph)[:, None]
    return Yc, np.angle(np.exp(1j * slope)), ph


def template(S, g, iters=3, min_corr=0.5):
    """Coherent template of one class; keeps the copies that match it after alignment. Returns their delays (as phase slopes) and
    indices."""
    Y = S[g][:, np.mod(KU, N)]; ref = Y[0]; keep = np.arange(len(g))
    for _ in range(iters):
        Yc, slope, ph = align(Y, ref)
        c = np.abs((Yc * np.conj(ref)[None, :]).sum(1)) / (np.linalg.norm(Yc, axis=1) * np.linalg.norm(ref))
        keep = np.where(c > min_corr)[0]; ref = Yc[keep].mean(0)
    return slope[keep], np.asarray(g)[keep]


def clock_from_symbols(S, pk, min_corr=0.5):
    """Sampling clock offset zeta (ppm) of one 105 ms window. The arrival times t_j of the copies of a reference symbol (symbol start
    plus the fractional delay) lie on a grid of the nominal period; a straight-line fit t_j = t_0 + j T gives zeta = (T / PER - 1) 1e6.
    Of the three largest classes, the fit with the most on-grid copies is kept."""
    best = None
    for g in symbol_classes(S)[:3]:
        slope, gk = template(S, g, min_corr=min_corr)
        if len(gk) < 10: continue
        t = pk[gk] - slope * N / (2 * np.pi)                     # copy start: integer peak + fractional delay
        dts = np.diff(np.sort(t))                                # 0.5 ms period on most models; otherwise taken from the data
        P0 = PER if np.mean(np.abs(dts / PER - np.round(dts / PER)) * PER < 20) > 0.5 else float(np.median(dts[dts > 0.25 * PER][:20]))
        best_ref, best_on = 0, None                              # copies on the period grid of the largest on-grid subset
        for r0 in range(min(len(t), 10)):
            on = np.abs((t - t[r0]) / P0 - np.round((t - t[r0]) / P0)) * P0 < 20
            if best_on is None or on.sum() > best_on.sum(): best_ref, best_on = r0, on
        t0 = t[best_ref]; t = t[best_on]; n = np.round((t - t0) / P0)
        if len(np.unique(n)) < 8: continue
        A = np.c_[np.ones_like(n), n]; coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        cand = dict(sco_ppm=(coef[1] / PER - 1) * 1e6 if P0 == PER else np.nan, t_resid=float(np.std(t - A @ coef)), n=len(t))
        if best is None or cand["n"] > best["n"]: best = cand
    return best


def record_zeta(path, fc, nwin=6, win=1 << 22):
    """zeta of one Dev02 record: the clock of nwin = 6 windows of 2^22 samples (105 ms) spread over the record; windows with a timing
    residual of 0.8 samples or more, or fewer than 20 copies, are dropped; median of the rest."""
    z = []
    for w in range(nwin):
        x = read_iq(path, "Dev02", (w + 0.5) / nwin, win) * np.exp(-2j * np.pi * fc * np.arange(win) / FS).astype(np.complex64)
        S, _, pk = ofdm_symbols(resample_poly(x, 48, 125).astype(np.complex64), distance=int(0.95 * N))
        if len(S) < 50: continue
        o = clock_from_symbols(S, pk)
        if o and o["t_resid"] < 0.8 and o["n"] >= 20 and not np.isnan(o["sco_ppm"]): z.append(o["sco_ppm"])
    return float(np.median(z)) if z else np.nan


def session_eps(zeta, session):
    """eps_i = zeta_i minus the median of zeta over the records of the same session: removes the receiver's own clock without labels.
    The record value is broadcast to its windows and shares the Dev02 availability."""
    z = pd.Series(np.asarray(zeta, float))
    return (z - z.groupby(np.asarray(session)).transform("median")).values
