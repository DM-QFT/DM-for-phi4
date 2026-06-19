"""Autocorrelation + Sokal windowing utilities for integrated autocorrelation time.

Extracted from dm_mc_ess_2d.py so observable scripts can estimate tau_int of an
HMC time series (e.g. for the autocorrelation-corrected reference error) without
importing the deprecated autocorrelation-ESS pipeline.
"""

import numpy as np


def acf_per_chain_fft(series):
    """
    Normalised ACF for each chain, computed via FFT.

    series: (T, M) array of an observable for M chains of length T.
    returns: rho of shape (T, M) with rho[0, :] = 1.
    """
    T, M = series.shape
    x = series - series.mean(axis=0, keepdims=True)
    n2 = 1 << int(np.ceil(np.log2(2 * T)))  # zero-pad to avoid wrap-around
    f = np.fft.rfft(x, n=n2, axis=0)
    acf = np.fft.irfft(f * np.conj(f), n=n2, axis=0)[:T]
    norm = (T - np.arange(T))[:, None]      # unbiased divisor
    acf = acf / norm
    rho = acf / acf[0:1]
    return rho


def sokal_window(rho_mean, c=5.0):
    """
    Sokal automatic windowing. rho_mean is the chain-averaged ACF (length T).

    tau_hat(W) = 1/2 + sum_{t=1..W} rho_mean(t)
    Pick the smallest W >= 1 with W >= c * tau_hat(W); fall back to last W if none.
    Return (tau_int, W_chosen, tau_hat_curve).
    """
    T = len(rho_mean)
    cum = np.cumsum(rho_mean[1:])           # cum[k-1] = sum_{t=1..k} rho(t)
    tau_hat = 0.5 + cum                     # tau_hat[W-1] for W = 1..T-1
    Ws = np.arange(1, T)
    ok = Ws >= c * tau_hat
    if ok.any():
        W = int(Ws[np.argmax(ok)])
    else:
        W = int(Ws[-1])
    tau = float(tau_hat[W - 1])
    return tau, W, tau_hat
