"""Diagonal-mode FFT and jackknife statistics shared by the analysis scripts."""

import numpy as np
import scipy.fft

SIGNS = np.array(((1, 1), (1, -1), (-1, 1), (-1, -1)), dtype=np.int32)
FFT_BATCH = 512
FFT_WORKERS = 8


def mode_layout(length):
    """Signed diagonal modes (n,n), (n,-n), (-n,n), (-n,-n) for n=1..L/2."""
    mode_n = np.arange(1, length // 2 + 1, dtype=np.int32)
    ky = (mode_n[:, None] * SIGNS[None, :, 0]) % length
    kx = (mode_n[:, None] * SIGNS[None, :, 1]) % length
    count = np.array([len(set(zip(y, x))) for y, x in zip(ky, kx)])
    khat2 = 8.0 * np.sin(np.pi * mode_n / length) ** 2
    return mode_n, ky, kx, count, np.sqrt(khat2), khat2


def diagonal_fft(fields, ky, kx):
    """Center each field and keep the four signed diagonal modes."""
    modes = np.empty((len(fields), ky.shape[0], 4), dtype=np.complex64)
    for start in range(0, len(fields), FFT_BATCH):
        block = np.asarray(fields[start:start + FFT_BATCH], dtype=np.float32)
        block = block - block.mean(axis=(1, 2), keepdims=True)
        spectrum = scipy.fft.fft2(block, workers=FFT_WORKERS)
        modes[start:start + len(block)] = spectrum[:, ky, kx]
    return modes


def power(modes, volume):
    """Equal-time diagonal power per trajectory, averaged over the four signs."""
    return np.mean((modes * np.conjugate(modes)).real, axis=-1) / volume


def cross(modes, i, j, volume):
    """Re<phi_k(t_i) phi_k(t_j)^*> per trajectory, averaged over the four signs."""
    return np.mean(
        (modes[i] * np.conjugate(modes[j])).real, axis=-1
    ) / volume


def jackknife_sem(leaves):
    n = len(leaves)
    deviation = leaves - leaves.mean(axis=0)
    return np.sqrt((n - 1) / n * np.sum(deviation * deviation, axis=0))


def paired_jackknife_sem(left, right):
    return jackknife_sem(left - right)


def statistics(cross_values, power_i, power_j):
    """F, its sem, rho = <F>/sqrt(<G_i><G_j>), its jackknife sem, and the leaves."""
    cross_values = np.asarray(cross_values, dtype=np.float64)
    power_i = np.asarray(power_i, dtype=np.float64)
    power_j = np.asarray(power_j, dtype=np.float64)
    n = len(cross_values)
    correlation = cross_values.mean(axis=0)
    power_i_mean = power_i.mean(axis=0)
    power_j_mean = power_j.mean(axis=0)
    leaves = (n * correlation - cross_values) / np.sqrt(
        (n * power_i_mean - power_i) * (n * power_j_mean - power_j)
    )
    return (
        correlation,
        cross_values.std(axis=0, ddof=1) / np.sqrt(n),
        correlation / np.sqrt(power_i_mean * power_j_mean),
        jackknife_sem(leaves),
        leaves,
    )


def moments(fields):
    """Per-trajectory M, <phi^2>, <phi^4> and the spatial variance."""
    volume = fields.shape[1] * fields.shape[2]
    mean = np.mean(fields, axis=(1, 2), dtype=np.float64)
    phi2 = np.einsum(
        "nij,nij->n", fields, fields, dtype=np.float64, optimize=True
    ) / volume
    phi4 = np.empty(len(fields), dtype=np.float64)
    for start in range(0, len(fields), 256):
        stop = min(start + 256, len(fields))
        square = np.asarray(fields[start:stop], dtype=np.float64) ** 2
        phi4[start:stop] = np.mean(square * square, axis=(1, 2))
    return mean, phi2, phi4, phi2 - mean * mean


def mean_sem(values):
    return (
        np.mean(values, axis=1, dtype=np.float64),
        np.std(values, axis=1, ddof=1, dtype=np.float64)
        / np.sqrt(values.shape[1]),
    )
