"""Diagonal-mode FFT and jackknife statistics shared by the analysis scripts."""

import h5py
import numpy as np
import scipy.fft
from scipy.fft import fft, ifft, next_fast_len

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


def load_series(path):
    """合并后的时间序列 -> (times, dataset)。

    dataset[i] 惰性读第 i 个时间点，布局统一为 [ensemble, x, y]。文件句柄不关，
    脚本跑完由解释器回收。
    """
    handle = h5py.File(path, "r")
    return np.asarray(handle["times"]), handle["phi_gen" if "phi_gen" in handle else "cfgs"]


def series_index(times, t, tol=1e-6):
    i = int(np.argmin(np.abs(times - t)))
    assert abs(times[i] - t) < tol, f"序列里没有 t={t}，最近的是 {times[i]}"
    return i


def load_fields(path, *, start=0, count=None, swap=False):
    """读成 [ensemble, x, y] float32。

    全库约定：磁盘上一律 [.., ensemble, x, y]。Julia 侧靠
    permutedims(phi_gen, (3, 2, 1)) 写出即满足，读时无需转置。
    swap=True 只留给尚未按此约定重存的历史文件。
    """
    stop = None if count is None else start + count
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return np.asarray(archive["phi_gen"][start:stop], dtype=np.float32)
    with h5py.File(path, "r") as handle:
        fields = np.asarray(handle["cfgs"][start:stop], dtype=np.float32)
    return fields.swapaxes(1, 2) if swap else fields


def tau_1e(t, c):
    """c(t) 降到 1/e 的时刻，对 log c 线性插值；不过阈值返回 nan。"""
    crossed = np.flatnonzero(c <= np.exp(-1.0))
    if crossed.size == 0:
        return np.nan
    i = crossed[0]
    if i == 0 or c[i] <= 0:
        return np.nan
    f = (np.log(c[i - 1]) + 1) / (np.log(c[i - 1]) - np.log(c[i]))
    return t[i - 1] + f * (t[i] - t[i - 1])


def cov_per_ens(x):
    """x[time,ens,...] 的时间自协方差，经 FFT，按 lag 计数归一后对 ens 平均。"""
    nt = x.shape[0]
    nlag = nt // 2 + 1
    tr = np.ascontiguousarray(x.transpose(2, 1, 0))
    nf = next_fast_len(2 * nt - 1)
    sp = fft(tr, nf, axis=-1, workers=FFT_WORKERS)
    cov = ifft(sp * sp.conj(), axis=-1, workers=FFT_WORKERS).real[..., :nlag]
    cov /= np.arange(nt, nt - nlag, -1)
    return cov.mean(axis=1)
