"""
Extract training loss from TensorBoard logs for 3D phi4 L=64 models.
Two kappa values on one figure, decomposed by noise level t.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "..", "draft", "figures", "3Dphi4")

L_TARGET = 64
K_VALUES = [0.18, 0.1923, 0.2]
LAMBDA = 0.9
EPOCH_MAX = 20000
EPOCH_MIN = 1

TAGS = ["train_loss_epoch", "loss_UV", "loss_mid", "loss_IR"]
PANEL_TITLES = {
    "train_loss_epoch": "(a) Total",
    "loss_UV":  r"(b) UV ($t<0.2$)",
    "loss_mid": r"(c) Mid ($0.2<t<0.8$)",
    "loss_IR":  r"(d) IR ($t>0.8$)",
}
COLORS_K = {0.18: "C0", 0.1923: "C1", 0.2: "C2"}
LABELS_K = {
    0.18:   r"$\kappa=0.18$",
    0.1923: r"$\kappa=0.1923\approx\kappa_c$",
    0.2:    r"$\kappa=0.2$",
}

# Each kappa run uses its own VE-SDE sigma; directory suffix differs.
SIGMA_TAG = {0.18: "sigma2000", 0.1923: "sigma2048", 0.2: "sigma2760"}

MIN_VERSION_EPOCHS = 1


def model_dir(k):
    return os.path.join(
        BASE, f"runs/phi4_3d_L{L_TARGET}_k{k}_l{LAMBDA}_ncsnpp_{SIGMA_TAG[k]}"
    )


def all_versions(log_dir):
    return sorted(glob.glob(os.path.join(log_dir, "version_*")))


def read_scalars(event_dir, tag, max_events=0):
    ea = event_accumulator.EventAccumulator(
        event_dir,
        size_guidance={event_accumulator.SCALARS: max_events},
    )
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def load_data():
    data = {}
    for k in K_VALUES:
        md = model_dir(k)
        log_dir = os.path.join(md, "lightning_logs")
        if not os.path.isdir(log_dir):
            print(f"  skip k={k}: no lightning_logs at {log_dir}")
            continue
        versions = all_versions(log_dir)
        if not versions:
            continue

        # Per-tag pieces (epoch, value) from each version; later pieces win on overlap.
        per_tag = {tag: [] for tag in TAGS}
        used = []
        for ver in versions:
            _, epoch_vals = read_scalars(ver, "epoch")
            if epoch_vals is None:
                continue
            piece_lens = {}
            for tag in TAGS:
                steps, vals = read_scalars(ver, tag)
                if steps is None or len(vals) == 0:
                    continue
                # Align step-indexed train_loss_epoch entries to epoch numbers.
                if len(epoch_vals) == len(vals):
                    ep = epoch_vals.astype(int)
                else:
                    # Fall back: assume one tag value per epoch, sampled from epoch curve.
                    idx = np.linspace(0, len(epoch_vals) - 1, len(vals)).astype(int)
                    ep = epoch_vals[idx].astype(int)
                per_tag[tag].append((ep, vals))
                piece_lens[tag] = len(vals)
            if piece_lens:
                n_repr = piece_lens.get("train_loss_epoch", max(piece_lens.values()))
                ep_lo = int(epoch_vals.min())
                ep_hi = int(epoch_vals.max())
                if n_repr < MIN_VERSION_EPOCHS:
                    print(
                        f"  k={k} {os.path.basename(ver)}: skip "
                        f"(only {n_repr} epoch points, range {ep_lo}-{ep_hi})"
                    )
                    for tag in TAGS:
                        if per_tag[tag] and per_tag[tag][-1] is not None:
                            # already appended above; pop the last entry
                            pass
                    # remove the just-appended entries (skip this version)
                    for tag in TAGS:
                        if tag in piece_lens:
                            per_tag[tag].pop()
                    continue
                used.append((os.path.basename(ver), ep_lo, ep_hi, n_repr))

        if not used:
            print(f"  k={k}: no usable versions")
            continue

        entry = {}
        for tag in TAGS:
            if not per_tag[tag]:
                continue
            ep_cat = np.concatenate([p[0] for p in per_tag[tag]])
            v_cat = np.concatenate([p[1] for p in per_tag[tag]])
            # Sort by epoch; on duplicates, keep the *last* occurrence (later version wins).
            order = np.argsort(ep_cat, kind="stable")
            ep_sorted = ep_cat[order]
            v_sorted = v_cat[order]
            keep = np.concatenate([np.diff(ep_sorted) != 0, [True]])
            entry[tag] = (ep_sorted[keep], v_sorted[keep])

        data[k] = entry
        n = len(entry["train_loss_epoch"][0]) if "train_loss_epoch" in entry else 0
        print(
            f"  loaded L={L_TARGET} k={k}: {len(entry)} tags, {n} unique epochs "
            f"from versions {[u[0] for u in used]}"
        )
        for v_name, lo, hi, n_pts in used:
            print(f"    {v_name}: epochs {lo}-{hi} ({n_pts} pts)")
    return data


def smooth(y, window=11):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def clip(epochs, vals):
    mask = (epochs >= EPOCH_MIN) & (epochs <= EPOCH_MAX)
    return epochs[mask], vals[mask]


def plot_decomposed(data, logx=False):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()

    YLIM_SKIP = 200

    for idx, tag in enumerate(TAGS):
        ax = axes[idx]
        all_v_tail = []
        for k in K_VALUES:
            if k not in data or tag not in data[k]:
                continue
            epochs, vals = data[k][tag]
            ep, v = clip(epochs, vals)
            sv = smooth(v)
            ax.plot(ep, v, color=COLORS_K[k], lw=0.3, alpha=0.35)
            ax.plot(ep, sv, label=LABELS_K[k],
                    color=COLORS_K[k], lw=1.4)
            tail_mask = ep >= YLIM_SKIP
            if tail_mask.any():
                all_v_tail.append(v[tail_mask])
        ax.set_title(PANEL_TITLES[tag], fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
            ax.set_xlim(max(EPOCH_MIN, 1), EPOCH_MAX)
        else:
            ax.set_xlim(EPOCH_MIN, EPOCH_MAX)

        if tag in ("train_loss_epoch", "loss_UV"):
            ax.legend(fontsize=9, loc="lower left")
        else:
            ax.legend(fontsize=9)

        if all_v_tail:
            vmin = min(a.min() for a in all_v_tail)
            vmax = max(a.max() for a in all_v_tail)
            span = max(abs(np.log10(vmax) - np.log10(vmin)), 0.05)
            if tag == "loss_IR":
                lo = 10**(np.log10(vmin) - 0.05 * span)
                hi = 10**(np.log10(vmax) + 0.35 * span)
                ax.set_ylim(lo, hi)
                import matplotlib.ticker as ticker
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
                ax.yaxis.get_major_formatter().set_scientific(False)
                ticks = []
                for exp in range(0, 5):
                    for m in [1, 2, 5]:
                        val = m * 10**exp
                        if lo <= val <= hi:
                            ticks.append(val)
                ax.set_yticks(ticks)
                ax.set_yticklabels([f"{int(t)}" if t >= 1 else f"{t}" for t in ticks])
            else:
                ax.set_ylim(10**(np.log10(vmin) - 0.02 * span),
                            10**(np.log10(vmax) + 0.02 * span))

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading TensorBoard data …")
    data = load_data()
    print(f"Loaded {len(data)} kappa values.\n")

    print("Plotting 3D L=64 decomposed loss (linear x) …")
    fig = plot_decomposed(data, logx=False)
    path = os.path.join(OUT_DIR, "loss_3d_L64_decomposed.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  saved {path}")

    print("Plotting 3D L=64 decomposed loss (log x) …")
    fig_log = plot_decomposed(data, logx=True)
    path_log = os.path.join(OUT_DIR, "loss_3d_L64_decomposed_logx.pdf")
    fig_log.savefig(path_log, bbox_inches="tight")
    print(f"  saved {path_log}")
    print("\nDone.")
