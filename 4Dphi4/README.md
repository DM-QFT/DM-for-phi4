---
pretty_name: 4D phi4 Wolff FAHMC configurations
tags:
- lattice-field-theory
- phi4
- monte-carlo
- hmc
- wolff
---

# 4D phi4 Wolff FAHMC configurations

This dataset contains 4D scalar phi4 lattice configurations generated with a
Wolff + Fourier-Accelerated HMC sampler.

## Files

| File | Lattice | Shape of `cfgs` | dtype |
|---|---:|---:|---|
| `cfgs_wolff_fahmc_k=0.145_l=0.9_8^4.npz` | `8^4` | `(8, 8, 8, 8, 5120)` | `float64` |
| `cfgs_wolff_fahmc_k=0.145_l=0.9_16^4.npz` | `16^4` | `(16, 16, 16, 16, 5120)` | `float64` |

Each `.npz` file contains:

- `cfgs`: field configurations, with samples on the last axis.
- `kappa`: hopping parameter.
- `lambda`: quartic coupling.
- `N`: lattice size.
- `n_samples`: number of stored configurations.
- `epsilon_final`: final HMC step size.
- `acc_rate`: production HMC acceptance rate.

In Python, one configuration is `cfgs[:, :, :, :, i]`, where `i` is the
zero-based sample index.

## Download

Install the Hugging Face Hub client:

```bash
pip install -U huggingface_hub
```

Download one file:

```bash
hf download YangyangTan/4Dphi4 \
  "cfgs_wolff_fahmc_k=0.145_l=0.9_8^4.npz" \
  --repo-type dataset \
  --local-dir .
```

## Load with NumPy

```python
import numpy as np

path = "cfgs_wolff_fahmc_k=0.145_l=0.9_8^4.npz"
data = np.load(path)

cfgs = data["cfgs"]
kappa = data["kappa"].item()
lam = data["lambda"].item()
N = data["N"].item()

phi0 = cfgs[:, :, :, :, 0]
print(cfgs.shape, cfgs.dtype)
print(kappa, lam, N)
```

You can also download directly from Python:

```python
from huggingface_hub import hf_hub_download
import numpy as np

path = hf_hub_download(
    repo_id="YangyangTan/4Dphi4",
    filename="cfgs_wolff_fahmc_k=0.145_l=0.9_8^4.npz",
    repo_type="dataset",
)

data = np.load(path)
cfgs = data["cfgs"]
```
