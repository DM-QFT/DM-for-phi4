# train_su2_dm.main with its module constants overridden at run time for beta = 50, L = 128: the first 5056 of the 5120
# heat-bath configurations (the last 64 are held out for the checks), NLAYERS from the command line, 100 epochs.  Layers > 3
# use the near-identity initialisation (bilinear linear part = identity, bilinear part x0.3, gate bias +3) that the
# beta = 200 deep runs needed.
import sys
import numpy as np
import torch
sys.path.insert(0, "/work/gw97/w97000/project/DM/2DSUN/src/SU2")
import train_su2_dm as T

NLAYERS = int(sys.argv[1])
DATA = "/work/gw97/w97000/project/DM/2DSUN/trainningdata/cfgs_su2_beta=50.0_128^2.h5"
N_TRAIN = 5056
orig_load = T.load_cfgs
T.load_cfgs = lambda n=N_TRAIN, path=DATA, dtype=torch.float32: orig_load(n, path, dtype)
if NLAYERS > 3:
    orig_init = T.LCNN.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        with torch.no_grad():
            for w, b in zip(self.bilin, self.gate_b):
                ch = w.shape[0]
                w[:, 1:, 1:] *= 0.3
                w[:, 0, 0] = 0
                w[:, 0, 1:] = 0.5 * torch.eye(ch)
                w[:, 1:, 0] = 0.5 * torch.eye(ch)
                b.bias += 3.0
    T.LCNN.__init__ = init
T.BETA = 50.0
T.DATA = DATA
T.NLAYERS = NLAYERS
T.EPOCHS = 100
T.SAVE_EPOCHS = sorted(set(np.rint(np.geomspace(1, T.EPOCHS, 10)).astype(int).tolist()) | {10})
T.OUT = f"/work/gw97/w97000/project/DM/2DSUN/results/su2_L128_beta50.0_lcnn_hk_c{T.CH}l{NLAYERS}_n{N_TRAIN}_b{T.BATCH}_e{T.EPOCHS}_betanoise"
print(f"beta 50 run via runtime overrides of train_su2_dm: NLAYERS={NLAYERS}, first {N_TRAIN} configurations of {DATA}, "
      f"near-identity init {'on' if NLAYERS > 3 else 'off'}", flush=True)
T.main()
