"""Re-probe normalisation using the collaborator's EXACT network (unet.py:
circular padding, GroupNorm(4) on gnorm1, score = kinematic/std, potential
unused). cos is sigma-independent."""
import sys, functools
from pathlib import Path
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[4]))   # diffusion_lightning (marginal_prob_std)
sys.path.insert(0, str(HERE.parent))       # unet.py, loss_eval.py
import numpy as np, torch, h5py
from diffusion_lightning import marginal_prob_std
from unet import ScoreNet as HisScoreNet
import loss_eval as le

DEV="cuda:0"; K,LAM,L=0.26,0.022,16; T=1e-4; N=512
CK=str(HERE.parent/"ckpt_L16_k0.26_l0.022.pth")
DATA=str(HERE.parents[3]/f"trainingdata/cfgs_wolff_fahmc_k={K}_l={LAM}_{L}^2.jld2")
ck=torch.load(CK,map_location="cpu",weights_only=False)
strip=lambda sd:{(k[7:] if k.startswith("module.") else k):v for k,v in sd.items()}
mp=functools.partial(marginal_prob_std,sigma=25.0)
model=HisScoreNet(mp).to(DEV).eval()

with h5py.File(DATA,"r") as f: cfgs=np.asarray(f["cfgs"])
idx=np.random.default_rng(42).choice(cfgs.shape[0],N,replace=False)
raw=cfgs[idx].astype(np.float32)
gmin,gmax=float(cfgs.min()),float(cfgs.max()); mabs=float(np.abs(cfgs).max())
mu=float(cfgs.mean()); sg=float(cfgs.std())
mk=lambda arr: torch.tensor(arr,dtype=torch.float32,device=DEV)[:,None]
norms={
 "global_minmax": (mk(((raw-gmin)/(gmax-gmin)-0.5)*2.0), gmin, gmax),
 "maxabs":        (mk(raw/mabs),                          -mabs, mabs),
 "zscore":        (mk((raw-mu)/sg),                       None, None),
 "raw_physical":  (mk(raw),                               None, None),
}
def cos_of(s,t):
    sm=s.reshape(N,-1); st=t.reshape(N,-1)
    return ((sm*st).sum(1)/(sm.norm(dim=1)*st.norm(dim=1)+1e-12)).mean().item()

sel=["epoch_0","epoch_240","epoch_480","epoch_960","epoch_1920","epoch_3000","epoch_4980"]
hdr=f"{'epoch':>6} "+" ".join(f"{k:>14}" for k in norms); print(hdr); print("-"*len(hdr))
with torch.no_grad():
    for key in sel:
        model.load_state_dict(strip(ck[key]),strict=True)
        row=[f"{int(key.split('_')[1]):>6}"]
        for lbl,(x,pmn,pmx) in norms.items():
            s=model(None,x,torch.full((N,),T,device=DEV))
            tgt=le.phi4_score(x,K,LAM,phi_min=pmn,phi_max=pmx,spatial_dims=2)
            row.append(f"{cos_of(s,tgt):14.4f}")
        print(" ".join(row))
