"""Definitive reproduction of the collaborator's score-quality-vs-epoch curves,
using HIS network (unet.py) + HIS metric (loss_eval.py).

Settings locked in by the probes:
  score      = kinematic(x,t)/marginal_prob_std(t)   (potential is dead code)
  norm       = global min/max -> [-1,1]
  t_eval     = 1e-4 ,  sigma = 25 (Song tutorial default; cos is sigma-free)
Metric is called via score_quality_each(param=0,...) to match his forward(param,x,t).
"""
import sys, functools
from pathlib import Path
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[4]))
sys.path.insert(0, str(HERE.parent))
import numpy as np, torch, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from diffusion_lightning import marginal_prob_std
from unet import ScoreNet as HisScoreNet
import loss_eval as le

DEV="cuda:0"; K,LAM,L=0.26,0.022,16; T=1e-4; N=512; SIGMA=25.0
CK=str(HERE.parent/"ckpt_L16_k0.26_l0.022.pth")
DATA=str(HERE.parents[3]/f"trainingdata/cfgs_wolff_fahmc_k={K}_l={LAM}_{L}^2.jld2")

ck=torch.load(CK,map_location="cpu",weights_only=False)
epochs=sorted(ck.keys(),key=lambda s:int(s.split("_")[1]))
ep=np.array([int(s.split("_")[1]) for s in epochs])
strip=lambda sd:{(k[7:] if k.startswith("module.") else k):v for k,v in sd.items()}
mp=functools.partial(marginal_prob_std,sigma=SIGMA)
model=HisScoreNet(mp).to(DEV).eval()

with h5py.File(DATA,"r") as f: cfgs=np.asarray(f["cfgs"])
gmin,gmax=float(cfgs.min()),float(cfgs.max())
idx=np.random.default_rng(42).choice(cfgs.shape[0],N,replace=False)
xn=((cfgs[idx]-gmin)/(gmax-gmin)-0.5)*2.0
x=torch.tensor(xn,dtype=torch.float32,device=DEV)[:,None]
print(f"norm [-1,1] from [{gmin:.4f},{gmax:.4f}]  N={N}  t={T}  sigma={SIGMA}\n")

cos=np.zeros(len(ep)); cse=np.zeros(len(ep)); mse=np.zeros(len(ep)); mag=np.zeros(len(ep))
with torch.no_grad():
    for i,key in enumerate(epochs):
        model.load_state_dict(strip(ck[key]),strict=True)
        c,m,g=le.score_quality_each(x,model,param=0,eps=T,k=K,l=LAM,
                                    phi_min=gmin,phi_max=gmax,spatial_dims=2)
        cos[i]=c.mean().item(); cse[i]=(c.std()/np.sqrt(N)).item()
        mse[i]=m.mean().item(); mag[i]=g.mean().item()
        print(f"{ep[i]:>6}  cos={cos[i]:.4f}  rel_mse={mse[i]:9.4f}  mag={mag[i]:.4f}")

ipk=cos.argmax()
# divergence onset: first epoch after the peak where cos drops below 0.9
after=np.where((ep>ep[ipk]) & (cos<0.9))[0]
onset=ep[after[0]] if len(after) else None
print(f"\npeak cos={cos[ipk]:.4f} @ epoch {ep[ipk]};  final cos={cos[-1]:.4f} @ epoch {ep[-1]}")
print(f"divergence onset (cos<0.9 after peak): epoch {onset}")

out=HERE.parent/"results"; out.mkdir(exist_ok=True)
fig,ax=plt.subplots(1,3,figsize=(13.5,4))
for a,(v,lbl) in zip(ax,[(cos,'cosine similarity'),(mse,'relative MSE'),(mag,'magnitude ratio')]):
    a.plot(ep,v,'o-',ms=3,color='C0'); a.set_xscale('log')
    a.set_xlabel('epoch'); a.set_ylabel(lbl); a.grid(alpha=0.3)
    a.axvline(ep[ipk],color='g',ls=':',lw=0.8)
    if onset: a.axvline(onset,color='r',ls='--',lw=0.8)
ax[2].axhline(1.0,color='gray',ls='--',lw=0.8)
fig.suptitle(f'his UNet (kinematic/std), L16 k=0.26, [-1,1] norm, t={T:.0e}',y=1.02)
fig.tight_layout()
pdf=out/"eval_hisnet_L16_k0.26.pdf"; fig.savefig(pdf,dpi=150,bbox_inches='tight')
np.savez(str(pdf).replace('.pdf','.npz'),epochs=ep,cos_sim=cos,cos_sem=cse,
         rel_mse=mse,mag_ratio=mag,phi_min=gmin,phi_max=gmax,t_eval=T,sigma=SIGMA)
print(f"saved {pdf}")
