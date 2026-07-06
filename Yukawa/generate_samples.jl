# High-quality sample generation for the 2D staggered Yukawa model using the
# verified matrix-free CG HMC. Saves decorrelated configs + observables to JLD2.
#   usage: julia generate_samples.jl g01   (or g03)
include("/work/gw97/w97000/project/DM/Yukawa/YukawaFermionHMC2D.jl")
using JLD2, Statistics, Printf, Random, LinearAlgebra
BLAS.set_num_threads(72)   # measured optimum for the 256x256 chiral inverse (BLAS=36 is ~4% slower)

# integrated autocorrelation time (Madras-Sokal automatic windowing), in trajectories
function tau_int(x)
    n=length(x); xc=x .- mean(x); c0=mean(abs2, xc)
    c0==0 && return 0.5
    tau=0.5
    for t in 1:n-1
        ct=mean(view(xc,1:n-t).*view(xc,t+1:n))/c0
        tau+=ct
        t >= 6*tau && break
    end
    return tau
end
binned_error(x; nb=25) = (bs=div(length(x),nb); bs==0 ? std(x)/sqrt(length(x)) :
    std([mean(x[(b-1)*bs+1:b*bs]) for b in 1:nb])/sqrt(nb))

function generate(label,m2,lambda,g; n_therm,n_gen,save_every,eps,n_steps,seed,outfile)
    N=16; V=N*N; rng=MersenneTwister(seed); phi=0.1 .*randn(rng,N,N)
    for i in 1:n_therm; hmc_step!(rng,phi,eps,n_steps,m2,lambda,g,0.0); end
    nconf=div(n_gen,save_every)
    configs=Array{Float64}(undef,N,N,nconf); Mhist=Vector{Float64}(undef,n_gen)
    chiral=Float64[]; nacc=0; ci=0
    for i in 1:n_gen
        acc,_=hmc_step!(rng,phi,eps,n_steps,m2,lambda,g,0.0); nacc+=acc
        Mhist[i]=abs(mean(phi))
        if i % save_every==0
            ci+=1; configs[:,:,ci]=phi; push!(chiral, abs(chiral_condensate(phi,g,0.0)))
        end
        i % 4000==0 && (@printf("[%s] %d/%d acc=%.3f\n",label,i,n_gen,nacc/i); flush(stdout))
    end
    Mconf=[abs(mean(view(configs,:,:,k))) for k in 1:nconf]; acc=nacc/n_gen; tM=tau_int(Mhist)
    jldsave(outfile; configs,Mhist,Mconf,chiral,m2,lambda,g,mf=0.0,N,eps,n_steps,
            n_therm,save_every,seed,accept=acc,tau_int_M=tM)
    @printf("[%s] saved %d configs -> %s\n",label,nconf,outfile)
    @printf("[%s] acc=%.3f  tau_int(|M|)=%.2f trajs  N_eff(M)~%d\n",label,acc,tM,round(Int,n_gen/(2tM)))
    @printf("[%s] RESULT <|M|>=%.6f +/- %.6f   <|psibar psi|>=%.6f +/- %.6f  (binned)\n",
            label,mean(Mconf),binned_error(Mconf),mean(chiral),binned_error(chiral)); flush(stdout)
end

# Single long chain per parameter set (no multi-seed pooling).
# Longer chain (250k trajectories -> 25k configs) so <|M|> converges to ~0.0003.
set=length(ARGS)>=1 ? ARGS[1] : "g01"
if set=="g01"
    generate("g0.1",-4.00,6.0,0.1; n_therm=5000,n_gen=1000000,save_every=10,
             eps=0.11,n_steps=18,seed=2,outfile="samples/yukawa_g0.1_L16_$(n_gen).jld2")
else
    generate("g0.3", -1.55, 2.4, 0.3; n_therm=5000, n_gen=1000000, save_every=10,
             eps=0.13,n_steps=15,seed=2,outfile="samples/yukawa_g0.3_L16_$(n_gen).jld2")
end
