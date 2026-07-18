# Pareto plot: accuracy (|bias|) vs speed (s per unit physical time) for EM / ETD / SOSRA.
using Plots, Printf
gr(); default(fontfamily="sans-serif", framestyle=:box, grid=true, gridalpha=0.15, dpi=150)
cd(@__DIR__)

# |bias|% at N_acc=256 (±0.235%); s/t single-field from bench_three (warmed up)
em    = [(0.005,0.271,0.03417),(0.010,0.717,0.01701),(0.020,1.599,0.00855)]
etd   = [(0.020,0.246,0.00558),(0.050,0.488,0.00228),(0.100,0.633,0.00115)]
sosra = [(0.010,0.243,0.03143),(0.020,0.301,0.02040),(0.050,4.180,0.00628)]
staterr = 0.235  # % bias statistical error (N_acc=256)

plt = plot(xlabel="wall time per unit physical time  (s/t)   ← faster | slower →",
    ylabel="|⟨φ²⟩ bias vs HMC|  (%)", title="Model B: accuracy vs speed (L=128, κ=0.2705)",
    xscale=:log10, yscale=:log10, legend=:topright, size=(780,540))
hspan!(plt, [0.05, staterr], c=:gray, alpha=0.12, label="stat. noise floor (±0.235%)")
for (data,name,mk,col) in ((em,"EM (fixed dt)",:circle,"#0072B2"),
                           (etd,"spectral ETD",:diamond,"#E69F00"),
                           (sosra,"SOSRA",:utriangle,"#009E73"))
    plot!(plt, [d[3] for d in data], [d[2] for d in data], marker=mk, ms=8, lw=2, c=col, label=name)
    for (dt,b,s) in data
        annotate!(plt, s, b*1.18, text(@sprintf("dt=%.3g",dt), 6, :center, :gray30))
    end
end
savefig(plt, "spectral/fig_bench_three_pareto.png")
println("written spectral/fig_bench_three_pareto.png")
