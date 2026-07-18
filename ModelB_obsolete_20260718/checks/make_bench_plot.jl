# Pareto plot for the L=128 solver benchmark (data from bench_L128.log, MC ref <phi^2>=3.0636).
using Plots, Printf
gr()
default(fontfamily = "sans-serif", framestyle = :box, grid = true, gridalpha = 0.15, dpi = 150)
cd(@__DIR__)

mcref = 3.0636
bias(p2) = abs(p2 / mcref - 1) * 100

# (label, walltime s, <phi^2>, marker offset)
em = [("EM dt=0.001", 126.6, 3.0543), ("EM dt=0.0025", 59.4, 3.0410),
    ("EM dt=0.005", 36.5, 3.0273), ("EM dt=0.01", 26.9, 2.9654)]
sra = [("SOSRA tol=0.5", 62.9, 2.9789), ("SOSRA tol=0.2", 135.9, 3.0202),
    ("SRA1 tol=0.5", 42.7, 2.9073)]
heun = [("Heun dt=0.005", 42.9, 2.9829), ("Heun dt=0.01", 30.1, 2.9140)]
lam = [("LambaEM tol=0.05", 39.1, 2.9700)]

plt = plot(xlabel = "wall time for t=600 at L=128  (s)", ylabel = "|<phi^2> bias vs MC|  (%)",
    title = "Model B solver Pareto, L=128, T=8", legend = :topright, size = (760, 520),
    ylims = (0, 5.6))
# statistical uncertainty band of a single run (EM vs EMZ same-config spread ~0.4%)
hspan!(plt, [0, 0.4], c = :gray, alpha = 0.12, label = "single-run stat. noise")
plot!(plt, [p[2] for p in em], [bias(p[3]) for p in em], marker = :circle, ms = 7, lw = 2,
    c = "#0072B2", label = "EM (fixed dt)")
scatter!(plt, [p[2] for p in sra], [bias(p[3]) for p in sra], marker = :diamond, ms = 8,
    c = "#E69F00", label = "SOSRA / SRA1 (adaptive)")
scatter!(plt, [p[2] for p in heun], [bias(p[3]) for p in heun], marker = :utriangle, ms = 8,
    c = "#009E73", label = "EulerHeun (fixed dt)")
scatter!(plt, [p[2] for p in lam], [bias(p[3]) for p in lam], marker = :square, ms = 7,
    c = "#CC79A7", label = "LambaEM (adaptive)")
for (lab, t, p2) in vcat(em, sra, heun, lam)
    if t > 110
        annotate!(plt, t - 3, bias(p2) + 0.13, text(lab, 7, :right, :gray30))
    else
        annotate!(plt, t + 3, bias(p2) + 0.13, text(lab, 7, :left, :gray30))
    end
end
savefig(plt, "fig_solver_pareto_L128.png")
println("written fig_solver_pareto_L128.png")
