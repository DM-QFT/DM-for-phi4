# Produce summary figures for the ModelB.jl verification + T_c search.
using DelimitedFiles, Plots, Printf, Statistics
gr()
default(fontfamily = "sans-serif", framestyle = :box, grid = true, dpi = 150)
cd(@__DIR__)

# ---- 1. Binder crossing (fine scan) ----
fine = readdlm("mc_scan_fine.dat", comments = true, comment_char = '#')
plt1 = plot(xlabel = "T", ylabel = "Binder U4", title = "Binder cumulant crossing (MC)",
    legend = :bottomleft)
for L in (32, 64, 128)
    sel = fine[:, 1] .== L
    plot!(plt1, fine[sel, 2], fine[sel, 5], marker = :circle, label = "L = $L")
end
savefig(plt1, "fig_binder_crossing.png")

# ---- 2. chi peak ----
plt2 = plot(xlabel = "T", ylabel = "chi", title = "Susceptibility (MC)", legend = :topleft, yscale = :log10)
for L in (32, 64, 128)
    sel = fine[:, 1] .== L
    plot!(plt2, fine[sel, 2], fine[sel, 6], marker = :circle, label = "L = $L")
end
savefig(plt2, "fig_chi.png")

# ---- 3. S(k) comparison in the disordered phase: Model B vs equilibrium MC ----
mb6 = readdlm("mblong_T6.000_L32_Sk.dat", comments = true, comment_char = '#')
mc6 = readdlm("mc_sk_T6_L32.dat", comments = true, comment_char = '#')
p3a = plot(xlabel = "k", ylabel = "S(k)", title = "T = 6, L = 32 (t up to 2000)",
    xscale = :log10, yscale = :log10, legend = :bottomleft)
plot!(p3a, mc6[:, 1], mc6[:, 2], marker = :circle, label = "equilibrium MC")
plot!(p3a, mb6[:, 1], mb6[:, 2], marker = :diamond, ls = :dash, label = "Model B (ModelB.jl)")
mb8 = readdlm("mblong_T8.000_L64_Sk.dat", comments = true, comment_char = '#')
mc8 = readdlm("mc_sk_T8_L64.dat", comments = true, comment_char = '#')
p3b = plot(xlabel = "k", ylabel = "S(k)", title = "T = 8, L = 64 (t up to 1000)",
    xscale = :log10, yscale = :log10, legend = :bottomleft)
plot!(p3b, mc8[:, 1], mc8[:, 2], marker = :circle, label = "equilibrium MC")
plot!(p3b, mb8[:, 1], mb8[:, 2], marker = :diamond, ls = :dash, label = "Model B (ModelB.jl)")
savefig(plot(p3a, p3b, size = (950, 400)), "fig_Sk_compare.png")

# ---- 4. <phi^2> vs T: MC curve + Model B points ----
coarse = readdlm("mc_scan_coarse.dat", comments = true, comment_char = '#')
sel32 = coarse[:, 1] .== 32
plt4 = plot(xlabel = "T", ylabel = "<phi^2>", title = "<phi^2> vs T", legend = :topright)
plot!(plt4, coarse[sel32, 2], coarse[sel32, 7], marker = :circle, label = "MC L=32 (canonical)")
mbT = Float64[]; mbp2 = Float64[]
for f in filter(f -> occursin(r"^mb_T[\d.]+_L64_obs\.dat$", f), readdir())
    T = parse(Float64, match(r"T([\d.]+)_", f).captures[1])
    d = readdlm(f, comments = true, comment_char = '#')
    nh = size(d, 1) ÷ 2
    push!(mbT, T); push!(mbp2, mean(d[nh+1:end, 3]))
end
scatter!(plt4, mbT, mbp2, marker = :star5, ms = 8, label = "Model B L=64 (fixed M=0)")
vline!(plt4, [4.45], ls = :dot, c = :gray, label = "T_c estimate")
savefig(plt4, "fig_phi2_vs_T.png")

# ---- 5. snapshots: below and above T_c ----
snapmin, snapmax = -3.5, 3.5
h1 = heatmap(readdlm("mb_T3.000_L64_final.dat"), c = :RdBu, clims = (snapmin, snapmax),
    title = "Model B, T=3 (t=500): phase separation", aspect_ratio = 1, size = (450, 400))
h2 = heatmap(readdlm("mb_T6.000_L64_final.dat"), c = :RdBu, clims = (snapmin, snapmax),
    title = "Model B, T=6 (t=500): disordered", aspect_ratio = 1, size = (450, 400))
savefig(plot(h1, h2, size = (950, 400)), "fig_snapshots.png")

# ---- 6. Binder crossing estimate (pairwise linear interpolation) ----
function crossT(fine, La, Lb)
    sa = sortslices(fine[fine[:, 1].==La, :], dims = 1, by = r -> r[2])
    sb = sortslices(fine[fine[:, 1].==Lb, :], dims = 1, by = r -> r[2])
    dU = sa[:, 5] .- sb[:, 5]
    for i in 1:length(dU)-1
        if dU[i] < 0 && dU[i+1] >= 0 || dU[i] > 0 && dU[i+1] <= 0
            t1, t2 = sa[i, 2], sa[i+1, 2]
            return t1 + (t2 - t1) * abs(dU[i]) / (abs(dU[i]) + abs(dU[i+1]))
        end
    end
    return NaN
end
@printf("Binder crossings: 32/64 -> T=%.3f ; 64/128 -> T=%.3f ; 32/128 -> T=%.3f\n",
    crossT(fine, 32, 64), crossT(fine, 64, 128), crossT(fine, 32, 128))
println("figures written")
