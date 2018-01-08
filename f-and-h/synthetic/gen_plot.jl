include("gen.jl")

K = 4
N = 50
params = Dict(
    :IMBALANCE => true, :INITIAL_ANGLE => 0, :COARSE_INC_ANGLE => 30,
    :ORTHO_ERR => 0.2)

epsilon = 0.8
X, y = gen(K; N=N, DIR_ERR=epsilon, params...)

markers = [:circle, :rect, :utriangle, :diamond, :hexagon, :pentagon]
markers_size = [2, 1.5, 2, 2, 2, 2]*1.6
linetypes = [:solid, :dash, :solid, :dashdot, :dot, :dashdotdot]
line_widths = [1, 1, 3]*0.5
lim = (minimum(X)-0.1, maximum(X)+0.1)

using Plots
scatter(X[:, 1], X[:, 2], color=:white, shape=markers[y],
        markersize=markers_size[y], label="", legend=:none, xlim=lim,
        ylim=lim, size=(300, 150), tickfont=Plots.Font("times", 6, :hcenter, :vcenter, 0.0, RGB{U8}(0.0,0.0,0.0)), legendfont=Plots.Font("times", 6, :hcenter, :vcenter, 0.0, RGB{U8}(0.0,0.0,0.0)))

