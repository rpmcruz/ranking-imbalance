include("../models/svm.jl")
include("../models/ranksvm.jl")
include("../models/ranksvm2.jl")
include("../models/prefix_vs_suffix.jl")
include("../models/our_prefix_vs_suffix.jl")
include("../models/svor.jl")
include("../scores.jl")
include("gen.jl")

USE_RANK = true
FIGURE = 1

K = 4
if FIGURE == 1  # all parallel
    d = Dict(:INITIAL_ANGLE=>60, :ORTHO_ERR=>0.2, :DIR_ERR=>0.2)
else  # orthogoanl
    d = Dict(:INITIAL_ANGLE=>0, :COARSE_INC_ANGLE=>30, :ORTHO_ERR=>0.2, :DIR_ERR=>0.2)
end
X, y = gen(K; N=75, IMBALANCE=true, d...)
Xts, yts = gen(K; N=75, IMBALANCE=true, d...)

mu = mean(X,1)
sd = std(X,1)
X = (X .- mu) ./ sd
Xts = (Xts .- mu) ./ sd

if USE_RANK
    models = (
        ("\\alpha=0", PrefixVsSuffix(RankSVM(SVM(0.01)))),
        ("\\alpha=0.01", OurPrefixVsSuffix(RankSVM2(SVM(0.01), "absolute"), RankSVM(SVM(0.01)))),
        ("\\alpha=1", OurPrefixVsSuffix(RankSVM2(SVM(0.01), "absolute"), RankSVM(SVM(1)))),
        #("\\alpha=100", RankSVM2(SVM(0.01), "absolute")),
    )
else
    models = (
        ("PvS SVM", PrefixVsSuffix(SVM(0.01))),
        ("idem \\alpha=0.01", OurPrefixVsSuffix(SVOR(0.01), SVM(0.01))),
        ("SVOR", SVOR(0.01)),
    )
end

metrics = (
    ("AMAE", average_mean_absolute_error),
    ("MMAE", maximum_mean_absolute_error),
    ("Acc", accuracy_score),
    ("meanF1", mean_f1_score),
    ("spearman", corspearman),
)

function get_coefs(m::SVM)
    m.b, m.w
end
function get_coefs(m::RankSVM)
    -m.th, m.svm.w
end
function get_coefs(m::RankSVM2)
    [(-th, m.svm.w) for th in m.thresholds]
end
function get_coefs(m::Union{PrefixVsSuffix,OurPrefixVsSuffix})
    [get_coefs(m) for m in m.models]
end
function get_coefs(m::SVOR)
    [(b, m.w) for b in m.b]
end

function plot_svm!(p, b::Float64, w::Array{Float64,1}, X::Array{Float64,2}, label::String, linetype::Symbol, linewidth::Float64)
    intercept = -b/w[2]
    slope = -w[1]/w[2]
    #xx = linspace(minimum(X[:, 1]), maximum(X[:, 1]))
    xx = [minimum(X), maximum(X)]
    yy = xx*slope + intercept
    plot!(p, xx, yy, label=label, color=:black, style=linetype, width=linewidth)
end

using Plots

markers = [:circle, :rect, :utriangle, :diamond, :hexagon, :pentagon]
markers_size = [2, 1.5, 2, 2, 2, 2]*1.6
linetypes = [:solid, :dash, :solid, :dashdot, :dot, :dashdotdot]
line_widths = [1, 1, 3]*0.5
lim = (minimum(X)-0.1, maximum(X)+0.1)

p = scatter(X[:, 1], X[:, 2], color=:white, shape=markers[y],
        markersize=markers_size[y], label="", legend=:none, xlim=lim,
        ylim=lim, size=(300, 150), tickfont=Plots.Font("times", 6, :hcenter, :vcenter, 0.0, RGB{U8}(0.0,0.0,0.0)), legendfont=Plots.Font("times", 6, :hcenter, :vcenter, 0.0, RGB{U8}(0.0,0.0,0.0)))

for (i, (name, m)) in enumerate(models)
    println(name)
    fit(m, X, y)
    coefs = get_coefs(m)
    first = true
    for (b, w) in coefs
        label = first ? name : "";
        plot_svm!(p, b, w, X, label, linetypes[i], line_widths[i])
        first = false
    end
    # draw intersection points
    for (b1, w1) in coefs
        for (b2, w2) in coefs
            if w1 != w2
                pt = hcat(w1,w2)' \ vcat(-b1,-b2)
                if minimum(X[:, 1]) <= pt[1] <= maximum(X[:, 1]) &&
                        minimum(X[:, 2]) <= pt[2] <= maximum(X[:, 2])
                    scatter!(p, [pt[1]], [pt[2]], label="", color=:black, markersize=2)
                end
            end
        end
    end
    yp = predict(m, Xts)
    for (metric_name, metric_fn) in metrics
        @printf("%20s %.4f\n", metric_name, metric_fn(y, yp))
    end
end
savefig(p, "synthetic$(FIGURE).pdf")

