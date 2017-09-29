addprocs(Sys.CPU_CORES)  # workers

@everywhere begin
include("../models/svm.jl")
include("../models/ranksvm.jl")
include("../models/ranksvm2.jl")
include("../models/prefix_vs_suffix.jl")
include("../models/our_prefix_vs_suffix.jl")
include("../models/svor.jl")
include("../scores.jl")
include("gen.jl")

models = (
    ("SVOR", SVOR(0.01)),
    ("PvS SVOR 1", OurPrefixVsSuffix(SVOR(0.01), SVM(1))),
    ("PvS SVOR 0.1", OurPrefixVsSuffix(SVOR(0.01), SVM(0.1))),
    ("PvS SVOR 0.01", OurPrefixVsSuffix(SVOR(0.01), SVM(0.01))),
    ("PvS SVM", PrefixVsSuffix(SVM(0.01))),

    ("RankSVM", RankSVM2(SVM(0.01), "absolute")),
    ("PvS RankSVM 1", OurPrefixVsSuffix(RankSVM2(SVM(0.01), "absolute"), RankSVM(SVM(1)))),
    ("PvS RankSVM 0.1", OurPrefixVsSuffix(RankSVM2(SVM(0.01), "absolute"), RankSVM(SVM(0.1)))),
    ("PvS RankSVM 0.01", OurPrefixVsSuffix(RankSVM2(SVM(0.01), "absolute"), RankSVM(SVM(0.01)))),
    ("PvS RankSVM", PrefixVsSuffix(RankSVM(SVM(0.01)))),
)

metrics = (
    ("MAE", mean_absolute_error, -1),
    ("AMAE", average_mean_absolute_error, -1),
    ("MMAE", maximum_mean_absolute_error, -1),
    ("Acc", accuracy_score, +1),
    ("meanF1", mean_f1_score, +1),
    ("spearman", corspearman, +1),
    ("kendall", corkendall, +1),
)

K = 4
N = 400

grid_params = (
    # vary k-delta angle
    Dict(:INITIAL_ANGLE => 0, :COARSE_INC_ANGLE => 0, :ORTHO_ERR => 0.2, :DIR_ERR => 0.2),
    Dict(:INITIAL_ANGLE => 0, :COARSE_INC_ANGLE => 15, :ORTHO_ERR => 0.2, :DIR_ERR => 0.2),
    Dict(:INITIAL_ANGLE => 0, :COARSE_INC_ANGLE => 30, :ORTHO_ERR => 0.2, :DIR_ERR => 0.2),
    Dict(:INITIAL_ANGLE => 0, :COARSE_INC_ANGLE => 45, :ORTHO_ERR => 0.2, :DIR_ERR => 0.2),
    # dir error
    Dict(:INITIAL_ANGLE => 0, :COARSE_INC_ANGLE => 30, :ORTHO_ERR => 0.2, :DIR_ERR => 0),
    Dict(:INITIAL_ANGLE => 0, :COARSE_INC_ANGLE => 30, :ORTHO_ERR => 0.2, :DIR_ERR => 0.2),
    Dict(:INITIAL_ANGLE => 0, :COARSE_INC_ANGLE => 30, :ORTHO_ERR => 0.2, :DIR_ERR => 0.4),
    # ortho error
    Dict(:INITIAL_ANGLE => 0, :COARSE_INC_ANGLE => 30, :ORTHO_ERR => 0, :DIR_ERR => 0.2),
    Dict(:INITIAL_ANGLE => 0, :COARSE_INC_ANGLE => 30, :ORTHO_ERR => 0.2, :DIR_ERR => 0.2),
    Dict(:INITIAL_ANGLE => 0, :COARSE_INC_ANGLE => 30, :ORTHO_ERR => 0.4, :DIR_ERR => 0.2),
)

function run(params)
    res = zeros(length(metrics), length(models))

    X, y = gen(K; N=N, IMBALANCE=true, params...)
    Xts, yts = gen(K; N=N, IMBALANCE=true, params...)

    mu = mean(X,1)
    sd = std(X,1)
    X = (X .- mu) ./ sd
    Xts = (Xts .- mu) ./ sd

    for (i, (name, m)) in enumerate(models)
        println("\t", name)
        fit(m, X, y)
        yp = predict(m, Xts)
        for (j, (_, metric_fn, __)) in enumerate(metrics)
            res[j,i] = metric_fn(yts, yp)
        end
    end
    res
end
end  # @everywhere

table = zeros(length(metrics), length(grid_params), length(models))
folds = 30

for (di, params) in enumerate(grid_params)
    println(di, " ", params)
    r = pmap(run, [params for _ in 1:folds])
    table[:,di,:] = mean(r)
end

for (j, (metric_name, _, __)) in enumerate(metrics)
    writecsv("sim-$(metric_name).csv", table[j,:,:])
end
