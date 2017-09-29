#using MLBase

include("models/svm.jl")
include("models/ranksvm.jl")
include("models/ranksvm2.jl")
include("models/prefix_vs_suffix.jl")
include("models/our_prefix_vs_suffix.jl")
include("models/svor.jl")
include("scores.jl")

models = (
    ("SVOR", SVOR(0.01)),
    ("RankSVM", RankSVM2(SVM(0.01), "absolute")),
    ("PvS SVM", PrefixVsSuffix(SVM(0.01))),
    ("PvS SVOR 0.001", OurPrefixVsSuffix(SVOR(0.01), SVM(0.001))),
    ("PvS SVOR 0.01", OurPrefixVsSuffix(SVOR(0.01), SVM(0.01))),
    ("PvS SVOR 0.1", OurPrefixVsSuffix(SVOR(0.01), SVM(0.1))),
    ("PvS SVOR 1", OurPrefixVsSuffix(SVOR(0.01), SVM(0.1))),
    ("PvS SVOR 10", OurPrefixVsSuffix(SVOR(0.01), SVM(10))),
    ("PvS SVOR 100", OurPrefixVsSuffix(SVOR(0.01), SVM(100))),
    ("PvS SVOR 1000", OurPrefixVsSuffix(SVOR(0.01), SVM(1000))),
    ("PvS RankSVM", PrefixVsSuffix(RankSVM(SVM(0.01)))),
    ("PvS RankSVM 0.001", OurPrefixVsSuffix(RankSVM2(SVM(0.01), "absolute"), RankSVM(SVM(0.001)))),
    ("PvS RankSVM 0.01", OurPrefixVsSuffix(RankSVM2(SVM(0.01), "absolute"), RankSVM(SVM(0.01)))),
    ("PvS RankSVM 0.1", OurPrefixVsSuffix(RankSVM2(SVM(0.01), "absolute"), RankSVM(SVM(0.1)))),
    ("PvS RankSVM 1", OurPrefixVsSuffix(RankSVM2(SVM(0.01), "absolute"), RankSVM(SVM(1)))),
    ("PvS RankSVM 10", OurPrefixVsSuffix(RankSVM2(SVM(0.01), "absolute"), RankSVM(SVM(10)))),
    ("PvS RankSVM 100", OurPrefixVsSuffix(RankSVM2(SVM(0.01), "absolute"), RankSVM(SVM(100)))),
    ("PvS RankSVM 1000", OurPrefixVsSuffix(RankSVM2(SVM(0.01), "absolute"), RankSVM(SVM(1000)))),
)

CROSS_VALIDATE = false

function cross_validation(model, X, y, fit_fn::Function, folds::Int)
    n = size(X, 1)
    scores = MLBase.cross_validate(
        ix -> fit_fn(model, X[ix,:], y[ix])[1],
        (m, ix) -> average_mean_absolute_error(y[ix], predict(m, X[ix,:])),
        n,
        MLBase.StratifiedKfold(y, folds)
    )
    mean(scores)
end

LAMBDAS = logspace(-3, 0, 4)
LAMBDA_FOLDS = 5

function fit_search_lambda(model, X, y)
    best_lambda = 0
    best_score = Inf  # minimize
    if has_param(model, "lambda")
        for lambda in LAMBDAS
            set_param(model, "lambda", lambda)
            avg_score = cross_validation(model, X, y, fit, LAMBDA_FOLDS)
            if avg_score < best_score  # minimize
                best_lambda = lambda
                best_score = avg_score
            end
        end
        set_param(model, "lambda", best_lambda)
    end
    println("best lambda:", best_lambda)
    fit(model, X, y)
end

function run_file(dataset)
    println("* $(dataset)")
    for fold in 0:29
        tr_filename = "../../data/$(dataset)/matlab/train_$(dataset).$(fold)"
        #if fold == 0 && countlines(tr_filename) > 1000
        #    break
        #end
        tr = readdlm(tr_filename, ' ')
        Xtr = tr[:, 1:end-1]
        ytr = Array{Int,1}(tr[:, end])

        cross_validate = CROSS_VALIDATE
        if CROSS_VALIDATE && minimum(counts(ytr)) < LAMBDA_FOLDS
            println("Warning: lambda search impossible for $(dataset)/$(fold) [using fixed lambda]")
            cross_validate = false
        end

        ts_filename = "../../data/$(dataset)/matlab/test_$(dataset).$(fold)"
        ts = readdlm(ts_filename, ' ')
        Xts = ts[:, 1:end-1]
        yts = Array{Int,1}(ts[:, end])

        mu = mean(Xtr,1)
        sd = std(Xtr,1)
        Xtr = (Xtr .- mu) ./ sd
        Xts = (Xts .- mu) ./ sd

        for (i, (name, model)) in enumerate(models)
            println(name)
            outname = "../out/$(dataset)-$(name)-fold$(fold).csv"
            if isfile(outname)
                println("Already exists... [skipping]")
                continue
            end
            #if cross_validate
            #    model, = fit_search_lambda(model, Xtr, ytr)
            #else
            #    model, = fit(model, Xtr, ytr)
            #end
            #yp = predict(model, Xts)
            yp = 1 + (rand(size(Xts,1)) .> 0.5)
            writecsv(outname, yp)
        end
    end
    scores
end
