include("threshold.jl")
include("../scores.jl")

function weighted_choice(choices)
    total = sum(w for (c, w) in choices)
    r = rand()*total
    upto = 0
    for (c, w) in choices
        if upto + w >= r
            return c
        end
        upto += w
    end
    error("Shouldn't get here")
end

n = 5000
K = 5
noise = 0.30
choices = vcat([(-v, 1. / (2 ^ v)) for v in 1:K], [(v, 1. / (2 ^ v)) for v in 1:K])

scores = sort(rand(n))
thresholds = scores[sort(rand(1:n, K - 1))]
println("thresholds: ", thresholds)

y = [K - sum(s .< thresholds) for s in scores]
y = [min(K, max(1, s + if rand() < noise; weighted_choice(choices) else 0 end)) for s in y]

strategies = ["uniform", "inverse", "absolute"]
# n_samples / (n_classes * np.bincount(y))
for strategy in strategies
    println()
    println("strategy: ", strategy)
    tic()
    learned_thresholds = decide_thresholds(scores, y, K, strategy)
    toc()
    println("\tlearned_threshold: ", learned_thresholds)
    yp = [1 + sum(s .>= learned_thresholds) for s in scores]
    @printf("\taccuracy: %.2f\n", accuracy_score(y, yp))
    @printf("\tf1 score: %.2f\n", mean_f1_score(y, yp))
    @printf("\tmae:      %.2f\n", mean_absolute_error(y, yp))
    @printf("\tamae:     %.2f\n", average_mean_absolute_error(y, yp))
    @printf("\tmmae:     %.2f\n", maximum_mean_absolute_error(y, yp))
end
