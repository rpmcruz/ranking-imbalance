include("models/svm.jl")
include("models/ranksvm.jl")
include("models/ranksvm2.jl")
include("models/prefix_vs_suffix.jl")
include("models/our_prefix_vs_suffix.jl")
include("models/svor.jl")
include("scores.jl")
include("synthetic/gen.jl")

# Compare how both families handle stress

first_model1 = SVOR(0.01)
first_model2 = RankSVM2(SVM(0.01), "absolute")

fhall_model1 = (first_model, lambda) -> OurPrefixVsSuffix(first_model, SVM(lambda))
fhall_model2 = (first_model, lambda) -> OurPrefixVsSuffix(first_model, RankSVM(SVM(lambda)))

K = 4
N = 250
params = Dict(:IMBALANCE => true, :INITIAL_ANGLE => 0, :ORTHO_ERR => 0.2)

EPSILONS = linspace(0, 0.8, 9)
LAMBDAS = logspace(-3, 0, 4)
NSIM = 50

sum_amae1 = zeros(length(EPSILONS), length(LAMBDAS))
sum2_amae1 = zeros(length(EPSILONS), length(LAMBDAS))

sum_amae2 = zeros(length(EPSILONS), length(LAMBDAS))
sum2_amae2 = zeros(length(EPSILONS), length(LAMBDAS))

for (ei, epsilon) in enumerate(EPSILONS)
    println("epsilon: $(epsilon)")

    for i in 1:NSIM
        println("* nsim: $(i)")

        Xtr1, ytr1 = gen(K; N=N, DIR_ERR=epsilon, params...)
        Xtr2, ytr2 = gen(K; N=N, DIR_ERR=0.2, params...)
        Xts, yts = gen(K; N=N, DIR_ERR=0.2, params...)
        ny = counts(ytr2)
        cumy = [0;cumsum(ny)[1:end-1]]

        first_model, = fit(first_model1, Xtr1, ytr1)

        for (j, lambda) in enumerate(LAMBDAS)
            fh, = fit(fhall_model1(first_model, lambda), Xtr2, ytr2, false)
            amae = average_mean_absolute_error(yts, predict(fh, Xts))
            sum_amae1[ei,j] += amae
            sum2_amae1[ei,j] += amae^2

            fh, = fit(fhall_model2(first_model, lambda), Xtr2, ytr2, false)
            amae = average_mean_absolute_error(yts, predict(fh, Xts))
            sum_amae2[ei,j] += amae
            sum2_amae2[ei,j] += amae^2
        end
    end
end

writecsv("sum_amae1.csv", sum_amae1)
writecsv("sum2_amae1.csv", sum2_amae1)
writecsv("sum_amae2.csv", sum_amae2)
writecsv("sum2_amae2.csv", sum2_amae2)

#-
sum_amae1 = readcsv("sum_amae1.csv")
sum2_amae1 = readcsv("sum2_amae1.csv")
sum_amae2 = readcsv("sum_amae2.csv")
sum2_amae2 = readcsv("sum2_amae2.csv")
-#

ix = findmin(sum_amae1, 2)[2]
lambdas1 = LAMBDAS[Array{Int}(floor((ix-1) / size(sum_amae1,1))+1)]*0.9
sum_amae1 = sum_amae1[ix]
sum2_amae1 = sum2_amae1[ix]

ix = findmin(sum_amae2, 2)[2]
lambdas2 = LAMBDAS[Array{Int}(floor((ix-1) ./ size(sum_amae2,1))+1)]*1.1
sum_amae2 = sum_amae2[ix]
sum2_amae2 = sum2_amae2[ix]

avg_amae1 = sum_amae1 / NSIM
avg_amae2 = sum_amae2 / NSIM

std_amae1 = sqrt(abs(sum2_amae1 - ((sum_amae1.^2)/NSIM)) / (NSIM-1))
std_amae2 = sqrt(abs(sum2_amae2 - ((sum_amae2.^2)/NSIM)) / (NSIM-1))

using Plots

p = plot(EPSILONS, avg_amae1, yerr=std_amae1, label="SVOR", color=:black, style=:solid, width=2, xlabel="Epsilon used by the primary model", ylabel="AMAE", tickfont=Plots.Font("times", 6, :hcenter, :vcenter, 0.0, RGB{U8}(0.0,0.0,0.0)), legendfont=Plots.Font("times", 6, :hcenter, :vcenter, 0.0, RGB{U8}(0.0,0.0,0.0)))
plot!(p, EPSILONS, avg_amae2, yerr=std_amae2, label="RankSVM", color=:black, style=:dash, width=2)
savefig(p, "stress-amae.pdf")

p = plot(EPSILONS, lambdas1, label="SVOR", color=:black, style=:solid, width=2, xlabel="Epsilon used by the primary model", ylabel="\\lambda", yticks=LAMBDAS, ylim=(LAMBDAS[1], LAMBDAS[end]), yscale=:log10, tickfont=Plots.Font("times", 6, :hcenter, :vcenter, 0.0, RGB{U8}(0.0,0.0,0.0)), legendfont=Plots.Font("times", 6, :hcenter, :vcenter, 0.0, RGB{U8}(0.0,0.0,0.0)))
plot!(p, EPSILONS, lambdas2, label="RankSVM", color=:black, style=:dash, width=2)
savefig(p, "stress-lambdas.pdf")

