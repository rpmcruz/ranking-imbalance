include("../models/svm.jl")
include("../models/ranksvm.jl")
include("../models/ranksvm2.jl")
include("../models/prefix_vs_suffix.jl")
include("../models/our_prefix_vs_suffix.jl")
include("../models/svor.jl")
include("../scores.jl")

dataset = ARGS[1]

println("Doing stress test for $(dataset)")

filename = "../../../data/$(dataset)/matlab/train_$(dataset).0"
df = readdlm(filename, ' ')
Xtr = df[:, 1:end-1]
ytr = Array{Int,1}(df[:, end])

filename = "../../../data/$(dataset)/matlab/test_$(dataset).0"
df = readdlm(filename, ' ')
Xts = df[:, 1:end-1]
yts = Array{Int,1}(df[:, end])

# Compare how both families handle stress

first_model1 = SVOR(0.01)
first_model2 = RankSVM2(SVM(0.01), "absolute")

fhall_model1 = (first_model, lambda) -> OurPrefixVsSuffix(first_model, SVM(lambda))
fhall_model2 = (first_model, lambda) -> OurPrefixVsSuffix(first_model, RankSVM(SVM(lambda)))

FRACTIONS = 0.1:0.05:1
LAMBDAS = logspace(-3, +3, 7)
NSIM = 50

sum_amae1 = zeros(length(FRACTIONS), length(LAMBDAS))
sum2_amae1 = zeros(length(FRACTIONS), length(LAMBDAS))

sum_amae2 = zeros(length(FRACTIONS), length(LAMBDAS))
sum2_amae2 = zeros(length(FRACTIONS), length(LAMBDAS))

ny = counts(ytr)
cumy = [0;cumsum(ny)[1:end-1]]

for (fi, fraction) in enumerate(FRACTIONS)
    println("fraction: $(fraction)")

    new_n = round(fraction .* ny)
    new_n = Array{Int}(new_n + Array{Int}(new_n .== 0))

    for i in 1:NSIM
        println("* nsim: $(i)")
        sampling = [randperm(nn)[1:n]+cy for (nn, n, cy) in zip(ny, new_n, cumy)]
        sampling = vcat(sampling...)
        Xtr_ = Xtr[sampling,:]
        ytr_ = ytr[sampling]

        first_model, = fit(first_model1, Xtr_, ytr_)

        for (j, lambda) in enumerate(LAMBDAS)
            fh, = fit(fhall_model1(first_model, lambda), Xtr, ytr, false)
            amae = average_mean_absolute_error(yts, predict(fh, Xts))
            sum_amae1[fi,j] += amae
            sum2_amae1[fi,j] += amae^2

            fh, = fit(fhall_model2(first_model, lambda), Xtr, ytr, false)
            amae = average_mean_absolute_error(yts, predict(fh, Xts))
            sum_amae2[fi,j] += amae
            sum2_amae2[fi,j] += amae^2
        end
    end
end

writecsv("$(dataset)_sum_amae1.csv", sum_amae1)
writecsv("$(dataset)_sum2_amae1.csv", sum2_amae1)
writecsv("$(dataset)_sum_amae2.csv", sum_amae2)
writecsv("$(dataset)_sum2_amae2.csv", sum2_amae2)

#-
sum_amae1 = readcsv("$(dataset)_sum_amae1.csv")
sum2_amae1 = readcsv("$(dataset)_sum2_amae1.csv")
sum_amae2 = readcsv("$(dataset)_sum_amae2.csv")
sum2_amae2 = readcsv("$(dataset)_sum2_amae2.csv")
-#

ix = findmin(sum_amae1, 2)[2]
lambdas1 = LAMBDAS[Array{Int}(floor((ix-1) / size(sum_amae1,1))+1)]*0.9
sum_amae1 = sum_amae1[ix]
sum2_amae1 = sum2_amae1[ix]

ix = findmin(sum_amae2, 2)[2]
lambdas2 = LAMBDAS[Array{Int}(floor((ix-1) / size(sum_amae2,1))+1)]*1.1
sum_amae2 = sum_amae2[ix]
sum2_amae2 = sum2_amae2[ix]

avg_amae1 = sum_amae1 ./ NSIM
avg_amae2 = sum_amae2 ./ NSIM

std_amae1 = sqrt(abs(sum2_amae1 - ((sum_amae1.^2)/NSIM)) / (NSIM-1))
std_amae2 = sqrt(abs(sum2_amae2 - ((sum_amae2.^2)/NSIM)) / (NSIM-1))

using Plots

p = plot(FRACTIONS, avg_amae1, yerr=std_amae1, label="SVOR", color=:black, style=:solid, width=2, xlabel="Fraction used by the primary model", ylabel="AMAE", tickfont=Plots.Font("times", 6, :hcenter, :vcenter, 0.0, RGB{U8}(0.0,0.0,0.0)), legendfont=Plots.Font("times", 6, :hcenter, :vcenter, 0.0, RGB{U8}(0.0,0.0,0.0)))
plot!(p, FRACTIONS, avg_amae2, yerr=std_amae2, label="RankSVM", color=:black, style=:dash, width=2)
savefig(p, "stress-amae.pdf")

p = plot(FRACTIONS, lambdas1, label="SVOR", color=:black, style=:solid, width=2, xlabel="Fraction used by the primary model", ylabel="\\lambda", yticks=LAMBDAS, ylim=(LAMBDAS[1], LAMBDAS[end]), yscale=:log10, tickfont=Plots.Font("times", 6, :hcenter, :vcenter, 0.0, RGB{U8}(0.0,0.0,0.0)), legendfont=Plots.Font("times", 6, :hcenter, :vcenter, 0.0, RGB{U8}(0.0,0.0,0.0)))
plot!(p, FRACTIONS, lambdas2, label="RankSVM", color=:black, style=:dash, width=2)
savefig(p, "stress-lambdas.pdf")

