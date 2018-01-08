using StatsBase
using Formatting

function overlap1(X, y)  # rate of positives having negative as closest neighbor
    diff = 0
    #X = (X .- mean(X,1)) ./ std(X,1)
    for i in 1:size(X,1)
        # (cannot be optimized because we need correct indices)
        other = 0
        d = [norm(X[i,:] - X[j,:]) for j in 1:size(X,1) if i != j]
        diff += y[i] != y[findmin(d)[2]]
    end
    diff / size(X,1)
end

function mean_IR(X, y)
    K = maximum(y)
    N = counts(y)
    IR_per_class = [sum(N[1:K .!= k]) / ((K-1)*N[k]) for k in 1:K]
    mean(IR_per_class)
end

metrics = (
    ("N", (X, y) -> size(X,1), 0),
    ("Features", (X, y) -> size(X,2), 0),
    ("K", (X, y) -> length(unique(y)), 0),
    ("\\textbf{IR}", mean_IR, 2),
    ("OR", overlap1, 2),
    #("IR2", (X, y) -> maximum(counts(y))/minimum(counts(y)), 2),
)

datasets = readdir("../../data")
table = zeros(length(datasets), length(metrics))

for (i, dataset) in enumerate(datasets)
    filename1 = "../../data/$(dataset)/matlab/train_$(dataset).0"
    filename2 = "../../data/$(dataset)/matlab/test_$(dataset).0"
    df = vcat(readdlm(filename1, ' '), readdlm(filename2, ' '))
    X = df[:, 1:end-1]
    y = Array{Int,1}(df[:, end])

    table[i,:] = [metric(X, y) for (_, metric) in metrics]
end

# save csv file
writecsv("table-data.csv", table)

o = sortperm(table[:,4])
datasets = datasets[o]
table = table[o,:]  # order by IR

# save latex tabular file
open("table-data.tex", "w") do f
    write(f, """\\documentclass{standalone}
\\begin{document}
\\begin{tabular}{|l|""")
    write(f, repeat("r", length(metrics)))
    write(f, "|}\n\\hline\nDataset&\n")
    write(f, join([m[1] for m in metrics], " & "))
    write(f, "\\\\\n\\hline\n")
    for j in 1:length(datasets)
        write(f, datasets[j])
        write(f, " & ")
        line = [format(v, precision=prec) for (v, (_, __, prec)) in zip(table[j,:], metrics)]
        write(f, join(line, " & "))
        write(f, "\\\\")
    end
    write(f, """\\hline
\\end{tabular}
\\end{document}""")
end
