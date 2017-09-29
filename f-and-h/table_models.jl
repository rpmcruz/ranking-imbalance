using Formatting
using StatsBase
using Images
include("scores.jl")
include("stats/paired_ttest.jl")

function truncate(s::String)
    if length(s) >= 15
        s = string(s[1:13], "...")
    end
    s
end

models = (
    "SVOR",
    "PvS SVOR 1",
    "PvS SVOR 0.1",
    "PvS SVOR 0.01",
    "PvS SVM",
    #"PvS SVOR 0.001",
    #"PvS SVOR 10",
    #"PvS SVOR 100",
    #"PvS SVOR 1000",
    "RankSVM",
    "PvS RankSVM 1",
    "PvS RankSVM 0.1",
    "PvS RankSVM 0.01",
    "PvS RankSVM",
    #"PvS RankSVM 0.001",
    #"PvS RankSVM 10",
    #"PvS RankSVM 100",
    #"PvS RankSVM 1000",
)

hrow1 = "\\multicolumn{1}{c|}{SVOR} & \\multicolumn{4}{c|}{F\\&H w/ SVM} & \\multicolumn{1}{c|}{RankSVM} & \\multicolumn{4}{c|}{F\\&H w/ RankSVM}"
hrow2 = "\\multicolumn{1}{c|}{} & \$\\lambda{=}1\$ & \$\\lambda{=}0.1\$ & \$\\lambda{=}0.01\$ & \$\\lambda{=}0\$ & \\multicolumn{1}{c|}{} & \$\\lambda{=}1\$ & \$\\lambda{=}0.1\$ & \$\\lambda{=}0.01\$ & \$\\lambda{=}0\$"
hsep = "rrrrr|rrrrr"

metrics = (
    ("MAE", mean_absolute_error, -1),
    ("AMAE", average_mean_absolute_error, -1),
    ("MMAE", maximum_mean_absolute_error, -1),
    ("Acc", accuracy_score, +1),
    ("meanF1", mean_f1_score, +1),
    ("spearman", corspearman, +1),
    ("kendall", corkendall, +1),
)

# use same order as table-data IR column
o = sortperm(readcsv("table-data.csv")[:,3])
datasets = readdir("../../data")[o]

for (metric_name, metric_fn, optimize) in metrics
    println(metric_name)
    scores = zeros(length(datasets), length(models))
    bests = zeros(Bool, length(datasets), length(models))

    for (i, dataset) in enumerate(datasets)
        folds = zeros(30, length(models))
        for fold in 0:29
            ts_filename = "../../data/$(dataset)/matlab/test_$(dataset).$(fold)"
            ts = readdlm(ts_filename, ' ')
            yts = Array{Int,1}(ts[:, end])

            for (j, model) in enumerate(models)
                yp_filename = "../out/$(dataset)-$(model)-fold$(fold).csv"
                if !isfile(yp_filename)
                    println("$(yp_filename) not found [ignoring]")
                    continue
                end
                yp = Array{Int,1}(readdlm(yp_filename)[:, 1])
                score = metric_fn(yts, yp)
                scores[i,j] += score / 30
                folds[fold+1,j] = score
            end
        end

        best = findmax(scores[i,:] * optimize)[2]
        for j in 1:length(models)
            p = ttest_paired(folds[:,j] * optimize, folds[:,best] * optimize)
            if !(p < 0.05)  # cannot reject
                bests[i,j] = true
            end
        end
    end

    # save csv file
    writecsv("table-$(metric_name).csv", scores)

    # save latex tabular file
    open("table-$(metric_name).tex", "w") do f
        write(f, """\\documentclass{standalone}
\\begin{document}
\\begin{tabular}{|l|""")
        write(f, hsep)
        write(f, "|}\n\\hline\n&\n")
        write(f, hrow1)
        write(f, "\\\\\n&")
        write(f, hrow2)
        write(f, "\\\\\n")
        write(f, "\\hline\n")
        for i in 1:length(datasets)
            write(f, truncate(datasets[i]))
            write(f, " & ")
            line = scores[i,:]
            _line = [if isnan(v) "---" else format(v, precision=2) end for v in line]
            _line = [if bold "\\textbf{$(v)}" else v end for (v, bold) in zip(_line, bests[i,:])]
            write(f, join(_line, " & "))
            write(f, "\\\\\n")
        end
        write(f, "\n\\hline\n")
        write(f, "Average & ")
        write(f, join([format(v, precision=2) for v in meanfinite(scores,1)], " & "))
        write(f, "\\\\\n")
        write(f, "Winner & ")
        write(f, join([@sprintf("%d\\%%", v) for v in mean(bests,1)*100], " & "))
        write(f, "\\\\\n")
        write(f, """\\hline
\\end{tabular}
\\end{document}""")
    end
end
