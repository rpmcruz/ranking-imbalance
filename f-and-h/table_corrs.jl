using StatsBase
using Formatting

metrics = ("AMAE", "MMAE")
features = (1, 3, 4)#, 5)

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
dcols = ("N", "p", "IR", "OR")#, "IR2")

# exploratory analysis

d = readcsv("table-data.csv")
for metric in metrics
    println("* $(metric)")
    s = readcsv("table-$(metric).csv")
    for i in features
        println("** $(dcols[i])")
        for j in 1:size(s,2)
            rho = corkendall(d[:,i], s[:,j])
            if abs(rho) >= 0.1
                @printf("\t%-20s %-.3f\n", models[j], rho)
            end
        end
    end
end

println("rho between IR and OR: ", corkendall(d[:,3], d[:,4]))

# generate table

hrow1 = "\\multicolumn{1}{c|}{SVOR} & \\multicolumn{4}{c|}{F\\&H w/ SVM} & \\multicolumn{1}{c|}{RankSVM} & \\multicolumn{4}{c|}{F\\&H w/ RankSVM}"
hrow2 = "\\multicolumn{1}{c|}{} & \$\\lambda{=}1\$ & \$\\lambda{=}0.1\$ & \$\\lambda{=}0.01\$ & \$\\lambda{=}0\$ & \\multicolumn{1}{c|}{} & \$\\lambda{=}1\$ & \$\\lambda{=}0.1\$ & \$\\lambda{=}0.01\$ & \$\\lambda{=}0\$"
hsep = "rrrrr|rrrrr"

for metric in metrics
    open("table-corrs-$(metric).tex", "w") do f
        s = readcsv("table-$(metric).csv")
        write(f, """\\documentclass{standalone}
\\begin{document}
\\begin{tabular}{|l|""")
        write(f, hsep)
        write(f, "|}\n\\hline\n&")
        write(f, hrow1)
        write(f, "\\\\\n&")
        write(f, hrow2)
        write(f, "\\\\\n")
        write(f, "\\hline\n")
        for i in features
            write(f, dcols[i])
            write(f, " & ")
            line = [corkendall(d[:,i], s[:,j]) for j in 1:size(s,2)]
            _line = [format(v, precision=2) for v in line]
            write(f, join(_line, " & "))
            write(f, "\\\\\n")
        end
        write(f, """\\hline
\\end{tabular}
\\end{document}""")
    end
end
