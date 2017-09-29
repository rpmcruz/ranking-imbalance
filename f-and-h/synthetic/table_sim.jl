table = readcsv("sim-AMAE.csv")

hrow1 = "\\multicolumn{1}{c|}{SVOR} & \\multicolumn{4}{c|}{F\\&H w/ SVM} & \\multicolumn{1}{c|}{RankSVM} & \\multicolumn{4}{c|}{F\\&H w/ RankSVM}"
hrow2 = "\\multicolumn{1}{c|}{} & \$\\lambda{=}1\$ & \$\\lambda{=}0.1\$ & \$\\lambda{=}0.01\$ & \$\\lambda{=}0\$ & \\multicolumn{1}{c|}{} & \$\\lambda{=}1\$ & \$\\lambda{=}0.1\$ & \$\\lambda{=}0.01\$ & \$\\lambda{=}0\$"
hsep = "rrrrr|rrrrr"

cols1 = ("\$\\Delta a\$", "", "", "", "\$\\varepsilon\$", "", "", "\$\\varepsilon'\$", "", "")
multirow1 = (4, 0, 0, 0, 3, 0, 0, 3, 0, 0)
cols2 = ("0", "15", "\\textbf{30}", "45", "0", "\\textbf{0.2}", "0.4", "0", "\\textbf{0.2}", "0.4")

@assert length(cols1) == length(cols2)
@assert length(cols1) == size(table,1)

metrics = (
    ("MAE", nothing, -1),
    ("AMAE", nothing, -1),
    ("MMAE", nothing, -1),
    ("Acc", nothing, +1),
    ("meanF1", nothing, +1),
    ("spearman", nothing, +1),
    ("kendall", nothing, +1),
)

for (metric_name, _, metric_optim) in metrics
    open("table-sim-$(metric_name).tex", "w") do f
        write(f, """\\documentclass{standalone}
\\usepackage{multirow,makecell}
\\begin{document}
\\begin{tabular}{|lr|""")
        write(f, hsep)
        write(f, "|}\n\\hline\n&&\n")
        write(f, hrow1)
        write(f, "\\\\\n&&")
        write(f, hrow2)
        write(f, "\\\\\n")
        write(f, "\\hline\n")

        for i in 1:length(cols1)
            if i > 1 && multirow1[i] > 0
                write(f, "\\hline\n")
            end
            if multirow1[i] > 0
                write(f, @sprintf("\\multirowcell{%d}[0pt][l]{%s} & ", multirow1[i], cols1[i]))
            else
                write(f, " & ")
            end
            write(f, cols2[i])
            write(f, " & ")
            #best_th = maximum(table[i,:]*metric_optim) * 0.95
            best = maximum(table[i,:]*metric_optim)*metric_optim
            values = [if v == best @sprintf("\\textbf{%.3f}", v) else @sprintf("%.3f", v) end for v in table[i,:]]
            write(f, join(values, " & "))
            write(f, "\\\\\n")
        end
        write(f, "\n\\hline\n")
        write(f, """\\end{tabular}
\\end{document}""")
    end
end
