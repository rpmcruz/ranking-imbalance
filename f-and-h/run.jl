addprocs(Sys.CPU_CORES-1)  # workers
@everywhere include("run_file.jl")

datasets = readdir("../../data")
datasets = [dataset for dataset in datasets if isdir("../../data/$(dataset)")]
pmap(run_file, datasets)
