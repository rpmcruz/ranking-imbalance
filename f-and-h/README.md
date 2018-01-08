# Binary Ranking for Ordinal Class Imbalance

Authors: Ricardo Cruz • Kelwin Fernandes • Joaquim F. Pinto Costa • María Pérez Ortiz • Jaime S. Cardoso

- Article **(being reviewed)**
- [Source code](https://github.com/rpmcruz/ranking-imbalance/f-and-h/src)
- Datasets were obtained from UCI repository, as used in [Pérez-Ortiz et al. Graph-Based Approaches for Over-Sampling in the Context of Ordinal Regression](http://ieeexplore.ieee.org/document/6940273)

Usage:

1. `julia run.jl` will execute `run_file.jl` for each dataset in `data/`. You might have to tweak `run_file.jl` to adjust the structure of the datasets to your case.
2. `table*.jl` output the tables used in the paper from the results created by the previous command.
3. The most important contents are: (a) the various models implemented from scratch inside `models/` and (b) `synthetic` for generation of synthetic ordinal data in a highly parametrized fashion.
