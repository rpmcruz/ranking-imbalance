using Distributions

# H0: x1 = x2
# H1: x1 < x2

function ttest_paired(x1, x2)
    x = x1 - x2
    n = length(x)
    mu = mean(x)
    stderr = sqrt(var(x)/n)
    if stderr == 0
        return 1
    end
    t = mu / stderr
    df = n-1
    cdf(TDist(df), t)
end
