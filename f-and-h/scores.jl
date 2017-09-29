function accuracy_score(y, yp)
    sum(y .== yp) / length(y)
end

function mean_absolute_error(y, yp)
    mean(abs(y - yp))
end

function mean_absolute_error_per_class(y, yp)
    classes = unique(y)
    [mean_absolute_error(y[y .== k], yp[y .== k]) for k in classes]
end

function maximum_mean_absolute_error(y, yp)
    maximum(mean_absolute_error_per_class(y, yp))
end

function average_mean_absolute_error(y, yp)
    mean(mean_absolute_error_per_class(y, yp))
end

function f1_score(y, yp)
    TP = sum((y .== 1) & (yp .== 1))
    FN = sum((y .== 1) & (yp .== 0))
    FP = sum((y .== 0) & (yp .== 1))
    (2*TP) / (2*TP + FN + FP)
end

function mean_f1_score(y, yp)
    mean([f1_score(y .== k, yp .== k) for k in unique(y)])
end
