using StatsBase

function f(i, yi, y, k, dp_matrix, w)
    if i > length(y) || yi > k
        return 0
    end

    if dp_matrix[i,yi] != -1
        return dp_matrix[i,yi]
    end

    error = w[y[i],yi]
    if yi == k
        dp_matrix[i,yi] =
            error +
            f(i + 1, yi, y, k, dp_matrix, w)
    else
        dp_matrix[i,yi] =
            min(error +
                f(i + 1, yi, y, k, dp_matrix, w),
                f(i, yi + 1, y, k, dp_matrix, w))
    end
    dp_matrix[i,yi]
end

function _decide_thresholds(scores, y, k, w)
    function traverse_matrix(dp_matrix, w)
        (n, k) = size(dp_matrix)
        (i, yi) = (1, 1)
        ret = []
        while i+1 <= n && yi+1 <= k
            current = dp_matrix[i,yi]
            keep = dp_matrix[i+1,yi]
            error = w[y[i],yi]
            if abs((current - error) - keep) < 1e-5
                i += 1
            else
                push!(ret, i)
                yi += 1
            end
        end
        ret
    end

    dp_matrix = -ones(length(y), k)
    f(1, 1, y, k, dp_matrix, w)
    path = traverse_matrix(dp_matrix, w)

    #return scores[path]
    # return midpoints:
    [(scores[p]+scores[max(1,p-1)])/2 for p in path]
end

function decide_thresholds(scores, y, k, strategy)
    if strategy == "uniform"
        w = 1-eye(k)
    elseif strategy == "inverse"
        w = reshape(repmat(length(y) ./ (k*(counts(y)+1)), k), k, k)
        w .*= 1-eye(k)  # remove diagonal
    elseif strategy == "absolute"
        w = [abs(i-j) for i in 1:k, j in 1:k]
    else
        error("No such threshold strategy: ", strategy)
    end
    _decide_thresholds(scores, y, k, w)
end
