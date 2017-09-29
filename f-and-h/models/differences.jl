function build_differences(X::Array{Float64,2}, y::Array{Int,1}, nclasses::Int)
    if nclasses == 2
        classes = 0:1
    else
        classes = 1:nclasses
    end
    count = [sum(k .== y) for k in classes]
    @assert all(count .!= 0)

    # N: number of obs from all combinations
    # K: number of combinations
    N = 0
    for ki in 1:nclasses
        for kj in ki+1:nclasses
            N += 2*count[ki]*count[kj]
        end
    end
    K = Int(nclasses*(nclasses-1)/2)

    dX = zeros(N, size(X,2))
    dy = zeros(Int, N)
    w = zeros(N)
    i = 1
    for ki in 1:nclasses
        Xi = X[y .== classes[ki],:]
        for kj in ki+1:nclasses
            Xj = X[y .== classes[kj],:]

            n = 2*count[ki]*count[kj]
            _w = N/(K*n)

            for ii in 1:size(Xi,1)
                for jj in 1:size(Xj,1)
                    dX[i,:] = Xi[ii,:] - Xj[jj,:]
                    dy[i] = 0
                    w[i] = _w
                    i += 1
                    dX[i,:] = Xj[jj,:] - Xi[ii,:]
                    dy[i] = 1
                    w[i] = _w
                    i += 1
                end
            end
        end
    end
    @assert i-1 == N
    (dX, dy, w)
end
