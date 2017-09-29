function proj(x, y, theta_rad)
    c = cos(theta_rad)
    s = sin(theta_rad)
    hcat(x*c-y*s, x*s+y*c)
end

function gen(K=3; N=400, INITIAL_ANGLE=60, COARSE_INC_ANGLE=0, COARSE_DIST=0.2,
        DIR_ERR=0, ORTHO_ERR=0, PERMUT_LABEL_PROB=0, IMBALANCE=false)
    X = zeros(N, 2)
    y = zeros(Int, N)

    theta_rad = INITIAL_ANGLE*pi/180
    i = 1
    b = [0. 0.]
    for k in 1:K
        if k == K
            n = N-i+1  # last gets everything left
        else
            if IMBALANCE
                #n = (2/3) * ((1/3)^(k-1)) * N
                n = N * ((1/2)^k)
            else
                n = N/K
            end
            n = Int(round(n))
        end
        println("class ", k, " n=", n)

        xx = rand(n)/K
        vert_err = (2*rand(n)-1)*ORTHO_ERR
        hort_err = (2*rand(n)-1)*DIR_ERR
        X[i:i+n-1,:] = proj(xx+hort_err, zeros(n)+vert_err, theta_rad) .+ b

        yy = fill(k, n)
        p = rand(n) .< PERMUT_LABEL_PROB
        if k == 0
            yy[p] += 1
        elseif k == K-1
            yy[p] -= 1
        else
            yy[p] += sign(2*rand(sum(p))-1)
        end
        y[i:i+n-1] = yy
        i += n
        b += proj(1/K + COARSE_DIST, 0, theta_rad)
        theta_rad += COARSE_INC_ANGLE*pi/180
    end
    X, y
end
