# Using SVOR implementation from (Li and Lin, 2007)

abstract BaseModel

type SVOR <: BaseModel
    lambda::Float64

    # internal
    w::Array{Float64,1}
    b::Array{Float64,1}

    SVOR(lambda) = new(lambda, [], [])
end

function has_param(self::SVOR, param::String)
    param == "lambda"
end

function set_param(self::SVOR, param::String, value::Float64)
    self.lambda = value
end

function get_weights(self::SVOR)
    self.b, self.w
end

function compute_loss(self::SVOR, X::Array{Float64,2}, y::Array{Int,1}, K::Int)
    loss = 0
    dw = zeros(size(self.w))
    db = zeros(size(self.b))
    den = size(X,1)*(K-1)

    for k in 1:K-1
        _y = Array{Int,1}(y .> k)*2-1
        margins = max(0, 1 - _y.*(X*self.w+self.b[k]))
        loss += sum(margins) / den

        for i in 1:size(X,1)
            if margins[i] > 0
                dw += -_y[i] * X[i,:] / den
                db[k] += -_y[i] / den
            end
        end
    end

    # regularization
    loss += self.lambda * sum(self.w.^2)
    dw += 2 * self.lambda * self.w

    loss, dw, db
end

function fit(self::SVOR, X::Array{Float64,2}, y::Array{Int,1})
    K = length(unique(y))
    self.w = zeros(size(X,2))
    self.b = zeros(K-1)

    loss = 0
    oldloss = Inf
    for it in 1:1e4
        loss, dw, db = compute_loss(self, X, y, K)
        eta = 1 / (self.lambda*it)
        self.w -= eta*dw
        self.b -= eta*db
        if abs(loss-oldloss) < 1e-3  # stopping criteria
            break
        end
    end
    self, loss
end

function predict(self::SVOR, X::Array{Float64,2})
    yp = ones(Int, size(X,1))
    xx = X*self.w
    for k in 1:length(self.b)
        yp += xx+self.b[k] .> 0
    end
    yp
end
