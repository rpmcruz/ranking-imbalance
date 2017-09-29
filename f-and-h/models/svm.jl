abstract BaseModel

REG_BIAS = true

type SVM <: BaseModel
    lambda::Float64
    fit_intercept::Bool

    # internal
    w::Array{Float64,1}
    b::Float64

    SVM(lambda, fit_intercept=true) = new(lambda, fit_intercept, [], 0)
end

function has_param(self::SVM, param::String)
    param == "lambda"
end

function set_param(self::SVM, param::String, value::Float64)
    self.lambda = value
end

function compute_loss(self::SVM, X::Array{Float64,2}, y::Array{Int,1}, sw::Array{Float64,1})
    loss = 0
    dw = zeros(size(self.w))
    db = 0
    den = sum(sw)

    margins = max(0, 1 - y.*(X*self.w+self.b))
    loss += sum(sw .* margins) / den

    for i in 1:size(X,1)
        if margins[i] > 0
            dw += -y[i] * sw[i] * X[i,:] / den
            if self.fit_intercept
                db += -(sw[i]*y[i]) / den
            end
        end
    end

    # regularization
    loss += self.lambda * sum(self.w.^2)
    dw += 2 * self.lambda * self.w

    loss, dw, db
end

function fit(self::SVM, X::Array{Float64,2}, y::Array{Int,1}, sw::Nullable{Array{Float64,1}}=Nullable{Array{Float64,1}}())
    self.w = zeros(size(X,2))
    self.b = 0

    _sw = isnull(sw) ? ones(length(y)) : get(sw)
    _y = y*2-1
    loss = 0
    oldloss = Inf
    for it in 1:1e4
        loss, dw, db = compute_loss(self, X, _y, _sw)
        eta = 1 / (self.lambda*it)
        self.w -= eta*dw
        self.b -= eta*db
        if abs(loss-oldloss) < 1e-3  # stopping criteria
            break
        end
    end
    self, loss
end

function compute_loss_reg(self::SVM, X::Array{Float64,2}, y::Array{Int,1}, ow::Array{Float64,1})
    loss = 0
    dw = zeros(size(self.w))
    db = 0
    den = length(y)

    margins = max(0, 1 - y.*(X*self.w+self.b))
    loss += sum(margins) / den

    for i in 1:size(X,1)
        if margins[i] > 0
            dw += -y[i] * X[i,:] / den
            if self.fit_intercept
                db += -y[i] / den
            end
        end
    end

    # regularization
    loss += self.lambda * sum((self.w-ow).^2)
    dw += 2 * self.lambda * (self.w-ow)

    loss, dw, db
end

function fit_reg(self::SVM, X::Array{Float64,2}, y::Array{Int,1}, ow::Array{Float64,1}, ob::Float64)
    self.w = copy(ow)
    if REG_BIAS
        self.b = ob
    else
        self.b = 0
    end

    _y = y*2-1
    loss = 0
    oldloss = Inf
    for it in 1:1e4
        loss, dw, db = compute_loss_reg(self, X, _y, ow)
        eta = 1 / (self.lambda*it)
        self.w -= eta*dw
        self.b -= eta*db
        if abs(loss-oldloss) < 1e-3  # stopping criteria
            break
        end
    end
    self, loss
end

function decision_function(self::SVM, X::Array{Float64,2})
    X*self.w+self.b
end

function predict(self::SVM, X::Array{Float64,2})
    Array{Int,1}(X*self.w+self.b .> 0)
end
