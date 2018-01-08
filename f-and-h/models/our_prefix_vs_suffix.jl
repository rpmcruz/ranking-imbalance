abstract BaseModel

type OurPrefixVsSuffix <: BaseModel
    first_model::BaseModel
    base_model ::BaseModel

    # internal
    models     ::Array{BaseModel}

    OurPrefixVsSuffix(first_model, base_model) =
        new(first_model, base_model, [])
end

function has_param(self::OurPrefixVsSuffix, param::String)
    has_param(self.first_model, param)
end

function set_param(self::OurPrefixVsSuffix, param::String, value::Float64)
    set_param(self.first_model, param, value)
end

function fit(self::OurPrefixVsSuffix, X::Array{Float64,2}, y::Array{Int64}, train_first_model=true::Bool)
    @assert size(X,1) == length(y)
    nclasses = maximum(y)
    if train_first_model
        fit(self.first_model, X, y)
    end
    bs, ws = get_weights(self.first_model)
    self.models = Array{BaseModel}(nclasses-1)
    for k in 1:nclasses-1
        m = deepcopy(self.base_model)
        _y = Array{Int}(y .> k)
        fit_reg(m, X, _y, ws, bs[k])
        self.models[k] = m
    end
    self, NaN
end

function predict(self::OurPrefixVsSuffix, X::Array{Float64,2})
    yp = ones(Int, size(X,1))
    for k in 1:length(self.models)
        yp += predict(self.models[k], X)
    end
    yp
end
