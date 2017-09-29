abstract BaseModel

type PrefixVsSuffix <: BaseModel
    base_model ::BaseModel

    # internal
    models     ::Array{BaseModel}

    PrefixVsSuffix(base_model) =
        new(base_model, [])
end

function has_param(self::PrefixVsSuffix, param::String)
    has_param(self.base_model, param)
end

function set_param(self::PrefixVsSuffix, param::String, value::Float64)
    set_param(self.base_model, param, value)
end

function fit(self::PrefixVsSuffix, X::Array{Float64,2}, y::Array{Int64})
    @assert size(X,1) == length(y)
    nclasses = length(unique(y))
    self.models = Array{BaseModel}(nclasses-1)
    for k in 1:nclasses-1
        m = deepcopy(self.base_model)
        _y = Array{Int}(y .> k)
        fit(m, X, _y)
        self.models[k] = m
    end
    self, NaN
end

function predict(self::PrefixVsSuffix, X::Array{Float64,2})
    yp = ones(Int, size(X,1))
    for k in 1:length(self.models)
        yp += predict(self.models[k], X)
    end
    yp
end
