include("svm.jl")
include("differences.jl")
include("threshold.jl")

type RankSVM2 <: BaseModel
    svm       ::SVM
    strategy  ::String

    # internal
    thresholds::Array{Float64}

    RankSVM2(svm, strategy) =
        new(svm, strategy, [])
end

function has_param(self::RankSVM2, param::String)
    has_param(self.svm, param)
end

function set_param(self::RankSVM2, param::String, value::Float64)
    set_param(self.svm, param, value)
end

function get_weights(self::RankSVM2)
    -self.thresholds, self.svm.w
end

function fit(self::RankSVM2, X::Array{Float64,2}, y::Array{Int64,1})
    K = maximum(y)
    (dX, dy, dw) = build_differences(X, y, K)
    self.svm.fit_intercept = false
    fit(self.svm, dX, dy, Nullable{Array{Float64,1}}(dw))

    scores = decision_function(self.svm, X)
    self.thresholds = decide_thresholds(scores, y, K, self.strategy)
    self, nothing
end

function decision_function(self::RankSVM2, X::Array{Float64,2})
    decision_function(self.svm, X)
end

function predict(self::RankSVM2, X::Array{Float64,2})
    scores = decision_function(self.svm, X)
    [1+sum(s .>= self.thresholds) for s in scores]
end
