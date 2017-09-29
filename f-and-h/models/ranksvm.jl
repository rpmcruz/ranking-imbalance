include("svm.jl")
include("differences.jl")
include("../scores.jl")

abstract BaseModel

type RankSVM <: BaseModel
    svm::SVM
    th::Float64

    RankSVM(svm) = new(svm, 0)
end

function has_param(self::RankSVM, param::String)
    has_param(self.svm, param)
end

function set_param(self::RankSVM, param::String, value::Float64)
    set_param(self.svm, param, value)
end

function choose_threshold(self::RankSVM, X::Array{Float64,2}, y::Array{Int64,1})
    ss = sort(decision_function(self, X))
    # TODO: this could be optimized, like we did for IJCNN
    best_f1 = 0
    best_th = 0
    for (i, s) in enumerate(ss)
        yp = ss .>= s
        f1 = f1_score(y, yp)
        if f1 > best_f1
            best_f1 = f1
            best_th = (ss[i]+ss[max(1, i-1)])/2
        end
    end
    best_th
end

function fit(self::RankSVM, X::Array{Float64,2}, y::Array{Int64,1})
    dX, dy = build_differences(X, y, 2)
    self.svm.fit_intercept = false
    fit(self.svm, dX, dy)
    self.th = choose_threshold(self, X, y)
    self, nothing
end

function fit_reg(self::RankSVM, X::Array{Float64,2}, y::Array{Int,1}, ow::Array{Float64,1}, ob::Float64)
    dX, dy = build_differences(X, y, 2)
    self.svm.fit_intercept = false
    fit_reg(self.svm, dX, dy, ow, 0.)
    self.th = choose_threshold(self, X, y)
    self, nothing
end

function decision_function(self::RankSVM, X::Array{Float64,2})
    decision_function(self.svm, X)
end

function predict(self::RankSVM, X::Array{Float64,2})
    Array{Int,1}(decision_function(self.svm, X) .>= self.th)
end
