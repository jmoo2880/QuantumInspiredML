include("MLJ_integration.jl")
using JLD2

dloc =  "Data/ecg200/datasets/ecg200.jld2"
f = jldopen(dloc, "r")
    X_train = read(f, "X_train")
    y_train = read(f, "y_train")
    X_test = read(f, "X_test")
    y_test = read(f, "y_test")
close(f)

X_train = MLJ.table(X_train)
X_test = MLJ.table(X_test)
y_train = coerce(y_train, OrderedFactor)
y_test = coerce(y_test, OrderedFactor)

Rdtype = Float64

verbosity = 0
test_run = false
track_cost = false
#
encoding = legendre(project=false)
encode_classes_separately = false
train_classes_separately = false
exit_early=false

#encoding = Basis("Legendre")

nsweeps=10
chi_max=29
eta=0.1
d=3

mps = MPSClassifier(nsweeps=nsweeps, chi_max=chi_max, eta=eta, d=d, encoding=:Legendre_No_Norm, exit_early=exit_early, init_rng=4567)

mach = machine(mps, X_train, y_train)
MLJ.fit!(mach)

yhat = MLJ.predict(mach, X_test)
@show MLJ.accuracy(yhat, y_test)
