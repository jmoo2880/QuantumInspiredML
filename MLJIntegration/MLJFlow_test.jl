include("MLJ_integration.jl")
using JLD2
using MLJFlow
using MLJParticleSwarmOptimization
# bash: run mlflow server

logger = MLJFlow.Logger("http://localhost:5000/api"; experiment_name="test")

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

nsweeps=5
chi_max=29
eta=0.1
d=3

mps = MPSClassifier(nsweeps=nsweeps, chi_max=chi_max, eta=eta, d=d, encoding=:Legendre_No_Norm, exit_early=exit_early, init_rng=4567)

swarm = AdaptiveParticleSwarm(rng=MersenneTwister(0))
r1 = MLJ.range(mps, :eta, lower=0.001, upper=10, scale=:log);
r2 = MLJ.range(mps, :d, lower=2, upper=15)
r3 = MLJ.range(mps, :chi_max, lower=15, upper=50)
r4 = MLJ.range(mps, :exit_early, values=[true,false])
r5 = MLJ.range(mps, :sigmoid_transform, values=[true,false])

self_tuning_mps = TunedModel(
    model=mps,
    resampling=StratifiedCV(nfolds=5, rng=MersenneTwister(1)),
    tuning=swarm,
    range=[r1, r2, r3],
    measure=MLJ.misclassification_rate,
    n=3,
    acceleration=CPUThreads(), # acceleration=CPUProcesses()
    logger=logger
);
mach = machine(self_tuning_mps, Xs, ys)
MLJ.fit!(mach)

yhat = MLJ.predict(mach, X_test)
@show MLJ.accuracy(yhat, y_test)
