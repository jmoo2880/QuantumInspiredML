# goal here is to evaluate the time complexity w.r.t. training set size

using JLD2
using Random
using Distributions
using StatsBase
include("../MLJ_integration/MLJ_integration.jl");


# make function to generate synthetic data to train on (for interpolation?)
# takes in the target training set size
function GenerateSimpleTrendySine(T::Int = 100; eta::Float64 = 0.1, m::Float64 = 3.0,
        phi::Float64 = 0.0, tau::Float64 = 20.0)

    t = 0:1:(T-1)
    signal = sin.((2π .* t ) ./ tau .+ phi) .+ m/T .* t .+ eta * randn(T)

    return signal

end

function GenerateSyntheticDataset(T::Int = 100, ntrain::Int, ntest::Int)
    ntot = ntrain + ntest
    dataset = Matrix{Float64}(undef, ntot, T)
    for i in 1:ntot
        phi = rand(Uniform(0, 2π))
        dataset[i, :] = GenerateSimpleTrendySine(T; phi=phi)
    end
    train_idxs = sample(collect(1:ntot), ntrain, replace=false)
    test_idxs = setdiff(collect(1:ntot), train_idxs)
    X_train = dataset[train_idxs, :]
    y_train = Int.(zeros(ntrain))
    X_test = dataset[test_idxs, :]
    y_test = Int.(zeros(ntest))

    return X_train, y_train, X_test, y_test

end
# make function to train mps with fixed params and log the time
# loops over different dataset sizes

