# goal here is to evaluate the time complexity w.r.t. training set size
using JLD2
using Random
using Distributions
using BenchmarkTools
using StatsBase
include("../../MLJIntegration/MLJ_integration.jl");


# make function to generate synthetic data to train on (for interpolation?)
# takes in the target training set size
function GenerateSimpleTrendySine(T::Int = 100; eta::Float64 = 0.1, m::Float64 = 3.0,
        phi::Float64 = 0.0, tau::Float64 = 20.0)

    t = 0:1:(T-1)
    signal = sin.((2π .* t ) ./ tau .+ phi) .+ m/T .* t .+ eta * randn(T)

    return signal

end

function GenerateSyntheticDataset(T::Int, ntrain::Int, ntest::Int)
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
# fix the MPS params as we are only interested in the scaling behaviour w.r.t dataset size
Rdtype = Float64
verbosity = 0
test_run = false
track_cost = false
nsweeps = 5
chi_max = 25
eta = 0.1
d = 3

mps = MPSClassifier(nsweeps=nsweeps, chi_max=chi_max, eta=eta, d=d, 
    encoding=:Legendre_No_Norm, exit_early=exit_early, init_rng=4567)

ntrials = 10
sizes = [25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# rows are trials, columns are sizes
results_mat = Matrix{Float64}(undef, ntrials, length(sizes))
for (sidx, s) in enumerate(sizes)
    # iterate over each train set size
    ts = Vector{Float64}(undef, ntrials)
    for i = 1:ntrials
        # keep test set size fixed to 100 and vary train set size
        X_train, y_train, X_test, y_test = GenerateSyntheticDataset(100, s, 100)
        X_train = MLJ.table(X_train)
        X_test = MLJ.table(X_test)
        y_train = coerce(y_train, OrderedFactor)
        y_test = coerce(y_test, OrderedFactor)
        mach = machine(mps, X_train, y_train)
        t = @elapsed MLJ.fit!(mach)
        ts[i] = t
    end
    results_mat[:, sidx] = ts
end

# jldopen("training_set_size_time_complexity.jld2", "w") do f
#     f["results_mat"] = results_mat
# end

# plot results
mean_times = mean(results_mat, dims=1)[1, :]
std_times = std(results_mat, dims=1)[1, :]
p = plot(sizes, mean_times, legend=:none, c=:lightblue, lw=2)
scatter!(sizes, mean_times, legend=:none, title="Training Set Size Time Complexity",
   xlabel="training set size", ylabel="mean training time (s)", c=:lightblue,
    yerr=std_times, xscale=:log10, yscale=:log10, minorgrid=true)
#savefig("training_set_size_time_complexity_log.svg")

# repeat for time-series length
mps = MPSClassifier(nsweeps=nsweeps, chi_max=chi_max, eta=eta, d=d, 
    encoding=:Legendre_No_Norm, exit_early=exit_early, init_rng=4567)
lengths = [25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
results_mat_lengths = Matrix{Float64}(undef, ntrials, length(lengths))
for (lidx, l) in enumerate(lengths)
    # iterate over time series length
    println("TESTING T = $l")
    ts = Vector{Float64}(undef, ntrials)
    for i = 1:ntrials
        # keep test set size fixed and train set size fixed, vary time series length
        X_train, y_train, X_test, y_test = GenerateSyntheticDataset(l, 50, 50)
        X_train = MLJ.table(X_train)
        X_test = MLJ.table(X_test)
        y_train = coerce(y_train, OrderedFactor)
        y_test = coerce(y_test, OrderedFactor)
        mach = machine(mps, X_train, y_train)
        t = @elapsed MLJ.fit!(mach)
        ts[i] = t
    end
    results_mat_lengths[:, lidx] = ts
end

# plot results
mean_times_length = mean(results_mat_lengths, dims=1)[1, :]
std_times_length = std(results_mat_lengths, dims=1)[1, :]
p2 = plot(lengths, mean_times_length, legend=:none, c=:red, lw=2,
    xscale=:log10, yscale=:log10)
scatter!(lengths, mean_times_length, legend=:none, title="Time Series Length Time Complexity",
   xlabel="time series length", ylabel="mean training time (s)", c=:red,
    yerr=std_times, xscale=:log10, yscale=:log10, minorgrid=true)
#savefig("time_series_length_time_complexity_log.svg")