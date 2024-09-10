# goal here is to evaluate the time complexity w.r.t. training set size
using JLD2
using Random
using Distributions
using BenchmarkTools
using StatsBase
using CurveFit
include("../../MLJIntegration/MLJ_integration.jl");

savepath = "FinalBenchmarks/TimeComplexity/";
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
_, slope_size = linear_fit(log10.(sizes), log10.(mean_times))
p = plot(sizes, mean_times, legend=:none, c=:blue, lw=2)
scatter!(sizes, mean_times, legend=:none, title="Training Set Size Time Complexity, m = $(round(slope_size, digits=4))",
   xlabel="training set size", ylabel="mean training time (s)", c=:blue,
    yerr=std_times, xscale=:log10, yscale=:log10, minorgrid=true)
# savefig("FinalBenchmarks/TimeComplexity/training_set_size_time_complexity_log.svg")


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
_, slope_length = linear_fit(log10.(lengths), log10.(mean_times_length))
p2 = plot(lengths, mean_times_length, legend=:none, c=:red, lw=2,
    xscale=:log10, yscale=:log10)
scatter!(lengths, mean_times_length, legend=:none, title="Time Series Length Time Complexity, m = $(round(slope_length, digits=4))",
   xlabel="time series length", ylabel="mean training time (s)", c=:red,
    yerr=std_times, xscale=:log10, yscale=:log10, minorgrid=true)
#savefig("FinalBenchmarks/TimeComplexity/time_series_length_time_complexity_log.svg")

# now vary d
Rdtype = Float64
verbosity = 0
test_run = false
track_cost = false
nsweeps = 5
chi_max = 25
eta = 0.1
ds = collect(2:1:20)
results_mat_d = Matrix{Float64}(undef, ntrials, length(ds))
for (didx, d) in enumerate(ds)
    # iterate over each d 
    println("TESTING d = $d")
    mps = MPSClassifier(nsweeps=nsweeps, chi_max=chi_max, eta=eta, d=d, 
        encoding=:Legendre_No_Norm, exit_early=exit_early, init_rng=4567)
    ts = Vector{Float64}(undef, ntrials)
    for i = 1:ntrials
        # keep test set size fixed to 50 and vary d
        X_train, y_train, X_test, y_test = GenerateSyntheticDataset(100, 50, 50)
        X_train = MLJ.table(X_train)
        X_test = MLJ.table(X_test)
        y_train = coerce(y_train, OrderedFactor)
        y_test = coerce(y_test, OrderedFactor)
        mach = machine(mps, X_train, y_train)
        t = @elapsed MLJ.fit!(mach)
        ts[i] = t
    end
    results_mat_d[:, didx] = ts
end

# plot results
mean_times_chimean_times_d = mean(results_mat_d, dims=1)[1, :]
std_times_d = std(results_mat_d, dims=1)[1,:]
_, slope_d = linear_fit(log10.(ds), log10.(mean_times_d))
p3 = plot(ds, mean_times_d, legend=:none, title="Local dimension d Time Complexity, m = $(round(slope_d, digits=4))",
    xlabel="local dimension d", ylabel="mean training time (s)", yscale=:log10,
    xscale=:log10, c=:magenta, lw=2, minorgrid=true)
scatter!(ds, mean_times_d, legend=:none, c=:magenta, yerr=std_times_d)
#savefig(savepath*"d_time_complexity_log.svg")


# vary χmax - more subtle because chi < chi_max if insufficient training data
Rdtype = Float64
verbosity = 0
test_run = false
track_cost = false
nsweeps = 5
d = 3
eta = 0.1
chis = collect(4:4:100)
results_mat_chi = Matrix{Float64}(undef, ntrials, length(chis))
for (chidx, chi) in enumerate(chis)
    # iterate over each d 
    println("TESTING chi = $chi")
    mps = MPSClassifier(nsweeps=nsweeps, chi_max=chi, eta=eta, d=d, 
        encoding=:Legendre_No_Norm, exit_early=exit_early, init_rng=4567)
    ts = Vector{Float64}(undef, ntrials)
    for i = 1:ntrials
        # keep test set size fixed to 50 and vary chi max
        # need to ensure enough training data so that maximum bond dimension is used
        X_train, y_train, X_test, y_test = GenerateSyntheticDataset(100, 100, 50)
        X_train = MLJ.table(X_train)
        X_test = MLJ.table(X_test)
        y_train = coerce(y_train, OrderedFactor)
        y_test = coerce(y_test, OrderedFactor)
        mach = machine(mps, X_train, y_train)
        # check the mps max bond dim
        t = @elapsed MLJ.fit!(mach)
        ts[i] = t
        chi_max_actual = maximum([maxdim(mach.fitresult[3][i]) for i in 1:length(mach.fitresult[3])])
        println("Expected χ = $chi | Actual χ = $chi_max_actual")
    end
    results_mat_chi[:, chidx] = ts
end

mean_times_chi = mean(results_mat_chi, dims=1)[1,:]
std_times_chi = std(results_mat_chi, dims=1)[1,:]
_, slope_chi = linear_fit(log10.(chis), log10.(mean_times_chi))
p4 = plot(chis, mean_times_chi, legend=:none, title="Bond dimension χ Time Complexity, m = $(round(slope_chi, digits=4))",
    xlabel="bond dimension χ", ylabel="mean training time (s)", yscale=:log10,
    xscale=:log10, c=:orange, lw=2, minorgrid=true)
scatter!(chis, mean_times_chi, c=:orange, yerr=std_times_chi)
savefig(savepath*"chi_time_complexity_log.svg")