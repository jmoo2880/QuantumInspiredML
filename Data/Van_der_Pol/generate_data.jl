using DifferentialEquations
using SciMLBase
using Plots
using StatsPlots
using Plots.PlotMeasures
using Random
using Distributions
using JLD2


function determ_van_der_pol_ode!(du, u, p, t)
    c, k = p
    du[1] = u[2]
    du[2] = c * (1 - u[1]^2)*u[2] - k * u[1]
end

"""
    T: Number of samples
    eta: Measurement noise standard deviation σ
    c: Damping term
    k: Oscillation frequency
    sample_rate: Sampling frequency (Hz)
    n_transients: number of intial samples to discard
"""
function van_der_pol(T::Int, eta::Number=0.0, c::Union{Number, Tuple{Number, Number}, Vector}=1.0, 
    k::Union{Number, Tuple{Number, Number}, Vector}=1.0, sample_rate::Float64=1/6, 
    n_transients::Int=500)
    # generate random initial conditions
    u0 = [rand(Uniform(-1.0, 1.0)), rand(Uniform(-1.0, 1.0))] # x, y
    # determine the time span to simulate
    t_final = ceil((T + n_transients) * sample_rate)
    tspan = (0.0, t_final)
    c_val = c isa Vector ? rand(c) : c isa Float64 ? c : rand(Uniform(c...))
    k_val = k isa Vector ? rand(k) : k isa Float64 ? k : rand(Uniform(c...))
    p = (c_val, k_val)
    problem = ODEProblem(determ_van_der_pol_ode!, u0, tspan, p)
    sol = solve(problem, Tsit5(), saveat=sample_rate)
    x_sol = sol[2, (n_transients+1):(n_transients+T)] # only return x component w/o transients 
    # add gaussian distributed noise
    x_sol .+= rand(Normal(0, eta), length(x_sol))
    # store the info for the samples
    info = Dict(:T => T, :eta => eta, :c => c_val, :k => k_val, :transients => n_transients)
    return x_sol, info 
end

function construct_dataset(T::Int=100, ntrain::Int=100, ntest::Int=100, eta::Float64=0.0, 
    c::Union{Number, Tuple{Number, Number}, Vector}=1.0, 
    k::Union{Number, Tuple{Number, Number}, Vector}=1.0; random_state::Union{Nothing, Int}=nothing, 
    plot_param_dists::Bool=true)
    # pre-allocate data matrices
    if !isnothing(random_state)
        Random.seed!(random_state)
    end
    X_train = Matrix{Float64}(undef, ntrain, T)
    y_train = zeros(Int64, ntrain)
    X_test = Matrix{Float64}(undef, ntest, T)
    y_test = zeros(Int64, ntest)
    train_metadata = Vector{Dict}(undef, ntrain)
    test_metadata = Vector{Dict}(undef, ntest)
    for i in 1:ntrain
        X_train[i, :], train_metadata[i] = van_der_pol(T, eta, c, k)
    end
    for i in 1:ntest
        X_test[i, :], test_metadata[i] = van_der_pol(T, eta, c, k)
    end
    if plot_param_dists
        # Plot the parameter distributions
        pal = palette(:tab10)
        pc_tr = histogram([train_metadata[i][:c] for i in 1:length(train_metadata)], 
            xlabel="c", ylabel="Count", title="train", label="", c=pal[1])
        pk_tr = histogram([train_metadata[i][:k] for i in 1:length(train_metadata)],
            xlabel="k", ylabel="Count", title="train", label="", c=pal[1])
        pc_te = histogram([test_metadata[i][:c] for i in 1:length(test_metadata)], 
            xlabel="c", ylabel="Count", title="test", label="", c=pal[2])
        pk_te = histogram([test_metadata[i][:k] for i in 1:length(test_metadata)], 
            xlabel="k", ylabel="Count", title="test", label="", c=pal[2])
        p = plot(pc_tr, pk_tr, pc_te, pk_te, lower_margin=5mm, left_margin=5mm, upper_margin=5mm)
        display(p)
    end
    return (X_train, y_train), (X_test, y_test), (train_metadata, test_metadata)
end

# dataset1
T = 100
ntrain = 500
ntest = 100
eta = 0.1
c = (0.5, 4.0)
k = (0.5, 4.0)
(X_train, y_train), (X_test, y_test), (train_metadata, test_metadata) = construct_dataset(T, ntrain, ntest, eta, c, k);
jldopen("Data/Van_der_Pol/datasets/vdp_eta_01_c_0.5:4.0_k_0.5:4.0.jld2", "w") do f
    f["X_train"] = X_train
    f["X_test"] = X_test
    f["y_train"] = y_train
    f["y_test"] = y_test
    f["train_meta"] = train_metadata
    f["test_meta"] = test_metadata
end
close(f)