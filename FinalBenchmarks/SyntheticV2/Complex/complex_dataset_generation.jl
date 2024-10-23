using Plots
using JLD2
using Distributions
using Random
using StatsPlots
using Plots.PlotMeasures

"""Generate a noisy trendy sinusoid of length T samples, trend m, period tau and corrupted 
with noise of standard deviation eta.
"""
function trendysine(T::Int=100, 
    eta::Union{Number, Tuple{Number, Number}, Vector}=0.0, 
    m::Union{Number, Tuple{Number, Number}, Vector}=3.0,
    phi::Union{Number, Tuple{Number, Number}, Vector}=(0, 2π),
    tau::Union{Number, Tuple{Number, Number}, Vector}=20.0)

    t = 0:1:(T-1)
    tau_val = tau isa Vector ? rand(tau) : tau isa Float64 ? tau : rand(Uniform(tau...))
    phi_val = phi isa Vector ? rand(phi) : phi isa Float64 ? phi : rand(Uniform(phi...))
    m_val = m isa Vector ? rand(m) : m isa Float64 ? m : rand(Uniform(m...))
    eta_val = eta isa Vector ? rand(eta) : eta isa Float64 ? eta : rand(Uniform(eta...))

    x = sin.((2π .* t) ./ tau_val .+ phi_val) .+ m_val/T .* t .+ eta_val * randn(T)

    info = Dict(:T => T, :eta => eta_val, :tau => tau_val, :m => m_val, :phi => phi_val)
    return x, info 
end
"""
Construct a dataset of ntrain training samples and ntest test samples.\n
For each parameter - eta, m, phi, tau:
- Scalar -> Fix to a constant value
- Tuple (a, b) -> Sample from a continuous uniform distribution val ~ U(a, b)
- Vector [a, b, c, ..., d] -> Sample from a discrete uniform distribution with outcomes [a, b, c, ..., d]
"""
function construct_dataset(T::Int=100, ntrain::Int=100, ntest::Int=100, 
    eta::Union{Number, Tuple{Number, Number}, Vector}=0.0, 
    m::Union{Number, Tuple{Number, Number}, Vector}=3.0,
    phi::Union{Number, Tuple{Number, Number}, Vector}=(0, 2π),
    tau::Union{Number, Tuple{Number, Number}, Vector}=20.0;
    state::Union{Nothing, Int}=nothing,
    plot_param_dists::Bool=true,
    save::Bool=false
    )

    if !isnothing(state)
        Random.seed!(state)
    end
    X_train = Matrix{Float64}(undef, ntrain, T)
    y_train = zeros(Int64, ntrain)
    X_test = Matrix{Float64}(undef, ntest, T)
    y_test = zeros(Int64, ntest)
    train_metadata = Vector{Dict}(undef, ntrain)
    test_metadata = Vector{Dict}(undef, ntest)
    for i in 1:ntrain
        X_train[i, :], train_metadata[i] = trendysine(T, eta, m, phi, tau)
    end
    for i in 1:ntest
        X_test[i, :], test_metadata[i] = trendysine(T, eta, m, phi, tau)
    end

    if plot_param_dists
        pal = palette(:tab10)
        p_eta_tr = histogram([train_metadata[i][:eta] for i in 1:length(train_metadata)],
            xlabel="eta", ylabel="Count", title="train", label="", c=pal[1])
        p_tau_tr = histogram([train_metadata[i][:tau] for i in 1:length(train_metadata)],
            xlabel="tau", ylabel="Count", title="train", label="", c=pal[1])
        p_m_tr = histogram([train_metadata[i][:m] for i in 1:length(train_metadata)],
            xlabel="m", ylabel="Count", title="train", label="", c=pal[1])
        p_phi_tr = histogram([train_metadata[i][:phi] for i in 1:length(train_metadata)],
            xlabel="phi", ylabel="Count", title="train", label="", c=pal[1])
        train_ps = [p_eta_tr, p_tau_tr, p_m_tr, p_phi_tr]

        p_eta_te = histogram([test_metadata[i][:eta] for i in 1:length(test_metadata)], 
            xlabel="eta", ylabel="Count", title="test", label="", c=pal[2])
        p_tau_te = histogram([test_metadata[i][:tau] for i in 1:length(test_metadata)], 
            xlabel="tau", ylabel="Count", title="test", label="", c=pal[2])
        p_m_te = histogram([test_metadata[i][:m] for i in 1:length(test_metadata)],
            xlabel="m", ylabel="Count", title="test", label="", c=pal[2])
        p_phi_te = histogram([test_metadata[i][:phi] for i in 1:length(test_metadata)],
            xlabel="phi", ylabel="Count", title="test", label="", c=pal[2])
        test_ps = [p_eta_te, p_tau_te, p_m_te, p_phi_te]
        
        all_ps = vcat(train_ps, test_ps)
        p = plot(all_ps..., lower_margin=5mm, left_margin=5mm, upper_margin=5mm)
        display(p)
    end

    if save
        fname = filename(eta, m, phi, tau)
        @save fname X_train y_train X_test y_test train_metadata test_metadata
    end

    return (X_train, y_train), (X_test, y_test), (train_metadata, test_metadata)

end

function param_str(param)
    if param isa Number
        return "_"*string(param)
    elseif param isa Tuple
        return "_cont_range_" * string(first(param)) * "_" * string(last(param))
    elseif param isa Vector
        return "_disc_range_" * string(length(param))
    else
        error("Unsupported parameter type.")
    end
end

function filename(eta, m, phi, tau)
    eta_str = param_str(eta)
    m_str = param_str(m)
    phi_str = param_str(phi)
    tau_str = param_str(tau)
    fname = "eta$(eta_str)_m$(m_str)_phi$(phi_str)_tau$(tau_str).jld2"
    return fname
end
