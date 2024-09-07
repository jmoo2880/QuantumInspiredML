using ITensors
using Random
using QuadGK
using Roots
using Plots, StatsPlots
using StatsBase
using Base.Threads
using KernelDensity, Distributions
using LegendrePolynomials
include("../LogLoss/structs/structs.jl")
include("../LogLoss/encodings.jl")


function get_state(x::Float64, opts::Options)
    """Get the state for a time independent encoding"""
    if !opts.encoding.istimedependent    
        enc_args = []
        state = opts.encoding.encode(x, opts.d, enc_args...)
    else
        error("Expected a time independent encoding.")
    end

    return state

end

function get_state(x::Float64, opts::Options, enc_args::Vector{Vector{Any}}, 
    j::Int)
    """Get the state for a time dependent encoding at site j"""
    if opts.encoding.istimedependent
        enc_args_concrete = convert(Vector{Vector{Vector{Int}}}, enc_args) # https://i.imgur.com/cmFIJmS.png
        state = opts.encoding.encode(x, opts.d, j, enc_args_concrete...)
    else
        error("Expected a time dependent encoding.")
    end

    return state
    
end


function get_conditional_probability(s::Index, state::AbstractVector{<:Number}, rdm::ITensor)
    """For a given site, and its associated conditional reduced 
    density matrix (rdm), obtain the conditional
    probability of a state ϕ(x)."""
    # get σ_k = |⟨x_k | ρ | x_k⟩|
    return state' * Matrix(rdm, [s', s]) * state |> abs

end

function get_conditional_probability(state::ITensor, rdm::ITensor)
    """For a given site, and its associated conditional reduced 
    density matrix (rdm), obtain the conditional
    probability of a state ϕ(x)."""
    # get σ_k = |⟨x_k | ρ | x_k⟩|
    return abs(getindex(dag(state)' * rdm * state, 1))

end

function get_conditional_probability(x::Float64, rdm::Matrix, opts::Options)
    """For a given site, and its associated conditional reduced 
    density matrix (rdm), obtain the conditional
    probability of a state ϕ(x)."""
    # get σ_k = |⟨x_k | ρ | x_k⟩|
    state = get_state(x, opts)

    return abs(state' * rdm * state)

end

function get_conditional_probability(x::Float64, rdm::Matrix, opts::Options, 
    enc_args::Vector{Vector{Any}}, j::Int)

    state = get_state(x, opts, enc_args, j)

    return abs(state' * rdm * state) # Vector(state)' * Matrix(rdm, [s', s]) * Vector(state) |> abs

end

function get_normalisation_constant(s::Index, rdm::ITensor, args...)
    """Compute the normalisation constant, Z_k, such that 
    the conditional distribution integrates to one.
    """
    return get_normalisation_constant(Matrix(rdm, [s', s]), args...)

end

function get_normalisation_constant(rdm::Matrix, opts::Options)
    """Compute the normalisation constant, Z_k, such that 
    the conditional distribution integrates to one.
    """
    # make an anonymous function which allows us to integrate over x
    prob_density_wrapper(x) = get_conditional_probability(x, rdm, opts)
    # integrate over data domain xk 
    lower, upper = opts.encoding.range
    Z, _ = quadgk(prob_density_wrapper, lower, upper)

    return Z

end


function get_normalisation_constant(rdm::Matrix, opts::Options, enc_args::Vector{Vector{Any}},
    j::Int)
    prob_density_wrapper(x) = get_conditional_probability(x, rdm, opts, enc_args, j)
    lower, upper = opts.encoding.range
    Z, _ = quadgk(prob_density_wrapper, lower, upper)

    return Z
end

function get_cdf(x::Float64, rdm::Matrix, Z::Float64, opts::Options)
    """Compute the cumulative dist. function 
    via numerical integration of the probability density 
    function. Returns cdf evaluated at x where x is the proposed 
    value i.e., F(x)."""
    prob_density_wrapper(x_prime) = (1/Z) * get_conditional_probability(x_prime, rdm, opts)
    lower, _ = opts.encoding.range
    cdf_val, _ = quadgk(prob_density_wrapper, lower, x)

    return cdf_val

end

function get_cdf(x::Float64, rdm::Matrix, Z::Float64, opts::Options, 
    enc_args::Vector{Vector{Any}}, j::Int)
    prob_density_wrapper(x_prime) = (1/Z) * get_conditional_probability(x_prime, rdm, opts, enc_args, j)
    lower, _ = opts.encoding.range
    cdf_val, _ = quadgk(prob_density_wrapper, lower, x)

    return cdf_val

end

function get_sample_from_rdm(rdm::Matrix, opts::Options)
    """Sample an x value, and its corresponding state,
    ϕ(x) from a conditional density matrix using inverse 
    transform sampling."""
    Z = get_normalisation_constant(rdm, opts)
    # sample a uniform random value from U(0,1)
    u = rand()
    # solve for x by defining an auxilary function g(x) such that g(x) = F(x) - u
    cdf_wrapper(x) = get_cdf(x, rdm, Z, opts) - u
    sampled_x = find_zero(cdf_wrapper, opts.encoding.range; rtol=0.0)
    # map sampled x_k back to a state
    sampled_state = get_state(sampled_x, opts)

    return sampled_x, sampled_state

end

function get_sample_from_rdm(rdm::Matrix, opts::Options, enc_args::Vector{Vector{Any}},
    j::Int)
    Z = get_normalisation_constant(rdm, opts, enc_args, j)
    # sample a uniform random value from U(0,1)
    u = rand()
    # solve for x by defining an auxilary function g(x) such that g(x) = F(x) - u
    cdf_wrapper(x) = get_cdf(x, rdm, Z, opts, enc_args, j) - u
    sampled_x = find_zero(cdf_wrapper, opts.encoding.range; rtol=0)
    # map sampled x_k back to a state
    sampled_state = get_state(sampled_x, opts, enc_args, j)

    return sampled_x, sampled_state

end

function check_inverse_sampling(rdm::Matrix, opts::Options; dx::Float64=0.01)
    """Check the inverse sampling approach to ensure 
    that samples represent the numerical conditional
    probability distribution."""
    Z = get_normalisation_constant(rdm, opts)
    lower, upper = opts.encoding.range
    xvals = collect(lower:dx:upper)
    probs = Vector{Float64}(undef, length(xvals))
    for (index, xval) in enumerate(xvals)
        prob = (1/Z) * get_conditional_probability(xval, rdm, opts)
        probs[index] = prob
    end

    return xvals, probs

end

function check_inverse_sampling(rdm::Matrix, opts::Options, 
    enc_args::Vector{Vector{Any}}, j::Int; dx::Float64=0.01)
    Z = get_normalisation_constant(rdm, opts, enc_args, j)
    lower, upper = opts.encoding.range
    xvals = collect(lower:dx:upper)
    probs = Vector{Float64}(undef, length(xvals))
    for (index, xval) in enumerate(xvals)
        prob = (1/Z) * get_conditional_probability(xval, rdm, opts, enc_args, j)
        probs[index] = prob
    end

    return xvals, probs

end

function plot_samples_from_rdm(rdm::Matrix, opts::Options, n_samples::Int,
    show_plot::Bool=false)
    """Plot a histogram of the samples drawn 
    from the conditional distribution specified
    by the conditional density matrix ρ_k."""
    samples = Vector{Float64}(undef, n_samples)
    bins = sqrt(n_samples)
    @threads for i in eachindex(samples)
        samples[i], _ = get_sample_from_rdm(rdm, opts)
    end
    population_mean = mean(samples)
    h = StatsPlots.histogram(samples, num_bins=bins, normalize=true, 
        label="Inverse Transform Samples", 
        xlabel="x",
        ylabel="Density", 
        title="Conditional Density Matrix, $n_samples samples")
    h = vline!([population_mean], lw=3, label="Population Mean, μ = $(round(population_mean, digits=4))", c=:red)
    xvals, numerical_probs = check_inverse_sampling(rdm, opts)
    h = plot!(xvals, numerical_probs, label="Numerical Solution", lw=3, ls=:dot, c=:black)
    if show_plot
        display(h)
    end

    return h, samples

end

function plot_samples_from_rdm(rdm::Matrix, opts::Options, enc_args::Vector{Vector{Any}}, j::Int,
    n_samples::Int, show_plot::Bool=false)
    """Plot a histogram of the samples drawn 
    from the conditional distribution specified
    by the conditional density matrix ρ_k.
    """
    samples = Vector{Float64}(undef, n_samples)
    bins = sqrt(n_samples)
    @threads for i in eachindex(samples)
        samples[i], _ = get_sample_from_rdm(rdm, opts, enc_args, j)
    end
    population_mean = mean(samples)
    h = StatsPlots.histogram(samples, num_bins=bins, normalize=true, 
        label="Inverse Transform Samples", 
        xlabel="x",
        ylabel="Density", 
        title="Conditional Density Matrix, $n_samples samples, site $j")
    h = vline!([population_mean], lw=3, label="Sample Mean, μ = $(round(population_mean, digits=4))", c=:red)
    xvals, numerical_probs = check_inverse_sampling(rdm, opts, enc_args, j)
    h = plot!(xvals, numerical_probs, label="Numerical Solution", lw=3, ls=:dot, c=:black)
    if show_plot
        display(h)
    end

    return h, samples

end

function inspect_known_state_pdf(x::Float64, opts::Options, 
    n_samples::Int; show_plot=true)
    """ Inspect the distribution corresponding to 
    a conditional density matrix, given a
    known state ϕ(x_k). For an in ideal encoding with minimal uncertainty, 
    the mean of the distribution should align closely with the known value."""
    state = get_state(x, opts)
    # reduced density matrix is given by |x⟩⟨x|
    rdm = state * state'
    h, samples = plot_samples_from_rdm(rdm, opts, n_samples)
    if show_plot
        title!("$(opts.encoding.name) encoding, d=$(opts.d), true x=$x")
        vline!([x], label="Known value: $x", lw=3, c=:green)
        display(h)
    end

    return samples

end

function inspect_known_state_pdf(x::Float64, opts::Options, enc_args::Vector{Vector{Any}}, j::Int,
    n_samples::Int; show_plot=true)
    state = get_state(x, opts, enc_args, j)
    # reduced density matrix is given by |x⟩⟨x|
    rdm = state * state'
    h, samples = plot_samples_from_rdm(rdm, opts, enc_args, j, n_samples)
    if show_plot
        title!("$(opts.encoding.name) encoding, d=$(opts.d), \n aux basis dim = $(opts.aux_basis_dim), site $j")
        vline!([x], label="Known value: $x", lw=3, c=:green)
        display(h)
    end

    return samples

end

function get_encoding_uncertainty(opts::Options, xvals::Vector)
    """Computes the error as the abs. diff between
    a known x value (or equivalently, known state) and the
    expectation obtained by sampling from the rdm defined by the
    encoding"""
    expects = Vector{Float64}(undef, length(xvals))
    stds = Vector{Float64}(undef, length(xvals))
    @threads for i in eachindex(xvals)
        xval = xvals[i]
        # make the rdm
        state = get_state(xval, opts)
        rdm = state * state'
        expect_x, std_val, _ = get_cpdf_mean_std(rdm, opts)
        expects[i] = expect_x
        stds[i] = std_val
    end
    # compute the abs. diffs
    abs_diffs = abs.(expects - xvals)
    return xvals, abs_diffs, stds
end

function get_encoding_uncertainty(opts::Options, enc_args::Vector{Vector{Any}}, j::Int64, 
        xvals::Vector)
    """Computes the error as the abs. diff between
    a known x value (or equivalently, known state) and the
    expectation obtained by sampling from the rdm defined by the
    encoding"""
    expects = Vector{Float64}(undef, length(xvals))
    stds = Vector{Float64}(undef, length(xvals))
    @threads for i in eachindex(xvals)
        xval = xvals[i]
        # make the rdm
        state = get_state(xval, opts, enc_args, j)
        rdm = state * state'
        expect_x, std_val, _ = get_cpdf_mean_std(rdm, opts, enc_args, j)
        expects[i] = expect_x
        stds[i] = std_val
    end
    # compute the abs. diffs
    abs_diffs = abs.(expects - xvals)
    return xvals, abs_diffs, stds
end

function get_dist_mean_difference(eval_intervals::Int, opts::Options, n_samples::Int)
    """Get the difference between the known value
    and distribution mean for the given encoding 
    over the interval x_k ∈ [lower, upper]."""
    lower, upper = opts.encoding.range
    xvals = LinRange((lower+1E-8), (upper-1E-8), eval_intervals)
    deltas = Vector{Float64}(undef, length(xvals))
    for (index, xval) in enumerate(xvals)
        # get the state
        println("Computing x = $xval")
        state = get_state(xval, opts)
        # make the rdm 
        rdm = state * state'
        # get the
        samples = Vector{Float64}(undef, n_samples)
        @threads for i in eachindex(samples)
            samples[i], _ = get_sample_from_rdm(rdm, opts)
        end
        mean_val = mean(samples)
        delta = abs((xval - mean_val))
        deltas[index] = delta
    end 

    return collect(xvals), deltas

end

function get_cpdf_mode(rdm::ITensor, samp_xs::AbstractVector{Float64}, samp_states::AbstractVector{<:AbstractVector{<:Number}}, s::Index)
    """Much simpler approach to get the mode of the conditional 
    pdf (cpdf) for a given rdm. Simply evaluate P(x) over the x range,
    with interval dx, and take the argmax."""
    # don't even need to normalise since we just want the peak
    Z = get_normalisation_constant(s, rdm, opts)

    probs = Vector{Float64}(undef, length(samp_states))
    for (index, state) in enumerate(samp_states)
        prob = (1/Z) * get_conditional_probability(itensor(state, s), rdm)
        probs[index] = prob
    end
    
    # get the mode of the pdf
    mode_idx = argmax(probs)
    mode_x = samp_xs[mode_idx]
    mode_state = itensor(samp_states[mode_idx], s)

    return mode_x, mode_state

end

function get_cpdf_mode(rdm::Matrix, opts::Options;
    dx = 1E-4)
    """Much simpler approach to get the mode of the conditional 
    pdf (cpdf) for a given rdm. Simply evaluate P(x) over the x range,
    with interval dx, and take the argmax."""
    # don't even need to normalise since we just want the peak
    Z = get_normalisation_constant(rdm, opts)
    lower, upper = opts.encoding.range
    xvals = collect(lower:dx:upper)
    probs = Vector{Float64}(undef, length(xvals))
    for (index, xval) in enumerate(xvals)
        prob = (1/Z) * get_conditional_probability(xval, rdm, opts)
        probs[index] = prob
    end
    
    # get the mode of the pdf
    mode_idx = argmax(probs)
    mode_x = xvals[mode_idx]

    # convert xval back to state
    mode_state = get_state(mode_x, opts)

    return mode_x, mode_state

end

function get_cpdf_mode(rdm::Matrix, opts::Options, enc_args::Vector{Vector{Any}},
    j::Int; dx = 1E-4)
    Z = get_normalisation_constant(rdm, opts, enc_args, j)
    lower, upper = opts.encoding.range
    xvals = collect(lower:dx:upper)
    probs = Vector{Float64}(undef, length(xvals))
    for (index, xval) in enumerate(xvals)
        prob = (1/Z) * get_conditional_probability(xval, rdm, opts, enc_args, j)
        probs[index] = prob
    end
    
    # get the mode of the pdf
    mode_idx = argmax(probs)
    mode_x = xvals[mode_idx]

    # convert xval back to state
    mode_state = get_state(mode_x, opts, enc_args, j)

    return mode_x, mode_state

end

function get_cpdf_mean_std(rdm::Matrix, opts::Options;
    dx = 1E-4)

    Z = get_normalisation_constant(rdm, opts)
    lower, upper = opts.encoding.range
    xvals = collect(lower:dx:upper)
   
    probs = Vector{Float64}(undef, length(xvals))
    for (index, xval) in enumerate(xvals)
        prob = (1/Z) * get_conditional_probability(xval, rdm, opts)
        probs[index] = prob
    end

    # expectation
    expect_x = sum(xvals .* probs) * dx
    
    # variance
    squared_diffs = (xvals .- expect_x).^2
    var = sum(squared_diffs .* probs) * dx

    std_val = sqrt(var)
    expect_state = get_state(expect_x, opts)

    return expect_x, std_val, expect_state

end

function get_cpdf_mean_std(rdm::Matrix, opts::Options, enc_args::Vector{Vector{Any}}, 
    j::Int; dx = 1E-4)

    Z = get_normalisation_constant(rdm, opts, enc_args, j)
    lower, upper = opts.encoding.range
    xvals = collect(lower:dx:upper)
    
    probs = Vector{Float64}(undef, length(xvals))
    for (index, xval) in enumerate(xvals)
        prob = (1/Z) * get_conditional_probability(xval, rdm, opts, enc_args, j)
        probs[index] = prob
    end

    # expectation
    expect_x = sum(xvals .* probs) * dx
    
    # variance
    squared_diffs = (xvals .- expect_x).^2
    var = sum(squared_diffs .* probs) * dx

    std_val = sqrt(var)
    expect_state = get_state(expect_x, opts, enc_args, j)

    return expect_x, std_val, expect_state

end

function compute_entanglement_entropy_profile(class_mps::MPS)
    """Compute the entanglement entropy profile (page curve)
    for an un-labelled (class) mps.
        """
    mps = deepcopy(class_mps)
    @assert isapprox(norm(mps), 1.0; atol=1e-3) "MPS is not normalised!"
    mps_length = length(mps)
    entropy_vals = Vector{Float64}(undef, mps_length)
    # for each bi-partition coordinate, compute the entanglement entropy
    for oc in eachindex(mps)
        orthogonalize!(mps, oc) # shift orthogonality center to bipartition coordinate
        if oc == 1 || oc == mps_length
            _, S, _ = svd(mps[oc], (siteind(mps, oc)))
        else
            _, S, _ = svd(mps[oc], (linkind(mps, oc-1), siteind(mps, oc)))
        end
        SvN = 0.0
        # loop over the diagonal of the singular value matrix and extract the values
        for n = 1:ITensors.dim(S, 1)
            p = S[n, n]^2
            if (p > 1E-12) # to avoid log(0)
                SvN += -p * log(p)
            end
        end
        entropy_vals[oc] = SvN
    end

    return entropy_vals

end

function determine_num_hist_bins(x::Vector; bin_method::Symbol)
    """
    Determine the optimal number of bins to use for a histogram.
    - :Sturge: Sturge's Rule, based on the assumption of a normal distribution
    - :Scott: Scott's Rule, based on the assumption of normally distributed data
    - :Freedman: Freedman-Diaconis Rule, based on interquartile range and data size
    - :Sqrt: Square root rule, a simple rule for general use
    """
    n = length(x)

    # define the rules
    bin_rules = Dict(
        :Sturge => () -> 1 + log2(n),
        :Scott => () -> (maximum(x) - minimum(x)) / ((3.5 * std(x)) / (n^(1/3))),
        :Freedman => () -> (maximum(x) - minimum(x)) / ((2 * iqr(x)) / n^(1/3)),
        :Sqrt => () -> sqrt(n)
    )

    num_bins = get(bin_rules, bin_method, () -> error("Invalid method. Choose one of the following: :Sqrt, 
        :Freedman, :Scott, :Sturge"))()

    # round up to get number of bins
    return ceil(Int, num_bins)

end

function get_hist_mode(x::Vector; bin_method::Union{Symbol, Int})
    """Compute the mode of the histogram, 
    taking the center of the bin as the final statistic."""
    if typeof(bin_method) == Symbol
        num_bins = determine_num_hist_bins(x; bin_method=bin_method)
    elseif typeof(bin_method) == Int
        num_bins = bin_method
    else
        error("Invalid bin method")
    end
    h = StatsBase.fit(StatsBase.Histogram, x, nbins=num_bins)
    mode_bin_index = argmax(h.weights)
    mode_bin_edges = h.edges[1][mode_bin_index:mode_bin_index+1]
    # take the midpt as the mode
    mode_bin_midpt = mean(mode_bin_edges)

    return mode_bin_midpt

end

function get_kde_mode(x::Vector; kernel=Epanechnikov)
    # fit KDE
    U = kde(x, kernel=kernel)
    mode_idx = argmax(U.density)

    return U.x[mode_idx]

end

function bootstrap_mode_estimator(estimator::Function, x::Vector, n_resamples::Int)
    """Simple bootstrap with replacement"""
    n = length(x)
    bootstrap_indices = [StatsBase.sample(collect(1:n), n; replace=true) for _ in 1:n_resamples]
    resample_modes = Vector{Float64}(undef, n_resamples)
    for (index, resample_indices) in enumerate(bootstrap_indices)
        x_resampled = x[resample_indices]
        mode_est = estimator(x_resampled)
        resample_modes[index] = mode_est
    end
    # compute mean, std. error and 95 CI
    mean_mode = mean(resample_modes)
    standard_error_mode = std(resample_modes)
    ci_mode = 1.95996 * standard_error_mode/sqrt(n_resamples)

    return mean_mode, standard_error_mode, ci_mode

end

# function test_mode_with_known_state(x::Float64, basis::Basis, d::Int, 
#     num_samples::Int, resamples::Int=1000)

#     # sample from the rdm
#     samples = inspect_known_state_pdf(x, basis, d, num_samples; show_plot=false)
#     # get the population mean
#     mean_val_hist = mean(samples)
#     mode_val_hist = get_hist_mode(samples; bin_method=:Sqrt)
#     num_bins = determine_num_hist_bins(samples; bin_method=:Sqrt)
#     mode_bootstrap, mode_std_err, mode_ci = bootstrap_mode_estimator(get_kde_mode, samples, resamples)

#     println("Actual value: $x")
#     println("$num_samples sample mean: $mean_val_hist")
#     println("$num_bins bin $num_samples samples, histogram mode $mode_val_hist")
#     println("$resamples resample mode: $mode_bootstrap | mode std error: $mode_std_err | mode 95 ci: $mode_ci")

#     # plotting
#     h1 = histogram(samples, bins=num_bins, label="Samples", alpha=0.3)
#     h1 = vline!([x], label="True value", ls=:dot, lw=2)
#     h1 = vline!([mean_val_hist], label="Population Mean", lw=3)
#     title!("Mean, d = $d")
    
#     h2 = histogram(samples, bins=num_bins, label="Samples", alpha=0.3)
#     h2 = vline!([x], label="True value", ls=:dot, lw=2)
#     h2 = vline!([mode_val_hist], label="Population Mode (Single)", lw=3)
#     title!("Mode (Single), d = $d")

#     h3 = histogram(samples, bins=num_bins, label="Samples", alpha=0.3)
#     h3 = vline!([x], label="True value", ls=:dot, lw=2)
#     h3 = vline!([mode_bootstrap], label="Population Mode (Bootstrapped)", lw=3)
#     h3 = vline!([(mode_bootstrap+mode_std_err), (mode_bootstrap-mode_std_err)], label="Standard Error")
#     title!("Mode (Bootstrapped, $resamples resamples), d = $d")

#     plot(h1, h2, h3, size=(1200, 600), xlabel="x", ylabel="Count", bottom_margin=5mm, left_margin=5mm)

# end
