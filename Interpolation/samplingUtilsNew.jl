using ITensors
using Random
using QuadGK
using Roots
using Plots, StatsPlots
using StatsBase
using Base.Threads
using KernelDensity, Distributions
using LegendrePolynomials
include("../LogLoss/structs.jl")
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
        state = opts.encoding.encode(x, opts.d, j, enc_args...)
    else
        error("Expected a time dependent encoding.")
    end

    return state
    
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

    return abs(state' * rdm * state)

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
    n = length(xvals)
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
    expect_state = get_state(expect_x, opts, enc_args, j)

    return expect_x, std_val, expect_state

end

function get_cpdf_mean_std(rdm::Matrix, opts::Options, enc_args::Vector{Vector{Any}}, 
    j::Int; dx = 1E-4)

    Z = get_normalisation_constant(rdm, opts, enc_args, j)
    lower, upper = opts.encoding.range
    xvals = collect(lower:dx:upper)
    n = length(xvals)
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

#---------------------------------------------------------------------------#

function forecast_mps_sites(class_mps::MPS, known_values::Vector{Float64}, 
    forecast_start_site::Int, basis::Basis, d::Int; diagnostic::Bool=false)
    """
    - mps is unlabelled
    Forecasting is sequential, so the forecasting horizon is inferred from the 
    difference between the known values and the mps length.
    Note: also adds known values to x_samps 
    """
    # check that the basis local dimension matches the loaded mps
    mps = deepcopy(class_mps)
    s = siteinds(mps)
    @assert maxdim(class_mps[1]) == d "The d specified does not match the MPS local dimension."
    # put the mps into right canonical form - orthogonality center is set to the first site
    orthogonalize!(mps, 1)
    x_samps = Vector{Float64}(undef, length(mps)) 
    # set A to the first MPS site
    A = mps[1]
    # condition on known sites
    for i in 1:length(known_values)
        xk = known_values[i]
        xk_state = get_state(xk, basis, d)
        xk_state_as_ITensor = ITensor(xk_state, s[i])
        # make the measurment at the site k
        Am = A * dag(xk_state_as_ITensor)
        # absorb orthogonality center into the next site
        A_new = mps[(i+1)] * Am
        # normalise the next tensor by the probability of the known state
        rdm_xk = matrix(prime(A, s[i]) * dag(A))
        proba_xk = get_conditional_probability(xk, rdm_xk, basis, d)
        A_new *= 1/sqrt(proba_xk)
        A = A_new
        x_samps[i] = xk
    end
    # forecast on the remaining sites, A should now be the first forecasting site
    for i in forecast_start_site:length(mps)
        # get the reduced density matrix at site i
        rdm = prime(A, s[i]) * dag(A)
        if diagnostic
            @assert isapprox(tr(matrix(rdm)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        # sample a state from the rdm using inverse transform sampling
        sampled_x, sampled_state = get_sample_from_rdm(matrix(rdm), basis, d)
        x_samps[i] = sampled_x
        if i != length(mps)
            # absorb information into the next site if exists
            sampled_state_as_ITensor = ITensor(sampled_state, s[i])
            # get the probability of the state for normalising the next site
            proba_state = get_conditional_probability(sampled_x, matrix(rdm), basis, d)
            Am = A * dag(sampled_state_as_ITensor)
            # absorb into the next site
            A_new = mps[(i+1)] * Am
            # normalise by the probability
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
    end

    return x_samps

end

############################### NON-SAMPLING ("ANALYTIC") APPROACHES ##########################
function forecast_mps_sites_analytic_mean(class_mps::MPS, known_values::Vector{Float64}, 
    forecast_start_site::Int, basis::Basis, d::Int)
    """Same as above, but instead of sampling from the rdm to get the population 
    mean/mode, just evaluate the pdf over the x range and get the expectation/variance."""

    mps = deepcopy(class_mps)
    s = siteinds(mps)
    @assert maxdim(class_mps[1]) == d "The d specified does not match the MPS local dimension."
    # put the mps into right canonical form - orthogonality center is set to the first site
    orthogonalize!(mps, 1)
    x_samps = Vector{Float64}(undef, length(mps))
    x_stds = Vector{Float64}(undef, length(mps))
    # set A to the first MPS site
    A = mps[1]
    # condition on known sites
    for i in 1:length(known_values)
        xk = known_values[i]
        xk_state = get_state(xk, basis, d)
        xk_state_as_ITensor = ITensor(xk_state, s[i])
        # make the measurment at the site k
        Am = A * dag(xk_state_as_ITensor)
        # absorb orthogonality center into the next site
        A_new = mps[(i+1)] * Am
        # normalise the next tensor by the probability of the known state
        rdm_xk = matrix(prime(A, s[i]) * dag(A))
        proba_xk = get_conditional_probability(xk, rdm_xk, basis, d)
        A_new *= 1/sqrt(proba_xk)
        A = A_new
        x_samps[i] = xk
        x_stds[i] = 0.0
    end
    # forecast on the remaining sites, A should now be the first forecasting site
    for i in forecast_start_site:length(mps)
        # get the reduced density matrix at site i
        rdm = prime(A, s[i]) * dag(A)
        # get the mode of the pdf corresponding to the conditional rdm
        expect_x, std_val, expect_state = get_cdf_mean_std(matrix(rdm), basis, d)
        #mode_x, mode_state = get_cpdf_mean(matrix(rdm), basis, d)
        x_samps[i] = expect_x
        x_stds[i] = std_val
        if i != length(mps)
            # absorb information into the next site if exists
            sampled_state_as_ITensor = ITensor(expect_state, s[i])
            # get the probability of the state for normalising the next site
            proba_state = get_conditional_probability(expect_x, matrix(rdm), basis, d)
            Am = A * dag(sampled_state_as_ITensor)
            # absorb into the next site
            A_new = mps[(i+1)] * Am
            # normalise by the probability
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
    end

    return x_samps, x_stds

end


function forecast_mps_sites_analytic_mode(class_mps::MPS, known_values::Vector{Float64}, 
    forecast_start_site::Int, basis::Basis, d::Int)
    """Same as above, but instead of sampling from the rdm to get the population 
    mean/mode, just get the pdf and then find the peak. This is the most
    probable value."""
    mps = deepcopy(class_mps)
    s = siteinds(mps)
    @assert maxdim(class_mps[1]) == d "The d specified does not match the MPS local dimension."
    # put the mps into right canonical form - orthogonality center is set to the first site
    orthogonalize!(mps, 1)
    x_samps = Vector{Float64}(undef, length(mps)) 
    # set A to the first MPS site
    A = mps[1]
    # condition on known sites
    for i in 1:length(known_values)
        xk = known_values[i]
        xk_state = get_state(xk, basis, d)
        xk_state_as_ITensor = ITensor(xk_state, s[i])
        # make the measurment at the site k
        Am = A * dag(xk_state_as_ITensor)
        # absorb orthogonality center into the next site
        A_new = mps[(i+1)] * Am
        # normalise the next tensor by the probability of the known state
        rdm_xk = matrix(prime(A, s[i]) * dag(A))
        proba_xk = get_conditional_probability(xk, rdm_xk, basis, d)
        A_new *= 1/sqrt(proba_xk)
        A = A_new
        x_samps[i] = xk
    end
    # forecast on the remaining sites, A should now be the first forecasting site
    for i in forecast_start_site:length(mps)
        # get the reduced density matrix at site i
        rdm = prime(A, s[i]) * dag(A)
        # get the mode of the pdf corresponding to the conditional rdm
        mode_x, mode_state = get_cpdf_mode(matrix(rdm), basis, d)
        x_samps[i] = mode_x
        if i != length(mps)
            # absorb information into the next site if exists
            sampled_state_as_ITensor = ITensor(mode_state, s[i])
            # get the probability of the state for normalising the next site
            proba_state = get_conditional_probability(mode_x, matrix(rdm), basis, d)
            Am = A * dag(sampled_state_as_ITensor)
            # absorb into the next site
            A_new = mps[(i+1)] * Am
            # normalise by the probability
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
    end

    return x_samps

end

########################## INTERPOLATION #########################
function interpolate_mps_sites_mode(class_mps::MPS, basis::Basis, 
    time_series::Vector{Float64}, interpolation_sites::Vector{Int})

    mps = deepcopy(class_mps)
    d_mps = maxdim(mps[1])
    s = siteinds(mps)
    known_sites = setdiff(collect(1:length(mps)), interpolation_sites)
    x_samps = Vector{Float64}(undef, length(mps))
    original_mps_length = length(mps)

    # condition the mps on the known values
    for i in 1:original_mps_length
        if i in known_sites
            # condition the mps at the known site
            site_loc = findsite(mps, s[i]) # use the original indices
            known_x = time_series[i]
            x_samps[i] = known_x
            # pretty sure calling orthogonalize is the computational bottleneck
            orthogonalize!(mps, site_loc)
            A = mps[site_loc]
            # get the reduced density matrix
            rdm = prime(A, s[i]) * dag(A)
            known_state = get_state(known_x, basis, d_mps)
            known_state_as_ITensor = ITensor(known_state, s[i])
            # make projective measurement by contracting with the site
            Am = A * dag(known_state_as_ITensor)
            # if the next site exists, absorb the previous tensor
            if site_loc != length(mps)
                A_new = mps[(site_loc+1)] * Am
            else
                # if at the end of the mps, absorb into previous site
                A_new = mps[(site_loc-1)] * Am
            end
            # normalise by the probability
            proba_state = get_conditional_probability(known_x, matrix(rdm), basis, d_mps)
            A_new *= 1/sqrt(proba_state)

            # check the norm 
            if !isapprox(norm(A_new), 1.0)
                error("Site not normalised")
            end

            # make a new mps out of the remaining un-measured sites
            current_site = site_loc

            if current_site == 1
                next = (current_site+2):length(mps)
                new_mps = MPS(vcat(A_new, mps[next]))
            elseif current_site == length(mps)
                prev = 1:(current_site-2)
                new_mps = MPS(vcat(mps[prev], A_new))
            else
                prev = 1:current_site-1
                next = (current_site+2):length(mps)
                new_mps = MPS(vcat(mps[prev], A_new, mps[next]))
            end

            mps = new_mps
        end
    end

    if !isapprox(norm(mps), 1.0)
        error("MPS is not normalised after conditioning: $(norm(mps))")
    end

    # place the mps into right canonical form
    orthogonalize!(mps, 1)
    s = siteinds(mps)
    A = mps[1]

    count = 1
    for i in eachindex(mps)
        # same as for regular forecsting/sampling
        rdm = prime(A, s[i]) * dag(A)
        mode_x, mode_state = get_cpdf_mode(matrix(rdm), basis, d_mps)
        x_samps[interpolation_sites[count]] = mode_x
        if i != length(mps)
            sampled_state_as_ITensor = ITensor(mode_state, s[i])
            proba_state = get_conditional_probability(mode_x, matrix(rdm), basis, d_mps)
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end

        count += 1

    end 

    return x_samps

end



function interpolate_mps_sites(class_mps::MPS, basis::Basis, time_series::Vector{Float64},
    interpolation_sites::Vector{Int})
    """Interpolate mps sites without respecting time ordering, i.e., 
    condition on all known values first, then interpolate remaining sites
    one-by-one."""

    mps = deepcopy(class_mps)
    d_mps = maxdim(mps[1])
    s = siteinds(mps)
    known_sites = setdiff(collect(1:length(mps)), interpolation_sites)
    x_samps = Vector{Float64}(undef, length(mps))
    original_mps_length = length(mps)

    # condition the mps on the known values
    for i in 1:original_mps_length
        if i in known_sites
            # condition the mps at the known site
            site_loc = findsite(mps, s[i]) # use the original indices
            known_x = time_series[i]
            x_samps[i] = known_x
            # pretty sure calling orthogonalize is the computational bottleneck
            orthogonalize!(mps, site_loc)
            A = mps[site_loc]
            # get the reduced density matrix
            rdm = prime(A, s[i]) * dag(A)
            known_state = get_state(known_x, basis, d_mps)
            known_state_as_ITensor = ITensor(known_state, s[i])
            # make projective measurement by contracting with the site
            Am = A * dag(known_state_as_ITensor)
            # if the next site exists, absorb the previous tensor
            if site_loc != length(mps)
                A_new = mps[(site_loc+1)] * Am
            else
                # if at the end of the mps, absorb into previous site
                A_new = mps[(site_loc-1)] * Am
            end
            # normalise by the probability
            proba_state = get_conditional_probability(known_x, matrix(rdm), basis, d_mps)
            A_new *= 1/sqrt(proba_state)

            # check the norm 
            if !isapprox(norm(A_new), 1.0)
                error("Site not normalised")
            end

            # make a new mps out of the remaining un-measured sites
            current_site = site_loc

            if current_site == 1
                next = (current_site+2):length(mps)
                new_mps = MPS(vcat(A_new, mps[next]))
            elseif current_site == length(mps)
                prev = 1:(current_site-2)
                new_mps = MPS(vcat(mps[prev], A_new))
            else
                prev = 1:current_site-1
                next = (current_site+2):length(mps)
                new_mps = MPS(vcat(mps[prev], A_new, mps[next]))
            end

            mps = new_mps
        end
    end

    # sample from the remaining mps which contains unmeasured (interpolation)
    # sites. 
    if !isapprox(norm(mps), 1.0)
        error("MPS is not normalised after conditioning: $(norm(mps))")
    end

    # place the mps into right canonical form
    orthogonalize!(mps, 1)
    s = siteinds(mps)
    A = mps[1]

    count = 1
    for i in eachindex(mps)
        # same as for regular forecsting/sampling
        rdm = prime(A, s[i]) * dag(A)
        sampled_x, sampled_state = get_sample_from_rdm(matrix(rdm), basis, d_mps)
        x_samps[interpolation_sites[count]] = sampled_x
        if i != length(mps)
            sampled_state_as_ITensor = ITensor(sampled_state, s[i])
            proba_state = get_conditional_probability(sampled_x, matrix(rdm), basis, d_mps)
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end

        count += 1

    end 

    return x_samps

end

############################### OTHER ################################

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

function test_mode_with_known_state(x::Float64, basis::Basis, d::Int, 
    num_samples::Int, resamples::Int=1000)

    # sample from the rdm
    samples = inspect_known_state_pdf(x, basis, d, num_samples; show_plot=false)
    # get the population mean
    mean_val_hist = mean(samples)
    mode_val_hist = get_hist_mode(samples; bin_method=:Sqrt)
    num_bins = determine_num_hist_bins(samples; bin_method=:Sqrt)
    mode_bootstrap, mode_std_err, mode_ci = bootstrap_mode_estimator(get_kde_mode, samples, resamples)

    println("Actual value: $x")
    println("$num_samples sample mean: $mean_val_hist")
    println("$num_bins bin $num_samples samples, histogram mode $mode_val_hist")
    println("$resamples resample mode: $mode_bootstrap | mode std error: $mode_std_err | mode 95 ci: $mode_ci")

    # plotting
    h1 = histogram(samples, bins=num_bins, label="Samples", alpha=0.3)
    h1 = vline!([x], label="True value", ls=:dot, lw=2)
    h1 = vline!([mean_val_hist], label="Population Mean", lw=3)
    title!("Mean, d = $d")
    
    h2 = histogram(samples, bins=num_bins, label="Samples", alpha=0.3)
    h2 = vline!([x], label="True value", ls=:dot, lw=2)
    h2 = vline!([mode_val_hist], label="Population Mode (Single)", lw=3)
    title!("Mode (Single), d = $d")

    h3 = histogram(samples, bins=num_bins, label="Samples", alpha=0.3)
    h3 = vline!([x], label="True value", ls=:dot, lw=2)
    h3 = vline!([mode_bootstrap], label="Population Mode (Bootstrapped)", lw=3)
    h3 = vline!([(mode_bootstrap+mode_std_err), (mode_bootstrap-mode_std_err)], label="Standard Error")
    title!("Mode (Bootstrapped, $resamples resamples), d = $d")

    plot(h1, h2, h3, size=(1200, 600), xlabel="x", ylabel="Count", bottom_margin=5mm, left_margin=5mm)

end
