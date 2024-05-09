using ITensors
using Random
using QuadGK
using Roots
using Plots, StatsPlots
using Base.Threads

############################ FORECASTING ###########################
function get_state(x::Float64)
    """REPLACE WITH FEATURE MAP"""
    state = [exp(1im * (3π/2) * x) * cospi(0.5 * x), exp(-1im * (3π/2) * x) * sinpi(0.5 * x)]
    return state
end

function get_conditional_probability(x::Float64, rdm::Matrix)
    """For a given site, and its associated conditional reduced 
    density matrix (rdm), obtain the conditional
    probability of a state ϕ(x).
    """
    # get σ_k = |⟨x_k | ρ | x_k⟩|
    state = get_state(x)
    return abs(state' * rdm * state)
end

function get_normalisation_constant(rdm::Matrix)
    """Compute the normalisation constant, Z_k, such that 
    the conditional distribution integrates to one.
    """
    # make an anonymous function which allows us to integrate over x
    prob_density_wrapper(x) = get_conditional_probability(x, rdm)
    # integrate over data domain xk ∈ [0, 1]
    Z, _ = quadgk(prob_density_wrapper, 0, 1)
    return Z
end

function get_cdf(x::Float64, rdm::Matrix, Z::Float64)
    """Compute the cumulative dist. function 
    via numerical integration of the probability density 
    function. Returns cdf evaluated at x where x is the proposed 
    value i.e., F(x)."""
    prob_density_wrapper(x_prime) = (1/Z) * get_conditional_probability(x_prime, rdm)
    # pdf has support on the interval [0, 1] so integrate accordingly
    cdf_val, _ = quadgk(prob_density_wrapper, 0, x)
    return cdf_val
end

function get_sample_from_rdm(rdm::Matrix)
    """Sample an x value, and its corresponding state,
    ϕ(x) from a conditional density matrix using inverse 
    transform sampling.
    Returns both the sampled value x_k at site k, and the corresponding
    state, ϕ(x_k). 
    """
    Z = get_normalisation_constant(rdm)
    # sample a uniform random value from U(0,1)
    u = rand()
    # solve for x by defining an auxilary function g(x) such that g(x) = F(x) - u
    cdf_wrapper(x) = get_cdf(x, rdm, Z) - u
    sampled_x = find_zero(cdf_wrapper, (0, 1); rtol=0)
    # map sampled x_k back to a state
    sampled_state = get_state(sampled_x)

    return sampled_x, sampled_state

end

# testing functions

function check_inverse_sampling(rdm::Matrix)
    """Check the inverse sampling approach to ensure 
    that samples represent the numerical conditional
    probability distribution."""
    Z = get_normalisation_constant(rdm)
    xvals = collect(0.0:0.01:1.0)
    probs = Vector{Float64}(undef, length(xvals))
    for (index, xval) in enumerate(xvals)
        prob = (1/Z) * get_conditional_probability(xval, rdm)
        probs[index] = prob
    end
    return xvals, probs
end

function plot_samples_from_rdm(rdm::Matrix, n_samples::Int, 
    show_plot::Bool=false)
    """Plot a histogram of the samples drawn 
    from the conditional distribution specified
    by the conditional density matrix ρ_k.
    """
    samples = Vector{Float64}(undef, n_samples)
    bins =  sqrt(n_samples)
    @threads for i in eachindex(samples)
        samples[i], _ = get_sample_from_rdm(rdm)
    end
    mean_val = mean(samples)
    h = histogram(samples, num_bins=bins, normalize=true, 
        label="Inverse Transform Samples", 
        xlabel="x",
        ylabel="Density", 
        title="Conditional Density Matrix, $n_samples samples")
    h = vline!([mean_val], lw=3, label="Sample Mean, μ = $(round(mean_val, digits=4))", c=:red)
    xvals, numerical_probs = check_inverse_sampling(rdm)
    h = plot!(xvals, numerical_probs, label="Numerical Solution", lw=3, ls=:dot, c=:black)
    if show_plot
        display(h)
    end
    return h
end

function inspect_known_state_pdf(x::Float64, n_samples::Int)
    """ Inspect the distribution corresponding to 
    a conditional density matrix, given a
    known state ϕ(x_k).
    For an in ideal encoding, the mean of the distribution
    should align closely with the known value. 
    """
    state = get_state(x)
    # reduced density matrix is given by |x⟩⟨x|
    rdm = state * state'
    h = plot_samples_from_rdm(rdm, n_samples, false)
    display(h)
    title!("Known value: $x")
    vline!([x], label="Known value: $x", lw=3, c=:green)
end

function get_dist_mean_difference(eval_intervals::Int, n_samples::Int)
    """Get the difference between the known value
    and distribution mean for the given encoding 
    over the interval x_k ∈ [0, 1].
    """
    xvals = LinRange(0.0, 1.0, eval_intervals)
    deltas = Vector{Float64}(undef, length(xvals))
    for (index, xval) in enumerate(xvals)
        # get the state
        #println("Computing x = $xval")
        state = get_state(xval)
        # make the rdm 
        rdm = state * state'
        # get the
        samples = Vector{Float64}(undef, n_samples)
        @threads for i in eachindex(samples)
            samples[i], _ = get_sample_from_rdm(rdm)
        end
        mean_val = mean(samples)
        delta = abs((xval - mean_val))
        deltas[index] = delta
    end 

    return collect(xvals), deltas

end

function forecast_mps_sites(class_mps::MPS, known_values::Vector{Float64}, 
    forecast_start_site::Int; diagnostic::Bool=false)
    """
    - mps is unlabelled
    Forecasting is sequential, so the forecasting horizon is inferred from the 
    difference between the known values and the mps length.
    Note: also adds known values to x_samps 
    """
    mps = deepcopy(class_mps)
    s = siteinds(mps)
    # put the mps into right canonical form - orthogonality center is set to the first site
    orthogonalize!(mps, 1)
    x_samps = Vector{Float64}(undef, length(mps)) 
    # set A to the first MPS site
    A = mps[1]
    # condition on known sites
    for i in 1:length(known_values)
        xk = known_values[i]
        xk_state = get_state(xk)
        xk_state_as_ITensor = ITensor(xk_state, s[i])
        # make the measurment at the site k
        Am = A * dag(xk_state_as_ITensor)
        # absorb orthogonality center into the next site
        A_new = mps[(i+1)] * Am
        # normalise the next tensor by the probability of the known state
        rdm_xk = matrix(prime(A, s[i]) * dag(A))
        proba_xk = get_conditional_probability(xk, rdm_xk)
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
        sampled_x, sampled_state = get_sample_from_rdm(matrix(rdm))
        x_samps[i] = sampled_x
        if i != length(mps)
            # absorb information into the next site if exists
            sampled_state_as_ITensor = ITensor(sampled_state, s[i])
            # get the probability of the state for normalising the next site
            proba_state = get_conditional_probability(sampled_x, matrix(rdm))
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
function interpolate_mps_sites(class_mps::MPS, time_series::Vector{Float64},
    interpolation_sites::Vector{Int})
    """Interpolate mps sites without respecting time ordering, i.e., 
    condition on all known values first, then interpolate remaining sites
    one-by-one."""

    mps = deepcopy(class_mps)
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
            known_state = get_state(known_x)
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
            proba_state = get_conditional_probability(known_x, matrix(rdm))
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
        sampled_x, sampled_state = get_sample_from_rdm(matrix(rdm))
        x_samps[interpolation_sites[count]] = sampled_x
        if i != length(mps)
            sampled_state_as_ITensor = ITensor(sampled_state, s[i])
            proba_state = get_conditional_probability(sampled_x, matrix(rdm))
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
