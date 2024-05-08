using ITensors
using Random
using QuadGK
using Roots

function get_probability_density(x::Float64, rdm::Matrix)
    """Takes in the 1-site reduced density matrix and 
    returns the probability of a given time series value, x 
    (note, x is before applying encoding, NOT ϕ(x))."""
    # convert time series value to encoded state by applying feature map
    # our complex feature map
    state = [exp(1im * (3π/2) * x) * cospi(0.5 * x), exp(-1im * (3π/2) * x) * sinpi(0.5 * x)]
    return abs(state' * rdm * state) # |<x|ρ|x>|
end

function get_normalisation_constant(rdm::Matrix)
    """Get the normalisation constant Z for the pdf"""
    prob_density_wrapper(x) = get_probability_density(x, rdm)
    norm_const, _ = quadgk(prob_density_wrapper, 0, 1) # integrate over data/x domain
    return norm_const
end

function get_cdf(x::Float64, rdm::Matrix, integral_norm_const::Float64)
    """Get the cumulative distribution function via numerical integration of
    the probability density function.

        Returns cdf evaluated at x where x is the proposed value i.e., F(x)."""

    prob_density_wrapper(x_prime) = (1/integral_norm_const) * get_probability_density(x_prime, rdm)

    cdf_val, _ = quadgk(prob_density_wrapper, 0, x) # pdf has support on the interval [0, 1] so integrate accordingly
    
    return cdf_val
end

function sample_state_from_rdm(rdm)
    """Given a 1 site RDM, samples a random value according to the
    conditional distribution encapsulated by the rdm using inverse transform sampling.
    Returns both the sampled value x (the un-feature mapped value) and the feature mapped
    value ϕ(x)."""
    norm_factor = get_normalisation_constant(rdm)
    #println("Norm factor: $norm_factor")
    u = rand() # sample a uniform random value from ~U(0,1)
    # solve for x by defining an auxilary function g(x) such that g(x) = F(x) - u and then use root finder to solve for x such that g(x) = 0
    cdf_wrapper(x) = get_cdf(x, rdm, norm_factor) - u
    sampled_x = find_zero(cdf_wrapper, (0, 1); rtol=0)
    # map sampled value back to a state
    sampled_state = [exp(1im * (3π/2) * sampled_x) * cospi(0.5 * sampled_x), exp(-1im * (3π/2) * sampled_x) * sinpi(0.5 * sampled_x)]
    
    return sampled_x, sampled_state

end

function sample_mps_with_contractions(label_mps::MPS)
    """Using the perfect sampling approach described in Vidal et al.'s paper
    on sampling from Unitaary Tensor Networks."""
    # do initial checks on the mps
    @assert isapprox(norm(label_mps), 1.0) "WARNING: MPS NOT NORMALISED!"
    # make a copy of the mps
    mps = deepcopy(label_mps)
    s = siteinds(mps)
    # put the mps into right canonical form - orthogonality center is set to the first site
    orthogonalize!(mps, 1)
    # create storage for samples
    x_samps = Vector{Float64}(undef, length(mps))
    #println(length(x_samps))

    # set A to the first mps site
    A = mps[1]

    for i in 1:length(mps)

        # get the reduced density matrix at site i
        rdm = prime(A, s[i]) * dag(A)
        # convert to matrix type
        rdm_matrix = matrix(rdm)
        # sample a state from the rdm using inverse transform sampling
        sampled_x, sampled_state = sample_state_from_rdm(rdm_matrix)
        x_samps[i] = sampled_x
        if i != length(mps)
            sampled_state_as_ITensor = ITensor(sampled_state, s[i])
            # get the probability of the state for normalising the next site
            proba_state = get_probability_density(sampled_x, rdm_matrix)
            #println("Prob of sampled state: $proba_state")
            # check that the trace of the rdm is equal to one
            sampled_x, sampled_state = sample_state_from_rdm(rdm_matrix)
            # make the measurment of the site
            Am = A * dag(sampled_state_as_ITensor)
            # absorb into the next site
            A_new = mps[(i+1)] * Am
            # normalise by the probability
            A_new *= 1/sqrt(proba_state)
            # set A to A_new
            A = A_new
        end
        #println("Trace of ρ$i: $(real(tr(rdm_matrix)))")
    end

    return x_samps

end

function forecast_mps_sites(label_mps::MPS, known_values::Vector{Float64}, forecast_start_site::Int)
    """Forecasts sites starting at forecast_start_site (inclusive).
    Known values are time series values before feature map i.e., x not ϕ(x)"""
    """Future versions should use product states as inputs becuase it might be easier to match/infer site indices"""
    # do initial checks on the mps
    #@assert isapprox(norm(label_mps), 1.0) "WARNING: MPS NOT NORMALISED!"
    # make a copy of the mps
    mps = deepcopy(label_mps)
    s = siteinds(mps)
    # put the mps into right canonical form - orthogonality center is set to the first site
    orthogonalize!(mps, 1)
    # create storage for samples
    x_samps = Vector{Float64}(undef, length(mps)) 

    # set A to the first MPS site
    A = mps[1]

    # condition on known sites
    for i in 1:length(known_values)
        
        # map known value to state
        known_x = known_values[i]
        known_state = [exp(1im * (3π/2) * known_x) * cospi(0.5 * known_x), exp(-1im * (3π/2) * known_x) * sinpi(0.5 * known_x)]
        known_state_as_ITensor = ITensor(known_state, s[i])

        # make the measurment at the site
        Am = A * dag(known_state_as_ITensor)

        # absorb orthogonality center into the next site
        A_new = mps[(i+1)] * Am
        # normalise by the probability of the known state
        proba_state = get_probability_density(known_x, matrix(prime(A, s[i]) * dag(A)))
        A_new *= 1/sqrt(proba_state)
        A = A_new
        x_samps[i] = known_x
    end

    # forecast on the remaining sites, A should now be the first forecasting site
    for i in forecast_start_site:length(mps)

        # get the reduced density matrix at site i
        rdm = prime(A, s[i]) * dag(A)
        # convert to matrix type
        rdm_matrix = matrix(rdm)
        # sample a state from the rdm using inverse transform sampling
        sampled_x, sampled_state = sample_state_from_rdm(rdm_matrix)
        x_samps[i] = sampled_x
        if i != length(mps)
            sampled_state_as_ITensor = ITensor(sampled_state, s[i])
            # get the probability of the state for normalising the next site
            proba_state = get_probability_density(sampled_x, rdm_matrix)
            #println("Prob of sampled state: $proba_state")
            # check that the trace of the rdm is equal to one
            #sampled_x, sampled_state = sample_state_from_rdm(rdm_matrix)
            # make the measurment of the site
            Am = A * dag(sampled_state_as_ITensor)
            # absorb into the next site
            A_new = mps[(i+1)] * Am
            # normalise by the probability
            A_new *= 1/sqrt(proba_state)
            # set A to A_new
            A = A_new
        end
        #println("Trace of ρ$i: $(real(tr(rdm_matrix)))")
    end

    return x_samps

end


function compute_entanglement_entropy_profile(label_mps::MPS)
    """Compute the entanglement entropy profile for a given label MPS.
    Taken from ITensor docs: https://itensor.org/docs.cgi?page=formulas/entanglement_mps"""

    mps = deepcopy(label_mps) # make a copy of the mps
    # initial checks
    @assert isapprox(norm(mps), 1.0; atol=1e-3) "MPS is not normalised!"
    # check that label index is not attached to mps?
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

function interpolate_time_ordered(label_mps::MPS, time_series::Vector{Float64}, interpolate_sites::Vector{Int})
    """Approach 1 - interpolate mps from start to finish, obeying time ordering.
    Takes the full (normalised) time series sample as input, as well as the indices of the sites to interpolate.
    Will treat the interpolation sites as `missing'."""

    # check that the length of the mps equals the sample length
    if length(label_mps) !== length(time_series)
        error("Time series length does not match MPS length.")
    end
    
    # check interpolation idxs exist

    mps = deepcopy(label_mps) # make a copy
    s = siteinds(mps)
    # put the mps into right canonical form - orthogonality center is set to the first site
    orthogonalize!(mps, 1)
    # create storage for samples
    x_samps = Vector{Float64}(undef, length(mps)) 

    # set A to the first MPS site
    A = mps[1]

    # loop through each site and determine whether to project or sample
    for i in eachindex(mps)
        # get the reduced density matrix at site i
        rdm = prime(A, s[i]) * dag(A)
        rdm_m = matrix(rdm) # convert to rdm

        if i in interpolate_sites
            #println("Sampling site $i from the reduced density matrix.")
            # sample a state from the rdm using inverse transform sampling
            sampled_x, sampled_state = sample_state_from_rdm(rdm_m)
            x_samps[i] = sampled_x
            if i != length(mps)
                sampled_state_as_ITensor = ITensor(sampled_state, s[i])
                # get the probability of the state for normalising the next site
                proba_state = get_probability_density(sampled_x, rdm_m)
                #println("Prob of sampled state: $proba_state")
                # check that the trace of the rdm is equal to one
                #sampled_x, sampled_state = sample_state_from_rdm(rdm_m)
                # make the measurment of the site
                Am = A * dag(sampled_state_as_ITensor)
                # absorb into the next site
                A_new = mps[(i+1)] * Am
                # normalise by the probability
                A_new *= 1/sqrt(proba_state)
                # set A to A_new
                A = A_new
            end

        else
            # make a projective measurement using known value
            #println("Measuring site $i to be a known value: $(time_series[i]).")
            known_x = time_series[i]
            known_state = [exp(1im * (3π/2) * known_x) * cospi(0.5 * known_x), exp(-1im * (3π/2) * known_x) * sinpi(0.5 * known_x)]
            known_state_as_ITensor = ITensor(known_state, s[i])
            
            # make the measurment at the site
            Am = A * dag(known_state_as_ITensor)

            if i != length(mps)
                A_new = mps[(i+1)] * Am
                # normalise by the probability of the known state
                proba_state = get_probability_density(known_x, rdm_m)
                A_new *= 1/sqrt(proba_state)
                A = A_new
            end
            x_samps[i] = known_x
        end
    end

    return x_samps

end

function interpolate_non_sequential(label_mps::MPS, time_series::Vector{Float64},
    interpolate_sites::Vector{Int})
    # pre-condition MPS on all known sites
    mps = deepcopy(label_mps)
    s = siteinds(mps)
    known_sites = setdiff(collect(1:length(mps)), interpolate_sites)
    x_samps = Vector{Float64}(undef, length(mps))

    # part 1 - condition MPS on known values using MPOs
    for i in eachindex(mps)
        if i in known_sites
            # value is known
            x = time_series[i]
            state = [exp(1im * (3π/2) * x) * cospi(0.5 * x), exp(-1im * (3π/2) * x) * sinpi(0.5 * x)]
            # make projector
            projector = state * state'
            # check properties
            if !isapprox(tr(projector), 1.0) @warn "tr(|x⟩⟨x| != 1 @ site $i)" end
            println(projector)
            # convert into single site MPO at site id
            println("Making projector operator @ site $i")
            proj_op = op(projector, s, i)            
            # apply projector to the mps
            mps_projected = apply(proj_op, mps)
            normalize!(mps_projected) # should be fine, if not, need to normalise by 1/sqrt(prob) of state
            # done, proceed to next known site
            mps = mps_projected            
        end

    end

    # part 2 - interpolate unknown sites
    orthogonalize!(mps, 1) # put mps into right canonical form and start at site 1
    s = siteinds(mps)
    A = mps[1]
    for i in eachindex(mps)
        # determine whether site is known
        rdm = prime(A, s[i]) * dag(A)
        rdm_m = matrix(rdm) # convert to rdm
        if i in known_sites
            # site is known 
            known_x = time_series[i]
            known_state = [exp(1im * (3π/2) * known_x) * cospi(0.5 * known_x), exp(-1im * (3π/2) * known_x) * sinpi(0.5 * known_x)]
            known_state_as_ITensor = ITensor(known_state, s[i])
            # make the measurment at the site
            Am = A * dag(known_state_as_ITensor)
            if i != length(mps)
                A_new = mps[(i+1)] * Am
                # normalise by the probability of the known state
                proba_state = get_probability_density(known_x, rdm_m)
                A_new *= 1/sqrt(proba_state)
                A = A_new
            end
            x_samps[i] = known_x
        else
            # site is unkown - interpolate
            sampled_x, sampled_state = sample_state_from_rdm(rdm_m)
            x_samps[i] = sampled_x
            if i != length(mps)
                sampled_state_as_ITensor = ITensor(sampled_state, s[i])
                # get the probability of the state for normalising the next site
                proba_state = get_probability_density(sampled_x, rdm_m)
                #println("Prob of sampled state: $proba_state")
                # check that the trace of the rdm is equal to one
                #sampled_x, sampled_state = sample_state_from_rdm(rdm_m)
                # make the measurment of the site
                Am = A * dag(sampled_state_as_ITensor)
                # absorb into the next site
                A_new = mps[(i+1)] * Am
                # normalise by the probability
                A_new *= 1/sqrt(proba_state)
                # set A to A_new
                A = A_new
            end
        end
    end

    return x_samps

end

function interpolate_acausal(label_mps::MPS, time_series::Vector{Float64},
    interpolate_sites::Vector{Int})

    mps = deepcopy(label_mps)
    s = siteinds(mps)
    known_sites = setdiff(collect(1:length(mps)), interpolate_sites)
    x_samps = Vector{Float64}(undef, length(mps))
    original_mps_length = length(mps)

    # condition the mps on known values
    for i in 1:original_mps_length
        if i in known_sites
            #println("Conditioning MPS at site n = $i using known value")
            site_loc = findsite(mps, s[i]) # use original site indices
            #println("Site Loc: $site_loc")
            known_x = time_series[i]
            x_samps[i] = known_x
            orthogonalize!(mps, site_loc)
            A = mps[site_loc]
            # get the rdm
            rdm = prime(A, s[i]) * dag(A)
            rdm_m = matrix(rdm) # convert to matrix
            known_state = [exp(1im * (3π/2) * known_x) * cospi(0.5 * known_x), exp(-1im * (3π/2) * known_x) * sinpi(0.5 * known_x)]
            known_state_as_ITensor = ITensor(known_state, s[i]);
            # make measurement by contracting with the site
            Am = A * dag(known_state_as_ITensor)
            # absorb into next site if it exists
            if site_loc != length(mps)

                A_new = mps[(site_loc+1)] * Am
            else
                # absorb into previous site
                A_new = mps[(site_loc-1)] * Am
            end
            # normalise by the probability
            proba_state = get_probability_density(known_x, rdm_m)
            A_new *= 1/sqrt(proba_state);
            # check the norm
            if !isapprox(norm(A_new), 1.0)
                error("Site not normalised")
            end

            # make a new mps
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
            #println(mps)

        end

    end

    # now sample from the interpolation sites
    # check the normalisation of the mps
    if !isapprox(norm(mps), 1.0)
        error("MPS is not normalised after conditioning: $(norm(mps))")
    end

    # place the mps into right canonical form
    #samples = []
    orthogonalize!(mps, 1)
    s = siteinds(mps)
    A = mps[1]
    count = 1
    for i in eachindex(mps)
        rdm = prime(A, s[i]) * dag(A)
        rdm_m = matrix(rdm)
        sampled_x, sampled_state = sample_state_from_rdm(rdm_m)
        # use the site index to determine the location to save to
        x_samps[interpolate_sites[count]] = sampled_x
        #push!(samples, sampled_x)
        if i != length(mps)
            sampled_state_as_ITensor = ITensor(sampled_state, s[i])
            proba_state = get_probability_density(sampled_x, rdm_m)
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
        count += 1
    end

    return x_samps
    
end

function run_sampling_validation(x::Float64, num_samples::Int)
    # make into rdm
    state = [exp(1im * (3π/2) * x) * cospi(0.5 * x), exp(-1im * (3π/2) * x) * sinpi(0.5 * x)]
    rdm = state * state'
    samples = Vector{Float64}(undef, num_samples)
    for i in eachindex(samples)
        samples[i], _ = sample_state_from_rdm(rdm)
    end

    # get the numerical solution
    norm_factor = get_normalisation_constant(rdm)
    xvals = 0.0:0.01:1.0
    probs = []
    for xval in xvals
        prob = (1/norm_factor) * get_probability_density(xval, rdm)
        push!(probs, prob)
    end
    
    p = histogram(samples, bins=30, normalize=:pdf, label="Inverse Transform Samples")
    plot!(xvals, probs, label="Analytical PDF", lw=5)
    title!("Abs8")
    xlabel!("State")
    ylabel!("Probability Density")
    display(p)
    return p
end




