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
    return real(state' * rdm * state) # |<x|ρ|x>|
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
    cdf_val, _ = quadgk(prob_density_wrapper, 0, x) # intgerate over the domain [-∞, x]
    return cdf_val
end

function sample_state_from_rdm(rdm)
    """Given a 1 site RDM, samples a random value according to the
    conditional distribution encapsulated by the rdm using inverse transform sampling.
    Returns both the sampled value x (the un-feature mapped value) and the feature mapped
    value ϕ(x)."""
    norm_factor = get_normalisation_constant(rdm)
    u = rand() # sample a uniform random value from ~U(0,1)
    cdf_wrapper(x) = get_cdf(x, rdm, norm_factor) - u
    sampled_x = find_zero(cdf_wrapper, (0, 1))
    sampled_state = [exp(1im * (3π/2) * sampled_x) * cospi(0.5 * sampled_x), exp(-1im * (3π/2) * sampled_x) * sinpi(0.5 * sampled_x)]
    return sampled_x, sampled_state
end

function sample_mps_with_projectors(label_mps::MPS, normalise=false)
    """Samples a single configuration using direct/perfect sampling from the
    label mps, i.e., the mps corresponding to a single class."""
    """Approach one: uses a projective measurment to project the MPS into the 
    subspace where each site has the sampled value (conditional distribution)."""
    @assert isapprox(norm(label_mps), 1.0) "WARNING: MPS NOT NORMALISED!"
    # make a copy of the mps
    mps = deepcopy(label_mps)
    s = siteinds(mps)
    # put mps into right canonical form
    x_samples = Vector{Float64}(undef, length(mps))

    for i in eachindex(mps)

        orthogonalize!(mps, i)
        # since mps is in orthogonal form, can easily compute rdm
        rdm = prime(mps[i], s[i]) * dag(mps[i])
        # check properties of the rdm
        if !isapprox(real(tr(rdm)), 1.0; atol=1E-3) @warn "Trace of RDM ρ at site $i not equal to 1 ($(abs(tr(rdm))))." end
        if !isequal(rdm.tensor, adjoint(rdm).tensor) @warn "RDM at site $i not Hermitian." end
        rdm_m = matrix(rdm)
        # now sample a random state from the rdm
        sampled_x, sampled_state = sample_state_from_rdm(rdm_m)
        x_samples[i] = sampled_x

        # now make a projective measurement
        sampled_state_as_itensor = ITensor(sampled_state, s[i])
        # shenanigans to make to make a projector operator MPO
        # need to first make a 1 site MPS and then convert to an MPO
        m = MPS(1)
        m[1] = sampled_state_as_itensor
        state_projector = projector(m)
        # make the projector into an MPO
        state_projector_mpo = op(matrix(state_projector[1]), s[i])
        # apply the mpo to the mps at the site i
        mps[i] *= state_projector_mpo
        # unprime indices on the updated site from the mpo application
        noprime!(mps[i])

        # normalise the resulting mps. 
        # We have two options: (1) call normalize! or (2) multiply updated site by 1/p(xi)
        if normalise
            # use normalize!
            normalize!(mps)
        else
            # normalise by probability of sampled state 1/P(xi)
            proba_sampled_state = get_probability_density(sampled_x, rdm_m)
            mps[i] *= 1/sqrt(proba_sampled_state) # either 1/P(x) or 1/sqrt(P(x))
        end

    end

    return x_samples

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
            println("Prob of sampled state: $proba_state")
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
        println("Trace of ρ$i: $(real(tr(rdm_matrix)))")
    end

    return x_samps

end

