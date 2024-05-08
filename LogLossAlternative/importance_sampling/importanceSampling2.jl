using ITensors
using Random
using QuadGK
using Roots

function get_probability_density_target(x::Float64, rdm::Matrix)
    """Evaluate the probability density for the target distribution, i.e., 
    the original distribution."""
    state = [exp(1im * (3π/2) * x) * cospi(0.5 * x), exp(-1im * (3π/2) * x) * sinpi(0.5 * x)]
    return abs(state' * rdm * state) # |<x|ρ|x>|
end

function get_normalisation_constant_target(rdm::Matrix)
    """Get the normalisation constant Z for the target pdf
    such that it integrates to 1 over the interval x ∈ [0, 1]"""
    prob_density_wrapper(x) = get_probability_density_target(x, rdm)
    Z, _ = quadgk(prob_density_wrapper, 0, 1) # integrate over data/x domain
    return Z
end

function normalised_target_pdf(x::Float64, rdm::Matrix, Z::Float64)
    return get_probability_density_target(x, rdm) / Z
end

function get_probability_density_proposal(x::Float64, rdm::Matrix)
    """Evaluate the probability density for the proposal distribution."""
    state = [exp(1im * (3π/2) * x) * cospi(0.5 * x), exp(-1im * (3π/2) * x) * sinpi(0.5 * x)]
    return (abs(state' * rdm * state))^8 + 1E-25 # |<x|ρ|x>|^2
end

function get_normalisation_constant_proposal(rdm::Matrix)
    """Get the normalisation constant Z for the target pdf
    such that it integrates to 1 over the interval x ∈ [0, 1]"""
    prob_density_wrapper(x) = get_probability_density_proposal(x, rdm)
    Z, _ = quadgk(prob_density_wrapper, 0, 1) # integrate over data/x domain
    return Z
end

function get_cdf_proposal(x::Float64, rdm::Matrix, Z::Float64)
    """Get the cumulative distribution function via numerical integration of
    the probability density function.

        Returns cdf evaluated at x where x is the proposed value i.e., F(x)."""

    prob_density_wrapper(x_prime) = (1/Z) * get_probability_density_proposal(x_prime, rdm)

    cdf_val, _ = quadgk(prob_density_wrapper, 0, x) # pdf has support on the interval [0, 1] so integrate accordingly
    
    return cdf_val

end

function normalised_proposal_pdf(x::Float64, rdm::Matrix, Z::Float64)

    return get_probability_density_proposal(x, rdm) / Z
    
end

function sample_state_from_proposal(rdm::Matrix)
    """Given a 1 site RDM, samples a random value according to the
    conditional distribution encapsulated by the rdm using inverse transform sampling."""
    norm_factor = get_normalisation_constant_proposal(rdm)
    #println("Norm factor: $norm_factor")
    u = rand() # sample a uniform random value from ~U(0,1)
    # solve for x by defining an auxilary function g(x) such that g(x) = F(x) - u and then use root finder to solve for x such that g(x) = 0
    cdf_wrapper(x) = get_cdf_proposal(x, rdm, norm_factor) - u
    sampled_x = find_zero(cdf_wrapper, (0, 1); rtol=0)
    # map sampled value back to a state
    sampled_state = [exp(1im * (3π/2) * sampled_x) * cospi(0.5 * sampled_x), exp(-1im * (3π/2) * sampled_x) * sinpi(0.5 * sampled_x)]

    return sampled_x, sampled_state

end

function generate_sample_and_importance_weight(rdm::Matrix)
    """Generates a single sample, x_i, from the proposal distribution, q(x), 
    and computes the importance weight, w_i, as p(x_i)/q(x_i)."""
    sampled_x, sampled_state = sample_state_from_proposal(rdm)
    Z_proposal = get_normalisation_constant_proposal(rdm)
    q_of_xi = normalised_proposal_pdf(sampled_x, rdm, Z_proposal)
    Z_target = get_normalisation_constant_target(rdm)
    p_of_xi = normalised_target_pdf(sampled_x, rdm, Z_target)
    importance = p_of_xi/q_of_xi
    #println("Sampled state: $sampled_x | importance = $importance")

    return sampled_x, sampled_state, importance

end

function test_importance_sampling(rdm::Matrix, num_samples::Int)
    samples = Vector{Float64}(undef, num_samples)
    importance_weights = Vector{Float64}(undef, num_samples)

    Z_proposal = get_normalisation_constant_proposal(rdm)
    Z_target = get_normalisation_constant_target(rdm)

    for i in 1:num_samples
        sampled_x, _ = sample_state_from_proposal(rdm)
        samples[i] = sampled_x
        q_of_xi = normalised_proposal_pdf(sampled_x, rdm, Z_proposal)
        p_of_xi = normalised_target_pdf(sampled_x, rdm, Z_target)
        importance_weights[i] = p_of_xi / q_of_xi
    end

    normalized_weights = importance_weights / sum(importance_weights)

    p = histogram(samples, bins=50, normalize=:pdf, title="Importance Sampling Histogram",
              xlabel="Sampled Values", ylabel="Probability Density", label="Samples")
    display(p)
    expectation = sum(samples .* normalized_weights)
    variance = sum((samples .- expectation).^2 .* normalized_weights)

    println("Expectation: $expectation")
    println("Variance: $variance")

end

function forecast_mps_sites(label_mps::MPS, known_values::Vector{Float64}, forecast_start_site::Int)
    mps = deepcopy(label_mps)
    s = siteinds(mps)
    orthogonalize!(mps, 1)
    # create storage for samples
    num_forecasted_sites = length(forecast_start_site:length(mps))
    x_samps = Vector{Float64}(undef, length(mps)) 
    weights = ones(length(mps))

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
        proba_state = get_probability_density_target(known_x, matrix(prime(A, s[i]) * dag(A)))
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
        # sample a state from the proposal distribution and get the importance weight
        sampled_x, sampled_state, importance = generate_sample_and_importance_weight(rdm_matrix)
        x_samps[i] = sampled_x
        weights[i] = importance

        if i != length(mps)
            sampled_state_as_ITensor = ITensor(sampled_state, s[i])
            # get the probability of the state for normalising the next site
            proba_state = get_probability_density_target(sampled_x, rdm_matrix)
            # make the measurment of the site
            Am = A * dag(sampled_state_as_ITensor)
            # absorb into the next site
            A_new = mps[(i+1)] * Am
            # normalise by the probability
            #normalize!(A_new)
            A_new *= 1/sqrt(proba_state)
            # set A to A_new
            A = A_new
        end
        println("Trace of ρ$i: $(real(tr(rdm_matrix)))")
    end

    return x_samps, weights

end

