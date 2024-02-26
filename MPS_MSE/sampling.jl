using ITensors
using Plots
using HDF5
using Statistics
using Folds
using Base.Threads
using Random
using Distributions

function LoadMPS(fname::String)
    """Function to load a trained MPS"""
    f = h5open(fname, "r")
    W = read(f, "mps", MPS)

    return W

end

function SliceMPS(W::MPS)
    """Gets the label index of the MPS and slices according to the number of classes (dim of the label index)"""
    """Assume one-hot encoding scheme i.e. class 0 = [1, 0], class 1 = [0, 1], etc. """
    dec_index = findindex(W[end], "f(x)")
    if dec_index == nothing
        error("Label index not found on the last site of the MPS!")
    end

    n_states = ITensors.dim(dec_index)
    states = []
    for i=1:n_states
        state = deepcopy(W)
        if !isapprox(norm(state), 0)
            normalize!(state)
        end
        decision_state = onehot(dec_index => (i))
        println("Class $(i-1) state: $(vector(decision_state))")
        state[end] *= decision_state
        normalize!(state)
        push!(states, state)
    end

    return states

end;

# function SampleMPS(W::MPS; num_discrete_pts=100, rng=MersenneTwister())
#     """Function to sample from a MPS.
#     Takes in an MPS corresponding to a state that overlaps with a given
#     class (no label index attached)."""

#     # grid the angle range 
#     theta_range = collect(range(0, stop=pi/2, length=num_discrete_pts));
#     states = [[cos(theta), sin(theta)] for theta in theta_range];

#     mps = deepcopy(W)
#     sites = siteinds(mps)
#     num_sites = length(W)
#     sampled_angles = Vector{Float64}(undef, num_sites)

#     for i=1:num_sites
#         orthogonalize!(mps, i) # shift orthogonality center to site i
#         ρ = prime(mps[i], sites[i]) * dag(mps[i])
#         ρ = matrix(ρ)
#         # checks on the rdm
#         if !isapprox(tr(ρ), 1)
#             error("Reduced density matrix at site $i does not have a tr(ρ) ≈ 1")
#         end

#         probas = Vector{Float64}(undef, num_discrete_pts)
#         for j=1:length(states)
#             psi = states[j]
#             expect = psi' * ρ * psi
#             probas[j] = expect
#         end

#         # normalise the probabilities
#         probas_normalised = probas ./ sum(probas)
#         # get the cdf
#         cdf = cumsum(probas_normalised)
#         # check that the last index is approximately 1
#         if !isapprox(cdf[end], 1)
#             error("Probabilities do not sum to one. Check normalisation!")
#         end
        
#         # generate a uniform random number
#         r = rand(rng, Uniform(0, 1))
#         # choose a bin
#         sampled_bin_index = findfirst(>=(r), cdf)
#         selected_angle = theta_range[sampled_bin_index]

#         sampled_angles[i] = selected_angle

#         # now make the measurement at the site
#         sampled_state = states[sampled_bin_index]

#         # make the projector |x><x|
#         site_projector_matrix = sampled_state * sampled_state'
#         site_projector_operator = op(site_projector_matrix, sites[i])
#         site_before_measure = deepcopy(mps[i])

#         # apply single site mpo
#         site_after_measure = site_before_measure * site_projector_operator
#         noprime!(site_after_measure)

#         # add measured site back into the mps
#         mps[i] = site_after_measure
#         normalize!(mps)
#     end

#     # convert angles back to time-series values

#     sampled_time_series_values = (2 .* sampled_angles) ./ pi

#     return sampled_time_series_values

# end

function SampleMPS(W::MPS; num_discrete_pts=100)
    """Function to sample from a MPS.
    Takes in an MPS corresponding to a state that overlaps with a given
    class (no label index attached)."""

    # grid the angle range 
    theta_range = collect(range(0, stop=pi/2, length=num_discrete_pts));
    states = [[cos(theta), sin(theta)] for theta in theta_range];

    mps = deepcopy(W)
    sites = siteinds(mps)
    num_sites = length(W)
    sampled_angles = Vector{Float64}(undef, num_sites)

    for i=1:num_sites
        orthogonalize!(mps, i) # shift orthogonality center to site i
        ρ = prime(mps[i], sites[i]) * dag(mps[i])
        ρ = matrix(ρ)
        # checks on the rdm
        if !isapprox(tr(ρ), 1)
            error("Reduced density matrix at site $i does not have a tr(ρ) ≈ 1")
        end

        probas = Vector{Float64}(undef, num_discrete_pts)
        for j=1:length(states)
            psi = states[j]
            expect = psi' * ρ * psi
            probas[j] = expect
        end

        # normalise the probabilities
        probas_normalised = probas ./ sum(probas)
        # ensure normalised
        if !isapprox(sum(probas_normalised), 1)
            error("Probabilities not normalised!")
        end

        # create a categorical distribution with the normalised probabilities
        dist = Categorical(probas_normalised)
        sampled_index = rand(dist)
    
        sampled_angles[i] = theta_range[sampled_index]

        # now make the measurement at the site
        sampled_state = states[sampled_index]

        # make the projector |x><x|
        site_projector_matrix = sampled_state * sampled_state'
        site_projector_operator = op(site_projector_matrix, sites[i])
        site_before_measure = deepcopy(mps[i])

        # apply single site mpo
        site_after_measure = site_before_measure * site_projector_operator
        noprime!(site_after_measure)

        # add measured site back into the mps
        mps[i] = site_after_measure
        normalize!(mps)
    end

    # convert angles back to time-series values

    sampled_time_series_values = (2 .* sampled_angles) ./ π

    return sampled_time_series_values

end

function MPSMonteCarlo(W::MPS, num_trials::Int; num_discrete_pts::Int=100)

    all_samples = Matrix{Float64}(undef, num_trials, length(W))

    # each threads gets its own local RNG to ensure no correlation between the RN in each thread
    #rngs = [MersenneTwister(i) for i in 1: Threads.nthreads()];
    
    Threads.@threads for trial=1:num_trials
        
        s = SampleMPS(W; num_discrete_pts=num_discrete_pts)
        all_samples[trial, :] = s
    end

    mean_values = mean(all_samples, dims=1)
    std_values = std(all_samples, dims=1)

    return mean_values, std_values
        
end


