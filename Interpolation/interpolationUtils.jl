include("./samplingUtilsNew.jl")



function forward_interpolate_trajectory(class_mps::MPS, known_values::Vector{Float64},
    forecast_start_site::Int, opts::Options; check_rdm::Boolean=false)
    """Applies inverse transform sampling to obtain samples from the conditional 
    rdm and returns a single realisation/shot resulting from sequentially sampling
    each site."""

    mps = deepcopy(class_mps)
    s = siteinds(mps)
    @assert maxdim(class_mps[1]) == opts.d "The encoding dimension, d ($(opts.d)), does not match the MPS local dimension ($(maxdim(class_mps[1])))."
    # put the mps into right canonical form w/ ortho centre on site 1
    orthogonalize!(mps, 1)
    x_samps = Vector{Float64}(undef, length(mps))
    # set A to the first site in the MPS
    A = mps[1]
    # condition on known sites
    for i in 1:length(known_values)
        xk = known_values[i]
        xk_state = get_state(xk, opts)
        xk_state_as_ITensor = ITensor(xk_state, s[i])
        # make a projective measurement at site k
        Am = A * dag(xk_state_as_ITensor)
        # absorb the orthogonality centre into the next site to keep canonical form
        A_new = mps[(i+1)] * Am
        # normalise the next tensor by the proba. of the known state
        rdm_xk = matrix(prime(A, s[i]) * dag(A))
        if check_rdm
            @assert isapprox(tr(matrix(rdm_xk)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        proba_xk = get_conditional_probability(xk, rdm_xk, opts)
        A_new *= 1/sqrt(proba_xk)
        A = A_new
        x_samps[i] = xk
    end
    # interpolate the remaining sites by sampling from the condititonal rdm, sequentially
    # mps should still be in canonical form with ortho centre at first interp. site
    for i in forecast_start_site:length(mps)
        # get the rdm at site i
        rdm = prime(A, s[i]) * dag(A)
        if check_rdm
            @assert isapprox(tr(matrix(rdm)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        # get the sampled x and sampled state from the rdm using inverse transform sampling
        sx, ss = get_sample_from_rdm(matrix(rdm), opts)
        x_samps[i] = sx
        if i != length(mps)
            # absorb information into next site
            sampled_state_as_ITensor = ITensor(ss, s[i])
            # get the proba. of the sampled state for normalisation
            proba_state = get_conditional_probability(sx, matrix(rdm), opts)
            # make projective measurment using the sampled state
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            # normalise by the probability
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
    end

    return x_samps

end

function forward_interpolate_trajectory(class_mps::MPS, known_values::Vector{Float64},
    forecast_start_site::Int, opts::Options, enc_args::Vector{Vector{Any}}; 
    check_rdm::Boolean=false)
    # same as above, but for time dep. encoding

    mps = deepcopy(class_mps)
    s = siteinds(mps)
    @assert maxdim(class_mps[1]) == opts.d "The encoding dimension, d ($(opts.d)), does not match the MPS local dimension ($(maxdim(class_mps[1])))."
    # put the mps into right canonical form w/ ortho centre on site 1
    orthogonalize!(mps, 1)
    x_samps = Vector{Float64}(undef, length(mps))
    # set A to the first site in the MPS
    A = mps[1]
    # condition on known sites
    for i in 1:length(known_values)
        xk = known_values[i]
        xk_state = get_state(xk, opts, enc_args, i)
        xk_state_as_ITensor = ITensor(xk_state, s[i])
        # make a projective measurement at site k
        Am = A * dag(xk_state_as_ITensor)
        # absorb the orthogonality centre into the next site to keep canonical form
        A_new = mps[(i+1)] * Am
        # normalise the next tensor by the proba. of the known state
        rdm_xk = matrix(prime(A, s[i]) * dag(A))
        if check_rdm
            @assert isapprox(tr(matrix(rdm_xk)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        proba_xk = get_conditional_probability(xk, rdm_xk, opts, enc_args, i)
        A_new *= 1/sqrt(proba_xk)
        A = A_new
        x_samps[i] = xk
    end
    # interpolate the remaining sites by sampling from the condititonal rdm, sequentially
    # mps should still be in canonical form with ortho centre at first interp. site
    for i in forecast_start_site:length(mps)
        # get the rdm at site i
        rdm = prime(A, s[i]) * dag(A)
        if check_rdm
            @assert isapprox(tr(matrix(rdm)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        # get the sampled x and sampled state from the rdm using inverse transform sampling
        sx, ss = get_sample_from_rdm(matrix(rdm), opts, enc_args, i)
        x_samps[i] = sx
        if i != length(mps)
            # absorb information into next site
            sampled_state_as_ITensor = ITensor(ss, s[i])
            # get the proba. of the sampled state for normalisation
            proba_state = get_conditional_probability(sx, matrix(rdm), opts, enc_args, i)
            # make projective measurment using the sampled state
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            # normalise by the probability
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
    end

    return x_samps

end

function forward_interpolate_directMean(class_mps::MPS, known_values::Vector{Float64},
    forecast_start_site::Int, opts::Options; check_rdm::Boolean=false)
    """Evaluate different values of x for the conditional PDF and select the 
    mean (expectation) + variance"""

    mps = deepcopy(class_mps)
    s = siteinds(mps)
    @assert maxdim(class_mps[1]) == opts.d "The encoding dimension, d ($(opts.d)), does not match the MPS local dimension ($(maxdim(class_mps[1])))."
    # put the mps into right canonical form w/ ortho centre on site 1
    orthogonalize!(mps, 1)
    x_samps = Vector{Float64}(undef, length(mps))
    x_stds = Vector{Float64}(undef, length(mps))
    # set A to the first site in the MPS
    A = mps[1]
    # condition on known sites
    for i in 1:length(known_values)
        xk = known_values[i]
        xk_state = get_state(xk, opts)
        xk_state_as_ITensor = ITensor(xk_state, s[i])
        # make a projective measurement at site k
        Am = A * dag(xk_state_as_ITensor)
        # absorb the orthogonality centre into the next site to keep canonical form
        A_new = mps[(i+1)] * Am
        # normalise the next tensor by the proba. of the known state
        rdm_xk = matrix(prime(A, s[i]) * dag(A))
        if check_rdm
            @assert isapprox(tr(matrix(rdm_xk)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        proba_xk = get_conditional_probability(xk, rdm_xk, opts)
        A_new *= 1/sqrt(proba_xk)
        A = A_new
        x_samps[i] = xk
        x_stds[i] = 0.0
    end
    # interpolate the remaining sites by sampling from the condititonal rdm, sequentially
    # mps should still be in canonical form with ortho centre at first interp. site
    for i in forecast_start_site:length(mps)
        # get the rdm at site i
        rdm = prime(A, s[i]) * dag(A)
        if check_rdm
            @assert isapprox(tr(matrix(rdm)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        # get the expected x, std. of x and expected state from the rdm directly
        ex, stdx, es = get_cpdf_mean_std(matrix(rdm), opts)
        x_samps[i] = ex
        x_stds[i] = stdx
        if i != length(mps)
            # absorb information into next site
            sampled_state_as_ITensor = ITensor(es, s[i])
            # get the proba. of the sampled state for normalisation
            proba_state = get_conditional_probability(ex, matrix(rdm), opts)
            # make projective measurment using the sampled state
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            # normalise by the probability
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
    end

    return x_samps, x_stds

end

function forward_interpolate_directMean(class_mps::MPS, known_values::Vector{Float64},
    forecast_start_site::Int, opts::Options, enc_args::Vector{Vector{Any}};
    check_rdm::Boolean=false)
    # time dep version

    mps = deepcopy(class_mps)
    s = siteinds(mps)
    @assert maxdim(class_mps[1]) == opts.d "The encoding dimension, d ($(opts.d)), does not match the MPS local dimension ($(maxdim(class_mps[1])))."
    # put the mps into right canonical form w/ ortho centre on site 1
    orthogonalize!(mps, 1)
    x_samps = Vector{Float64}(undef, length(mps))
    x_stds = Vector{Float64}(undef, length(mps))
    # set A to the first site in the MPS
    A = mps[1]
    # condition on known sites
    for i in 1:length(known_values)
        xk = known_values[i]
        xk_state = get_state(xk, opts, enc_args, i)
        xk_state_as_ITensor = ITensor(xk_state, s[i])
        # make a projective measurement at site k
        Am = A * dag(xk_state_as_ITensor)
        # absorb the orthogonality centre into the next site to keep canonical form
        A_new = mps[(i+1)] * Am
        # normalise the next tensor by the proba. of the known state
        rdm_xk = matrix(prime(A, s[i]) * dag(A))
        if check_rdm
            @assert isapprox(tr(matrix(rdm_xk)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        proba_xk = get_conditional_probability(xk, rdm_xk, opts, enc_args, i)
        A_new *= 1/sqrt(proba_xk)
        A = A_new
        x_samps[i] = xk
        x_stds[i] = 0.0
    end
    # interpolate the remaining sites by sampling from the condititonal rdm, sequentially
    # mps should still be in canonical form with ortho centre at first interp. site
    for i in forecast_start_site:length(mps)
        # get the rdm at site i
        rdm = prime(A, s[i]) * dag(A)
        if check_rdm
            @assert isapprox(tr(matrix(rdm)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        # get the expected x, std. of x and expected state from the rdm directly
        ex, stdx, es = get_cpdf_mean_std(matrix(rdm), opts, enc_args, i)
        x_samps[i] = ex
        x_stds[i] = stdx
        if i != length(mps)
            # absorb information into next site
            sampled_state_as_ITensor = ITensor(es, s[i])
            # get the proba. of the sampled state for normalisation
            proba_state = get_conditional_probability(ex, matrix(rdm), opts, enc_args, i)
            # make projective measurment using the sampled state
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            # normalise by the probability
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
    end

    return x_samps, x_stds

end

function forward_interpolate_directMode(class_mps::MPS, known_values::Vector{Float64},
    forecast_start_site::Int, opts::Options; check_rdm::Boolean=false)
    """Evaluate different values of x for the conditional PDF and select the 
    mode."""

    mps = deepcopy(class_mps)
    s = siteinds(mps)
    @assert maxdim(class_mps[1]) == opts.d "The encoding dimension, d ($(opts.d)), does not match the MPS local dimension ($(maxdim(class_mps[1])))."
    # put the mps into right canonical form w/ ortho centre on site 1
    orthogonalize!(mps, 1)
    x_samps = Vector{Float64}(undef, length(mps))
    # set A to the first site in the MPS
    A = mps[1]
    # condition on known sites
    for i in 1:length(known_values)
        xk = known_values[i]
        xk_state = get_state(xk, opts)
        xk_state_as_ITensor = ITensor(xk_state, s[i])
        # make a projective measurement at site k
        Am = A * dag(xk_state_as_ITensor)
        # absorb the orthogonality centre into the next site to keep canonical form
        A_new = mps[(i+1)] * Am
        # normalise the next tensor by the proba. of the known state
        rdm_xk = matrix(prime(A, s[i]) * dag(A))
        if check_rdm
            @assert isapprox(tr(matrix(rdm_xk)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        proba_xk = get_conditional_probability(xk, rdm_xk, opts)
        A_new *= 1/sqrt(proba_xk)
        A = A_new
        x_samps[i] = xk
   
    end
    # interpolate the remaining sites by sampling from the condititonal rdm, sequentially
    # mps should still be in canonical form with ortho centre at first interp. site
    for i in forecast_start_site:length(mps)
        # get the rdm at site i
        rdm = prime(A, s[i]) * dag(A)
        if check_rdm
            @assert isapprox(tr(matrix(rdm)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        # get the expected x, std. of x and expected state from the rdm directly
        # get the x value as the mode of the rdm, mx, and the corresponding state, ms, directly
        mx, ms = get_cpdf_mode(matrix(rdm), opts)
        x_samps[i] = mx
       
        if i != length(mps)
            # absorb information into next site
            sampled_state_as_ITensor = ITensor(ms, s[i])
            # get the proba. of the sampled state for normalisation
            proba_state = get_conditional_probability(mx, matrix(rdm), opts)
            # make projective measurment using the sampled state
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            # normalise by the probability
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
    end

    return x_samps

end

function forward_interpolate_directMode(class_mps::MPS, known_values::Vector{Float64},
    forecast_start_site::Int, opts::Options, enc_args::Vector{Vector{Any}};
    check_rdm::Boolean=false)
    # time dependent version
    mps = deepcopy(class_mps)
    s = siteinds(mps)
    @assert maxdim(class_mps[1]) == opts.d "The encoding dimension, d ($(opts.d)), does not match the MPS local dimension ($(maxdim(class_mps[1])))."
    # put the mps into right canonical form w/ ortho centre on site 1
    orthogonalize!(mps, 1)
    x_samps = Vector{Float64}(undef, length(mps))
    # set A to the first site in the MPS
    A = mps[1]
    # condition on known sites
    for i in 1:length(known_values)
        xk = known_values[i]
        xk_state = get_state(xk, opts, enc_args, i)
        xk_state_as_ITensor = ITensor(xk_state, s[i])
        # make a projective measurement at site k
        Am = A * dag(xk_state_as_ITensor)
        # absorb the orthogonality centre into the next site to keep canonical form
        A_new = mps[(i+1)] * Am
        # normalise the next tensor by the proba. of the known state
        rdm_xk = matrix(prime(A, s[i]) * dag(A))
        if check_rdm
            @assert isapprox(tr(matrix(rdm_xk)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        proba_xk = get_conditional_probability(xk, rdm_xk, opts, enc_args, i)
        A_new *= 1/sqrt(proba_xk)
        A = A_new
        x_samps[i] = xk
   
    end
    # interpolate the remaining sites by sampling from the condititonal rdm, sequentially
    # mps should still be in canonical form with ortho centre at first interp. site
    for i in forecast_start_site:length(mps)
        # get the rdm at site i
        rdm = prime(A, s[i]) * dag(A)
        if check_rdm
            @assert isapprox(tr(matrix(rdm)), 1.0) "Trace of rdm @ site $i not ≈ 1."
        end
        # get the expected x, std. of x and expected state from the rdm directly
        # get the x value as the mode of the rdm, mx, and the corresponding state, ms, directly
        mx, ms = get_cpdf_mode(matrix(rdm), opts, enc_args, i)
        x_samps[i] = mx
       
        if i != length(mps)
            # absorb information into next site
            sampled_state_as_ITensor = ITensor(ms, s[i])
            # get the proba. of the sampled state for normalisation
            proba_state = get_conditional_probability(mx, matrix(rdm), opts, enc_args, i)
            # make projective measurment using the sampled state
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            # normalise by the probability
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
    end

    return x_samps

end

function any_interpolate_trajectory(class_mps::MPS, opts::Options, time_series::Vector{Float64},
    interpolation_sites::Vector{Int})
    """Interpolate mps sites without respecting time ordering, i.e., 
    condition on all known values first, then interpolate remaining sites one-by-one.
    Use inverse transform sampling to get a single trajectory."""
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
            known_state = get_state(known_x, opts)
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
            proba_state = get_conditional_probability(known_x, matrix(rdm), opts)
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
    # sample from the remaining mps which contains unmeasured (interpolation) sites. 
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
        sampled_x, sampled_state = get_sample_from_rdm(matrix(rdm), opts)
        x_samps[interpolation_sites[count]] = sampled_x
        if i != length(mps)
            sampled_state_as_ITensor = ITensor(sampled_state, s[i])
            proba_state = get_conditional_probability(sampled_x, matrix(rdm), opts)
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end

        count += 1

    end 

    return x_samps

end

function any_interpolate_trajectory(class_mps::MPS, opts::Options, enc_args::Options,
    time_series::Vector{Float64}, interpolation_sites::Vector{Int})

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
            known_state = get_state(known_x, opts, enc_args, i)
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
            proba_state = get_conditional_probability(known_x, matrix(rdm), opts, enc_args, i)
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
    # sample from the remaining mps which contains unmeasured (interpolation) sites. 
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
        sampled_x, sampled_state = get_sample_from_rdm(matrix(rdm), opts, enc_args, i)
        x_samps[interpolation_sites[count]] = sampled_x
        if i != length(mps)
            sampled_state_as_ITensor = ITensor(sampled_state, s[i])
            proba_state = get_conditional_probability(sampled_x, matrix(rdm), opts, enc_args, i)
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end

        count += 1

    end 

    return x_samps

end

function any_interpolate_directMean(class_mps::MPS, opts::Options, time_series::Vector{Float64},
    interpolation_sites::Vector{Int})
    """Interpolate mps sites without respecting time ordering, i.e., 
    condition on all known values first, then interpolate remaining sites one-by-one.
    
    Use direct mean/variance"""
    mps = deepcopy(class_mps)
    s = siteinds(mps)
    known_sites = setdiff(collect(1:length(mps)), interpolation_sites)
    x_samps = Vector{Float64}(undef, length(mps))
    x_stds = Vector{Float64}(undef, length(mps))
    original_mps_length = length(mps)

    # condition the mps on the known values
    for i in 1:original_mps_length
        if i in known_sites
            # condition the mps at the known site
            site_loc = findsite(mps, s[i]) # use the original indices
            known_x = time_series[i]
            x_samps[i] = known_x
            x_stds[i] = 0.0
            # pretty sure calling orthogonalize is a massive computational bottleneck
            orthogonalize!(mps, site_loc)
            A = mps[site_loc]
            # get the reduced density matrix
            rdm = prime(A, s[i]) * dag(A)
            known_state = get_state(known_x, opts)
            known_state_as_ITensor = ITensor(known_state, s[i])
            # make projective measurement by contracting with the site
            Am = A * dag(known_state_as_ITensor)
            if site_loc != length(mps)
                A_new = mps[(site_loc+1)] * Am
            else
                A_new = mps[(site_loc-1)] * Am
            end
            proba_state = get_conditional_probability(known_x, matrix(rdm), opts)
            A_new *= 1/sqrt(proba_state)

            if !isapprox(norm(A_new), 1.0)
                error("Site not normalised")
            end

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
    orthogonalize!(mps, 1)
    s = siteinds(mps)
    A = mps[1]
    count = 1
    for i in eachindex(mps)
        rdm = prime(A, s[i]) * dag(A)
        ex, stdx, es = get_cpdf_mean_std(matrix(rdm), opts)
        x_samps[interpolation_sites[count]] = ex
        x_stds[interpolation_sites[count]] = stdx
        if i != length(mps)
            sampled_state_as_ITensor = ITensor(es, s[i])
            proba_state = get_conditional_probability(ex, matrix(rdm), opts)
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
        count += 1
    end 
    return x_samps, x_stds
end

function any_interpolate_directMean(class_mps::MPS, opts::Options, enc_args::Options,
    time_series::Vector{Float64}, interpolation_sites::Vector{Int})
    mps = deepcopy(class_mps)
    s = siteinds(mps)
    known_sites = setdiff(collect(1:length(mps)), interpolation_sites)
    x_samps = Vector{Float64}(undef, length(mps))
    x_stds = Vector{Float64}(undef, length(mps))
    original_mps_length = length(mps)

    # condition the mps on the known values
    for i in 1:original_mps_length
        if i in known_sites
            # condition the mps at the known site
            site_loc = findsite(mps, s[i]) # use the original indices
            known_x = time_series[i]
            x_samps[i] = known_x
            x_stds[i] = 0.0
            # pretty sure calling orthogonalize is a massive computational bottleneck
            orthogonalize!(mps, site_loc)
            A = mps[site_loc]
            # get the reduced density matrix
            rdm = prime(A, s[i]) * dag(A)
            known_state = get_state(known_x, opts, enc_args, i)
            known_state_as_ITensor = ITensor(known_state, s[i])
            # make projective measurement by contracting with the site
            Am = A * dag(known_state_as_ITensor)
            if site_loc != length(mps)
                A_new = mps[(site_loc+1)] * Am
            else
                A_new = mps[(site_loc-1)] * Am
            end
            proba_state = get_conditional_probability(known_x, matrix(rdm), opts, enc_args, i)
            A_new *= 1/sqrt(proba_state)

            if !isapprox(norm(A_new), 1.0)
                error("Site not normalised")
            end

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
    orthogonalize!(mps, 1)
    s = siteinds(mps)
    A = mps[1]
    count = 1
    for i in eachindex(mps)
        rdm = prime(A, s[i]) * dag(A)
        ex, stdx, es = get_cpdf_mean_std(matrix(rdm), opts, enc_args, i)
        x_samps[interpolation_sites[count]] = ex
        x_stds[interpolation_sites[count]] = stdx
        if i != length(mps)
            sampled_state_as_ITensor = ITensor(es, s[i])
            proba_state = get_conditional_probability(ex, matrix(rdm), opts, enc_args, i)
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
        count += 1
    end 
    return x_samps, x_stds
end

function any_interpolate_directMode(class_mps::MPS, opts::Options, time_series::Vector{Float64},
    interpolation_sites::Vector{Int})
    """Interpolate mps sites without respecting time ordering, i.e., 
    condition on all known values first, then interpolate remaining sites one-by-one.
    Use direct mode."""
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
            
            # pretty sure calling orthogonalize is a massive computational bottleneck
            orthogonalize!(mps, site_loc)
            A = mps[site_loc]
            # get the reduced density matrix
            rdm = prime(A, s[i]) * dag(A)
            known_state = get_state(known_x, opts)
            known_state_as_ITensor = ITensor(known_state, s[i])
            # make projective measurement by contracting with the site
            Am = A * dag(known_state_as_ITensor)
            if site_loc != length(mps)
                A_new = mps[(site_loc+1)] * Am
            else
                A_new = mps[(site_loc-1)] * Am
            end
            proba_state = get_conditional_probability(known_x, matrix(rdm), opts)
            A_new *= 1/sqrt(proba_state)

            if !isapprox(norm(A_new), 1.0)
                error("Site not normalised")
            end

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
    orthogonalize!(mps, 1)
    s = siteinds(mps)
    A = mps[1]
    count = 1
    for i in eachindex(mps)
        rdm = prime(A, s[i]) * dag(A)
        mx, ms = get_cpdf_mode(matrix(rdm), opts)
        x_samps[interpolation_sites[count]] = mx
       
        if i != length(mps)
            sampled_state_as_ITensor = ITensor(ms, s[i])
            proba_state = get_conditional_probability(mx, matrix(rdm), opts)
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
        count += 1
    end 
    return x_samps
end

function any_interpolate_directMode(class_mps::MPS, opts::Options, enc_args::Vector{Vector{Any}},
    time_series::Vector{Float64}, interpolation_sites::Vector{Int})
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
            
            # pretty sure calling orthogonalize is a massive computational bottleneck
            orthogonalize!(mps, site_loc)
            A = mps[site_loc]
            # get the reduced density matrix
            rdm = prime(A, s[i]) * dag(A)
            known_state = get_state(known_x, opts, enc_args, i)
            known_state_as_ITensor = ITensor(known_state, s[i])
            # make projective measurement by contracting with the site
            Am = A * dag(known_state_as_ITensor)
            if site_loc != length(mps)
                A_new = mps[(site_loc+1)] * Am
            else
                A_new = mps[(site_loc-1)] * Am
            end
            proba_state = get_conditional_probability(known_x, matrix(rdm), opts, enc_args, i)
            A_new *= 1/sqrt(proba_state)

            if !isapprox(norm(A_new), 1.0)
                error("Site not normalised")
            end

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
    orthogonalize!(mps, 1)
    s = siteinds(mps)
    A = mps[1]
    count = 1
    for i in eachindex(mps)
        rdm = prime(A, s[i]) * dag(A)
        mx, ms = get_cpdf_mode(matrix(rdm), opts, enc_args, i)
        x_samps[interpolation_sites[count]] = mx
       
        if i != length(mps)
            sampled_state_as_ITensor = ITensor(ms, s[i])
            proba_state = get_conditional_probability(mx, matrix(rdm), opts, enc_args, i)
            Am = A * dag(sampled_state_as_ITensor)
            A_new = mps[(i+1)] * Am
            A_new *= 1/sqrt(proba_state)
            A = A_new
        end
        count += 1
    end 
    return x_samps
end

