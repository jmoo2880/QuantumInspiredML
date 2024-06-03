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



