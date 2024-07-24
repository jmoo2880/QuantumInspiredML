using GenericLinearAlgebra
using ITensors
using Optim
using OptimKit
using Random
using Distributions
using DelimitedFiles
using Folds
using JLD2
using HDF5
using StatsBase
using Plots


include("structs.jl")
include("summary.jl")
include("utils_PBC.jl")

#testlist = []

function random_indices_by_label(labels, target_label, num_samples)
    # Find all indices with the target label
    indices = findall(x -> x == target_label, labels)
    
    # Check if we have enough indices to sample from
    if length(indices) < num_samples
        error("Not enough indices to sample the requested number of samples.")
    end
    # Randomly select 'num_samples' indices from the found indices
    random_seed = UInt(floor(time_ns()))
    Random.seed!(random_seed)  # Seed for reproducibility, remove or change seed for different results each time
    selected_indices = StatsBase.sample(indices, num_samples, replace=false)

    label_list = []
    for i = 1:length(selected_indices)
        push!(label_list, target_label)
    end
    label_list = Int.(label_list)
    return selected_indices, label_list
end

function split_data(data, labels, num_test_samples, classes)
    test_indices = []
    y_test = []
    for i = 1:length(num_test_samples)
        num_test_sample = num_test_samples[i]
        class = classes[i]
        selected_indices, label_list = random_indices_by_label(labels, class, num_test_sample)
        push!(test_indices, selected_indices)
        push!(y_test, label_list)
    end
    test_indices = vcat(test_indices...)
    y_test = vcat(y_test...)
    X_test = data[test_indices, :]
    all_indices = 1:size(data, 1)
    train_indices = setdiff(all_indices, test_indices)
    X_train = data[train_indices, :]
    y_train = labels[train_indices]
    return X_train, y_train, X_test, y_test, test_indices
end

function read_and_parse(file_path)
    data = []
    open(file_path, "r") do file
        for line in eachline(file)
            # Use regex to split on any sequence of whitespace and filter out empty strings
            entries = filter(!isempty, split(line, r"\s+"))
            # Convert each entry to Float64, replacing empty strings or malformed data with NaN
            parsed_entries = [isempty(s) ? NaN : parse(Float64, s) for s in entries]
            push!(data, parsed_entries)
        end
    end
    return data
end

function extract_class_samples(training_data, testing_data, training_labels, testing_labels, class_1, class_2)

    training_indices = findall(x -> x == class_1 || x == class_2, training_labels)
    testing_indices = findall(x -> x == class_1 || x == class_2, testing_labels)
    
    selected_training_data = training_data[training_indices, :]
    selected_training_labels = training_labels[training_indices]

    selected_testing_data = testing_data[testing_indices, :]
    selected_testing_labels = testing_labels[testing_indices]

    selected_training_labels = map(x -> x == class_1 ? 0 : 1, selected_training_labels)
    selected_testing_labels = map(x -> x == class_1 ? 0 : 1, selected_testing_labels)

    return selected_training_data, selected_testing_data, selected_training_labels, selected_testing_labels
end

# function generate_data(N_sine_train, N_sine_test, N_noise_train, N_noise_test, noise_factor)
#     Random.seed!(0)
#     sine_train = Matrix{Float64}(undef, N_sine_train, 40)
#     sine_test = Matrix{Float64}(undef, N_sine_test, 40)
#     noise_train = Matrix{Float64}(undef, N_noise_train, 40)
#     noise_test = Matrix{Float64}(undef, N_noise_test, 40)
#     ground_truth_train = []
#     ground_truth_test = []
#     x = range(0, stop = 2pi, length = 40) #length specifies how long time series data is
#     for i = 1:N_sine_train
#         random_phase = 2pi * rand()
#         y = sin.(x .+ random_phase)
#         noise = noise_factor * rand(length(x))
#         y_noisy = y .+ noise
#         y_scaled = ((y_noisy .- minimum(y_noisy)) ./ (maximum(y_noisy) - minimum(y_noisy)))
#         sine_train[i, :] = y_scaled
#         push!(ground_truth_train, 0)
#     end
#     for i = 1:N_sine_test
#         random_phase = 2pi * rand()
#         y = sin.(x .+ random_phase)
#         noise = noise_factor * rand(length(x))
#         y_noisy = y .+ noise
#         y_scaled = ((y_noisy .- minimum(y_noisy)) ./ (maximum(y_noisy) - minimum(y_noisy)))
#         sine_test[i, :] = y_scaled
#         push!(ground_truth_test, 0)
#     end
#     for i = 1:N_noise_train
#         y = rand(length(x))
#         noise_train[i, :] = y
#         push!(ground_truth_train, 1)
#     end
#     for i = 1:N_noise_test
#         y = rand(length(x))
#         noise_test[i, :] = y
#         push!(ground_truth_test, 1)
#     end
#     train = vcat(sine_train, noise_train)
#     test = vcat(sine_test, noise_test)
#     return train, test, ground_truth_train, ground_truth_test #returns training and testing data, as well as vectors represneting ground_truths
# end

function generate_data(N, N_sine_train, N_sine_test, N_circle_train, N_circle_test, noise_factor)
    sine_train = Matrix{Float64}(undef, N_sine_train, N)
    sine_test = Matrix{Float64}(undef, N_sine_test, N)
    circle_train = Matrix{Float64}(undef, N_circle_train, N)
    circle_test = Matrix{Float64}(undef, N_circle_test, N)
    train_labels = []
    test_labels = []

    x_min = 0
    x_max = 2π

    dx = (x_max - x_min) / N
    x = range(0, stop = (2π - dx), length = N)

    for i = 1:N_sine_train
        y = (π / 2) * sin.(x)
        phase = rand() * N
        shift = floor(Int, phase)
        y = circshift(y, shift)
        noise = noise_factor * rand(N)
        y_noisy = y .+ noise
        y_scaled = ((y_noisy .- minimum(y_noisy)) ./ (maximum(y_noisy) - minimum(y_noisy)))
        sine_train[i, :] = y_scaled
        push!(train_labels, 0)
    end

    for i = 1:N_sine_test
        y = (π / 2) * sin.(x)
        phase = rand() * N
        shift = floor(Int, phase)
        y = circshift(y, shift)
        noise = noise_factor * rand(N)
        y_noisy = y .+ noise
        y_scaled = ((y_noisy .- minimum(y_noisy)) ./ (maximum(y_noisy) - minimum(y_noisy)))
        sine_test[i, :] = y_scaled
        push!(test_labels, 0)
    end

    for i = 1:N_circle_train
        y = zeros(length(x))
        y[(x .>= 0) .& (x .< π)] .= sqrt.(π^2/4 .- (x[(x .>= 0) .& (x .< π)] .- π/2).^2)
        y[(x .>= π) .& (x .< 2π)] .= -sqrt.(π^2/4 .- (x[(x .>= π) .& (x .< 2π)] .- 3π/2).^2)
        phase = rand() * N
        shift = floor(Int, phase)
        y = circshift(y, shift)
        noise = noise_factor * rand(N)
        y_noisy = noise_factor * rand(N)
        y_noisy = y .+ noise
        y_scaled = ((y_noisy .- minimum(y_noisy)) ./ (maximum(y_noisy) - minimum(y_noisy)))
        circle_train[i, :] = y_scaled
        push!(train_labels, 1)
    end

    for i = 1:N_circle_test
        y = zeros(length(x))
        y[(x .>= 0) .& (x .< π)] .= sqrt.(π^2/4 .- (x[(x .>= 0) .& (x .< π)] .- π/2).^2)
        y[(x .>= π) .& (x .< 2π)] .= -sqrt.(π^2/4 .- (x[(x .>= π) .& (x .< 2π)] .- 3π/2).^2)
        phase = rand() * N
        shift = floor(Int, phase)
        y = circshift(y, shift)
        noise = noise_factor * rand(N)
        y_noisy = noise_factor * rand(N)
        y_noisy = y .+ noise
        y_scaled = ((y_noisy .- minimum(y_noisy)) ./ (maximum(y_noisy) - minimum(y_noisy)))
        circle_test[i, :] = y_scaled
        push!(test_labels, 1)
    end
    return x, sine_train, sine_test, circle_train, circle_test, train_labels, test_labels
end

function find_stable_accuracy(training_accuracies::Vector{Float64}, n::Float64)
    # Convert n% to a decimal for comparison
    threshold = n / 100

    # Start from the second element as there's no previous accuracy for the first one
    for i in 2:length(training_accuracies)
        # Calculate the relative change compared to the previous accuracy
        if abs(training_accuracies[i] - training_accuracies[i - 1]) / training_accuracies[i - 1] < threshold
            return i-1  # Return the index where change is less than the threshold
        end
    end

    return length(training_accuracies)  # Return index of final element if above condition isn't satisfied  
end

function loadMPS(path::String; id::String="W")
    """Loads an MPS from a .h5 file. Returns and ITensor MPS."""
    file = path[end-2:end] != ".h5" ? path * ".h5" : path
    f = h5open("$file","r")
    mps = read(f, "$id", MPS)
    close(f)
    return mps
end

# function generate_startingMPS(chi_init, site_indices::Vector{Index{Int64}};
#     num_classes = 2, random_state=nothing, opts::Options=Options(), label_tag::String="f(x)")
#     """Generate the starting weight MPS, W using values sampled from a 
#     Gaussian (normal) distribution. Accepts a chi_init parameter which
#     specifies the initial (uniform) bond dimension of the MPS."""
    
#     if random_state !== nothing
#         # use seed if specified
#         Random.seed!(random_state)
#         println("Generating initial weight MPS with bond dimension χ_init = $chi_init
#         using random state $random_state.")
#     else
#         println("Generating initial weight MPS with bond dimension χ_init = $chi_init.")
#     end

#     W = randomMPS(opts.dtype, site_indices, linkdims=chi_init)

#     label_idx = Index(num_classes, label_tag)

#     # get the site of interest and copy over the indices at the last site where we attach the label 
#     old_site_idxs = inds(W[end])
#     new_site_idxs = old_site_idxs, label_idx
#     new_site = randomITensor(opts.dtype,new_site_idxs)

#     # add the new site back into the MPS
#     W[end] = new_site

#     # normalise the MPS
#     normalize!(W)

#     # canonicalise - bring MPS into canonical form by making all tensors 1,...,j-1 left orthogonal
#     # here we assume we start at the right most index
#     last_site = length(site_indices)
#     orthogonalize!(W, last_site)

#     return W

# end

function generate_startingMPS(chi_init, site_indices::Vector{Index{Int64}};
    num_classes = 2, random_state=nothing, opts::Options=Options(), label_tag::String="f(x)")
    """Generate the starting weight MPS, W using values sampled from a 
    Gaussian (normal) distribution. Accepts a chi_init parameter which
    specifies the initial (uniform) bond dimension of the MPS."""
    
    if random_state !== nothing
        # use seed if specified
        Random.seed!(random_state)
        println("Generating initial weight MPS with bond dimension χ_init = $chi_init
        using random state $random_state.")
    else
        println("Generating initial weight MPS with bond dimension χ_init = $chi_init.")
    end

    N = length(site_indices)
    bonds = [Index(chi_init, "Link,l=$l") for l=1:N]
    W = Vector{ITensor}(undef, N)

    for i=1:N
        if i == 1
            W[i] = randomITensor(opts.dtype, site_indices[i], bonds[i], bonds[N])
        elseif i == N
            W[i] = randomITensor(opts.dtype, bonds[i-1], site_indices[i], bonds[N])
        else
            W[i] = randomITensor(opts.dtype, bonds[i-1], site_indices[i], bonds[i])
        end
    end

    W = MPS(W)
    #W = randomMPS(opts.dtype, site_indices, linkdims=chi_init)

    label_idx = Index(num_classes, label_tag)

    # get the site of interest and copy over the indices at the last site where we attach the label 
    old_site_idxs = inds(W[end])
    new_site_idxs = old_site_idxs, label_idx
    new_site = randomITensor(opts.dtype,new_site_idxs)

    # add the new site back into the MPS
    W[end] = new_site

    # normalise the MPS
    normalize!(W)

    # canonicalise - bring MPS into canonical form by making all tensors 1,...,j-1 left orthogonal
    # here we assume we start at the right most index
    last_site = length(site_indices)
    orthogonalize!(W, last_site)
    #@show W
    return W
end

function construct_caches(W::MPS, training_pstates::timeSeriesIterable; going_left=true, dtype::DataType=ComplexF64)
    """Function to pre-compute tensor contractions between the MPS and the product states. """

    # get the num of training samples to pre-allocate a caching matrix
    N_train = length(training_pstates) 
    # get the number of MPS sites
    N = length(W)

    # pre-allocate left and right environment matrices 
    LE = PCache(undef, N, N_train) 
    RE = PCache(undef, N, N_train)

    if going_left
        # backward direction - initialise the LE with the first site
        for i = 1:N_train
            LE[1,i] =  conj(training_pstates[i].pstate[1]) * W[1] 
        end

        for j = 2 : N
            for i = 1:N_train
                LE[j,i] = LE[j-1, i] * (conj(training_pstates[i].pstate[j]) * W[j])
            end
        end
    
    else
        # going right
        # initialise RE cache with the terminal site and work backwards
        for i = 1:N_train
            RE[N,i] = conj(training_pstates[i].pstate[N]) * W[N]
        end

        for j = (N-1):-1:1
            for i = 1:N_train
                RE[j,i] =  RE[j+1,i] * (W[j] * conj(training_pstates[i].pstate[j]))
            end
        end
    end

    @assert !isa(eltype(eltype(RE)), dtype) || !isa(eltype(eltype(LE)), dtype)  "Caches are not the correct datatype!"

    return LE, RE

end


function realise(B::ITensor, C_index::Index{Int64}; dtype::DataType=ComplexF64)
    """Converts a Complex {s} dimension r itensor into a eal 2x{s} dimension itensor. Increases the rank from rank{s} to 1+ rank{s} by adding a 2-dimensional index "C_index" to the start"""
    ib = inds(B)
    inds_c = C_index,ib
    B_m = Array{dtype}(B, ib)

    out = Array{real(dtype)}(undef, 2,size(B)...)
    
    ls = eachslice(out; dims=1)
    
    ls[1] = real(B_m)
    ls[2] = imag(B_m)

    return ITensor(real(dtype), out, inds_c)
end


# function complexify(B::ITensor, C_index::Index{Int64}; dtype::DataType=ComplexF64)
#     """Converts a real 2x{s} dimension itensor into a Complex {s} dimension itensor. Reduces the rank from rank{s}+1 to rank{s} by removing the first index"""
#     ib = inds(B)
#     C_index, c_inds... = ib
#     B_ra = NDTensors.array(B, ib) # should return a view


# #     re_part = selectdim(B_ra, 1,1);
# #     im_part = selectdim(B_ra, 1,2);

# #     return ITensor(dtype, complex.(re_part,im_part), c_inds)
# # end


function yhat_phitilde(BT::ITensor, LEP::PCacheCol, REP::PCacheCol, 
    product_state::PState, lid::Int, rid::Int)
    """Return yhat and phi_tilde for a bond tensor and a single product state"""
    ps= product_state.pstate
    phi_tilde = conj(ps[lid] * ps[rid]) # phi tilde 


    if lid == 1
        # at the first site, no LE
        # formatted from left to right, so env - product state, product state - env
        phi_tilde *=  REP[rid+1]
    elseif rid == length(ps)
        # terminal site, no RE
        phi_tilde *= LEP[lid-1] 
    else
        # we are in the bulk, both LE and RE exist
        phi_tilde *= LEP[lid-1] * REP[rid+1]

    end


    yhat = BT * phi_tilde # NOT a complex inner product !! 

    return yhat, phi_tilde

end

# function getmethestuff(W::MPS, product_state::PState, lid::int)
#     ps = product_state.pstate
#     phi_tilde_before = W[2] * conj(ps[2])
#     for k = 3:lid-1
#         phi_tilde_before *= W[k] * conj(ps[k])
#     end
#     return phi_tilde_before
# end 

function yhat_phitilde_terminal(W::MPS, BT::ITensor, product_state::PState, lid::Int, rid::Int)
    """Return yhat and phi_tilde for a bond tensor and a single product state"""
    ps= product_state.pstate
    #println(ps)
    phi_tilde = conj(ps[lid] * ps[rid]) # phi tilde initially given by the outer product of first and terminal PS site

    for k = 2:lid-1
        phi_tilde *= W[k] * conj(ps[k])
    end

    yhat = BT * phi_tilde # NOT a complex inner product !! 

    return yhat, phi_tilde

end

function MSE_iter(W, BT_c::ITensor, LEP::Union{PCacheCol, Nothing}, REP::Union{PCacheCol, Nothing},
    product_state::PState, lid::Int, rid::Int) 
    """Computes the Mean squared error loss function derived from KL divergence and its gradient"""

    if typeof(LEP) == PCacheCol && typeof(REP) == PCacheCol
        yhat, phi_tilde = yhat_phitilde(BT_c, LEP, REP, product_state, lid, rid)
    else
        yhat, phi_tilde = yhat_phitilde_terminal(W, BT_c, product_state, lid, rid)
    end

    # convert the label to ITensor
    label_idx = first(inds(yhat))
    y = onehot(label_idx => (product_state.label + 1))

    diff_sq = abs2.(yhat - y)
    sum_of_sq_diff = sum(diff_sq)
    loss = 0.5 * real(sum_of_sq_diff)

    # construct the gradient - return dC/dB
    gradient = (yhat - y) * conj(phi_tilde)

    return [loss, gradient]

end

function KLD_iter(W, BT_c::ITensor, LEP::Union{PCacheCol, Nothing}, REP::Union{PCacheCol, Nothing},
    product_state::PState, lid::Int, rid::Int) 
    """Computes the complex valued logarithmic loss function derived from KL divergence and its gradient"""


    if typeof(LEP) == PCacheCol && typeof(REP) == PCacheCol
        yhat, phi_tilde = yhat_phitilde(BT_c, LEP, REP, product_state, lid, rid)
    else
        yhat, phi_tilde = yhat_phitilde_terminal(W, BT_c, product_state, lid, rid)
    end

    # convert the label to ITensor
    label_idx = first(inds(yhat))
    y = onehot(label_idx => (product_state.label + 1))
    f_ln = first(yhat *y)
    loss = -log(abs2(f_ln))

    # construct the gradient - return dC/dB
    gradient = -y * conj(phi_tilde / f_ln) # mult by y to account for delta_l^lambda



    return [loss, gradient]

end

function mixed_iter(BT_c::ITensor, LEP::PCacheCol, REP::PCacheCol,
    product_state::PState, lid::Int, rid::Int; alpha=5) 
    """Returns the loss and gradient that results from mixing the logarithmic loss and mean squared error loss with mixing parameter alpha"""

    yhat, phi_tilde = yhat_phitilde(BT_c, LEP, REP, product_state, lid, rid)

    # convert the label to ITensor
    label_idx = first(inds(yhat))
    y = onehot(label_idx => (product_state.label + 1))
    f_ln = first(yhat *y)
    log_loss = -log(abs2(f_ln))

    # construct the gradient - return dC/dB
    log_gradient = -y * conj(phi_tilde / f_ln) # mult by y to account for delta_l^lambda

    # MSE
    diff_sq = abs2.(yhat - y)
    sum_of_sq_diff = sum(diff_sq)
    MSE_loss = 0.5 * real(sum_of_sq_diff)

    # construct the gradient - return dC/dB
    MSE_gradient = (yhat - y) * conj(phi_tilde)


    return [log_loss + alpha*MSE_loss, log_gradient + alpha*MSE_gradient]

end

function loss_grad(W, BT::ITensor, LE::Union{PCache, Nothing}, RE::Union{PCache, Nothing},
    TSs::timeSeriesIterable, lid::Int, rid::Int; lg_iter::Function=KLD_iter)
    """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
        Allows the input to be complex if that is supported by lg_iter"""
    
    if typeof(LE) == PCache && typeof(RE) == PCache
        loss,grad = Folds.mapreduce((LEP,REP, prod_state) -> lg_iter(W, BT,LEP,REP,prod_state,lid,rid),+, eachcol(LE), eachcol(RE),TSs)
    else
        loss,grad = Folds.mapreduce((prod_state) -> lg_iter(W, BT, nothing, nothing, prod_state, lid, rid), +, TSs)
    end

    loss /= length(TSs)
    grad ./= length(TSs)

    return loss, grad

end


function loss_grad_enforce_real(BT::ITensor, LE::PCache, RE::PCache,
    TSs::timeSeriesIterable, lid::Int, rid::Int, C_index::Union{Index{Int64},Nothing}; dtype::DataType=ComplexF64, lg_iter::Function=KLD_iter)
    """Function for computing the loss function and the gradient over all samples using a left and right cache. 
        Takes a real itensor and will convert it to complex before calling loss_grad if dtype is complex. Returns a real gradient. """
    

    if isnothing(C_index) # the itensor is real
        loss, grad = loss_grad(BT, LE, RE, TSs, lid, rid; lg_iter=lg_iter)
    else
        # pass in a complex itensor
        BT_c = complexify(BT, C_index; dtype=dtype)

        loss, grad = loss_grad(BT_c, LE, RE, TSs, lid, rid; lg_iter=lg_iter)

        grad = realise(grad, C_index; dtype=dtype)
    end


    return loss, grad

end

function loss_grad!(F,G,B_flat::AbstractArray, b_inds::Tuple{Vararg{Index{Int64}}}, LE::PCache, RE::PCache,
    TSs::timeSeriesIterable, lid::Int, rid::Int, C_index::Union{Index{Int64},Nothing}; dtype::DataType=ComplexF64, lg_iter::Function=KLD_iter)

    """Calculates the loss and gradient in a way compatible with Optim. Takes a flat, real array and converts it into an itensor before it passes it lg_iter """
    BT = itensor(real(dtype), B_flat, b_inds) # convert the bond tensor from a flat array to an itensor

    loss, grad = loss_grad_enforce_real(BT, LE, RE, TSs, lid, rid, C_index; dtype=dtype, lg_iter=lg_iter)

    if !isnothing(G)
        G .= NDTensors.array(grad,b_inds)
    end

    if !isnothing(F)
        return loss
    end

end

testlist = []
function apply_update(W, BT_init::ITensor, LE::Union{PCache, Nothing}, RE::Union{PCache, Nothing}, lid::Int, rid::Int,
    TSs::timeSeriesIterable; iters=10, verbosity::Real=1, dtype::DataType=ComplexF64, lg_iter::Function=KLD_iter, bbopt::BBOpt=BBOpt("Optim"),
    track_cost::Bool=false, eta=0.01, rescale = [true, false])
    """Apply update to bond tensor using the method specified by BBOpt. Will normalise B before and/or after it computes the update B+dB depending on the value of rescale [before::Bool,after::Bool]"""

    iscomplex = !(dtype <: Real)

    if rescale[1]
        normalize!(BT_init)
    end

    if bbopt.name == "CustomGD"
        BT_old = BT_init
        for i in 1:iters
            # get the gradient
            loss, grad = loss_grad(W, BT_old, LE, RE, TSs, lid, rid; lg_iter=lg_iter)
            #zygote_gradient_per_batch(bt_old, LE, RE, pss, lid, rid)
            # update the bond tensor
            if length(testlist) < 1
                push!(testlist, grad)
            end
            BT_new = BT_old - eta * grad
            if verbosity >=1 && track_cost
                # get the new loss
                println("Loss at step $i: $loss")
            end

            BT_old = BT_new
        end
    else
        # break down the bond tensor to feed into optimkit or optim
        if iscomplex
            C_index = Index(2, "C")
            bt_re = realise(BT_init, C_index; dtype=dtype)
        else
            C_index = nothing
            bt_re = BT_init
        end

        if bbopt.name == "Optim" 
             # flatten bond tensor into a vector and get the indices
            bt_inds = inds(bt_re)
            bt_flat = NDTensors.array(bt_re, bt_inds) # should return a view

            # create anonymous function to feed into optim, function of bond tensor only
            fgcustom! = (F,G,B) -> loss_grad!(F, G, B, bt_inds, LE, RE, TSs, lid, rid, C_index; dtype=dtype, lg_iter=lg_iter)
            # set the optimisation manfiold
            # apply optim using specified gradient descent algorithm and corresp. paramters 
            # set the manifold to either flat, sphere or Stiefel 
            if bbopt.fl == "CGD"
                method = Optim.ConjugateGradient(eta=eta)
            else
                method = Optim.GradientDescent(alphaguess=eta)
            end
            #method = Optim.LBFGS()
            res = Optim.optimize(Optim.only_fg!(fgcustom!), bt_flat; method=method, iterations = iters, 
            show_trace = (verbosity >=1),  g_abstol=1e-20)
            result_flattened = Optim.minimizer(res)

            BT_new = itensor(real(dtype), result_flattened, bt_inds)


        elseif bbopt.name == "OptimKit"

            lg = BT -> loss_grad_enforce_real(BT, LE, RE, TSs, lid, rid, C_index; dtype=dtype, lg_iter=lg_iter)
            if bbopt.fl == "CGD"
                alg = OptimKit.ConjugateGradient(; verbosity=verbosity, maxiter=iters)
            else
                alg = OptimKit.GradientDescent(; verbosity=verbosity, maxiter=iters)
            end
            BT_new, fx, _ = OptimKit.optimize(lg, bt_re, alg)


        else
            error("Unknown Black Box Optimiser $bbopt, options are [CustomGD, Optim, OptimKit]")
        end

        if iscomplex # convert back to a complex itensor
            BT_new = complexify(BT_new, C_index; dtype=dtype)
        end
    end

    if rescale[2]
        normalize!(BT_new)
    end

    if track_cost
        loss, grad = loss_grad(BT_new, LE, RE, TSs, lid, rid; lg_iter=lg_iter)
        println("Loss at site $lid*$rid: $loss")
    end

    return BT_new

end

function decomposeBT(BT::ITensor, MPS_length::Int, lid::Int, rid::Int; 
    chi_max=nothing, cutoff=nothing, going_left=true, dtype::DataType=ComplexF64)
    """Decompose an updated bond tensor back into two tensors using SVD"""
    left_site_index = findindex(BT, "n=$lid")
    label_index = findindex(BT, "f(x)")


    if going_left
        # need to make sure the label index is transferred to the next site to be updated
        if lid == 1
            bond_index = findindex(BT, "Link,l=$MPS_length")
            U, S, V = svd(BT, (left_site_index, label_index, bond_index); maxdim=chi_max, cutoff=cutoff)
        else
            bond_index = findindex(BT, "Link,l=$(lid-1)")
            U, S, V = svd(BT, (left_site_index, label_index, bond_index); maxdim=chi_max, cutoff=cutoff)
        end
        # absorb singular values into the next site to update to preserve canonicalisation
        left_site_new = U * S
        right_site_new = V
        # fix tag names 
        replacetags!(left_site_new, "Link,v", "Link,l=$lid")
        replacetags!(right_site_new, "Link,v", "Link,l=$lid")
    else
        # going right, label index automatically moves to the next site
        if lid == 1
            bond_index = findindex(BT, "Link,l=$MPS_length")
            U, S, V = svd(BT, (bond_index, left_site_index); maxdim=chi_max, cutoff=cutoff)
        else
            bond_index = findindex(BT, "Link,l=$(lid-1)")
            U, S, V = svd(BT, (bond_index, left_site_index); maxdim=chi_max, cutoff=cutoff)
        end
        # absorb into next site to be updated 
        left_site_new = U
        right_site_new = S * V
        # fix tag names 
        replacetags!(left_site_new, "Link,u", "Link,l=$lid")
        replacetags!(right_site_new, "Link,u", "Link,l=$lid")
    end


    return left_site_new, right_site_new

end

function update_caches!(left_site_new::ITensor, right_site_new::ITensor, 
    LE::PCache, RE::PCache, lid::Int, rid::Int, product_states; going_left::Bool=true)
    """Given a newly updated bond tensor, update the caches."""
    num_train = length(product_states)
    num_sites = size(LE)[1]
    if going_left
        for i = 1:num_train
            if rid == num_sites
                RE[num_sites,i] = right_site_new * conj(product_states[i].pstate[rid])
            else
                RE[rid,i] = RE[rid+1,i] * right_site_new * conj(product_states[i].pstate[rid])
            end
        end

    else
        # going right
        for i = 1:num_train
            if lid == 1
                LE[1,i] = left_site_new * conj(product_states[i].pstate[lid])
            else
                LE[lid,i] = LE[lid-1,i] * conj(product_states[i].pstate[lid]) * left_site_new
            end
        end
    end

end

function fitMPS(path::String; id::String="W", opts::Options=Options(), test_run=false)
    W_old, training_states, validation_states, testing_states = loadMPS_tests(path; id=id, opts=opts)

    return W_old, fitMPS(W_old, training_states, validation_states, testing_states; opts=opts, test_run=test_run)...
end

function fitMPS(X_train::Matrix, y_train::Vector, X_val::Matrix, y_val::Vector, X_test::Matrix, y_test::Vector; random_state=nothing, chi_init=4, opts::Options=Options(), test_run=false, algorithm::String="left")

    # first, create the site indices for the MPS and product states 
    num_mps_sites = size(X_train)[2]
    sites = siteinds(opts.d, num_mps_sites)

    # generate the starting MPS with unfirom bond dimension chi_init and random values (with seed if provided)
    num_classes = length(unique(y_train))
    W = generate_startingMPS(chi_init, sites; num_classes=num_classes, random_state=random_state, opts=opts)

    return fitMPS(W, X_train, y_train, X_val, y_val, X_test, y_test; opts=opts, test_run=test_run, algorithm=algorithm)
end

function fitMPS(W::MPS, X_train::Matrix, y_train::Vector, X_val::Matrix, y_val::Vector, X_test::Matrix, y_test::Vector; opts::Options=Options(),test_run=false, algorithm::String="left")

    @assert eltype(W[1]) == opts.dtype  "The MPS elements are of type $(eltype(W[1])) but the datatype is opts.dtype=$(opts.dtype)"

    # first, get the site indices for the product states from the MPS
    sites = get_siteinds(W)
    num_mps_sites = length(sites)
    @assert num_mps_sites == size(X_train)[2] == size(X_val)[2] == size(X_test)[2] "The number of sites supported by the MPS, training, testing, and validation data do not match! "


    @assert size(X_train)[1] == size(y_train)[1] "Size of training dataset and number of training labels are different!"
    @assert size(X_val)[1] == size(y_val)[1] "Size of validation dataset and number of validation labels are different!"
    @assert size(X_test)[1] == size(y_test)[1] "Size of testing dataset and number of testing labels are different!"

    
    # now let's handle the training/validation/testing data
    # rescale using a robust sigmoid transform
    scaler = fit_scaler(RobustSigmoidTransform, X_train; range=opts.encoding.range);
    X_train_scaled = transform_data(scaler, X_train)
    X_val_scaled = transform_data(scaler, X_val)
    X_test_scaled = transform_data(scaler, X_test)

    # generate product states using rescaled data
    if opts.encoding.iscomplex
        if opts.dtype <: Real
            error("Using a complex valued encoding but the MPS is real")
        end

    elseif !(opts.dtype <: Real)
        @warn "Using a complex valued MPS but the encoding is real"
    end

    training_states = generate_all_product_states(X_train_scaled, y_train, "train", sites; opts=opts)
    validation_states = generate_all_product_states(X_val_scaled, y_val, "valid", sites; opts=opts)
    testing_states = generate_all_product_states(X_test_scaled, y_test, "test", sites; opts=opts)

    # generate the starting MPS with uniform bond dimension chi_init and random values (with seed if provided)
    num_classes = length(unique(y_train))
    _, l_index = find_label(W)

    @assert num_classes == ITensors.dim(l_index) "Number of Classes in the training data doesn't match the dimension of the label index!"

    return fitMPS(W, training_states, validation_states, testing_states; opts=opts, test_run=test_run, algorithm=algorithm)
end

function fitMPS(training_states::timeSeriesIterable, validation_states::timeSeriesIterable, testing_states::timeSeriesIterable;
    random_state=nothing, chi_init=4, opts::Options=Options(), test_run=false, algorithm::String="left") # optimise bond tensor)
    # first, create the site indices for the MPS and product states 

    @assert opts.d == ITensors.dim(siteinds(training_states[1].pstate)[1]) "Dimension of site indices must match feature map dimension"
    sites = siteinds(testing_states[1].pstate)

    # generate the starting MPS with unfirom bond dimension chi_init and random values (with seed if provided)
    num_classes = length(unique([ps.label for ps in training_states]))
    W = generate_startingMPS(chi_init, sites; num_classes=num_classes, random_state=random_state, opts=opts)

    fitMPS(W, training_states, validation_states, testing_states; opts=opts, test_run=test_run, algorithm=algorithm)

end

function fitMPS(W::MPS, training_states::timeSeriesIterable, validation_states::timeSeriesIterable, testing_states::timeSeriesIterable; 
     opts::Options=Options(), test_run=false, algorithm::String="left") # optimise bond tensor)

    if test_run
        println("Encoding completed! Returning initial states without training.")
        return W, [], training_states, testing_states
    end

    #println(W)
    @unpack_Options opts # unpacks the attributes of opts into the local namespace
    println("Using $update_iters iterations per update.")
    # construct initial caches
    # LE, RE = construct_caches(W, training_states; going_left=true, dtype=dtype)

    # compute initial training and validation acc/loss
    init_train_loss, init_train_acc = MSE_loss_acc(W, training_states)
    init_val_loss, init_val_acc = MSE_loss_acc(W, validation_states)
    init_test_loss, init_test_acc = MSE_loss_acc(W, testing_states)

    train_KL_div = KL_div(W, training_states)
    val_KL_div = KL_div(W, validation_states)
    init_KL_div = KL_div(W, testing_states)
    sites = siteinds(W)

    # print loss and acc

    println("Validation MSE loss: $init_val_loss | Validation acc. $init_val_acc." )
    println("Training MSE loss: $init_train_loss | Training acc. $init_train_acc." )
    println("Testing MSE loss: $init_test_loss | Testing acc. $init_test_acc." )
    println("")
    println("Validation KL Divergence: $val_KL_div.")
    println("Training KL Divergence: $train_KL_div.")
    println("Test KL Divergence: $init_KL_div.")


    running_train_loss = init_train_loss
    running_val_loss = init_val_loss
    

    # create structures to store training information
    training_information = Dict(
        "train_loss" => Float64[],
        "train_acc" => Float64[],
        "val_loss" => Float64[],
        "val_acc" => Float64[],
        "test_loss" => Float64[],
        "test_acc" => Float64[],
        "time_taken" => Float64[], # sweep duration
        "train_KL_div" => Float64[],
        "test_KL_div" => Float64[],
        "val_KL_div" => Float64[]
    )

    push!(training_information["train_loss"], init_train_loss)
    push!(training_information["train_acc"], init_train_acc)
    push!(training_information["val_loss"], init_val_loss)
    push!(training_information["val_acc"], init_val_acc)
    push!(training_information["test_loss"], init_test_loss)
    push!(training_information["test_acc"], init_test_acc)
    push!(training_information["time_taken"], 0)
    push!(training_information["train_KL_div"], train_KL_div)
    push!(training_information["val_KL_div"], val_KL_div)
    push!(training_information["test_KL_div"], init_KL_div)


    # initialising loss algorithms
    if typeof(lg_iter) <: AbstractArray
        @assert length(lg_iter) == nsweeps "lg_iter(::MPS,::PState)::(loss,grad) must be a loss function or an array of loss functions with length nsweeps"
    elseif typeof(lg_iter) <: Function
        lg_iter = [lg_iter for _ in 1:nsweeps]
    else
        error("lg_iter(::MPS,::PState)::(loss,grad) must be a loss function or an array of loss functions with length nsweeps")
    end

    if typeof(bbopt) <: AbstractArray
        @assert length(bbopt) == nsweeps "bbopt must be an optimiser or an array of optimisers to use with length nsweeps"
    elseif typeof(bbopt) <: BBOpt
        bbopt = [bbopt for _ in 1:nsweeps]
    else
        error("bbopt must be an optimiser or an array of optimisers to use with length nsweeps")
    end
    test_lists = []
    if algorithm == "left"
        # start the sweep
        for itS = 1:nsweeps
            test_list = []
            start = time()
            println("Using optimiser $(bbopt[itS].name) with the \"$(bbopt[itS].fl)\" algorithm")
            println("Starting left sweeep: [$itS/$nsweeps]")
            LE, RE = construct_caches(W, training_states; going_left=true, dtype=dtype)
            for j = (length(sites)-1):-1:1
                push!(test_list, find_label(W)[1])
                #println(W)
                #print("Bond $j")
                # j tracks the LEFT site in the bond tensor (irrespective of sweep direction)
                BT = W[j] * W[(j+1)] # create bond tensor
                BT_new = apply_update(W, BT, LE, RE, j, (j+1), training_states; iters=update_iters, verbosity=verbosity, 
                                        dtype=dtype, lg_iter=lg_iter[itS], bbopt=bbopt[itS],
                                        track_cost=track_cost, eta=eta, rescale = rescale) # optimise bond tensor
                # if length(testlist) < 1
                #     push!(testlist, BT_new)
                # end
                # decompose the bond tensor using SVD and truncate according to chi_max and cutoff
                lsn, rsn = decomposeBT(BT_new, length(W), j, (j+1); chi_max=chi_max, cutoff=cutoff, going_left=true, dtype=dtype)
                    
                # update the caches to reflect the new tensors
                update_caches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=true)
                # place the updated sites back into the MPS
                W[j] = lsn
                W[(j+1)] = rsn
            end
            
            # now we do optimisation over link between originally terminal sites of MPS
            push!(test_list, find_label(W)[1])
            left_id = length(sites)
            right_id = 1
            BT = W[left_id] * W[right_id] # f(x) will be on site 1
            #println(W)
            BT_new = apply_update(W, BT, nothing, nothing, left_id, right_id, training_states; iters=update_iters, verbosity=verbosity, 
                                        dtype=dtype, lg_iter=lg_iter[itS], bbopt=bbopt[itS],
                                        track_cost=track_cost, eta=eta, rescale = rescale) # optimise bond tensor
            lsn, rsn = decomposeBT(BT_new, length(W), left_id, right_id; chi_max=chi_max, cutoff=cutoff, going_left=true, dtype=dtype)
            
            W[left_id] = lsn
            W[right_id] = rsn
            push!(test_list, find_label(W)[1])
            # add time taken for backward sweep.
            println("Left sweep finished.")
            
            
            finish = time()

            time_elapsed = finish - start

            # compute the loss and acc on both training and validation sets
            train_loss, train_acc = MSE_loss_acc(W, training_states)
            val_loss, val_acc = MSE_loss_acc(W, validation_states)
            test_loss, test_acc = MSE_loss_acc(W, testing_states)
            train_KL_div = KL_div(W, training_states)
            val_KL_div = KL_div(W, validation_states)
            test_KL_div = KL_div(W, testing_states)

            # dot_errs = test_dot(W, testing_states)

            # if !isempty(dot_errs)
            #     @warn "Found mismatching values between inner() and MPS_contract at Sites: $dot_errs"
            # end
            println("Validation MSE loss: $val_loss | Validation acc. $val_acc." )
            println("Training MSE loss: $train_loss | Training acc. $train_acc." )
            println("Testing MSE loss: $test_loss | Testing acc. $test_acc." )
            println("")
            println("Validation KL Divergence: $val_KL_div.")
            println("Training KL Divergence: $train_KL_div.")
            println("Test KL Divergence: $test_KL_div.")

            running_train_loss = train_loss
            running_val_loss = val_loss

            push!(training_information["train_loss"], train_loss)
            push!(training_information["train_acc"], train_acc)
            push!(training_information["val_loss"], val_loss)
            push!(training_information["val_acc"], val_acc)
            push!(training_information["test_loss"], test_loss)
            push!(training_information["test_acc"], test_acc)
            push!(training_information["time_taken"], time_elapsed)
            push!(training_information["train_KL_div"], train_KL_div)
            push!(training_information["val_KL_div"], val_KL_div)
            push!(training_information["test_KL_div"], test_KL_div)
            push!(test_lists, test_list)
        end
    elseif algorithm == "right"
        for itS = 1:nsweeps
            test_list = []
            start = time()
            println("Using optimiser $(bbopt[itS].name) with the \"$(bbopt[itS].fl)\" algorithm")
            println("Starting right sweeep: [$itS/$nsweeps]")
            push!(test_list, find_label(W)[1])
            left_id = length(sites)
            right_id = 1
            BT = W[left_id] * W[right_id] # f(x) will be on last site
            BT_new = apply_update(W, BT, nothing, nothing, left_id, right_id, training_states; iters=update_iters, verbosity=verbosity, 
                                        dtype=dtype, lg_iter=lg_iter[itS], bbopt=bbopt[itS],
                                        track_cost=track_cost, eta=eta, rescale = rescale) # optimise bond tensor
            lsn, rsn = decomposeBT(BT_new, length(W), left_id, right_id; chi_max=chi_max, cutoff=cutoff, going_left=false, dtype=dtype)
            W[left_id] = lsn
            W[right_id] = rsn
            LE, RE = construct_caches(W, training_states; going_left=false, dtype = dtype)
            for j = 1:(length(sites)-1)
                push!(test_list, find_label(W)[1])
                #print("Bond $j")
                BT = W[j] * W[(j+1)]
                BT_new = apply_update(BT, LE, RE, j, (j+1), training_states; iters=update_iters, verbosity=verbosity, 
                                        dtype=dtype, lg_iter=lg_iter[itS], bbopt=bbopt[itS],
                                        track_cost=track_cost, eta=eta, rescale=rescale) # optimise bond tensor
                
                lsn, rsn = decomposeBT(BT_new, length(W), j, (j+1); chi_max=chi_max, cutoff=cutoff, going_left=false, dtype=dtype)
                update_caches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=false)
                W[j] = lsn
                W[(j+1)] = rsn
            end
            # now we do optimisation over link between originally terminal sites of MPS
            
            finish = time()

            time_elapsed = finish - start
            
            # add time taken for full sweep 
            println("Finished right sweep $itS.")

            # compute the loss and acc on both training and validation sets
            train_loss, train_acc = MSE_loss_acc(W, training_states)
            val_loss, val_acc = MSE_loss_acc(W, validation_states)
            test_loss, test_acc = MSE_loss_acc(W, testing_states)
            train_KL_div = KL_div(W, training_states)
            val_KL_div = KL_div(W, validation_states)
            test_KL_div = KL_div(W, testing_states)

            # dot_errs = test_dot(W, testing_states)

            # if !isempty(dot_errs)
            #     @warn "Found mismatching values between inner() and MPS_contract at Sites: $dot_errs"
            # end
            println("Validation MSE loss: $val_loss | Validation acc. $val_acc." )
            println("Training MSE loss: $train_loss | Training acc. $train_acc." )
            println("Testing MSE loss: $test_loss | Testing acc. $test_acc." )
            println("")
            println("Validation KL Divergence: $val_KL_div.")
            println("Training KL Divergence: $train_KL_div.")
            println("Test KL Divergence: $test_KL_div.")

            running_train_loss = train_loss
            running_val_loss = val_loss

            push!(training_information["train_loss"], train_loss)
            push!(training_information["train_acc"], train_acc)
            push!(training_information["val_loss"], val_loss)
            push!(training_information["val_acc"], val_acc)
            push!(training_information["test_loss"], test_loss)
            push!(training_information["test_acc"], test_acc)
            push!(training_information["time_taken"], time_elapsed)
            push!(training_information["train_KL_div"], train_KL_div)
            push!(training_information["val_KL_div"], val_KL_div)
            push!(training_information["test_KL_div"], test_KL_div)
            push!(test_lists, test_list)
        end
    elseif algorithm == "both_one"
        for itS = 1:nsweeps
            test_list = []
            start = time()
            println("Using optimiser $(bbopt[itS].name) with the \"$(bbopt[itS].fl)\" algorithm")
            if itS % 2 == 1
                println("Starting left sweeep: [$itS/$nsweeps]")
                LE, RE = construct_caches(W, training_states; going_left=true, dtype = dtype)
                push!(test_list, find_label(W)[1])
                for j = (length(sites)-1):-1:1
                    #println(W)
                    #print("Bond $j")
                    # j tracks the LEFT site in the bond tensor (irrespective of sweep direction)
                    BT = W[j] * W[(j+1)] # create bond tensor
                    BT_new = apply_update(W, BT, LE, RE, j, (j+1), training_states; iters=update_iters, verbosity=verbosity, 
                                            dtype=dtype, lg_iter=lg_iter[itS], bbopt=bbopt[itS],
                                            track_cost=track_cost, eta=eta, rescale = rescale) # optimise bond tensor
                    # if length(testlist) < 1
                    #     push!(testlist, BT_new)
                    # end
                    # decompose the bond tensor using SVD and truncate according to chi_max and cutoff
                    lsn, rsn = decomposeBT(BT_new, length(W), j, (j+1); chi_max=chi_max, cutoff=cutoff, going_left=true, dtype=dtype)
                        
                    # update the caches to reflect the new tensors
                    update_caches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=true)
                    # place the updated sites back into the MPS
                    W[j] = lsn
                    W[(j+1)] = rsn
                    push!(test_list, find_label(W)[1])
                end
            
                # now we do optimisation over link between originally terminal sites of MPS - f(x) goes from site[1] to site[end]
                left_id = length(sites)
                right_id = 1
                BT = W[left_id] * W[right_id] # f(x) will be on site[1]
                #println(W)
                BT_new = apply_update(W, BT, nothing, nothing, left_id, right_id, training_states; iters=update_iters, verbosity=verbosity, 
                                            dtype=dtype, lg_iter=lg_iter[itS], bbopt=bbopt[itS],
                                            track_cost=track_cost, eta=eta, rescale = rescale) # optimise bond tensor
                lsn, rsn = decomposeBT(BT_new, length(W), left_id, right_id; chi_max=chi_max, cutoff=cutoff, going_left=true, dtype=dtype)
                
                W[left_id] = lsn
                W[right_id] = rsn
                push!(test_list, find_label(W)[1])
                println("Left sweep finished.")
            elseif itS % 2 == 0
                println("Starting right sweeep: [$itS/$nsweeps]")
                push!(test_list, find_label(W)[1])
                # now go other direction - f(x) goes from site[end] to site 1
                left_id = length(sites)
                right_id = 1
                BT = W[left_id] * W[right_id] # f(x) will be on site[end]
                #println(W)
                BT_new = apply_update(W, BT, nothing, nothing, left_id, right_id, training_states; iters=update_iters, verbosity=verbosity, 
                                            dtype=dtype, lg_iter=lg_iter[itS], bbopt=bbopt[itS],
                                            track_cost=track_cost, eta=eta, rescale = rescale) # optimise bond tensor
                lsn, rsn = decomposeBT(BT_new, length(W), left_id, right_id; chi_max=chi_max, cutoff=cutoff, going_left=false, dtype=dtype)
                
                W[left_id] = lsn
                W[right_id] = rsn
                push!(test_list, find_label(W)[1])
                LE, RE = construct_caches(W, training_states; going_left=false)
                for j = 1:(length(sites)-1)
                    #print("Bond $j")
                    BT = W[j] * W[(j+1)]
                    BT_new = apply_update(BT, LE, RE, j, (j+1), training_states; iters=update_iters, verbosity=verbosity, 
                                            dtype=dtype, lg_iter=lg_iter[itS], bbopt=bbopt[itS],
                                            track_cost=track_cost, eta=eta, rescale=rescale) # optimise bond tensor
                    
                    lsn, rsn = decomposeBT(BT_new, length(W), j, (j+1); chi_max=chi_max, cutoff=cutoff, going_left=false, dtype=dtype)
                    update_caches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=false)
                    W[j] = lsn
                    W[(j+1)] = rsn
                    push!(test_list, find_label(W)[1])
                end
                println("Right sweep finished.")
            end
            finish = time()

            time_elapsed = finish - start

            # compute the loss and acc on both training and validation sets
            train_loss, train_acc = MSE_loss_acc(W, training_states)
            val_loss, val_acc = MSE_loss_acc(W, validation_states)
            test_loss, test_acc = MSE_loss_acc(W, testing_states)
            train_KL_div = KL_div(W, training_states)
            val_KL_div = KL_div(W, validation_states)
            test_KL_div = KL_div(W, testing_states)

            # dot_errs = test_dot(W, testing_states)

            # if !isempty(dot_errs)
            #     @warn "Found mismatching values between inner() and MPS_contract at Sites: $dot_errs"
            # end
            println("Validation MSE loss: $val_loss | Validation acc. $val_acc." )
            println("Training MSE loss: $train_loss | Training acc. $train_acc." )
            println("Testing MSE loss: $test_loss | Testing acc. $test_acc." )
            println("")
            println("Validation KL Divergence: $val_KL_div.")
            println("Training KL Divergence: $train_KL_div.")
            println("Test KL Divergence: $test_KL_div.")

            running_train_loss = train_loss
            running_val_loss = val_loss

            push!(training_information["train_loss"], train_loss)
            push!(training_information["train_acc"], train_acc)
            push!(training_information["val_loss"], val_loss)
            push!(training_information["val_acc"], val_acc)
            push!(training_information["test_loss"], test_loss)
            push!(training_information["test_acc"], test_acc)
            push!(training_information["time_taken"], time_elapsed)
            push!(training_information["train_KL_div"], train_KL_div)
            push!(training_information["val_KL_div"], val_KL_div)
            push!(training_information["test_KL_div"], test_KL_div)
            push!(test_lists, test_list)
        end
    elseif algorithm == "both_two"
        for itS = 1:nsweeps
            test_list = []
            start = time()
            println("Using optimiser $(bbopt[itS].name) with the \"$(bbopt[itS].fl)\" algorithm")
            if itS % 4 == 1 || itS % 4 == 2
                println("Starting left sweeep: [$itS/$nsweeps]")
                LE, RE = construct_caches(W, training_states; going_left=true, dtype = dtype)
                push!(test_list, find_label(W)[1])
                for j = (length(sites)-1):-1:1
                    #println(W)
                    #print("Bond $j")
                    # j tracks the LEFT site in the bond tensor (irrespective of sweep direction)
                    BT = W[j] * W[(j+1)] # create bond tensor
                    BT_new = apply_update(W, BT, LE, RE, j, (j+1), training_states; iters=update_iters, verbosity=verbosity, 
                                            dtype=dtype, lg_iter=lg_iter[itS], bbopt=bbopt[itS],
                                            track_cost=track_cost, eta=eta, rescale = rescale) # optimise bond tensor
                    # if length(testlist) < 1
                    #     push!(testlist, BT_new)
                    # end
                    # decompose the bond tensor using SVD and truncate according to chi_max and cutoff
                    lsn, rsn = decomposeBT(BT_new, length(W), j, (j+1); chi_max=chi_max, cutoff=cutoff, going_left=true, dtype=dtype)
                        
                    # update the caches to reflect the new tensors
                    update_caches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=true)
                    # place the updated sites back into the MPS
                    W[j] = lsn
                    W[(j+1)] = rsn
                    push!(test_list, find_label(W)[1])
                end
            
                # now we do optimisation over link between originally terminal sites of MPS - f(x) goes from site[1] to site[end]
                left_id = length(sites)
                right_id = 1
                BT = W[left_id] * W[right_id] # f(x) will be on site[1]
                #println(W)
                BT_new = apply_update(W, BT, nothing, nothing, left_id, right_id, training_states; iters=update_iters, verbosity=verbosity, 
                                            dtype=dtype, lg_iter=lg_iter[itS], bbopt=bbopt[itS],
                                            track_cost=track_cost, eta=eta, rescale = rescale) # optimise bond tensor
                lsn, rsn = decomposeBT(BT_new, length(W), left_id, right_id; chi_max=chi_max, cutoff=cutoff, going_left=true, dtype=dtype)
                
                W[left_id] = lsn
                W[right_id] = rsn
                push!(test_list, find_label(W)[1])
                println("Left sweep finished.")
            elseif itS % 4 == 3 || itS % 4 == 0
                println("Starting right sweeep: [$itS/$nsweeps]")
                push!(test_list, find_label(W)[1])
                # now go other direction - f(x) goes from site[end] to site 1
                left_id = length(sites)
                right_id = 1
                BT = W[left_id] * W[right_id] # f(x) will be on site[end]
                #println(W)
                BT_new = apply_update(W, BT, nothing, nothing, left_id, right_id, training_states; iters=update_iters, verbosity=verbosity, 
                                            dtype=dtype, lg_iter=lg_iter[itS], bbopt=bbopt[itS],
                                            track_cost=track_cost, eta=eta, rescale = rescale) # optimise bond tensor
                lsn, rsn = decomposeBT(BT_new, length(W), left_id, right_id; chi_max=chi_max, cutoff=cutoff, going_left=false, dtype=dtype)
                
                W[left_id] = lsn
                W[right_id] = rsn
                push!(test_list, find_label(W)[1])
                LE, RE = construct_caches(W, training_states; going_left=false)
                for j = 1:(length(sites)-1)
                    #print("Bond $j")
                    BT = W[j] * W[(j+1)]
                    BT_new = apply_update(BT, LE, RE, j, (j+1), training_states; iters=update_iters, verbosity=verbosity, 
                                            dtype=dtype, lg_iter=lg_iter[itS], bbopt=bbopt[itS],
                                            track_cost=track_cost, eta=eta, rescale=rescale) # optimise bond tensor
                    
                    lsn, rsn = decomposeBT(BT_new, length(W), j, (j+1); chi_max=chi_max, cutoff=cutoff, going_left=false, dtype=dtype)
                    update_caches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=false)
                    W[j] = lsn
                    W[(j+1)] = rsn
                    push!(test_list, find_label(W)[1])
                end
                println("Right sweep finished.")
            end
            finish = time()

            time_elapsed = finish - start

            # compute the loss and acc on both training and validation sets
            train_loss, train_acc = MSE_loss_acc(W, training_states)
            val_loss, val_acc = MSE_loss_acc(W, validation_states)
            test_loss, test_acc = MSE_loss_acc(W, testing_states)
            train_KL_div = KL_div(W, training_states)
            val_KL_div = KL_div(W, validation_states)
            test_KL_div = KL_div(W, testing_states)

            # dot_errs = test_dot(W, testing_states)

            # if !isempty(dot_errs)
            #     @warn "Found mismatching values between inner() and MPS_contract at Sites: $dot_errs"
            # end
            println("Validation MSE loss: $val_loss | Validation acc. $val_acc." )
            println("Training MSE loss: $train_loss | Training acc. $train_acc." )
            println("Testing MSE loss: $test_loss | Testing acc. $test_acc." )
            println("")
            println("Validation KL Divergence: $val_KL_div.")
            println("Training KL Divergence: $train_KL_div.")
            println("Test KL Divergence: $test_KL_div.")

            running_train_loss = train_loss
            running_val_loss = val_loss

            push!(training_information["train_loss"], train_loss)
            push!(training_information["train_acc"], train_acc)
            push!(training_information["val_loss"], val_loss)
            push!(training_information["val_acc"], val_acc)
            push!(training_information["test_loss"], test_loss)
            push!(training_information["test_acc"], test_acc)
            push!(training_information["time_taken"], time_elapsed)
            push!(training_information["train_KL_div"], train_KL_div)
            push!(training_information["val_KL_div"], val_KL_div)
            push!(training_information["test_KL_div"], test_KL_div)
            push!(test_lists, test_list)
        end
    end
    normalize!(W)
    println("\nMPS normalised!\n")
    # compute the loss and acc on both training and validation sets post normalisation
    train_loss, train_acc = MSE_loss_acc(W, training_states)
    val_loss, val_acc = MSE_loss_acc(W, validation_states)
    test_loss, test_acc = MSE_loss_acc(W, testing_states)
    train_KL_div = KL_div(W, training_states)
    val_KL_div = KL_div(W, validation_states)
    test_KL_div = KL_div(W, testing_states)


    println("Validation MSE loss: $val_loss | Validation acc. $val_acc." )
    println("Training MSE loss: $train_loss | Training acc. $train_acc." )
    println("Testing MSE loss: $test_loss | Testing acc. $test_acc." )
    println("")
    println("Validation KL Divergence: $val_KL_div.")
    println("Training KL Divergence: $train_KL_div.")
    println("Test KL Divergence: $test_KL_div.")

    running_train_loss = train_loss
    running_val_loss = val_loss

    push!(training_information["train_loss"], train_loss)
    push!(training_information["train_acc"], train_acc)
    push!(training_information["val_loss"], val_loss)
    push!(training_information["val_acc"], val_acc)
    push!(training_information["test_loss"], test_loss)
    push!(training_information["test_acc"], test_acc)
    push!(training_information["time_taken"], training_information["time_taken"][end]) # no time has passed
    push!(training_information["train_KL_div"], train_KL_div)
    push!(training_information["val_KL_div"], val_KL_div)
    push!(training_information["test_KL_div"], test_KL_div)
   
    return W, training_information, training_states, testing_states, test_lists

end
# if abspath(PROGRAM_FILE) == @__FILE__ 
#     (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits_txt("LogLoss/datasets/ECG_train.txt", 
#         "LogLoss/datasets/ECG_val.txt", "LogLoss/datasets/ECG_test.txt")

#     X_train_final = vcat(X_train, X_val)
#     y_train_final = vcat(y_train, y_val)

#     training_data_matrix = Matrix{Float64}(training_data[2:end, :])
#     raining_data_matrix = Matrix(transpose(training_data_matrix))

#     setprecision(BigFloat, 128)
#     Rdtype = Float64

#     verbosity = 0


#     opts=Options(; nsweeps=1, chi_max=20,  update_iters=1, verbosity=verbosity, dtype=Rdtype, lg_iter=KLD_iter,
#     bbopt=BBOpt("CustomGD"), track_cost=false, eta=0.05, rescale = [false, true], d=2, encoding=Encoding("Stoudenmire"))
#     W, info, train_states, test_states = fitMPS(X_train, y_train, X_val, y_val, X_test, y_test; random_state=456, chi_init=4, opts=opts)

#     # saveMPS(W, "LogLoss/saved/loglossout.h5")
#     print_opts(opts)

#     summary = get_training_summary(W, train_states, test_states; print_stats=true);
#     # plot_training_summary(info)

#     sweep_summary(info)
# end

