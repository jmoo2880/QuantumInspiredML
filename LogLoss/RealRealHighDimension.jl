using GenericLinearAlgebra
using ITensors
using Optim
using OptimKit
using Random
using Distributions
using DelimitedFiles
using Folds
using JLD2
using StatsBase
using Plots


include("structs.jl")
include("encodings.jl")
include("summary.jl")
include("utils.jl")
include("loss_functions.jl")



function generate_startingMPS(chi_init, site_indices::Vector{Index{Int64}};
    num_classes = 2, random_state=nothing, opts::Options=Options(), label_tag::String="f(x)")
    """Generate the starting weight MPS, W using values sampled from a 
    Gaussian (normal) distribution. Accepts a chi_init parameter which
    specifies the initial (uniform) bond dimension of the MPS."""
    verbosity = opts.verbosity

    if random_state !== nothing
        # use seed if specified
        Random.seed!(random_state)
        verbosity >= 0 && println("Generating initial weight MPS with bond dimension χ_init = $chi_init
        using random state $random_state.")
    else
        verbosity >= 0 && println("Generating initial weight MPS with bond dimension χ_init = $chi_init.")
    end

    W = randomMPS(opts.dtype, site_indices, linkdims=chi_init)

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

    return W

end

function construct_caches(W::MPS, training_pstates::TimeseriesIterable; going_left=true, dtype::DataType=ComplexF64)
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


function complexify(B::ITensor, C_index::Index{Int64}; dtype::DataType=ComplexF64)
    """Converts a real 2x{s} dimension itensor into a Complex {s} dimension itensor. Reduces the rank from rank{s}+1 to rank{s} by removing the first index"""
    ib = inds(B)
    C_index, c_inds... = ib
    B_ra = NDTensors.array(B, ib) # should return a view


    re_part = selectdim(B_ra, 1,1);
    im_part = selectdim(B_ra, 1,2);

    return ITensor(dtype, complex.(re_part,im_part), c_inds)
end





function loss_grad_enforce_real(tsep::TrainSeparate, BT::ITensor, LE::PCache, RE::PCache,
    ETSs::EncodedTimeseriesSet, lid::Int, rid::Int, C_index::Union{Index{Int64},Nothing}; dtype::DataType=ComplexF64, loss_grad::Function=loss_grad_KLD)
    """Function for computing the loss function and the gradient over all samples using a left and right cache. 
        Takes a real itensor and will convert it to complex before calling loss_grad if dtype is complex. Returns a real gradient. """
    

    if isnothing(C_index) # the itensor is real
        loss, grad = loss_grad(tsep, BT, LE, RE, ETSs, lid, rid)
    else
        # pass in a complex itensor
        BT_c = complexify(BT, C_index; dtype=dtype)

        loss, grad = loss_grad(tsep, BT_c, LE, RE, ETSs, lid, rid)

        grad = realise(grad, C_index; dtype=dtype)
    end


    return loss, grad

end

function loss_grad!(tsep::TrainSeparate, F,G,B_flat::AbstractArray, b_inds::Tuple{Vararg{Index{Int64}}}, LE::PCache, RE::PCache,
    ETSs::EncodedTimeseriesSet, lid::Int, rid::Int, C_index::Union{Index{Int64},Nothing}; dtype::DataType=ComplexF64, loss_grad::Function=loss_grad_KLD)

    """Calculates the loss and gradient in a way compatible with Optim. Takes a flat, real array and converts it into an itensor before it passes it loss_grad """
    BT = itensor(real(dtype), B_flat, b_inds) # convert the bond tensor from a flat array to an itensor

    loss, grad = loss_grad_enforce_real(tsep, BT, LE, RE, ETSs, lid, rid, C_index; dtype=dtype, loss_grad=loss_grad)

    if !isnothing(G)
        G .= NDTensors.array(grad,b_inds)
    end

    if !isnothing(F)
        return loss
    end

end

function custGD(tsep::TrainSeparate, BT_init::ITensor, LE::PCache, RE::PCache, lid::Int, rid::Int, ETSs::EncodedTimeseriesSet;
    iters=10, verbosity::Real=1, dtype::DataType=ComplexF64, loss_grad::Function=loss_grad_KLD, track_cost::Bool=false, eta::Real=0.01)
    BT_old = BT_init
    BT_new = BT_old # Julia and its damn scoping

    for i in 1:iters
        # get the gradient
        loss, grad = loss_grad(tsep, BT_old, LE, RE, ETSs, lid, rid)
        #zygote_gradient_per_batch(bt_old, LE, RE, pss, lid, rid)
        # update the bond tensor
        BT_new = BT_old - eta * grad
        if verbosity >=1 && track_cost
            # get the new loss
            println("Loss at step $i: $loss")
        end

        BT_old = BT_new
    end

    return BT_new
end

function TSGO(tsep::TrainSeparate, BT_init::ITensor, LE::PCache, RE::PCache, lid::Int, rid::Int, ETSs::EncodedTimeseriesSet;
    iters=10, verbosity::Real=1, dtype::DataType=ComplexF64, loss_grad::Function=loss_grad_KLD, track_cost::Bool=false, eta::Real=0.01)
    BT_old = BT_init
    BT_new = BT_old # Julia and its damn scoping

    for i in 1:iters
        # get the gradient
        loss, grad = loss_grad(tsep, BT_old, LE, RE, ETSs, lid, rid)
        #zygote_gradient_per_batch(bt_old, LE, RE, pss, lid, rid)
        # update the bond tensor


        grad /= norm(grad)  # the TSGO difference
       

        BT_new = BT_old - eta * grad
        if verbosity >=1 && track_cost
            # get the new loss
            println("Loss at step $i: $loss")
        end

        BT_old = BT_new
    end
    return BT_new
end

function apply_update(tsep::TrainSeparate, BT_init::ITensor, LE::PCache, RE::PCache, lid::Int, rid::Int,
    ETSs::EncodedTimeseriesSet; iters=10, verbosity::Real=1, dtype::DataType=ComplexF64, loss_grad::Function=loss_grad_KLD, bbopt::BBOpt=BBOpt("Optim"),
    track_cost::Bool=false, eta::Real=0.01, rescale::Tuple{Bool,Bool} = (false, true))
    """Apply update to bond tensor using the method specified by BBOpt. Will normalise B before and/or after it computes the update B+dB depending on the value of rescale [before::Bool,after::Bool]"""

    iscomplex = !(dtype <: Real)

    if rescale[1]
        normalize!(BT_init)
    end

    if bbopt.name == "CustomGD"
        if uppercase(bbopt.fl) == "GD"
            BT_new = custGD(tsep, BT_init, LE, RE, lid, rid, ETSs; iters=iters, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad, track_cost=track_cost, eta=eta)

        elseif uppercase(bbopt.fl) == "TSGO"
            BT_new = TSGO(tsep, BT_init, LE, RE, lid, rid, ETSs; iters=iters, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad, track_cost=track_cost, eta=eta)

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
            fgcustom! = (F,G,B) -> loss_grad!(tsep, F, G, B, bt_inds, LE, RE, ETSs, lid, rid, C_index; dtype=dtype, loss_grad=loss_grad)
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

            lg = BT -> loss_grad_enforce_real(tsep, BT, LE, RE, ETSs, lid, rid, C_index; dtype=dtype, loss_grad=loss_grad)
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
        loss, grad = loss_grad(tsep, BT_new, LE, RE, ETSs, lid, rid)
        println("Loss at site $lid*$rid: $loss")
    end

    return BT_new

end

function decomposeBT(BT::ITensor, lid::Int, rid::Int; 
    chi_max=nothing, cutoff=nothing, going_left=true, dtype::DataType=ComplexF64)
    """Decompose an updated bond tensor back into two tensors using SVD"""
    left_site_index = findindex(BT, "n=$lid")
    label_index = findindex(BT, "f(x)")


    if going_left
        # need to make sure the label index is transferred to the next site to be updated
        if lid == 1
            U, S, V = svd(BT, (left_site_index, label_index); maxdim=chi_max, cutoff=cutoff)
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
            U, S, V = svd(BT, (left_site_index); maxdim=chi_max, cutoff=cutoff)
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
    LE::PCache, RE::PCache, lid::Int, rid::Int, product_states::TimeseriesIterable; going_left::Bool=true)
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

function fitMPS(X_train::Matrix, y_train::Vector, X_val::Matrix, y_val::Vector, X_test::Matrix, y_test::Vector; random_state=nothing, chi_init=4, opts::Options=Options(), test_run=false, return_sample_encoding::Bool=false)

    # first, create the site indices for the MPS and product states 
    num_mps_sites = size(X_train)[2]
    sites = siteinds(opts.d, num_mps_sites)

    # generate the starting MPS with uniform bond dimension chi_init and random values (with seed if provided)
    num_classes = length(unique(y_train))
    W = generate_startingMPS(chi_init, sites; num_classes=num_classes, random_state=random_state, opts=opts)

    return fitMPS(W, X_train, y_train, X_val, y_val, X_test, y_test; opts=opts, test_run=test_run, return_sample_encoding=return_sample_encoding)
    
end

function fitMPS(W::MPS, X_train::Matrix, y_train::Vector, X_val::Matrix, y_val::Vector, X_test::Matrix, y_test::Vector; opts::Options=Options(), test_run=false, return_sample_encoding::Bool=false, sortrng = MersenneTwister(1234))

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
    #  TODO permutedims earlier on in the code, check which array order is a good convention
    scaler = fit_scaler(RobustSigmoidTransform, X_train);
    range = opts.encoding.range
    X_train_scaled = permutedims(transform_data(scaler, X_train; range=range, minmax_output=opts.minmax))
    X_val_scaled = permutedims(transform_data(scaler, X_val; range=range, minmax_output=opts.minmax))
    X_test_scaled = permutedims(transform_data(scaler, X_test; range=range, minmax_output=opts.minmax))

    


    # generate product states using rescaled data
    if opts.encoding.iscomplex
        if opts.dtype <: Real
            error("Using a complex valued encoding but the MPS is real")
        end

    elseif !(opts.dtype <: Real)
        @warn "Using a complex valued MPS but the encoding is real"
    end

    @assert !(opts.encode_classes_separately && opts.encoding.isbalanced) "Attempting to balance classes while encoding separately is ambiguous"

    # generate the starting MPS with uniform bond dimension chi_init and random values (with seed if provided)
    classes = unique(vcat(y_train, y_test, y_val))
    num_classes = length(classes)
    _, l_index = find_label(W)

    @assert num_classes == ITensors.dim(l_index) "Number of Classes in the training data doesn't match the dimension of the label index!"

    # construct a key for the label index
    # may add support for non integer classes later
    # try
    #     sort!(classes)
    # catch e
    #     if e isa MethodError && e.f == isless
    #         if opts.allow_unsorted_class_labels
    #             @warn "The class labels contain the unsortable pair $(e.args). This may cause the summary statistics to be misleading"
    #         else
    #             throw(ArgumentError("The class labels contain the unsortable pair $(e.args). Set opts.allow_unsorted_class_labels to true to allow this type of input."))
    #         end
    #     else
    #         throw(e)
    #     end
    # end

    @assert eltype(classes) <: Integer "Classes must be integers"
    sort!(classes)
    class_keys = Dict(zip(classes, 1:num_classes))

    

    # sort the arrays by class. This will provide a speedup if classes are trained/encoded separately
    # the loss grad function assumes the timeseries are sorted! Removing the sorting now breaks the algorithm
    order_tr = sortperm(y_train)
    order_val = sortperm(y_val)
    order_test = sortperm(y_test)

    y_train = y_train[order_tr]
    y_val = y_val[order_val]
    y_test = y_test[order_test]

    X_train_scaled .= X_train_scaled[:, order_tr]
    X_val_scaled .= X_val_scaled[:, order_val]
    X_test_scaled .= X_test_scaled[:, order_test]


    s = EncodeSeparate{opts.encode_classes_separately}()
    training_states, enc_args_tr = encode_dataset(s, X_train_scaled, y_train, "train", sites; opts=opts, class_keys=class_keys)
    validation_states, enc_args_val = encode_dataset(s, X_val_scaled, y_val, "valid", sites; opts=opts, class_keys=class_keys)
    testing_states, enc_args_test = encode_dataset(s, X_test_scaled, y_test, "test", sites; opts=opts, class_keys=class_keys)
    
    enc_args = vcat(enc_args_tr, enc_args_val, enc_args_test)

    if return_sample_encoding || test_run
        num_ts = 500
        test_enc_meta = encoding_test(s, X_train_scaled, y_train, "test_enc", sites; opts=opts, num_ts=num_ts, class_keys=class_keys)
        test_enc = test_enc_meta.timeseries

        a,b = opts.encoding.range
        stp = (b-a)/(num_ts-1)
        xs = collect(a:stp:b)

        if opts.encode_classes_separately
            sample_states = [hcat([Vector.(te.pstate) for te in test_enc[(1 + num_ts*(ci-1)):(num_ts*ci)]]...) for ci in 1:num_classes]
            num_samps = num_classes
        else
            num_samps=1
            sample_states = [hcat([Vector.(te.pstate) for te in test_enc]...)]
        end
    end

    if test_run
        num_plts = 4
        plotinds = shuffle(MersenneTwister(), 1:num_mps_sites)[1:num_plts]
        opts.verbosity > -1 && println("Choosing $num_plts timepoints to plot the basis of at random")

       

        p1s = [histogram(X_train_scaled[i,:]; bins=25, title="Timepoint $i/$num_mps_sites", legend=:none, xlims=(0,1)) for i in plotinds]
        p2s = [ [plot(xs, real.(transpose(hcat(sample_states[s][i,:]...))); xlabel="x", ylabel="real{Encoding}", title="class $(unique([y_train; y_val; y_test])[s])", legend=:none) for i in plotinds] for s in 1:num_samps]

        ps = plot(vcat(p1s,p2s...)..., layout=(1 + num_samps,num_plts), size=(1600,600*(num_samps+1)))
        opts.verbosity > -1 && println("Encoding completed! Returning initial states without training.")
        return W, [], training_states, testing_states, ps
    end

    extra_args = []

    if return_sample_encoding
        push!(extra_args,  xs)
        push!(extra_args,  sample_states)
    end

    if opts.return_encoding_meta_info
        push!(extra_args, enc_args)
    end

    return [fitMPS(W, training_states, validation_states, testing_states; opts=opts, test_run=test_run)..., extra_args... ]
end

function fitMPS(training_states_meta::EncodedTimeseriesSet, validation_states_meta::EncodedTimeseriesSet, testing_states_meta::EncodedTimeseriesSet;
    random_state=nothing, chi_init=4, opts::Options=Options(), test_run=false) # optimise bond tensor)
    # first, create the site indices for the MPS and product states 

    training_states = training_states_meta.timeseries
    testing_states = testing_states_meta.timeseries

    @assert opts.d == ITensors.dim(siteinds(training_states[1].pstate)[1]) "Dimension of site indices must match feature map dimension"
    sites = siteinds(testing_states[1].pstate)

    # generate the starting MPS with unfirom bond dimension chi_init and random values (with seed if provided)
    num_classes = length(unique([ps.label for ps in training_states]))
    W = generate_startingMPS(chi_init, sites; num_classes=num_classes, random_state=random_state, opts=opts)

    fitMPS(W, training_states_meta, validation_states_meta, testing_states_meta; opts=opts, test_run=test_run)

end


"""
Options
    nsweeps::Int # Number of MPS optimisation sweeps to perform (Both forwards and Backwards)
    chi_max::Int # Maximum bond dimension allowed within the MPS during the SVD step
    cutoff::Float64 # Size based cutoff for the number of singular values in the SVD (See Itensors SVD documentation)
    update_iters::Int # Maximum number of optimiser iterations to perform for each bond tensor optimisation. E.G. The number of steps of (Conjugate) Gradient Descent used by CustomGD, Optim or OptimKit
    verbosity::Int # Represents how much info to print to the terminal while optimising the MPS. Higher numbers mean more output
    dtype::DataType # The datatype of the elements of the MPS as well as the encodings. Set to a complex value only if necessary for the encoding type. Supports the arbitrary precsion types BigFloat and Complex{BigFloat}
    loss_grad::Function # The type of cost function to use for training the MPS, typically Mean Squared Error or KL Divergence. Must return a vector or pair [cost, dC/dB]
    bbopt::BBOpt # Which Black Box optimiser to use, options are Optim or OptimKit derived solvers which work well for MSE costs, or CustomGD, which is a standard gradient descent algorithm with fixed stepsize which seems to give the best results for KLD cost 
    track_cost::Bool # Whether to print the cost at each Bond tensor site to the terminal while training, mostly useful for debugging new cost functions or optimisers
    eta::Float64 # The gradient descent step size for CustomGD. For Optim and OptimKit this serves as the initial step size guess input into the linesearch
    rescale::Tuple{Bool,Bool} # Has the form rescale = (before::Bool, after::Bool) and tells the optimisor where to enforce the normalisation of the MPS during training, either calling normalise!(BT) before or after BT is updated. Note that for an MPS that starts in canonical form, rescale = (true,true) will train identically to rescale = (false, true) but may be less performant.
    d::Int # The dimension of the feature map or "Encoding". This is the true maximum dimension of the feature vectors. For a splitting encoding, d = num_splits * aux_basis_dim
    aux_basis_dim::Int # If encoding::SplitBasis, serves as the auxilliary dimension of a basis mapped onto the split encoding, so that num_bins = d / aux_basis_dim. Unused if encoding::Basis
    encoding::Encoding # The type of encoding to use, see structs.jl and encodings.jl for the various options. Can be just a time (in)dependent orthonormal basis, or a time (in)dependent basis mapped onto a number of "splits" which distribute tighter basis functions where the sites of a timeseries are more likely to be measured.  
    train_classes_separately::Bool # whether the the trainer takes the average MPS loss over all classes or whether it considers each class as a separate problem
    encode_classes_separately::Bool # only relevant for a histogram splitbasis. If true, then the histogram used to determine the bin widths for encoding class A is composed of only data from class A, etc. Functionally, this causes the encoding method to vary depending on the class
    return_encoding_meta_info::Bool # Whether to return the normalised data as well as the histogram bins for the splitbasis types
    """
function fitMPS(W::MPS, training_states_meta::EncodedTimeseriesSet, validation_states_meta::EncodedTimeseriesSet, testing_states_meta::EncodedTimeseriesSet; 
     opts::Options=Options(), test_run=false) # optimise bond tensor)

    if test_run
        opts.verbosity > -1 && println("Encoding completed! Returning initial states without training.")
        return W, [], training_states, testing_states, []
    end

    @unpack_Options opts # unpacks the attributes of opts into the local namespace
    tsep = TrainSeparate{train_classes_separately}() # value type to determine training style

    

    training_states = training_states_meta.timeseries
    validation_states = validation_states_meta.timeseries
    testing_states = testing_states_meta.timeseries

    if opts.encode_classes_separately && !opts.train_classes_separately
        @warn "Classes are encoded separately, but not trained separately"
    elseif opts.train_classes_separately && !opts.encode_classes_separately
        @warn "Classes are trained separately, but not encoded separately"
    end
    
    verbosity > -1 && println("Using $update_iters iterations per update.")
    # construct initial caches
    LE, RE = construct_caches(W, training_states; going_left=true, dtype=dtype)

    # compute initial training and validation acc/loss
    init_train_loss, init_train_acc = MSE_loss_acc(W, training_states)
    init_val_loss, init_val_acc = MSE_loss_acc(W, validation_states)
    init_test_loss, init_test_acc = MSE_loss_acc(W, testing_states)

    train_KL_div = KL_div(W, training_states)
    val_KL_div = KL_div(W, validation_states)
    init_KL_div = KL_div(W, testing_states)
    sites = siteinds(W)

    # print loss and acc

    verbosity > -1 && println("Validation MSE loss: $init_val_loss | Validation acc. $init_val_acc." )
    verbosity > -1 && println("Training MSE loss: $init_train_loss | Training acc. $init_train_acc." )
    verbosity > -1 && println("Testing MSE loss: $init_test_loss | Testing acc. $init_test_acc." )
    verbosity > -1 && println("")
    verbosity > -1 && println("Validation KL Divergence: $val_KL_div.")
    verbosity > -1 && println("Training KL Divergence: $train_KL_div.")
    verbosity > -1 && println("Test KL Divergence: $init_KL_div.")


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
    if typeof(loss_grad) <: AbstractArray
        @assert length(loss_grad) == nsweeps "loss_grad(...)::(loss,grad) must be a loss function or an array of loss functions with length nsweeps"
    elseif typeof(loss_grad) <: Function
        loss_grad = [loss_grad for _ in 1:nsweeps]
    else
        error("loss_grad(...)::(loss,grad) must be a loss function or an array of loss functions with length nsweeps")
    end

    if train_classes_separately && !(eltype(loss_grad) <: typeof(loss_grad_KLD))
        @warn "Classes will be trained separately, but the cost function _may_ depend on measurements of multiple classes. Switch to a KLD style cost function or ensure your custom cost function depends only on one class at a time."
    end

    if typeof(bbopt) <: AbstractArray
        @assert length(bbopt) == nsweeps "bbopt must be an optimiser or an array of optimisers to use with length nsweeps"
    elseif typeof(bbopt) <: BBOpt
        bbopt = [bbopt for _ in 1:nsweeps]
    else
        error("bbopt must be an optimiser or an array of optimisers to use with length nsweeps")
    end

    # start the sweep
    for itS = 1:nsweeps
        
        start = time()
        verbosity > -1 && println("Using optimiser $(bbopt[itS].name) with the \"$(bbopt[itS].fl)\" algorithm")
        verbosity > -1 && println("Starting backward sweeep: [$itS/$nsweeps]")

        for j = (length(sites)-1):-1:1
            #print("Bond $j")
            # j tracks the LEFT site in the bond tensor (irrespective of sweep direction)
            BT = W[j] * W[(j+1)] # create bond tensor
            BT_new = apply_update(tsep, BT, LE, RE, j, (j+1), training_states_meta; iters=update_iters, verbosity=verbosity, 
                                    dtype=dtype, loss_grad=loss_grad[itS], bbopt=bbopt[itS],
                                    track_cost=track_cost, eta=eta, rescale = rescale) # optimise bond tensor

            # decompose the bond tensor using SVD and truncate according to chi_max and cutoff
            lsn, rsn = decomposeBT(BT_new, j, (j+1); chi_max=chi_max, cutoff=cutoff, going_left=true, dtype=dtype)
                
            # update the caches to reflect the new tensors
            update_caches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=true)
            # place the updated sites back into the MPS
            W[j] = lsn
            W[(j+1)] = rsn
        end
    
        # add time taken for backward sweep.
        verbosity > -1 && println("Backward sweep finished.")
        
        # finished a full backward sweep, reset the caches and start again
        # this can be simplified dramatically, only need to reset the LE
        LE, RE = construct_caches(W, training_states; going_left=false)
        
        verbosity > -1 && println("Starting forward sweep: [$itS/$nsweeps]")

        for j = 1:(length(sites)-1)
            #print("Bond $j")
            BT = W[j] * W[(j+1)]
            BT_new = apply_update(tsep, BT, LE, RE, j, (j+1), training_states_meta; iters=update_iters, verbosity=verbosity, 
                                    dtype=dtype, loss_grad=loss_grad[itS], bbopt=bbopt[itS],
                                    track_cost=track_cost, eta=eta, rescale=rescale) # optimise bond tensor

            lsn, rsn = decomposeBT(BT_new, j, (j+1); chi_max=chi_max, cutoff=cutoff, going_left=false, dtype=dtype)
            update_caches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=false)
            W[j] = lsn
            W[(j+1)] = rsn
        end

        LE, RE = construct_caches(W, training_states; going_left=true)
        
        finish = time()

        time_elapsed = finish - start
        
        # add time taken for full sweep 
        verbosity > -1 && println("Finished sweep $itS.")

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
        verbosity > -1 && println("Validation MSE loss: $val_loss | Validation acc. $val_acc." )
        verbosity > -1 && println("Training MSE loss: $train_loss | Training acc. $train_acc." )
        verbosity > -1 && println("Testing MSE loss: $test_loss | Testing acc. $test_acc." )
        verbosity > -1 && println("")
        verbosity > -1 && println("Validation KL Divergence: $val_KL_div.")
        verbosity > -1 && println("Training KL Divergence: $train_KL_div.")
        verbosity > -1 && println("Test KL Divergence: $test_KL_div.")

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
       
    end
    normalize!(W)
    verbosity > -1 && println("\nMPS normalised!\n")
    # compute the loss and acc on both training and validation sets post normalisation
    train_loss, train_acc = MSE_loss_acc(W, training_states)
    val_loss, val_acc = MSE_loss_acc(W, validation_states)
    test_loss, test_acc = MSE_loss_acc(W, testing_states)
    train_KL_div = KL_div(W, training_states)
    val_KL_div = KL_div(W, validation_states)
    test_KL_div = KL_div(W, testing_states)


    verbosity > -1 && println("Validation MSE loss: $val_loss | Validation acc. $val_acc." )
    verbosity > -1 && println("Training MSE loss: $train_loss | Training acc. $train_acc." )
    verbosity > -1 && println("Testing MSE loss: $test_loss | Testing acc. $test_acc." )
    verbosity > -1 && println("")
    verbosity > -1 && println("Validation KL Divergence: $val_KL_div.")
    verbosity > -1 && println("Training KL Divergence: $train_KL_div.")
    verbosity > -1 && println("Test KL Divergence: $test_KL_div.")

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
   
    return W, training_information, training_states_meta, testing_states_meta

end

# Demo 
# (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits_txt("LogLoss/datasets/ECG_train.txt", 
# "LogLoss/datasets/ECG_val.txt", "LogLoss/datasets/ECG_test.txt")

# X_train = vcat(X_train, X_val)
# y_train = vcat(y_train, y_val)


# setprecision(BigFloat, 128)
# Rdtype = Float64

# verbosity = 0
# test_run = true


# opts=Options(; nsweeps=20, chi_max=20,  update_iters=1, verbosity=verbosity, dtype=Complex{Rdtype}, loss_grad=KLD_iter,
# bbopt=BBOpt("CustomGD"), track_cost=false, eta=0.05, rescale = (false, true), d=12, aux_basis_dim=2, encoding=SplitBasis("Hist Split", "Stoudenmire"))



# # saveMPS(W, "LogLoss/saved/loglossout.h5")
# print_opts(opts)

# if test_run
#     W, info, train_states, test_states, p = fitMPS(X_train, y_train, X_val, y_val, X_test, y_test; random_state=456, chi_init=4, opts=opts, test_run=true)
#     plot(p)
# else
#     W, info, train_states, test_states = fitMPS(X_train, y_train, X_val, y_val, X_test, y_test; random_state=456, chi_init=4, opts=opts, test_run=false)

#     print_opts(opts)
#     summary = get_training_summary(W, train_states, test_states; print_stats=true);
#     sweep_summary(info)
# end


