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


function complexify(B::ITensor, C_index::Index{Int64}; dtype::DataType=ComplexF64)
    """Converts a real 2x{s} dimension itensor into a Complex {s} dimension itensor. Reduces the rank from rank{s}+1 to rank{s} by removing the first index"""
    ib = inds(B)
    C_index, c_inds... = ib
    B_ra = NDTensors.array(B, ib) # should return a view


    re_part = selectdim(B_ra, 1,1);
    im_part = selectdim(B_ra, 1,2);

    return ITensor(dtype, complex.(re_part,im_part), c_inds)
end


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

function MSE_iter(BT_c::ITensor, LEP::PCacheCol, REP::PCacheCol,
    product_state::PState, lid::Int, rid::Int) 
    """Computes the Mean squared error loss function derived from KL divergence and its gradient"""


    yhat, phi_tilde = yhat_phitilde(BT_c, LEP, REP, product_state, lid, rid)

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




function KLD_iter(BT_c::ITensor, LEP::PCacheCol, REP::PCacheCol,
    product_state::PState, lid::Int, rid::Int) 
    """Computes the complex valued logarithmic loss function derived from KL divergence and its gradient"""


    yhat, phi_tilde = yhat_phitilde(BT_c, LEP, REP, product_state, lid, rid)

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

function loss_grad(BT::ITensor, LE::PCache, RE::PCache,
    TSs::timeSeriesIterable, lid::Int, rid::Int; lg_iter::Function=KLD_iter)
    """Function for computing the loss function and the gradient over all samples using lg_iter and a left and right cache. 
        Allows the input to be complex if that is supported by lg_iter"""
 
    loss,grad = Folds.mapreduce((LEP,REP, prod_state) -> lg_iter(BT,LEP,REP,prod_state,lid,rid),+, eachcol(LE), eachcol(RE),TSs)
    
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

function custGD(BT_init::ITensor, LE::PCache, RE::PCache, lid::Int, rid::Int, TSs::timeSeriesIterable;
    iters=10, verbosity::Real=1, dtype::DataType=ComplexF64, lg_iter::Function=KLD_iter, track_cost::Bool=false, eta::Real=0.01)
    BT_old = BT_init
    BT_new = BT_old # Julia and its damn scoping

    for i in 1:iters
        # get the gradient
        loss, grad = loss_grad(BT_old, LE, RE, TSs, lid, rid; lg_iter=lg_iter)
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

function TSGO(BT_init::ITensor, LE::PCache, RE::PCache, lid::Int, rid::Int, TSs::timeSeriesIterable;
    iters=10, verbosity::Real=1, dtype::DataType=ComplexF64, lg_iter::Function=KLD_iter, track_cost::Bool=false, eta::Real=0.01)
    BT_old = BT_init
    BT_new = BT_old # Julia and its damn scoping

    for i in 1:iters
        # get the gradient
        loss, grad = loss_grad(BT_old, LE, RE, TSs, lid, rid; lg_iter=lg_iter)
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

function apply_update(BT_init::ITensor, LE::PCache, RE::PCache, lid::Int, rid::Int,
    TSs::timeSeriesIterable; iters=10, verbosity::Real=1, dtype::DataType=ComplexF64, lg_iter::Function=KLD_iter, bbopt::BBOpt=BBOpt("Optim"),
    track_cost::Bool=false, eta::Real=0.01, rescale::Tuple{Bool,Bool} = (false, true))
    """Apply update to bond tensor using the method specified by BBOpt. Will normalise B before and/or after it computes the update B+dB depending on the value of rescale [before::Bool,after::Bool]"""

    iscomplex = !(dtype <: Real)

    if rescale[1]
        normalize!(BT_init)
    end

    if bbopt.name == "CustomGD"
        if uppercase(bbopt.fl) == "GD"
            BT_new = custGD(BT_init, LE, RE, lid, rid, TSs; iters=iters, verbosity=verbosity, dtype=dtype, lg_iter=lg_iter, track_cost=track_cost, eta=eta)

        elseif uppercase(bbopt.fl) == "TSGO"
            BT_new = TSGO(BT_init, LE, RE, lid, rid, TSs; iters=iters, verbosity=verbosity, dtype=dtype, lg_iter=lg_iter, track_cost=track_cost, eta=eta)

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

function fitMPS(X_train::Matrix, y_train::Vector, X_val::Matrix, y_val::Vector, X_test::Matrix, y_test::Vector; random_state=nothing, chi_init=4, opts::Options=Options(), test_run=false)

    # first, create the site indices for the MPS and product states 
    num_mps_sites = size(X_train)[2]
    sites = siteinds(opts.d, num_mps_sites)

    # generate the starting MPS with unfirom bond dimension chi_init and random values (with seed if provided)
    num_classes = length(unique(y_train))
    W = generate_startingMPS(chi_init, sites; num_classes=num_classes, random_state=random_state, opts=opts)

    return fitMPS(W, X_train, y_train, X_val, y_val, X_test, y_test; opts=opts, test_run=test_run)
end

function fitMPS(W::MPS, X_train::Matrix, y_train::Vector, X_val::Matrix, y_val::Vector, X_test::Matrix, y_test::Vector; opts::Options=Options(),test_run=false)

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

    training_states = encode_dataset(X_train_scaled, y_train, "train", sites; opts=opts)
    validation_states = encode_dataset(X_val_scaled, y_val, "valid", sites; opts=opts)
    testing_states = encode_dataset(X_test_scaled, y_test, "test", sites; opts=opts)

    # generate the starting MPS with uniform bond dimension chi_init and random values (with seed if provided)
    num_classes = length(unique(y_train))
    _, l_index = find_label(W)

    @assert num_classes == ITensors.dim(l_index) "Number of Classes in the training data doesn't match the dimension of the label index!"

    if test_run
        num_plts = 4
        plotinds = shuffle(MersenneTwister(), 1:num_mps_sites)[1:num_plts]
        println("Choosing $num_plts timepoints to plot the basis of at random")
        test_enc = encode_dataset(X_train_scaled, y_train, "test_enc", sites; opts=opts)

        num_ts = 10*size(X_train_scaled)[1]
        a,b = opts.encoding.range
        stp = (b-a)/(num_ts-1)
        xs = collect(a:stp:b)

        states = hcat([real.(Vector.(te.pstate)) for te in test_enc]...)
        p1s = [histogram(X_train_scaled[:,i]; bins=25, title="Timepoint $i/$num_mps_sites", legend=:none) for i in plotinds]
        p2s = [plot(xs, transpose(hcat(states[i,:]...)); legend=:none) for i in plotinds]

        p = plot(vcat(p1s,p2s)..., layout=(2,num_plts), size=(1200,800))
        
        println("Encoding completed! Returning initial states without training.")
        return W, [], training_states, testing_states, p
    end



    return fitMPS(W, training_states, validation_states, testing_states; opts=opts, test_run=test_run)
end

function fitMPS(training_states::timeSeriesIterable, validation_states::timeSeriesIterable, testing_states::timeSeriesIterable;
    random_state=nothing, chi_init=4, opts::Options=Options(), test_run=false) # optimise bond tensor)
    # first, create the site indices for the MPS and product states 

    @assert opts.d == ITensors.dim(siteinds(training_states[1].pstate)[1]) "Dimension of site indices must match feature map dimension"
    sites = siteinds(testing_states[1].pstate)

    # generate the starting MPS with unfirom bond dimension chi_init and random values (with seed if provided)
    num_classes = length(unique([ps.label for ps in training_states]))
    W = generate_startingMPS(chi_init, sites; num_classes=num_classes, random_state=random_state, opts=opts)

    fitMPS(W, training_states, validation_states, testing_states; opts=opts, test_run=test_run)

end


function fitMPS(W::MPS, training_states::timeSeriesIterable, validation_states::timeSeriesIterable, testing_states::timeSeriesIterable; 
     opts::Options=Options(), test_run=false) # optimise bond tensor)

    if test_run
        println("Encoding completed! Returning initial states without training.")
        return W, [], training_states, testing_states, []
    end

    @unpack_Options opts # unpacks the attributes of opts into the local namespace

    println("Using $update_iters iterations per update.")
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
    # start the sweep
    for itS = 1:nsweeps
        
        start = time()
        println("Using optimiser $(bbopt[itS].name) with the \"$(bbopt[itS].fl)\" algorithm")
        println("Starting backward sweeep: [$itS/$nsweeps]")

        for j = (length(sites)-1):-1:1
            #print("Bond $j")
            # j tracks the LEFT site in the bond tensor (irrespective of sweep direction)
            BT = W[j] * W[(j+1)] # create bond tensor
            BT_new = apply_update(BT, LE, RE, j, (j+1), training_states; iters=update_iters, verbosity=verbosity, 
                                    dtype=dtype, lg_iter=lg_iter[itS], bbopt=bbopt[itS],
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
        println("Backward sweep finished.")
        
        # finished a full backward sweep, reset the caches and start again
        # this can be simplified dramatically, only need to reset the LE
        LE, RE = construct_caches(W, training_states; going_left=false)
        
        println("Starting forward sweep: [$itS/$nsweeps]")

        for j = 1:(length(sites)-1)
            #print("Bond $j")
            BT = W[j] * W[(j+1)]
            BT_new = apply_update(BT, LE, RE, j, (j+1), training_states; iters=update_iters, verbosity=verbosity, 
                                    dtype=dtype, lg_iter=lg_iter[itS], bbopt=bbopt[itS],
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
        println("Finished sweep $itS.")

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
   
    return W, training_information, training_states, testing_states

end

# Demo 
# (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits_txt("/Users/joshua/QuantumMay/QuantumInspiredML/LogLoss/datasets/ECG_train.txt", 
# "/Users/joshua/QuantumMay/QuantumInspiredML/LogLoss/datasets/ECG_val.txt", "/Users/joshua/QuantumMay/QuantumInspiredML/LogLoss/datasets/ECG_test.txt")

train_data = jldopen("/Users/joshua/QuantumMay/QuantumInspiredML/Interpolation/benchmarking/ecg200/data/train_unscaled.jld2", "r")
test_data = jldopen("/Users/joshua/QuantumMay/QuantumInspiredML/Interpolation/benchmarking/ecg200/data/test_unscaled.jld2", "r")
X_train = train_data["X_train"]
y_train = train_data["y_train"]
X_test = test_data["X_test"]
y_test = test_data["y_test"]

# X_train = vcat(X_train, X_val)
# y_train = vcat(y_train, y_val)

X_val = X_test
y_val = y_test

setprecision(BigFloat, 128)
Rdtype = Float64

verbosity = 0
test_run = false


opts=Options(; nsweeps=20, chi_max=20,  update_iters=1, verbosity=verbosity, dtype=Complex{Rdtype}, lg_iter=KLD_iter,
bbopt=BBOpt("CustomGD"), track_cost=false, eta=0.05, rescale = (false, true), d=8, aux_basis_dim=2, encoding=SplitBasis("Hist Split", "Stoudenmire"))

# opts=Options(; nsweeps=20, chi_max=25,  update_iters=1, verbosity=verbosity, dtype=Complex{Rdtype}, lg_iter=KLD_iter,
# bbopt=BBOpt("CustomGD"), track_cost=false, eta=0.05, rescale = (false, true), d=6, aux_basis_dim=2, encoding=Basis("Sahand"))



# # saveMPS(W, "LogLoss/saved/loglossout.h5")
print_opts(opts)

if test_run
    W, info, train_states, test_states, p = fitMPS(X_train, y_train, X_val, y_val, X_test, y_test; random_state=456, chi_init=4, opts=opts, test_run=true)
    plot(p)
else
    W, info, train_states, test_states = fitMPS(X_train, y_train, X_val, y_val, X_test, y_test; random_state=456, chi_init=4, opts=opts, test_run=false)

    print_opts(opts)
    summary = get_training_summary(W, train_states, test_states; print_stats=true);
    sweep_summary(info)
end


