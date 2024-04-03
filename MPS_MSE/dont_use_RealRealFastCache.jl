using ITensors
using OptimKit
using Random
using Distributions
using DelimitedFiles
using Folds
using JLD2
using StatsBase
using Plots

import NDTensors

include("summary.jl")
include("utils.jl")



function GenerateStartingMPS(χ_init, site_indices::Vector{Index{Int64}};
    num_classes = 2, random_state=nothing)
    """Generate the starting weight MPS, W using values sampled from a 
    Gaussian (normal) distribution. Accepts a χ_init parameter which
    specifies the initial (uniform) bond dimension of the MPS."""
    
    if random_state !== nothing
        # use seed if specified
        Random.seed!(random_state)
        println("Generating initial weight MPS with bond dimension χ = $χ_init
        using random state $random_state.")
    else
        println("Generating initial weight MPS with bond dimension χ = $χ_init.")
    end

    W = randomMPS(ComplexF64,site_indices, linkdims=χ_init)

    label_idx = Index(num_classes, "f(x)")

    # get the site of interest and copy over the indices at the last site where we attach the label 
    old_site_idxs = inds(W[end])
    new_site_idxs = old_site_idxs, label_idx
    new_site = randomITensor(new_site_idxs)

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

function ConstructCaches(W::MPS, training_pstates::timeSeriesIterable; going_left=true)
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
            LE[1,i] = training_pstates[i].pstate[1] * W[1] 
        end

        for j = 2 : N
            for i = 1:N_train
                LE[j,i] = LE[j-1, i] * (training_pstates[i].pstate[j] * W[j])
            end
        end
    
    else
        # going right
        # initialise RE cache with the terminal site and work backwards
        for i = 1:N_train
            RE[N,i] = training_pstates[i].pstate[N] * W[N]
        end

        for j = (N-1):-1:1
            for i = 1:N_train
                RE[j,i] = RE[j+1,i] * (W[j] * training_pstates[i].pstate[j])
            end
        end
    end

    return LE, RE

end

function realise!(B_ra::Array{Float64}, B_ca::Array{ComplexF64}, inds_c::Tuple{Vararg{Index{Int64}}})
    n_dims = length(inds_c)
    re_part = selectdim(B_ra, n_dims,1);
    im_part = selectdim(B_ra, n_dims,2);

    re_part .= real.(B_ca)
    im_part .= imag.(B_ca)

    return itensor(Float64, B_ra, inds_c)
end

function realise(T_c::ITensor, inds_c::Tuple{Vararg{Index{Int64}}})
    T_ca = NDTensors.array(T_c, inds_c[1:end-1]) # should return a view
    T_ra = Array{Float64}(undef, size(T_ca)...,2)
    
    n_dims = length(inds_c)
    re_part = selectdim(T_ra, n_dims,1);
    im_part = selectdim(T_ra, n_dims,2);

    re_part .= real.(T_ca)
    im_part .= imag.(T_ca)

    return itensor(Float64, T_ra, inds_c)
end




function complexify!(B_ca::Array{ComplexF64}, B_r::ITensor, inds_c::Tuple{Vararg{Index{Int64}}})
    n_dims = length(inds_c)
    B_ra = NDTensors.array(B_r, inds_c) # should return a view

    re_part = selectdim(B_ra, n_dims,1);
    im_part = selectdim(B_ra, n_dims,2);

    B_ca .= complex.(re_part,im_part)
end

function ComputeYhatAndDerivative(BT::ITensor, LEP::PCacheCol, REP::PCacheCol, 
    product_state::PState, lid::Int, rid::Int)
    """Return yhat and phi_tilde for a bond tensor and a single product state"""
    ps=product_state.pstate
    site_inds = inds(BT, "Site")
    if length(site_inds) !== 2
        error("Bond tensor does not contain two sites!")
    end

    phi_tilde = ps[lid] * ps[rid] # phi tilde 


    if lid == 1
        # at the first site, no LE
        # formatted from left to right, so env - product state, product state - env
        phi_tilde = ps[lid] * ps[rid] * REP[rid+1]
    elseif rid == length(ps)
        # terminal site, no RE
        phi_tilde = LEP[lid-1] * ps[lid] * ps[rid] 
    else
        # we are in the bulk, both LE and RE exist
        phi_tilde =  ps[lid] * ps[rid] * LEP[lid-1] * REP[rid+1]

    end


    yhat = BT * phi_tilde

    return yhat, phi_tilde

end


function LossGradPerSampleCache(B_c::ITensor, LEP::PCacheCol, REP::PCacheCol,
    product_state::PState, lid::Int, rid::Int)
    """In order to use OptimKit, we must format the function to return 
    the loss function evaluated for the sample, along with the gradient 
        of the loss function for that sample (fg)"""
    yhat, phi_tilde = ComputeYhatAndDerivative(B_c, LEP, REP, product_state, lid, rid)

    # convert the label to ITensor
    label_idx = first(inds(yhat))
    y = onehot(label_idx => (product_state.label + 1))
    diff_sq = abs2.(yhat - y)
    sum_of_sq_diff = sum(diff_sq)
    loss = 0.5 * real(sum_of_sq_diff)

    # construct the gradient - return -dC/dB
    g = (y - yhat) * conj(phi_tilde)

    return [loss, g]

end

function LossAndGradientCache(B_r::ITensor, B_c::ITensor, LE::PCache, RE::PCache,
    ϕs::timeSeriesIterable, lid::Int, rid::Int, inds_c::Tuple{Vararg{Index{Int64}}}, B_ca::Array{ComplexF64}, B_ra::Array{Float64})
    """Function for computing the loss function and the gradient
    over all samples. Need to specify a LE, RE,
    left id (lid) and right id (rid) for the bond tensor."""
    # get the complex itensor back. The optimisation algorithim is a black box, so we cant assume that B_ra points to B_r. We do know this
    # about B_c and B_ca however
    complexify!(B_ca, B_r, inds_c) # alters B_c through pointer nonsense


    loss,grad = Folds.mapreduce((LEP,REP, prod_state) -> LossGradPerSampleCache(B_c,LEP,REP,prod_state,lid,rid),+, eachcol(LE), eachcol(RE),ϕs)

    grad = realise(grad, inds_c)    # convert gradient back to a vector of reals
    loss /= length(ϕs)
    grad ./= length(ϕs)

    return loss, -grad

end

function ApplyUpdate(BT_init::ITensor, LE::PCache, RE::PCache, lid::Int, rid::Int,
    ϕs::timeSeriesIterable; rescale=false, iters=10, verbosity::Real=1)
    """Apply update to bond tensor using Optimkit"""
    # we want the loss and gradient fn to be a functon of only the bond tensor 
    # this is what optimkit updates and feeds back into the loss/grad function to re-evaluate on 
    # each iteration. 

    # break down the bond tensor to feed into optimkit
    # println("New B")
    # println()
    C_index = Index(2, "C")
    ib = inds(BT_init)
    inds_c = ib..., C_index
    B_ca = Array{ComplexF64}(BT_init, ib)
    B_ra = Array{Float64}(undef, size(BT_init)...,2)
    realise!(B_ra, B_ca, inds_c)

    # preserve reference to original array, tensor storage must be dense!!
    B_c = itensor(B_ca, ib) 
    B_r = itensor(B_ra, inds_c)

    
    normalize!(B_ra) # modify B_ra and test for changes in B_r

    @assert abs(norm(B_r) - 1) <= eps()  "ITensors is not passing the cache arrays by reference !! Aborting."

    lg = x -> LossAndGradientCache(x, B_c, LE, RE, ϕs, lid, rid, inds_c, B_ca, B_ra)
    alg = ConjugateGradient(; verbosity=verbosity, maxiter=iters)
    #alg = GradientDescent(; maxiter=iters)
    new_BT, fx, _ = optimize(lg, B_r, alg)

    complexify!(B_ca, new_BT, inds_c) # implicitly sets B_c

    # return the new bond tensor and the loss function
    return B_c

end

function DecomposeBondTensor(BT::ITensor, lid::Int, rid::Int; 
    χ_max=nothing, cutoff=nothing, going_left=true)
    """Decompose an updated bond tensor back into two tensors using SVD"""
    left_site_index = findindex(BT, "n=$lid")
    label_index = findindex(BT, "f(x)")
    if going_left
        # need to make sure the label index is transferred to the next site to be updated
        if lid == 1
            U, S, V = svd(BT, (left_site_index, label_index); maxdim=χ_max, cutoff=cutoff)
        else
            bond_index = findindex(BT, "Link,l=$(lid-1)")
            U, S, V = svd(BT, (left_site_index, label_index, bond_index); maxdim=χ_max, cutoff=cutoff)
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
            U, S, V = svd(BT, (left_site_index); maxdim=χ_max, cutoff=cutoff)
        else
            bond_index = findindex(BT, "Link,l=$(lid-1)")
            U, S, V = svd(BT, (bond_index, left_site_index); maxdim=χ_max, cutoff=cutoff)
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

function UpdateCaches!(left_site_new::ITensor, right_site_new::ITensor, 
    LE::PCache, RE::PCache, lid::Int, rid::Int, product_states; going_left=true)
    """Given a newly updated bond tensor, update the caches."""
    num_train = length(product_states)
    num_sites = size(LE)[1]
    if going_left
        for i = 1:num_train
            if rid == num_sites
                RE[num_sites,i] = right_site_new * product_states[i].pstate[num_sites]
            else
                RE[rid,i] = RE[rid+1,i] * right_site_new * product_states[i].pstate[rid]
            end
        end

    else
        # going right
        for i = 1:num_train
            if lid == 1
                LE[1,i] = left_site_new * product_states[i].pstate[lid]
            else
                LE[lid,i] = LE[lid-1,i] * product_states[i].pstate[lid] * left_site_new
            end
        end
    end

end

function fitMPS(X_train::Matrix, y_train::Vector, X_val::Matrix, 
    y_val::Vector, X_test, y_test; χ_init=4, nsweep=5, χ_max=25, cutoff=1E-10, 
    random_state=nothing, update_iters=10, verbosity=1)

    # first, create the site indices for the MPS and product states 
    num_mps_sites = size(X_train)[2]
    sites = siteinds("S=1/2", num_mps_sites)
    println("Using χ_init=$χ_init and a maximum of $nsweep sweeps...")
    println("Using $update_iters iterations per update.")

    # now let's handle the training/validation/testing data
    # rescale using a robust sigmoid transform
    scaler = fitScaler(RobustSigmoidTransform, X_train; positive=true);
    X_train_scaled = transformData(scaler, X_train)
    X_val_scaled = transformData(scaler, X_val)
    X_test_scaled = transformData(scaler, X_test)

    # generate product states using rescaled data
    
    training_states = GenerateAllProductStates(X_train_scaled, y_train, "train", sites)
    validation_states = GenerateAllProductStates(X_val_scaled, y_val, "valid", sites)
    testing_states = GenerateAllProductStates(X_test_scaled, y_test, "test", sites)

    # generate the starting MPS with unfirom bond dimension χ_init and random values (with seed if provided)
    num_classes = length(unique(y_train))
    W = GenerateStartingMPS(χ_init, sites; num_classes=num_classes, random_state=random_state)

    # construct initial caches
    LE, RE = ConstructCaches(W, training_states; going_left=true)

    # compute initial training and validation acc/loss
    init_train_loss, init_train_acc = ComputeLossAndAccuracyDataset(W, training_states)
    init_val_loss, init_val_acc = ComputeLossAndAccuracyDataset(W, validation_states)
    init_test_loss, init_test_acc = ComputeLossAndAccuracyDataset(W, testing_states)

    # print loss and acc
    println("Initial training loss: $init_train_loss | train acc: $init_train_acc")
    println("Initial validation loss: $init_val_loss | val acc: $init_val_acc")

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
        "time_taken" => Float64[] # sweep duration
    )

    push!(training_information["train_loss"], init_train_loss)
    push!(training_information["train_acc"], init_train_acc)
    push!(training_information["val_loss"], init_val_loss)
    push!(training_information["val_acc"], init_val_acc)
    push!(training_information["test_loss"], init_test_loss)
    push!(training_information["test_acc"], init_test_acc)

    # start the sweep
    for itS = 1:nsweep
        
        start = time()
        println("Starting backward sweeep: [$itS/$nsweep]")

        for j = 1(length(sites)-1):-1:1
            #print("Bond $j")
            # j tracks the LEFT site in the bond tensor (irrespective of sweep direction)
            BT = W[j] * W[(j+1)] # create bond tensor
            new_BT = ApplyUpdate(BT, LE, RE, j, (j+1), training_states; rescale=true, iters=update_iters, verbosity=verbosity) # optimise bond tensor
            # decompose the bond tensor using SVD and truncate according to χ_max and cutoff
            lsn, rsn = DecomposeBondTensor(new_BT, j, (j+1); χ_max=χ_max, cutoff=cutoff, going_left=true)
            # update the caches to reflect the new tensors
            UpdateCaches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=true)
            # place the updated sites back into the MPS
            W[j] = lsn
            W[(j+1)] = rsn
        end
        # add time taken for backward sweep.
        println("Backward sweep finished.")
        
        # finished a full backward sweep, reset the caches and start again
        # this can be simplified dramatically, only need to reset the LE
        LE, RE = ConstructCaches(W, training_states; going_left=false)
        
        println("Starting forward sweep: [$itS/$nsweep]")

        for j = 1:(length(sites)-1)
            #print("Bond $j")
            BT = W[j] * W[(j+1)]
            new_BT = ApplyUpdate(BT, LE, RE, j, (j+1), training_states; rescale=true, iters=update_iters, verbosity=verbosity)
            lsn, rsn = DecomposeBondTensor(new_BT, j, (j+1); χ_max=χ_max, cutoff=cutoff, going_left=false)
            UpdateCaches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=false)
            W[j] = lsn
            W[(j+1)] = rsn
        end

        LE, RE = ConstructCaches(W, training_states; going_left=true)
        
        finish = time()

        time_elapsed = finish - start
        
        # add time taken for full sweep 
        println("Finished sweep $itS.")

        # compute the loss and acc on both training and validation sets
        train_loss, train_acc = ComputeLossAndAccuracyDataset(W, training_states)
        val_loss, val_acc = ComputeLossAndAccuracyDataset(W, validation_states)
        test_loss, test_acc = ComputeLossAndAccuracyDataset(W, testing_states)

        println("Validation loss: $val_loss | Validation acc. $val_acc." )
        println("Training loss: $train_loss | Training acc. $train_acc." )
        println("testing loss: $test_loss | testing acc. $test_acc." )

        running_train_loss = train_loss
        running_val_loss = val_loss

        push!(training_information["train_loss"], train_loss)
        push!(training_information["train_acc"], train_acc)
        push!(training_information["val_loss"], val_loss)
        push!(training_information["val_acc"], val_acc)
        push!(training_information["test_loss"], test_loss)
        push!(training_information["test_acc"], test_acc)
        push!(training_information["time_taken"], time_elapsed)


       
    end
    normalize!(W)
    return W, training_information, training_states, testing_states

end


(X_train, y_train), (X_val, y_val), (X_test, y_test) = LoadSplitsFromTextFile("datasets/ECG_train.txt", 
    "datasets/ECG_val.txt", "datasets/ECG_test.txt")

X_train_final = vcat(X_train, X_val)
y_train_final = vcat(y_train, y_val)

W, info, train_states, test_states = fitMPS(X_train_final, y_train_final, X_val, y_val, 
    X_test, y_test; nsweep=15, χ_max=15, random_state=123456, 
    update_iters=13, verbosity=0)

summary = get_training_summary(W, train_states, test_states)

#PlotTrainingSummary(info)

println("Time taken: $(info["time_taken"]) | $(mean(info["time_taken"][2:end]))")
println("Accs: $(info["test_acc"]) | $(mean(info["test_acc"][2:end]))")