using ITensors
using OptimKit
using Random
using Distributions
using DelimitedFiles
using StatsBase
using Folds

# Define a function which returns the gradient and the output
function QuadraticProblem(B, y)
    function fg(x)
        g = B*(x-y) # gradient
        f = dot(x-y, g)/2 # function 
        return f, g
    end
    return fg
end

struct PState
    """Create a custom structure to store product state objects, 
    along with their associated label and type (i.e, train, test or valid)"""
    pstate::MPS
    label::Int
    type::String
end

function BinariseTimeSeries(time_series::Vector, method="median")

    # check valid method
    if method !== "median" && method !== "mean"
        error("Invalid binarisation method. Can be either `median' or `mean'.")
    end

    # construct empty array for binarised time series
    binarised = zeros(Int, length(time_series))

    if method == "mean"
        thresh = mean(time_series)
    elseif method == "median"
        thresh = median(time_series)
    end

    for (i, val) in enumerate(time_series)
        # 1 for above the mean/median and 0 for below 
        binarised[i] = (val >= thresh) ? 1 : 0
    end
    
    return binarised

end

function BinariseDataset(time_series::Matrix; method="median")
    """Function to binarise an entire dataset of time series"""
    binarised_matrix = zeros(Int, size(time_series)[1], size(time_series)[2])

    for i=1:size(time_series)[1]
        binarised_matrix[i, :] = BinariseTimeSeries(time_series[i, :], method)
    end

    return binarised_matrix

end

function BinaryToProductState(binarised_time_series::Vector, site_indices::Vector{Index{Int64}})
    """convert a binarised time series to a product state where 1 maps to 
    spin down and 0 maps to spin up as per bloch sphere conventions"""

    # intial checks
    if !all(x -> x == 1 || x == 0, binarised_time_series)
        error("Time sereies does not contain binary values.")
    end

    # check that the time series length is equal to the site indices length
    if length(binarised_time_series) !== length(site_indices)
        error("Mismatch betwee number of physical sites ($(length(site_indices))) and length of 
        the binarised time series ($(length(binarised_time_series)))")
    end

    # create empty product state container
    phi = MPS(site_indices; linkdims=1)

    # fill the product state container with tensors corresponding to the states at each site
    for s = 1:length(site_indices)
        T = ITensor(site_indices[s]) # extract the site
        if binarised_time_series[s] == 1
            # spin down for 1 state -> [0; 1]
            T[2] = 1
        else
            # spin up for 0 state -> [1; 0]
            T[1] = 1
        end
        # write new tensor to the product state container
        phi[s] = T
    end

    return phi

end


function GenerateAllProductStates(X_binarised::Matrix, y::Vector, type::String, 
    site_indices::Vector{Index{Int64}})
    """Convert an entire dataset of binarised time series to a corresponding 
    dataset of product states"""

    if type == "train"
        println("Initialising training states.")
    elseif type == "test"
        println("Initialising testing states.")
    elseif type == "valid"
        println("Initialising validation states.")
    else
        error("Invalid dataset type. Must be either train, test or valid!")
    end

    num_samples = size(X_binarised)[1]

    # create a vector of PStates to store all product states
    phis = Vector{PState}(undef, num_samples)

    Threads.@threads for sample = 1:num_samples
        sample_pstate = BinaryToProductState(X_binarised[sample, :], site_indices)
        sample_label = y[sample]
        ps = PState(sample_pstate, sample_label, type)
        phis[sample] = ps
    end

    return phis

end

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

    W = randomMPS(site_indices, linkdims=χ_init)

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

    W = randomMPS(site_indices, linkdims=χ_init)

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

function ConstructCaches(W::MPS, training_pstates::Vector{PState}; going_left=true)
    """Function to pre-compute tensor contractions between the MPS and the product states. """

    # get the num of training samples to pre-allocate a caching matrix
    N_train = length(training_pstates) 
    # get the number of MPS sites
    N = length(W)

    # pre-allocate left and right environment matrices 
    LE = Matrix{ITensor}(undef, N_train, N)
    RE = Matrix{ITensor}(undef, N_train, N)


    if going_left
        # backward direction - initialise the LE with the first site
        for i = 1:N_train
            LE[i, 1] = training_pstates[i].pstate[1] * W[1] 
        end

        for j = 2 : N
            for i = 1:N_train
                LE[i, j] = LE[i, j-1] * training_pstates[i].pstate[j] * W[j]
            end
        end
    
    else
        # going right
        # initialise RE cache with the terminal site and work backwards
        for i = 1:N_train
            RE[i, N] = training_pstates[i].pstate[N] * W[N]
        end

        for j = (N-1):-1:1
            for i = 1:N_train
                RE[i, j] = RE[i, j+1] * W[j] * training_pstates[i].pstate[j]
            end
        end
    end

    return LE, RE

end

function ContractMPSAndProductState(W::MPS, ϕ::PState)
    N_sites = length(W)
    res = 1
    for i=1:N_sites
        res *= W[i] * ϕ.pstate[i]
    end

    return res

end

function ModelOutputToProba(yhat::ITensor)
    """Convert raw scores to probabilities"""
    norm = (yhat*yhat)[]
    return [yhat[i]^2/norm for i=1:ITensors.dim(yhat)]
end

function ComputeLossPerSampleAndIsCorrect(W::MPS, ϕ::PState)
    """For a given sample, compute the NLL and whether or not it is 
    correctly classified"""
    yhat = ContractMPSAndProductState(W, ϕ)
    label = ϕ.label
    label_idx = inds(yhat)[1]
    y = onehot(label_idx => label + 1) # one hot encode, so 0 -> 1 and 1->2
    y_yhat = (y * yhat)[] # isolate the score of the ground truth index
    yhat_yhat = (yhat * yhat)[]
    prob = y_yhat^2 / yhat_yhat
    log_loss = -log(prob)

    # now get the predicted label
    probs = ModelOutputToProba(yhat)
    correct = 0
    if (argmax(probs) - 1) == ϕ.label
        correct = 1
    end

    return [log_loss, correct]

end

function ComputeLossAndAccuracyDataset(W::MPS, ϕs::Vector{PState})
    """Compute the NLL and accuracy for an entire dataset"""
    loss, acc = Folds.reduce(+, ComputeLossPerSampleAndIsCorrect(W, ϕ) for ϕ in ϕs)
    loss /= length(ϕs)
    acc /= length(ϕs)

    return loss, acc

end

function ComputeYhatAndDerivative(BT::ITensor, LE::Matrix, RE::Matrix, product_state,
     ps_id::Int, lid::Int, rid::Int)
    """Return yhat and d_yhat_dW for a bond tensor and a single product state"""
    # detect lid and rid from the bond tensor
    site_inds = inds(BT, "Site")
    if length(site_inds) !== 2
        error("Bond tensor does not contain two sites!")
    end
    #s1 = site_inds[1]
    #s2 = site_inds[2]

    if lid == 1
        # at the first site, no LE
        # formatted from left to right, so env - product state, product state - env
        d_yhat_dW = product_state.pstate[lid] * product_state.pstate[rid] * RE[ps_id, (rid+1)]
    elseif rid == length(product_state.pstate)
        # terminal site, no RE
        d_yhat_dW = LE[ps_id, (lid-1)] * product_state.pstate[lid] * product_state.pstate[rid] 
    else
        # we are in the bulk, both LE and RE exist
        d_yhat_dW = LE[ps_id, (lid-1)] * product_state.pstate[lid] * product_state.pstate[rid] * RE[ps_id, (rid+1)]
    end

    yhat = BT * d_yhat_dW

    return yhat, d_yhat_dW

end

function ComputeLossAndGradientPerSample(BT::ITensor, LE::Matrix, RE::Matrix,
    product_state, ps_id::Int, lid, rid)
    """In order to use OptimKit, we must format the function to return 
    the loss function evaluated for the sample, along with the gradient 
        of the loss function for that sample (fg)"""
    
    
    yhat, d_yhat_dW = ComputeYhatAndDerivative(BT, LE, RE, product_state, ps_id, lid, rid)

    # convert label to ITensor
    label_index = inds(yhat)[1]
    y = onehot(label_index => (product_state.label + 1))
    y_yhat = (y * yhat)[]
    yhat_yhat = (yhat * yhat)[]
    p = y_yhat^2 / yhat_yhat
    loss = -log(p)

    # now for the gradient
    part_one = y_yhat * (y * d_yhat_dW)
    part_one ./= yhat_yhat

    part_two = y_yhat^2 * (yhat * d_yhat_dW)
    part_two ./= (yhat_yhat)^2

    gradient = -2 * (part_one - part_two)
    gradient ./= p

    return [loss, gradient]

end

function LossAndGradient(BT::ITensor, LE::Matrix, RE::Matrix,
    ϕs::Vector{PState}, lid, rid)
    """Function for computing the loss function and the gradient over all
    samples. Need to specify a LE, RE, lid (left site id) for the bond tensor and 
    rid (right site id) for the bond tensor."""
    loss, grad = Folds.reduce(+, ComputeLossAndGradientPerSample(BT, LE, RE,
        prod_state, prod_state_id, lid, rid) for (prod_state_id, prod_state) in enumerate(ϕs))

    loss /= length(ϕs)
    grad ./= length(ϕs)

    return loss, grad

end

function ApplyUpdate(BT_init::ITensor, LE::Matrix, RE::Matrix, lid::Int, rid::Int,
    ϕs::Vector{PState}; rescale=false)
    """Apply update to bond tensor using Optimkit"""
    # we want the loss and gradient fn to be a functon of only the bond tensor 
    # this is what optimkit updates and feeds back into the loss/grad function to re-evaluate on 
    # each iteration. 
    lg = x -> LossAndGradient(x, LE, RE, ϕs, lid, rid)
    alg = ConjugateGradient(; verbosity=0, maxiter=5)
    new_BT, fx, _ = optimize(lg, BT_init, alg)

    if rescale
        # rescale the bond tensor so that the MPS remains normalised
        new_BT ./= sqrt(inner(dag(new_BT), new_BT))
    end

    # return the new bond tensor and the loss function
    return new_BT

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
    LE::Matrix, RE::Matrix, lid::Int, rid::Int, product_states; going_left=true)
    """Given a newly updated bond tensor, update the caches."""
    num_train = length(product_states)
    num_sites = size(LE)[2]
    if going_left
        for i = 1:num_train
            if rid == num_sites
                RE[i, num_sites] = right_site_new * product_states[i].pstate[num_sites]
            else
                RE[i, rid] = RE[i, rid+1] * right_site_new * product_states[i].pstate[rid]
            end
        end

    else
        # going right
        for i = 1:num_train
            if lid == 1
                LE[i, 1] = left_site_new * product_states[i].pstate[lid]
            else
                LE[i, lid] = LE[i, lid-1] * product_states[i].pstate[lid] * left_site_new
            end
        end
    end

end

function fitMPS(X_train::Matrix, y_train::Vector, X_val::Matrix, 
    y_val::Vector; χ_init=2, nsweep=5, χ_max=25, cutoff=1E-10, 
    random_state=nothing)

    # first, create the site indices for the MPS and product states 
    num_mps_sites = size(X_train)[2]
    sites = siteinds("S=1/2", num_mps_sites)
    println("Using χ_init=$χ_init and a maximum of $nsweep sweeps...")

    # now let's handle the training and validation data
    training_data_binarised = BinariseDataset(X_train; method="median")
    validation_data_binarised = BinariseDataset(X_val; method="median")
    # convert to product states
    training_states = GenerateAllProductStates(training_data_binarised, y_train, "train", sites)
    validation_states = GenerateAllProductStates(validation_data_binarised, y_val, "valid", sites)

    # generate the starting MPS with unfirom bond dimension χ_init and random values (with seed if provided)
    num_classes = length(unique(y_train))
    W = GenerateStartingMPS(χ_init, sites; num_classes=num_classes, random_state=random_state)

    # construct initial caches
    LE, RE = ConstructCaches(W, training_states; going_left=true)

    # compute initial training and validation acc/loss
    init_train_loss, init_train_acc = ComputeLossAndAccuracyDataset(W, training_states)
    init_val_loss, init_val_acc = ComputeLossAndAccuracyDataset(W, validation_states)

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
    )

    push!(training_information["train_loss"], init_train_loss)
    push!(training_information["train_acc"], init_train_acc)
    push!(training_information["val_loss"], init_val_loss)
    push!(training_information["val_acc"], init_val_acc)

    # start the sweep
    for itS = 1:nsweep
        
        println("Starting backward sweeep: [$itS/$nsweep]")

        for j = 1(length(sites)-1):-1:1
            # j tracks the LEFT site in the bond tensor (irrespective of sweep direction)
            BT = W[j] * W[(j+1)] # create bond tensor
            new_BT = ApplyUpdate(BT, LE, RE, j, (j+1), training_states; rescale=true) # optimise bond tensor
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
            BT = W[j] * W[(j+1)]
            new_BT = ApplyUpdate(BT, LE, RE, j, (j+1), training_states; rescale=true)
            lsn, rsn = DecomposeBondTensor(new_BT, j, (j+1); χ_max=χ_max, cutoff=cutoff, going_left=false)
            UpdateCaches!(lsn, rsn, LE, RE, j, (j+1), training_states; going_left=false)
            W[j] = lsn
            W[(j+1)] = rsn
        end

        # add time taken for full sweep 
        println("Finished sweep $itS.")

        # compute the loss and acc on both training and validation sets
        train_loss, train_acc = ComputeLossAndAccuracyDataset(W, training_states)
        val_loss, val_acc = ComputeLossAndAccuracyDataset(W, validation_states)

        println("Validation loss: $val_loss | Validation acc. $val_acc." )
        println("Training loss: $train_loss | Training acc. $train_acc." )


        running_train_loss = train_loss
        running_val_loss = val_loss

        push!(training_information["train_loss"], train_loss)
        push!(training_information["train_acc"], train_acc)
        push!(training_information["val_loss"], val_loss)
        push!(training_information["val_acc"], val_acc)
       
    end

    return W, training_information

end

X_train = rand(1000, 100)
y_train = rand([0, 1], 1000)

X_val = rand(200, 100)
y_val = rand([0, 1], 200)


W, info = fitMPS(X_train, y_train, X_val, y_val; nsweep=3, χ_max=35)

# sites = siteinds("S=1/2", 100)
# X_binarised = BinariseDataset(data)
# ϕs = GenerateAllProductStates(X_binarised, y, "train", sites)
# W = GenerateStartingMPS(2, sites)
# LE, RE = ConstructCaches(W, ϕs)
# #yhat, d_yhat_dW = ComputeYhatAndDerivative(W, LE, RE, 99, 100, ϕs[1], 1)
# BT = W[99] * W[100]
# #loss, grad = LossAndGradient(BT, LE, RE, ϕs, 99, 100)
# new_BT = ApplyUpdate(BT, LE, RE, 99, 100, ϕs; rescale=true)
# l_new, r_new = DecomposeBondTensor(new_BT, 99, 100; cutoff=1E-10, χ_max=10)
# UpdateCaches!(l_new, r_new, LE, RE, 99, 100, ϕs)

