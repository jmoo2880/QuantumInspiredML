using ITensors
using OptimKit
using Random
using Distributions
using DelimitedFiles
using Folds
using JLD2
using StatsBase
using Plots

include("utils.jl")

# type aliases
const PCache = Matrix{ITensor}
const PCacheCol = SubArray{ITensor, 1, PCache, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true} # for view mapping shenanigans
const Maybe{T} = Union{T,Nothing} 




struct PState
    """Create a custom structure to store product state objects, 
    along with their associated label and type (i.e, train, test or valid)"""
    pstate::MPS
    label::Int
    type::String
end

const timeSeriesIterable = Vector{PState}

function AngleEncoder(x::Float64) 
    """Function to convert normalised time series to an angle encoding."""
    if x <= 1.0 && x >= 0.0
        return [cospi(0.5 * x), sinpi(0.5 * x)] # use cospi/sinpi instead to avoid non-zero floating pt error 
    else
        println("Data points must be rescaled between 1 and 0 before encoding using the angle encoder.")
    end
end

function NormalisedDataToProductState(sample::Vector, site_indices::Vector{Index{Int64}})
    """Function to convert a single normalised sample to a product state
    with local dimension 2, as specified by the feature map."""

    n_sites = length(site_indices) # number of mps sites
    product_state = MPS(site_indices; linkdims=1)
    
    # check that the number of sites matches the length of the time series
    if n_sites !== length(sample)
        error("Number of MPS sites: $n_sites does not match the time series length: $(length(sample))")
    end

    for j=1:n_sites
        T = ITensor(site_indices[j])
        # map 0 to |0> and 1 to |1> 
        zero_state, one_state = AngleEncoder(sample[j])
        T[1] = zero_state
        T[2] = one_state
        product_state[j] = T
    end

    return product_state

end

function GenerateAllProductStates(X_normalised::Matrix, y::Vector{Int}, type::String, 
    site_indices::Vector{Index{Int64}})
    """"Convert an entire dataset of normalised time series to a corresponding 
    dataset of product states"""
    # check data is in the expected range first
    if all((0 .<= X_normalised) .& (X_normalised .<= 1)) == false
        error("Data must be rescaled between 0 and 1 before generating product states.")
    end

    types = ["train", "test", "valid"]
    if type in types
        println("Initialising $type states.")
    else
        error("Invalid dataset type. Must be train, test, or valid.")
    end

    num_samples = size(X_normalised)[1]
    # pre-allocate
    all_product_states = timeSeriesIterable(undef, num_samples)

    for i=1:num_samples
        sample_pstate = NormalisedDataToProductState(X_normalised[i, :], site_indices)
        sample_label = y[i]
        product_state = PState(sample_pstate, sample_label, type)
        all_product_states[i] = product_state
    end

    return all_product_states

end;

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

function ContractMPSAndProductState(W::MPS, ϕ::PState)
    N_sites = length(W)
    res = 1
    for i=1:N_sites
        res *= W[i] * ϕ.pstate[i]
    end

    return res

end

function ComputeLossPerSampleAndIsCorrect(W::MPS, ϕ::PState)
    """For a given sample, compute the Quadratic Cost and whether or not
    the corresponding prediction (using argmax on deicision func. output) is
    correctly classfified"""
    yhat = ContractMPSAndProductState(W, ϕ)
    label = ϕ.label # ground truth label
    label_idx = inds(yhat)[1]
    y = onehot(label_idx => label + 1) # one hot encode, so class 0 [1 0] is assigned using label_idx = 1
    # compute the loss using the ground-truth y and model prediction yhat
    diff_sq = (yhat - y).^2
    sum_of_sq_diff = sum(diff_sq)

    loss = 0.5 * sum_of_sq_diff

    # now get the predicted label
    correct = 0
    
    if (argmax(abs.(vector(yhat))) - 1) == ϕ.label
        correct = 1
    end

    return [loss, correct]

end

function ComputeLossAndAccuracyDataset(W::MPS, ϕs::timeSeriesIterable)
    """Compute the loss and accuracy for an entire dataset"""
    loss, acc = Folds.reduce(+, ComputeLossPerSampleAndIsCorrect(W, ϕ) for ϕ in ϕs)
    loss /= length(ϕs)
    acc /= length(ϕs)

    return loss, acc 

end



function ComputeYhatAndDerivative(BT::ITensor, LEP::PCacheCol, REP::PCacheCol, 
    product_state, lid::Int, rid::Int)
    """Return yhat and d_yhat_dW for a bond tensor and a single product state"""

    site_inds = inds(BT, "Site")
    if length(site_inds) !== 2
        error("Bond tensor does not contain two sites!")
    end

    if lid == 1
        # at the first site, no LE
        # formatted from left to right, so env - product state, product state - env
        d_yhat_dW = product_state.pstate[lid] * product_state.pstate[rid] * REP[rid+1]
    elseif rid == length(product_state.pstate)
        # terminal site, no RE
        d_yhat_dW = LEP[lid-1] * product_state.pstate[lid] * product_state.pstate[rid] 
    else
        # we are in the bulk, both LE and RE exist
        d_yhat_dW =  product_state.pstate[lid] * product_state.pstate[rid] * LEP[lid-1] * REP[rid+1]

    end


    yhat = BT * d_yhat_dW

    return yhat, d_yhat_dW

end
function ComputeYhatAndDerivative(BT::ITensor, LE::PCache, RE::PCache, 
    product_state, ps_id::Int, lid::Int, rid::Int)
    """Return yhat and d_yhat_dW for a bond tensor and a single product state"""

    site_inds = inds(BT, "Site")
    if length(site_inds) !== 2
        error("Bond tensor does not contain two sites!")
    end

    if lid == 1
        # at the first site, no LE
        # formatted from left to right, so env - product state, product state - env
        d_yhat_dW = product_state.pstate[lid] * product_state.pstate[rid] * RE[(rid+1), ps_id]
    elseif rid == length(product_state.pstate)
        # terminal site, no RE
        d_yhat_dW = LE[(lid-1), ps_id ] * product_state.pstate[lid] * product_state.pstate[rid] 
    else
        # we are in the bulk, both LE and RE exist
        d_yhat_dW = LE[(lid-1), ps_id] * product_state.pstate[lid] * product_state.pstate[rid] * RE[(rid+1),ps_id]

    end


    yhat = BT * d_yhat_dW

    return yhat, d_yhat_dW

end

function ComputeLossAndGradientPerSample(BT::ITensor, LE::PCache, RE::PCache,
    product_state, ps_id::Int, lid, rid)
    """In order to use OptimKit, we must format the function to return 
    the loss function evaluated for the sample, along with the gradient 
        of the loss function for that sample (fg)"""

    yhat, d_yhat_dW = ComputeYhatAndDerivative(BT, LE, RE, product_state, ps_id, lid, rid)

    # convert the label to ITensor
    label_idx = inds(yhat)[1]
    y = onehot(label_idx => (product_state.label + 1))
    diff_sq = (yhat - y).^2
    sum_of_sq_diff = sum(diff_sq)
    loss = 0.5 * sum_of_sq_diff

    # construct the gradien - return -dC/dB
    gradient = (y - yhat) * d_yhat_dW

    return [loss, gradient]

end

function LossGradPerSample(BT::ITensor, LEP::PCacheCol, REP::PCacheCol,
    product_state::PState, lid::Int, rid::Int)
    """In order to use OptimKit, we must format the function to return 
    the loss function evaluated for the sample, along with the gradient 
        of the loss function for that sample (fg)"""

    yhat, d_yhat_dW = ComputeYhatAndDerivative(BT, LEP, REP, product_state, lid, rid)

    # convert the label to ITensor
    label_idx = inds(yhat)[1]
    y = onehot(label_idx => (product_state.label + 1))
    diff_sq = (yhat - y).^2
    sum_of_sq_diff = sum(diff_sq)
    loss = 0.5 * sum_of_sq_diff

    # construct the gradien - return -dC/dB
    gradient = (y - yhat) * d_yhat_dW

    return [loss, gradient]

end

function LossAndGradient(BT::ITensor, LE::PCache, RE::PCache,
    ϕs::timeSeriesIterable, lid, rid)
    """Function for computing the loss function and the gradient
    over all samples. Need to specify a LE, RE,
    left id (lid) and right id (rid) for the bond tensor."""
    
    # loss, grad = Folds.reduce(+, ComputeLossAndGradientPerSample(BT, LE, RE, prod_state, prod_state_id, lid, rid) for 
    #     (prod_state_id, prod_state) in enumerate(ϕs))

    loss,grad = Folds.mapreduce((LEP,REP, prod_state) -> LossGradPerSample(BT,LEP,REP,prod_state,lid,rid),+, eachcol(LE), eachcol(RE),ϕs)
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
    lg = x -> LossAndGradient(x, LE, RE, ϕs, lid, rid)
    alg = ConjugateGradient(; verbosity=verbosity, maxiter=iters)
    #alg = GradientDescent(; maxiter=iters)
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

struct RobustSigmoidTransform{T<:Real} <: AbstractDataTransform
    median::T
    iqr::T
    k::T
    positive::Bool

    function RobustSigmoidTransform(median::T, iqr::T, k::T, positive=true) where T<:Real
        new{T}(median, iqr, k, positive)
    end
end

function robust_sigmoid(x::Real, median::Real, iqr::Real, k::Real, positive::Bool)
    xhat = 1.0 / (1.0 + exp(-(x - median) / (iqr / k)))
    if !positive
        xhat = 2*xhat - 1
    end
    return xhat
end

function fitScaler(::Type{RobustSigmoidTransform}, X::Matrix; k::Real=1.35, positive::Bool=true)
    medianX = median(X)
    iqrX = iqr(X)
    return RobustSigmoidTransform(medianX, iqrX, k, positive)
end

function transformData(t::RobustSigmoidTransform, X::Matrix)
    return map(x -> robust_sigmoid(x, t.median, t.iqr, t.k, t.positive), X)
end

# New SigmoidTransform
struct SigmoidTransform <: AbstractDataTransform
    positive::Bool
end

function sigmoid(x::Real, positive::Bool)
    xhat = 1.0 / (1.0 + exp(-x))
    if !positive
        xhat = 2*xhat - 1
    end
    return xhat
end

function fitScaler(::Type{SigmoidTransform}, X::Matrix; positive::Bool=true)
    return SigmoidTransform(positive)
end

function transformData(t::SigmoidTransform, X::Matrix)
    return map(x -> sigmoid(x, t.positive), X)
end;

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

    return W, training_information, training_states

end


(X_train, y_train), (X_val, y_val), (X_test, y_test) = LoadSplitsFromTextFile("MPS_MSE/datasets/ECG_train.txt", 
    "MPS_MSE/datasets/ECG_val.txt", "MPS_MSE/datasets/ECG_test.txt")

X_train_final = vcat(X_train, X_val)
y_train_final = vcat(y_train, y_val)

W, info, tstates = fitMPS(X_train_final, y_train_final, X_val, y_val, 
    X_test, y_test; nsweep=10, χ_max=30, random_state=0, 
    update_iters=20, verbosity=0)

PlotTrainingSummary(info)

println("Time taken: $(info["time_taken"]) | $(mean(info["time_taken"][2:end]))")
println("Accs: $(info["test_acc"]) | $(mean(info["test_acc"][2:end]))")