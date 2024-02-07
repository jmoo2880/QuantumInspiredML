using Distributions
using ITensors
using Base.Threads
using Folds
using Random


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
    old_site_idxs = inds(W[1])
    new_site_idxs = old_site_idxs, label_idx
    new_site = randomITensor(new_site_idxs)

    # add the new site back into the MPS
    W[1] = new_site

    # normalise the MPS
    normalize!(W)

    # canonicalise - bring MPS into canonical form by making all tensors 1,...,j-1 left orthogonal
    # here we assume we start at the right most index
    last_site = length(site_indices)
    orthogonalize!(W, last_site)

    return W

end


# each thread can handle its own product state sample
function ConstructCaches(W::MPS, training_pstates::Vector{PState}; direction::String="forward")
    """Function to pre-compute tensor contractions between the MPS and the product states."""

    N_train = length(training_pstates) # number of training samples
    N = length(W) # number of MPS sites

    LE = Matrix{ITensor}(undef, N_train, N) # Left environment for each training sample 
    RE = Matrix{ITensor}(undef, N_train, N) # Right environment for each training sample

    if direction == "forward"
        
        # intialise the RE with the terminal site and work backwards accumulating contractions site-by-site
        for i = 1:N_train
            RE[i, N] = training_pstates[i].pstate[N] * W[N]
            #normalize!(RE[i, N])
        end

        # accumulate all other sites
        for j = (N-1):-1:1
            for i=1:N_train
                RE[i, j] = RE[i, j+1] * W[j] * training_pstates[i].pstate[j]
                #normalize!(RE[i, j])
            end
        end

    elseif direction == "backward"

        for i = 1:N_train
            # intialise LE with first site and work forward accumularing contractions site-by-site
            LE[i, 1] = training_pstates[i].pstate[1] * W[1]
            #normalize!(LE[i, 1])
        end

        for j=2:N
            for i=1:N_train
                LE[i, j] = LE[i, j - 1] * training_pstates[i].pstate[j] * W[j]
                #normalize!(LE[i, j])
            end
        end

    else
        error("Invalid direction. Can either be forward or backward.")
    end

    return LE, RE
end

function ContractMPSAndProductState(W::MPS, ϕ::PState)
    """Contract the MPS with the product state representation of the data

                              |    
    PSI :     O-...-O-O-O-...-O
              |     | | |     |
              |     | | |     |
    Data:     O     O O O     O

    Gives:

        |
        O

    Function to manually contract the weight MPS with a single
    product state since ITensor `inner' function doesn't like it when
    there is a label index attached to the MPS. Returns RAW output
    (prior to conversion to prob. dist). Will return an ITensor
    corresponding to the label index with the raw scores of each class."""

    N_sites = length(W)
    res = 1
    for i=1:N_sites
        res *= W[i] * ϕ.pstate[i]
    end

    return res

end

function LossPerSampleAndCorrect(W::MPS, ϕ::PState)
    """ Evaluate the loss function for a single sample and whether the sample was 
    correctly (1) or incorrectly (0) classified"""

    yhat = ContractMPSAndProductState(W, ϕ) # model prediction
    label_index = inds(yhat)[1]
    label_index_copy = deepcopy(label_index)
    y = onehot(label_index_copy => ϕ.label + 1)

    # compute the quadratic cost
    diff_sq = (yhat - y).^2
    sum_of_sq_diff = sum(diff_sq)

    cost = 0.5 * sum_of_sq_diff

    correct = 0
    predicted_label = argmax(abs.(Vector(yhat))) - 1

    if predicted_label == ϕ.label
        correct = 1
    end

    return [cost, correct]
end

function LossAndAccDataset(W::MPS, pstates::Vector{PState})
    """Function to compute the loss and accuracy for an entire dataset (i.e. test/train/validation)"""

    running_loss = Vector{Float64}(undef, length(pstates))
    running_acc = Vector{Float64}(undef, length(pstates))

    for i=1:length(pstates)
        loss, acc = LossPerSampleAndCorrect(W, pstates[i])
        running_loss[i] = loss
        running_acc[i] = acc
    end

    loss_total = sum(running_loss)
    acc_total = sum(running_acc)

    return [loss_total/length(pstates), acc_total/length(pstates)]
end


function ComputeGradient(B::ITensor, LE::Matrix, RE::Matrix, product_states::Vector{PState};
    id::Int, direction::String)

    B_old = B

    num_samples = length(product_states)
    num_sites = length(product_states[1].pstate)

    gradient_accumulate = Vector{ITensor}(undef, num_samples)
    # get the gradient
    if direction == "forward"
        Threads.@threads for i=1:num_samples
            if id == (num_sites-1)
                effective_input = LE[i, id-1] * product_states[i].pstate[id] * product_states[i].pstate[id+1]
            elseif id == 1
                effective_input = RE[i, id+2] * product_states[i].pstate[id] * product_states[i].pstate[id+1]
            else
                effective_input = LE[i, id-1] * RE[i, id+2] * product_states[i].pstate[id] * product_states[i].pstate[id+1]
            end

            P = B_old * effective_input
            label_index = inds(P)[1]
            label_index_new = deepcopy(label_index)
            target_vector = onehot(label_index_new => (product_states[i].label + 1))

            dP = target_vector - P

            grad = dP * effective_input

            gradient_accumulate[i] = grad
        end

    elseif direction == "backward"
        Threads.@threads for i=1:num_samples
            if id == num_sites
                effective_input = LE[i, id-2] * product_states[i].pstate[id] * product_states[i].pstate[id-1]
            elseif id == 2
                effective_input = RE[i, id+1] * product_states[i].pstate[id] * product_states[i].pstate[id-1]
            else
                effective_input = LE[i, id-2] * RE[i, id+1] * product_states[i].pstate[id] * product_states[i].pstate[id-1]
            end

            P = B_old * effective_input
            label_index = inds(P)[1]
            label_index_new = deepcopy(label_index)
            target_vector = onehot(label_index_new => (product_states[i].label + 1))

            dP = target_vector - P

            grad = dP * effective_input

            gradient_accumulate[i] = grad

        end
    end

    ΔB = sum(gradient_accumulate)
    ΔB ./= num_samples

    return ΔB

end


function UpdateBondTensor(W::MPS, id::Int, direction::String, product_states::Vector{PState}, 
    LE::Matrix, RE::Matrix; α::Float64, χ_max=nothing, cutoff=nothing)

    """Function to apply gradient descent to a bond tensor"""
    d = 2
    N_train = length(product_states) # number of training samples
    N = length(W) # number of sites

    if direction == "forward"

        left_site = W[id]
        right_site = W[id+1]

        B_old = left_site * right_site
        gradient = ComputeGradient(B_old, LE, RE, product_states; id=id, direction=direction)
        B_new = B_old + α * gradient
        B_new ./= sqrt(inner(dag(B_new), B_new))

        # SVD B_new back into MPS tensors
        left_site_idx = findindex(B_new, "S=1/2,Site,n=$id") # retain left site physical index
      
        if id == 1
            if χ_max !== nothing && cutoff !== nothing
                U, S, V = svd(B_new, (left_site_idx); maxdim=χ_max, cutoff=cutoff)
            elseif χ_max !== nothing
                U, S, V = svd(B_new, (left_site_idx); maxdim=χ_max)
            else
                U, S, V = svd(B_new, (left_site_idx); cutoff=cutoff)
            end
        else
            bond_index = findindex(B_new, "Link,l=$(id-1)") # retain the bond dimension index
            # by convention, any specified indices are reatined on the U tensor
            if χ_max !== nothing && cutoff !== nothing
                U, S, V = svd(B_new, (bond_index, left_site_idx); maxdim=χ_max, cutoff=cutoff)
            elseif χ_max !== nothing
                U, S, V = svd(B_new, (bond_index, left_site_idx); maxdim=χ_max)
            else 
                U, S, V = svd(B_new, (bond_index, left_site_idx); cutoff=cutoff)
            end
        end

        left_site_new = U
        right_site_new = S * V

        # fix tag names
        replacetags!(left_site_new, "Link,u", "Link,l=$(id)")
        replacetags!(right_site_new, "Link,u", "Link,l=$(id)")

        # update environments
        for i=1:N_train
            if id == 1
                LE[i, 1] = left_site_new * product_states[i].pstate[id]

            else
                LE[i, id] = LE[i, id-1] * left_site_new * product_states[i].pstate[id]
  
            end
        end

    elseif direction == "backward"

        left_site = W[id-1]
        right_site = W[id]

        B_old = left_site * right_site
        gradient = ComputeGradient(B_old, LE, RE, product_states; id=id, direction=direction)
        B_new = B_old + α * gradient
        B_new ./= sqrt(inner(dag(B_new), B_new))

        left_site_idx = findindex(B_new, "S=1/2,Site,n=$(id-1)")
        label_idx = findindex(B_new, "f(x)")

        if id == 2
            if χ_max !== nothing && cutoff !== nothing
                U, S, V = svd(B_new, (left_site_idx, label_idx); maxdim=χ_max, cutoff=cutoff)
            elseif χ_max !== nothing
                U, S, V = svd(B_new, (left_site_idx, label_idx); maxdim=χ_max)
            else
                U, S, V = svd(B_new, (left_site_idx, label_idx); cutoff=cutoff)
            end
        else
            bond_index = findindex(B_new, "Link,l=$(id-2)")
            if χ_max !== nothing && cutoff !== nothing
                U, S, V = svd(B_new, (left_site_idx, bond_index, label_idx); maxdim=χ_max, cutoff=cutoff)
            elseif χ_max !== nothing
                U, S, V = svd(B_new, (left_site_idx, bond_index, label_idx); maxdim=χ_max)
            else
                U, S, V = svd(B_new, (left_site_idx, bond_index, label_idx); cutoff=cutoff)
            end
        end

        left_site_new = U * S
        right_site_new = V

        replacetags!(left_site_new, "Link,v", "Link,l=$(id-1)")
        replacetags!(right_site_new, "Link,v", "Link,l=$(id-1)")

        # updated environments
        for i=1:N_train
            if id == N
                RE[i, N] = right_site_new * product_states[i].pstate[N]
           
            else
                RE[i, id] = RE[i, id+1] * right_site_new * product_states[i].pstate[id]
    
            end
        end

    else
        error("Invalid direction. Specify either forward or backward.")
    end

    return left_site_new, right_site_new, LE, RE
end

function fitMPS(X_train::Matrix, y_train::Vector, X_val::Matrix, y_val::Vector; χ_init=5,
    nsweep=10, α=0.01, χ_max=25, cutoff=nothing, random_state=nothing, sweep_tol=nothing)

    num_mps_sites = size(X_train)[2]
    sites = siteinds("S=1/2", num_mps_sites)
    println("Using χ_init = $χ_init | α=$α | nsweep = $nsweep")

    # step one - binarise the data
    training_data_binarised = BinariseDataset(X_train; method="median")
    # step two - convert to product state
    training_states = GenerateAllProductStates(training_data_binarised, y_train, "train", sites)

    # step three - repeat for validation data
    validation_data_binarised = BinariseDataset(X_val; method="median")
    validation_states = GenerateAllProductStates(validation_data_binarised, y_val, "valid", sites)

    # generate starting MPS 
    W = GenerateStartingMPS(χ_init, sites; num_classes=2, random_state=random_state)

    # construct initial caches
    LE, RE = ConstructCaches(W, training_states; direction="forward")

    # compute initial training and validation loss
    init_train_loss, init_train_acc = LossAndAccDataset(W, training_states)
    init_valid_loss, init_valid_acc = LossAndAccDataset(W, validation_states)

    println("Initial train loss: $init_train_loss")
    println("Initial validation loss: $init_valid_loss")

    running_train_loss = init_train_loss
    running_valid_loss = init_valid_loss 

    for itS = (1:nsweep)
        
        println("Forward Sweep L -> R ($itS/$nsweep)")

        for j=1:(length(sites)-1)
            W[j], W[j+1], LE, RE = UpdateBondTensor(W, j, "forward", training_states, LE, RE; α=α, χ_max=χ_max, cutoff=cutoff)
        end

        # finished forward sweep, reset cache and begin backward sweep
        LE, RE = ConstructCaches(W, training_states; direction="backward")

        println("Backward Sweep R -> L ($itS/$nsweep)")
        

        for j=(length(sites)):-1:2
            W[j-1], W[j], LE, RE = UpdateBondTensor(W, j, "backward", training_states, LE, RE; α=α, χ_max=χ_max, cutoff=cutoff)
        end

        LE, RE = ConstructCaches(W, training_states, direction="forward")

        # compute new loss and acc
        train_loss, train_acc = LossAndAccDataset(W, training_states)
        valid_loss, valid_acc = LossAndAccDataset(W, validation_states)
       
        println("Training loss after Sweep $itS: $train_loss | Training Acc: $train_acc")
        println("Validation loss after Sweep $itS: $valid_loss | Validation Acc: $valid_acc")

        ΔC_val = running_valid_loss - valid_loss
        println("ΔC val after sweep $itS: $ΔC_val")

        if sweep_tol !== nothing
            if ΔC_val < sweep_tol
                println("Convergence reached. ΔC Val = $ΔC_val is less than the threshold $sweep_tol)!")
                break
            end
        end
        
        running_train_loss = train_loss
        running_valid_loss = valid_loss

    end

    return W, sites

end

function ScoreMPS(W, X_test, y_test; verbose=true)
    """Return the test accuracy of the final model"""
    test_states = InitialiseProductStates(X_test, y_test, "test")
    test_loss, test_acc = LossAndAccDataset(W, test_states)
    
    if verbose == true
        println("Test loss: $test_loss | Test Accuracy: $test_acc")
    end

    return [test_loss, test_acc]
end

ecg_dat = readdlm("../ECG200_TRAIN.txt")
X_train = ecg_dat[:, 2:end]
y_train = Int.(ecg_dat[:, 1])
remap = Dict(-1 => 0, 1 => 1)
y_train = [remap[label] for label in y_train];

ecg_dat_test = readdlm("../ECG200_TEST.txt")
X_test = ecg_dat_test[:, 2:end]
y_test = Int.(ecg_dat_test[:, 1])
y_test = [remap[label] for label in y_test]


W, sites = fitMPS(X_train, y_train, X_test, y_test; χ_max=20, α=0.01, nsweep=10)




# using MLBase:StratifiedKfold
# function GetKFold(y::Matrix, n_folds::Int; random_state=nothing)
#     """Returns folds in the (train_idx), (val_idx) format."""
#     if random_state !== nothing
#         Random.seed!(random_state)
#     end

#     skf = StratifiedKfold(y, n_folds)
#     folds = collect(skf)
#     all_idxs = 1:length(y)
#     all_folds = []

#     for i = 1:n_folds
#         train_idx = folds[i]
#         val_idx = setdiff(all_idxs, train_idx)
#         push!(all_folds, (train_idx, val_idx))
#     end

#     return all_folds

# end

# function CreateStratifiedKFold(y::Matrix, n_folds::Int; random_state=nothing)
#     """ Return the indices for stratified K-fold. Folds are made such that class distributions are preserved. """

#     if random_state !== nothing
#         rng = MersenneTwister(random_state)
#     else
#         rng = MersenneTwister()
#     end

#     folds = [Int[] for _ in 1:n_folds]
#     for label in unique(y)
#         label_idxs = findall(isequal(label), y[:, 1])
#         # shuffle label idxs
#         shuffle!(rng, label_idxs)

#         # distribute the indices across
#         for (idx, fold_idx) in enumerate(label_idxs)
#             fold_number = (idx % n_folds) + 1
#             push!(folds[fold_number], fold_idx)
#         end
#     end
    
#     return folds
# end

# function CrossValidate(X::Matrix, y::Matrix, params::Dict, n_folds::Int)
#     folds = CreateStratifiedKFold(y, n_folds)
#     scores = []

#     for i in 1:n_folds
#         val_idx = folds[i] # select validation set from the n_folds
#         train_idx = vcat(folds[1:i-1]..., folds[i+1:end]...) # use the remaining (n_folds - 1) indices for training 

#         X_train, y_train = X[train_idx, :], y[train_idx, :]
#         X_val, y_val = X[val_idx, :], y[val_idx, :]

#         # train the MPS classifier
#         W, sites = fitMPS(X_train, y_train, X_val, y_val; params...)

#         # compute validation accuracy 
#         # convert validation data to product states
#         all_test_states = InitialiseProductStates(X_val, y_val, "valid")
#         val_loss, val_acc = LossAndAccDataset(W, all_test_states)
        
#         #val_acc = TestAccuracy(W, X_val, y_val, sites, params[:feature_map])

#         push!(scores, val_acc)
#     end
#     println(mean(scores))
#     return mean(scores)

# end


# function GridSearchCV(X::Matrix, y::Matrix, param_dists::Dict, n_folds::Int)

#     results = Vector{Tuple{Dict{Symbol,Any}, Float64}}()
   
#     subset_dicts = [Dict(:χ_init => χ_init, :nsweep => nsweep, :random_state => random_state, :χ_max => χ_max, :α => α,
#         :cutoff => cutoff, :feature_map => feature_map, :sweep_tol => sweep_tol) for χ_init in param_dists[:χ_init] for nsweep in param_dists[:nsweep] 
#         for random_state in param_dists[:random_state] for cutoff in param_dists[:cutoff]
#             for feature_map in param_dists[:feature_map] for χ_max in param_dists[:χ_max] 
#                 for sweep_tol in param_dists[:sweep_tol] for α in param_dists[:α]]

#     num_combinations = length(subset_dicts)
#     for (index, subset) in enumerate(subset_dicts)
#         println("Combination: $index/$num_combinations.")
#         println(subset)
#         acc = CrossValidate(X, y, subset, n_folds)

#         push!(results, (subset, acc))
#     end

#     return results
# end

# # Additional utilities
# function TwoPointCorrelation(W::MPS, class_label::Int, operator::String = "Sz") 

#     """Function to return the two point correlation between each MPS site."""

#     # get the local hilbert space dimension
#     d = ITensors.dim(findindex(W[1], "Site"))

#     if d == 2
#         if operator == "Sz"
#             op = 1/2 .* [1 0; 0 -1]
#         elseif operator == "Sx"
#             op = 1/2 .* [0 1; 1 0]
#         else
#             error("Invalid operator. Can either be Sz or Sx.")
#         end
#     elseif d == 3
#         if operator == "Sz"
#             op = 1/sqrt(2) .* [0 1 0; 1 0 1; 0 1 0]
#         elseif operator == "Sx"
#             op = 1/sqrt(2) .* [1 0 0; 0 0 0; 0 0 -1]
#         else
#             error("Invalid operator. Can either be Sz or Sx.")
#         end
#     else
#         error("Unable to obtain the local Hilbert space dimension. Check the MPS indices.")
#     end
    
#     normalize!(W)
#     ψ = deepcopy(W)
#     decision_idx = findindex(ψ[1], "decision") # the label index is assumed to be attached to the first site
#     decision_state = onehot(decision_idx => (class_label+1)) # one-hot encoded indexing starts from zero
#     ψ[1] *= decision_state # filter out the MPS state corresponding to the target class
#     normalize!(ψ)

#     corr = correlation_matrix(ψ, op, op)

#     return corr
# end


# function EntanglementEntropyProfile(W::MPS, class_label::Int)

#     """Function to compute the entanglement entropy as a function of site/bond index."""
#     # isolate the MPS weights for the class of interest
#     normalize!(W)
#     ψ = deepcopy(W)
#     decision_idx = findindex(ψ[1], "decision") # assumes the label index is attached to the first site. 
#     decision_state = onehot(decision_idx => (class_label + 1))
#     ψ[1] *= decision_state
#     normalize!(ψ)
#     entropy = Vector{Float64}(undef, (length(W)-2))
#     # for MPS of length N we can compute the entanglement entropy of a bipartition of the system into a region "A"
#     # which consists of sites 1, 2, ..., b and region B of sites b+1, b+2,...N
#     # b tracks the orthogonality center
#     for b = (2:length(ψ)-1)
#         orthogonalize!(ψ, b) # shift the orthogonality center to the site b of the MPS 
#         U, S, V = svd(ψ[b], (linkind(ψ, b-1), siteind(ψ, b)))
#         SvN = 0.0
#         for n = 1:ITensors.dim(S, 1)
#             p = S[n, n]^2
#             SvN -= p * log2(p)
#         end
#         entropy[b-1] = SvN
#     end
#     return entropy
# end


# function SliceMPS(W::MPS, class_label::Int)
#     """General function to slice the MPS and return the state corresponding to a specific class label."""
#     ψ = deepcopy(W)
#     normalize!(ψ)
#     decision_idx = findindex(ψ[1], "decision")
#     decision_state = onehot(decision_idx => (class_label + 1))
#     ψ[1] *= decision_idx
#     normalize!(ψ)

#     return ψ
# end

# function DifferenceVector(ψ1::MPS, ψ2::MPS)
#     """Computes the difference (residual) vector between two state vectors, ψ1 and ψ2"""
#     # We subtract the component of class 0 that is parallel to class 1, leaving an orthogonal vector (state)
#     # this is the same as projecting state 2 onto state 1 and subtracting to get the residual vector 
#     # which component of ψ2 is most different from ψ1

#     normalize!(ψ1)
#     normalize!(ψ2)

#     Δ = ψ2 - (inner(ψ1, ψ2) * ψ1)

#     normalize!(Δ)

#     return Δ
# end

# function ComputeOverlapMatrix(X::Matrix, y::Matrix, params::Dict=LocalParameters; return_dmat=true)
#     """Compute the overlap between each product state in the dataset after being mapped by the feature map to Hilbert space."""
#     """The returned matrix can be interpreted as a dissimilarity matrix (1 - overlap)."""

#     feature_map = params[:feature_map]
#     d = params[:local_dimension]
    
#     if d == 2
#         sites = siteinds("S=1/2", size(X)[2])
#     elseif d == 3
#         sites = siteinds("S=1", size(X)[2])
#     else
#         error("Local dimension not found. Check the feature map used.")
#     end

#     overlap_matrix = Matrix{Any}(undef, size(X)[1], size(X)[1])
#     product_states = InitialiseProductStates(X, y, "train")
#     for i=1:length(product_states)
#         for j=1:length(product_states)
#         contract = inner(product_states[i].pstate, product_states[j].pstate)
#         overlap_matrix[i, j] = contract
#         end
#     end

#     if return_dmat == true
#         dmat = gramd2dmat(convert(Matrix{Float64}, overlap_matrix))
#         return dmat
#     else
#         return overlap_matrix
#     end
# end


# function InspectOverlap(X::Matrix, y::Matrix, W::MPS, params::Dict=LocalParameters) 
#     """Returns dataframe"""
#     n_classes = params[:num_classes]
#     product_states = InitialiseProductStates(X, y, "test")

#     contractions = Matrix{Union{Float64, Int}}(undef, size(X, 1), n_classes+2) 
    
#     for i=1:size(X, 1)
#         overlap = ContractMPSAndProdState(W, product_states[i])
#         contractions[i, 1:n_classes] = overlap

#         # get prediction based on overlaps
#         contractions[i, n_classes+1] = argmax(abs.(overlap)) - 1
#         # include ground truth label in last column
#         contractions[i, end] = product_states[i].label 
#     end

#     # convert matrix to dataframe
#     col_names = [Symbol("class_$i") for i=0:n_classes + 1] # double-check the indexing
#     # add the prediction and ground truth label column names
#     push!(col_names, :prediction, :ground_truth_label)
#     df = DataFrame(contractions, col_names)

#     return df
# end


