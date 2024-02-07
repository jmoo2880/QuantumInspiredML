using ITensors
using Random
using StatsBase
using Base.Threads
using DelimitedFiles

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

function ComputePhiTilde(LE::Matrix, RE::Matrix, product_states::Vector{PState};
    pid::Int, lid::Int, rid::Int)

    """Compute the derivative of the MPS with resepct to bond tensor for a single product state indexed by `pid'.
    
    MPS:  O-...-O-      -O-...-O
          |     |        |     |
          |     |  |  |  |     |
    Data: O     O  O  O  O     O
            LE     ^  ^    RE
                  LID RID

    Resulting in an order-4 tensor (in bulk, order-3 at ends):

    | | | |
    O O O O
    ^     ^
    LE    RE 

    Uses the caches to access pre-computed tensors.

    lid will always index the left-most tensor (of two sites being considered),
    irrespective of direction and rid will always be right most tensor.
    """

    N = length(product_states[1].pstate)

    # check whether at the ends, in which case one of either LE/RE does not exist
    if lid == 1
        # at the first site, no LE
        # formatted from left to right, so env - product state, product state - env

        phi_tilde = product_states[pid].pstate[lid] * product_states[pid].pstate[rid] * RE[pid, (rid+1)]

    elseif rid == N
        # terminal site, no RE
        phi_tilde = LE[pid, lid-1] * product_states[pid].pstate[lid] * product_states[pid].pstate[rid]
    
    else
        # we are in the bulk, both LE and RE exist
        phi_tilde = LE[pid, lid-1] * product_states[pid].pstate[lid] * product_states[pid].pstate[rid] * RE[pid, (rid+1)]
    end

    return phi_tilde

end

function ComputeQuadraticCost(W::MPS, product_states::Vector{PState})
    """
    Loss for an entire MPS. 
    The loss function is given as
        L = 1/(2T) * SUM_{i=1}^T (f^l(x)- y^l)^2
    """

    Ns = length(product_states)
    # sum over data dependent terms
    sum = 0
    for ps in product_states
        # get the ground truth label and one-hot encode
        overlap = ContractMPSAndProductState(W, ps)
        label_index = inds(overlap)[1]
        label_index_new = deepcopy(label_index)
        true_label = ps.label
        target_vector = onehot(label_index_new => (true_label + 1)) # plus one offset because labels start at 0 and julia starts at 1
        diff = overlap - target_vector
        sum += inner(diff, diff)
    end

    cost = 1/(2 * Ns) * sum

    return cost

end

function ComputeGradient(B::ITensor, LE::Matrix, RE::Matrix, product_states::Vector{PState};
    lid::Int, rid::Int)

    B_old = B

    num_samples = length(product_states)
    num_sites = length(product_states[1].pstate)

    gradient_accumulate = Vector{ITensor}(undef, num_samples)
    # get the gradient

    Threads.@threads for i=1:num_samples
        phi_tilde = ComputePhiTilde(LE, RE, product_states; pid=i, lid=lid, rid=rid)
        decision_func_output = B_old * phi_tilde
        true_label = product_states[i].label
        label_index = inds(decision_func_output)[1]
        label_index_new = deepcopy(label_index)
        target_vector = onehot(label_index_new => (true_label + 1))
        #println("REACHED")
        #println(decision_func_output)
        diff = target_vector - decision_func_output
        #println("ALSO REACHED")
        grad = diff * phi_tilde
        # check rank
        order_val = order(grad)
        if (rid == num_sites) || (lid == 1) 
            # end of mps, expect only rank 4 tensor (2 physical indices, 1 bond index and 1 label index)
            if order_val !== 4
                error("Unexpected gradient tenosr order. Expected rank 4 tensor but got rank $(order_val).")
            end
        else
            if order_val !== 5
                # in the bulk of the mps, we expect a rank 5 tensor (2 physical indices, 2 bond indices and 1 label index)
                error("Unexpected gradient tensor order. Expected rank 5 tensor but got rank $(order_val)")
            end
        end

        gradient_accumulate[i] = grad

    end

    sum_grad = sum(gradient_accumulate)
    final_grad = sum_grad ./ num_samples

    return final_grad

end

function UpdateBondTensor(W::MPS, lid::Int, rid::Int, product_states::Vector{PState},
    LE::Matrix, RE::Matrix; going_left=true, α::Float64=0.1, χ_max=nothing, cutoff=nothing)


    num_train = length(product_states)
    num_sites = length(W)

    if going_left
        
        left_site = W[lid]
        right_site = W[rid]

        # construct old bond tensor
        B_old = left_site * right_site

        gradient = ComputeGradient(B_old, LE, RE, product_states; lid=lid, rid=rid)
        #println(gradient)
        B_new = B_old + α * gradient # addition because gradient is -dC/dB
        # rescale the new bond tensor to keep the MPS normalised
        B_new ./= sqrt(inner(dag(B_new), B_new))

        # now that we have the updated bond tensor, let's SVD back into MPS tensors
        left_site_index = findindex(B_new, "S=1/2,Site,n=$lid" )
        # we also need to ensure we retain the bond index on the left site
        label_index = findindex(B_new, "f(x)")

        if lid == 1
            if χ_max !== nothing && cutoff !== nothing
                U, S, V = svd(B_new, (left_site_index, label_index); maxdim=χ_max, cutoff=cutoff)
            elseif χ_max !== nothing
                # just max bond dimension has been specified, use default cutoff
                U, S, V = svd(B_new, (left_site_index, label_index); maxdim=χ_max)
            else
                # just cutoff specified, use default max bond dimension
                U, S, V = svd(B_new, (left_site_index, label_index); cutoff=cutoff)
            end
        else
            # ensure that the U site has the bond id-1 where id is the left site
            bond_index = findindex(B_new, "Link,l=$(lid-1)")
            if χ_max !== nothing && cutoff !== nothing
                U, S, V = svd(B_new, (left_site_index, bond_index, label_index); maxdim=χ_max, cutoff=cutoff)
            elseif χ_max !== nothing
                U, S, V = svd(B_new, (left_site_index, bond_index, label_index); maxdim=χ_max)
            else
                U, S, V = svd(B_new, (left_site_index, bond_index, label_index); cutoff=cutoff)
            end
        end

        # to preserve canonicalisation, we need to absorb the singular values into the left site when moving left

        left_site_new = U * S
        right_site_new = V

        replacetags!(left_site_new, "Link,v", "Link,l=$lid")
        replacetags!(right_site_new, "Link,v", "Link,l=$lid")

        # update environments
        for i = 1:num_train
            if rid == num_sites
                RE[i, num_sites] = right_site_new * product_states[i].pstate[num_sites]
            else
                RE[i, rid] = RE[i, rid+1] * right_site_new * product_states[i].pstate[rid]
            end
        end

    else
        # going right 
        left_site = W[lid]
        right_site = W[rid]

        # construct old bond tensor
        B_old = left_site * right_site

        gradient = ComputeGradient(B_old, LE, RE, product_states; lid=lid, rid=rid)
        B_new = B_old + α * gradient # addition because gradient is -dC/dB

        # rescale the new bond tensor to keep the MPS normalised
        B_new ./= sqrt(inner(dag(B_new), B_new))

        # now that we have the updated bond tensor, let's SVD back into MPS tensors
        left_site_index = findindex(B_new, "S=1/2,Site,n=$lid" )
        
        if lid == 1
            if χ_max !== nothing && cutoff !== nothing
                U, S, V = svd(B_new, (left_site_index); maxdim=χ_max, cutoff=cutoff)
            elseif χ_max !== nothing
                U, S, V = svd(B_new, (left_site_index); maxdim=χ_max)
            else
                U, S, V = svd(B_new, (left_site_index); cutoff=cutoff)
            end
        else
            bond_index = findindex(B_new, "Link,l=$(lid-1)") # retain the bond dimension index
            # by convention, any specified indices are retained on the U tensor
            if χ_max !== nothing && cutoff !== nothing
                U, S, V = svd(B_new, (bond_index, left_site_index); maxdim=χ_max, cutoff=cutoff)
            elseif χ_max !== nothing
                U, S, V = svd(B_new, (bond_index, left_site_index); maxdim=χ_max)
            else
                U, S, V = svd(B_new, (bond_index, left_site_index); cutoff=cutoff)
            end
        end

        left_site_new = U
        right_site_new = S * V

        # fix tag names
        replacetags!(left_site_new, "Link,u", "Link,l=$(lid)")
        replacetags!(right_site_new, "Link,u", "Link,l=$(lid)")

        # update environments
        for i = 1:num_train
            if lid == 1
                LE[i, 1] = left_site_new * product_states[i].pstate[lid]
            else
                LE[i, lid] = LE[i, lid-1] * product_states[i].pstate[lid] * left_site_new
            end
        end

    end

    return left_site_new, right_site_new, LE, RE

end

function fitMPS(X_train::Matrix, y_train::Vector, X_val::Matrix, y_val::Vector; χ_init=5,
    nsweep=10, α=0.01, χ_max=50, cutoff=nothing, random_state=nothing, sweep_tol=nothing)

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
    LE, RE = ConstructCaches(W, training_states; going_left=true)

    # compute initial training and validation loss
    init_train_loss = ComputeQuadraticCost(W, training_states)
    init_valid_loss = ComputeQuadraticCost(W, validation_states)

    println("Initial train loss: $init_train_loss")
    println("Initial validation loss: $init_valid_loss")

    running_train_loss = init_train_loss
    running_valid_loss = init_valid_loss 

    # start the sweep
    for itS = 1:nsweep
        println("Starting backward sweep: ($itS/$nsweep)")

        for j = (length(sites)-1):-1:1
            # j tracks lid
            lsn, rsn, LE, RE = UpdateBondTensor(W, j, j+1, training_states, LE, RE; going_left=true, α=α, 
                χ_max=χ_max, cutoff=cutoff)
            W[j] = lsn
            W[j+1] = rsn
        end

        # finished the backward sweep, print output and reconstruct caches for next forward sweep
        println("Backward sweep finished.")

        # construct new caches for forward sweep
        LE, RE = ConstructCaches(W, training_states; going_left=false)

        println("Starting forward sweep: ($itS/$nsweep)")

        for j = 1:(length(sites) - 1)
            lsn, rsn, LE, RE = UpdateBondTensor(W, j, j+1, training_states, LE, RE; going_left=false, α=α, 
            χ_max=χ_max, cutoff=cutoff)
            W[j] = lsn
            W[j+1] = rsn
        end

        println("Finished sweep $itS")

        # compute new loss
        train_loss = ComputeQuadraticCost(W, training_states)
        valid_loss = ComputeQuadraticCost(W, validation_states)


        println("Validation loss after sweep $itS: $valid_loss")
        println("Training loss after sweep $itS: $train_loss ")

        if sweep_tol !== nothing
            if ΔC_valid < sweep_tol
                println("Convergence reached. ΔC Val = $ΔC_valid is less than the threshold $sweep_tol)!")
                break
            end
        end


        running_train_loss = train_loss
        running_valid_loss = valid_loss
    end

    return W, sites 
end

function ScoreMPS(W, sites, X_test, y_test; binarise_method="median")
    """Return the test accuracy of the final model"""
    # binarise the test data
    test_data_binarised = BinariseDataset(X_test; method=binarise_method)
    test_states = GenerateAllProductStates(test_data_binarised, y_test, "test", sites)

    num_correct = 0
    for i=1:length(test_states)
        overlap = ContractMPSAndProductState(W, test_states[i])
        overlap = abs.(vector(overlap))
        pred = argmax(overlap) - 1
        if pred == test_states[i].label
            num_correct += 1
        end
    end

    return num_correct / length(test_states)

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


W, sites = fitMPS(X_train, y_train, X_test, y_test; χ_max=50, α=0.1, nsweep=10)
ScoreMPS(W, sites, X_test, y_test)


# sites = siteinds("S=1/2", 100)
# W = GenerateStartingMPS(5, sites)
# phis = GenerateAllProductStates(x, y, "train", sites)
# LE, RE = ConstructCaches(W, phis)
# #phi_tilde = ComputePhiTilde(LE, RE, phis; pid=1, lid=90, rid=100)
# phi_tilde = ComputePhiTilde(LE, RE, phis; pid=1, lid=99, rid=100)
# B = W[99] * W[100]
# final_grad = ComputeGradient(B, LE, RE, phis; lid=99, rid=100)
# left_site_new, right_site_new, LE, RE = UpdateBondTensor(W, 99, 100, phis, LE, RE)
