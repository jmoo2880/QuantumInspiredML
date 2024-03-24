using ITensors
using Optim
using Folds
using Distributions


struct PState
    """Create a custom structure to store product state objects, 
    along with their associated label and type (i.e, train, test or valid)"""
    pstate::MPS
    label::Int
    id::Int
end

function complex_feature_map(x::Float64)
    s1 = exp(1im * (3π/2) * x) * cospi(0.5 * x)
    s2 = exp(-1im * (2π/2) * x) * sinpi(0.5 * x)
    return [s1, s2]
end

function sample_to_product_state(ts::Vector, site_inds::Vector{Index{Int64}})
    """Convert a SINGLE time series (ts) to a product state (mps)"""
    n_sites = length(site_inds)
    product_state = MPS(n_sites)
    for site in 1:n_sites
        # loop over each site, create an itensor and fill with encoded values
        T = ITensor(ComplexF64, site_inds[site])
        zero_state, one_state = complex_feature_map(ts[site])
        T[1] = zero_state
        T[2] = one_state
        product_state[site] = T
    end

    return product_state

end

function dataset_to_product_state(ts_dataset::Matrix, ts_labels::Vector{Int}, site_inds::Vector{Index{Int64}})
    """Convert ALL time series (ts) in a dataset to a vector of
    PStates"""
    dataset_shape = size(ts_dataset)

    all_product_states = Vector{PState}(undef, dataset_shape[1])
    for p in 1:length(all_product_states)
        time_series_as_product_state = sample_to_product_state(ts_dataset[p, :], site_inds)
        time_series_label = ts_labels[p]
        product_state = PState(time_series_as_product_state, time_series_label, p)
        all_product_states[p] = product_state
    end

    return all_product_states
end

function construct_caches(mps::MPS, training_product_states::Vector{PState}; going_left=true)
    """Function to pre-allocate tensor contractions between the MPS and the product states.
    LE stores the left environment, i.e. all accumulate contractions from site 1 to site N
    RE stores the right env., all contractions from site N to site 1."""

    # get the number of training samples to pre-allocated caches
    n_train = length(training_product_states)
    n = length(mps)
    # make the caches
    LE = Matrix{ITensor}(undef, n_train, n)
    RE = Matrix{ITensor}(undef, n_train, n)

    for i in 1:n_train 
        # get the product state for the current training sample
        ps = training_product_states[i].pstate

        if going_left
            # initialise the first contraction
            LE[i, 1] = mps[1] * ps[1]
            for j in 2:n
                LE[i, j] = LE[i, j-1] * ps[j] * mps[j]
            end
            
        else
            # going right
            RE[i, n] = ps[n] * mps[n]
            # accumulate remaining sites
            for j in n-1:-1:1
                RE[i, j] = RE[i, j+1] * ps[j] * mps[j]
            end
        end
    end

    return LE, RE

end


function loss_and_correct_per_sample(mps::MPS, product_state::PState)
    """Compute the loss for a single product state and the corresponding
    prediction. To be used with multi-threading/folds. Combine loss and
    accuracy into a single function to eliminate redundant calculations."""
    ps = product_state.pstate
    y = product_state.label # ground truth label

    yhat = 1
    num_sites = length(mps)
    # start at the terminal site and contract backwards
    for site = num_sites:-1:1
        yhat *= mps[site] * ps[site]
    end
    abs_diff_sq = norm(yhat[] - y)^2
    loss = 0.5 * abs_diff_sq

    # compare prediction to label, return 1 if correct else return 0
    pred = abs(yhat[])
    # ternery operator because i'm edgy and a Julian
    correct = (pred < 0.5 ? 0 : 1) == y ? 1.0 : 0.0

    return [loss, correct]

end

function loss_and_acc_batch(mps::MPS, pss::Vector{PState})
    """Function the loss and accuracy for an entire dataset of 
    product states using multithreading."""
    # check whether the MPS is normalised.
    if !isapprox(norm(mps), 1.0; atol=1e-2) @warn "MPS is not normalised!" end

    loss, acc = Folds.reduce(+, loss_and_correct_per_sample(mps, ps) for ps in pss)

    mean_loss = loss/length(pss)
    mean_acc = acc/length(pss)

    return mean_loss, mean_acc

end

function contract_mps_and_product_state(mps::MPS, product_state::PState)
    """Function to get the raw overlap for a single sample (for inspection).
    Returns a complex value"""
    if !isapprox(norm(mps), 1.0; atol=1e-2) @warn "MPS is not normalised!" end
    ps = product_state.pstate
    @assert length(mps) == length(ps) "Length of MPS does not match product state!"

    overlap = 1
    num_sites = length(mps)
    for site in num_sites:-1:1
        overlap *= mps[site] * ps[site]
    end

    return overlap[]

end

function get_overlaps_dataset(mps::MPS, pss::Vector{PState})
    # ASSUMES BINARY CLASSIFIER WITH CLASS 0 AND CLASS 1
    """Just print the stats, doesn't return anything"""
    overlaps_class_0 = []
    overlaps_class_1 = []

    for ps in pss
        class = ps.label
        raw_overlap = contract_mps_and_product_state(mps, ps)
        real_overlap = abs(raw_overlap)
        if class == 0
            push!(overlaps_class_0, real_overlap)
        else
            push!(overlaps_class_1, real_overlap)
        end
    end

    # get class-wise max/min/median
    c0_max, c0_min, c0_med = maximum(overlaps_class_0), minimum(overlaps_class_0), median(overlaps_class_0)
    c1_max, c1_min, c1_med = maximum(overlaps_class_1), minimum(overlaps_class_1), median(overlaps_class_1)

    println("Class ⟨0|ψ⟩ -> Max: $c0_max \t Min: $c0_min \t Median: $c0_med")
    println("Class ⟨1|ψ⟩ -> Max: $c1_max \t Min: $c1_min \t Median: $c1_med")

    return nothing

end

function flatten_bond_tensor(B::ITensor)
    """Function to flatten an ITensor so that it can be fed into Optim
    as a vector. Returns flattened tensor as a high dimensional 
    vector as well as the corresponding indices for reconstruction. """
    flattened_tensor = collect(Iterators.flatten(B))
    bond_tensor_indices = inds(B)

    return flattened_tensor, bond_tensor_indices

end

function reconstruct_bond_tensor(bond_tensor_flattened::Vector, bond_tensor_indices)
    """Function to reconstruct an ITensor, given a high dimensional vector
    and ITensor indices. """
    # check that the dimensions match up
    dim_flattened_tensor = length(bond_tensor_flattened)
    dim_indices = 1
    for i in bond_tensor_indices
        dim_indices *= i.space
    end
    @assert dim_indices == dim_flattened_tensor "Dimensions of flattened tensor do not match indices."
    BT = ITensor(bond_tensor_indices)
    for (n, val) in enumerate(bond_tensor_flattened)
        BT[n] = val
    end

    return BT
end

function fg!(F, G, B_flattened::Vector, B_inds::Any, pss::Vector{PState},
    LE::Matrix, RE::Matrix, lid::Int, rid::Int)
    # NOTE: B_flattened is entirely real with C index, splice inside of the loss and grad func to recover complex version
    B = reconstruct_bond_tensor(B_flattened, B_inds) 

    function loss_and_grad_single(product_state::PState)

        # get C index
        C_index = findinds(B, "C")[1]
        B_real = deepcopy(B) * onehot(C_index => 1) # extract the real component
        B_imag = deepcopy(B) * onehot(C_index => 2) # extract the imag component
        BT = B_real + im * B_imag
        ps = product_state.pstate
        ps_id = product_state.id
        y = product_state.label

        d_yhat_dW = ps[lid] * ps[rid]

        if lid == 1
            d_yhat_dW *= RE[ps_id, (rid+1)]
        elseif rid == length(ps)
            d_yhat_dW *= LE[ps_id, (lid-1)]
        else
            d_yhat_dW *= LE[ps_id, (lid-1)] * RE[ps_id, (rid+1)]
        end

        yhat = BT * d_yhat_dW
        dP = yhat[] - y
        abs_diff_sq = norm(dP)^2
        loss = 0.5 * abs_diff_sq
        grad = dP * conj(d_yhat_dW)

        # needs to return real valued 
        grad_real = real(grad)
        grad_imag = imag(grad)

        C_tensor_real = ITensor([1; 0], C_index)
        grad_real *= C_tensor_real

        C_tensor_imag = ITensor([0; 1], C_index)
        grad_imag *= C_tensor_imag

        grad_combined_real_imag = grad_real + grad_imag

        return [loss, grad_combined_real_imag]
    end

    loss_sum, grad_sum = Folds.reduce(+, loss_and_grad_single(ps) for ps in pss)
    # convert the gradient back to flattened/real tensor?

    if G !== nothing
        G .= grad_sum ./ length(pss)
    end

    if F !== nothing
        value = loss_sum / length(pss)
        return value
    end

end

function optimise_bond_tensor(BT::ITensor, pss::Vector{PState}, LE::Matrix, RE::Matrix,
    lid::Int, rid::Int; verbose=true, maxiters=2)
    """Handles all of the internal operations. Takes in complex valued BT
    then isolated real and imag components, adds C index to each,
    adds them back together into a real valued BT with C index, flattens 
    the new BT to pass to opimiser, optimises real valued tensor, reconstructs
    real valued tensor (from flattened version) and converts back to a complex valued BT."""

    C_index = Index(2, "C")
    bt_real = real(BT)
    bt_imag = imag(BT)

    bt_real_index_tensor = ITensor([1; 0], C_index)
    bt_imag_index_tensor = ITensor([0; 1], C_index)

    bt_real *= bt_real_index_tensor
    bt_imag *= bt_imag_index_tensor

    bt_combined_real_imag = bt_real + bt_imag

    # flatten bond tensor into a vector and get the indices
    bt_flat, bt_inds = flatten_bond_tensor(bt_combined_real_imag)
    # create anonymous function to feed into optim, function of bond tensor only
    fgcustom! = (F,G,B) -> fg!(F, G, B, bt_inds, pss, LE, RE, lid, rid)
    # set the optimisation manfiold
    # apply optim using specified gradient descent algorithm and corresp. paramters 
    # set the manifold to either flat, sphere or Stiefel 
    method = Optim.LBFGS(; alphaguess = Optim.LineSearches.InitialHagerZhang(),
        linesearch = Optim.LineSearches.HagerZhang(), P = nothing, precondprep = (P, x) -> nothing, manifold = Sphere())
    #method = Optim.LBFGS()
    res = Optim.optimize(Optim.only_fg!(fgcustom!), bt_flat, method=method, iterations = maxiters, show_trace = verbose)
    result_flattened = Optim.minimizer(res)

    # get the reconstructed ITensor as a real valued tensor
    result_as_ITensor = reconstruct_bond_tensor(result_flattened, bt_inds)

    # convert back to complex valued ITensor
    bt_real_reconst = deepcopy(result_as_ITensor) * onehot(C_index => 1)
    bt_imag_reconst = deepcopy(result_as_ITensor) * onehot(C_index => 2)

    BT_reconst = bt_real_reconst + im * bt_imag_reconst

    #normalize!(BT_reconst)

    return BT_reconst

end

function decompose_bond_tensor(BT::ITensor, lid::Int; χ_max=nothing, cutoff=nothing, going_left=true)
    """Decompose an updated bond tensor back into two tensors using SVD"""
    left_site_index = findindex(BT, "n=$lid")
    #label_index = findindex(BT, "f(x)")
    if going_left
         # need to make sure the label index is transferred to the next site to be updated
         if lid == 1
            U, S, V = svd(BT, (left_site_index); maxdim=χ_max, cutoff=cutoff)
        else
            bond_index = findindex(BT, "Link,l=$(lid-1)")
            U, S, V = svd(BT, (left_site_index, bond_index); maxdim=χ_max, cutoff=cutoff)
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
            U, S, V = svd(BT, (left_site_index, bond_index); maxdim=χ_max, cutoff=cutoff)
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

function update_caches(left_site_new::ITensor, right_site_new::ITensor, 
    LE::Matrix, RE::Matrix, lid::Int, rid::Int, pss::Vector{PState}; going_left=true)
    """Given a newly updated bond tensor, update the caches."""
    num_train = length(pss)
    num_sites = size(LE)[2]
    if going_left
        for i = 1:num_train
            if rid == num_sites
                RE[i, num_sites] = right_site_new * pss[i].pstate[num_sites]
            else
                RE[i, rid] = RE[i, rid+1] * right_site_new * pss[i].pstate[rid]
                
            end
        end

    else
        # going right
        for i = 1:num_train
            if lid == 1
                LE[i, 1] = left_site_new * pss[i].pstate[lid]
                
            else
                LE[i, lid] = LE[i, lid-1] * pss[i].pstate[lid] * left_site_new
                
            end
        end
    end

    return LE, RE

end

function basic_sweep(num_sweeps::Int, χ_max::Int=10, cutoff=nothing)

    Random.seed!(4574)
    s = siteinds("S=1/2", 96)

    mps = randomMPS(ComplexF64, s; linkdims=4)

    samples, labels = generate_training_data(100, 96)
    all_pstates = dataset_to_product_state(samples, labels, s)
    samples_test, labels_test = generate_training_data(100, 96)

    #(X_train, y_train), (X_val, y_val), (X_test, y_test) = GenerateToyDataset(20, 100)
    #scaler = fitScaler(RobustSigmoidTransform, X_train; positive=true);
    #X_train_scaled = transformData(scaler, X_train)
    #X_test_scaled = transformData(scaler, X_test)

    #all_pstates = dataset_to_product_state(X_train_scaled, y_train, s)
    all_test_pstates = dataset_to_product_state(samples_test, labels_test, s)

    LE, RE = construct_caches(mps, all_pstates; going_left=false)

    init_loss, init_acc = loss_and_acc_batch(mps, all_pstates)

    loss_per_sweep = [init_loss]
    acc_per_sweep = [init_acc]

    for sweep in 1:num_sweeps
        for i = 1:length(mps) - 1
            BT = mps[i] * mps[(i+1)]
            BT_new = optimise_bond_tensor(BT, all_pstates, LE, RE, (i), (i+1))
            left_site_new, right_site_new = decompose_bond_tensor(BT_new, (i); χ_max=χ_max, cutoff=cutoff, going_left=false)
            LE, RE = update_caches(left_site_new, right_site_new, LE, RE, (i), (i+1), all_pstates; going_left=false)
            mps[i] = left_site_new
            mps[(i+1)] = right_site_new
        end

        LE, RE = construct_caches(mps, all_pstates; going_left=true)

        for j = (length(mps)-1):-1:1
            BT = mps[j] * mps[(j+1)]
            BT_new = optimise_bond_tensor(BT, all_pstates, LE, RE, (j), (j+1))
            left_site_new, right_site_new = decompose_bond_tensor(BT_new, (j); χ_max=χ_max, cutoff=cutoff, going_left=true)
            LE, RE = update_caches(left_site_new, right_site_new, LE, RE, (j), (j+1), all_pstates; going_left=true)
            mps[j] = left_site_new
            mps[(j+1)] = right_site_new
        end

        LE, RE = construct_caches(mps, all_pstates; going_left=false)
        loss_sweep, acc_sweep = loss_and_acc_batch(mps, all_pstates)

        push!(loss_per_sweep, loss_sweep)
        push!(acc_per_sweep, acc_sweep)

        println("Sweep $sweep finished. Loss: $loss_sweep | Acc: $acc_sweep.")

    end

    test_loss, test_acc = loss_and_acc_batch(mps, all_test_pstates)
    println("Final test acc: $test_acc | test loss: $test_loss")

    return mps, all_pstates, loss_per_sweep, acc_per_sweep

end

# s = siteinds("S=1/2", 5)
# mps = randomMPS(ComplexF64, s; linkdims=4)
# samples, labels = generate_training_data(1000, 5)
# pstates = dataset_to_product_state(samples, labels, s)
# LE, RE = construct_caches(mps,pstates; going_left=false)
# bt = mps[1] * mps[2]
# BT_reconst = optimise_bond_tensor(bt, pstates, LE, RE, 1, 2)

# C_index = Index(2, "C")
# bt_real = real(bt)
# bt_imag = imag(bt)
# bt_real_index_tensor = ITensor([1; 0], C_index)
# bt_real *= bt_real_index_tensor
# bt_imag_index_tensor = ITensor([0; 1], C_index)
# bt_imag *= bt_imag_index_tensor
# bt_combined_real_imag = bt_real + bt_imag
# flattened_bt, flattened_bt_inds = flatten_bond_tensor(bt_combined_real_imag)
# reconstruct_bt_combined = reconstruct_bond_tensor(flattened_bt, flattened_bt_inds, bt_combined_real_imag)