using ITensors
using Optim
using Random
using StatsBase
using Distributions

struct PState
    """Define a custom struct for product states"""
    pstate::MPS # product state as a vector of ITenors (MPS)
    label::Int # ground truth class label
    id::Int # identifier for caching
end

function complex_feature_map(x::Float64)
    s1 = exp(1im * (3π/2) * x) * cospi(0.5 * x)
    s2 = exp(-1im * (2π/2) * x) * sinpi(0.5 * x)
    return [s1, s2]
end

function generate_training_data(samples_per_class::Int; data_pts::Int=5)

    class_A_samples = zeros(samples_per_class, data_pts)
    class_B_samples = ones(samples_per_class, data_pts)
    all_samples = vcat(class_A_samples, class_B_samples)
    all_labels = Int.(vcat(zeros(size(class_A_samples)[1]), ones(size(class_B_samples)[1])))

    shuffle_idxs = shuffle(1:samples_per_class*2)


    return all_samples[shuffle_idxs, :], all_labels[shuffle_idxs]

end

function sample_to_product_state(ts::Vector, site_inds::Vector{Index{Int64}})
    """Convert a SINGLE time series (ts) to a product state (mps)"""
    n_sites = length(site_inds)
    product_state = MPS(n_sites)
    for site in 1:n_sites
        # loop over each site, create an itensor and fill with encoded values
        T = ITensor(ComplexF64, site_inds[site])
        zero_state, one_state = complex_feature_map(ts[site]) # 
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
    #@assert dataset_shape[1] > dataset_shape[2] "Ensure time series are in rows"

    all_product_states = Vector{PState}(undef, dataset_shape[1])
    for p in 1:length(all_product_states)
        # note, now using column-major ordering, so ts stored in COLUMNS not rows
        time_series_as_product_state = sample_to_product_state(ts_dataset[p, :], site_inds)
        time_series_label = ts_labels[p]
        product_state = PState(time_series_as_product_state, time_series_label, p)
        all_product_states[p] = product_state
    end

    return all_product_states
end

function flatten_bond_tensor(BT::ITensor)
    """Function to flatten an ITensor so that it can be fed into Optim
    as a vector."""
    # should probably return the indices as well
    # might need checks to ensure correct assignment of indices to values
    flattened_tensor = collect(Iterators.flatten(BT))
    return flattened_tensor, inds(BT)
end

function reconstruct_bond_tensor(BT_flat::Vector, indices)
    BT = ITensor(indices)
    # ORDER OF ASSIGNMENT MUST MATCH THE ORDER OF FLATTENING
    for (n, val) in enumerate(BT_flat)
        BT[n] = val
    end

    return BT

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

function loss_single(B::ITensor, LE::Matrix, RE::Matrix, product_state::PState, lid::Int, rid::Int)
    # regular loss for a single product state
    ps = product_state.pstate
    phi_tilde = ps[lid] * ps[rid]
    n = size(LE, 2)

    if lid == 1
        # LE does not exist
        phi_tilde *= RE[product_state.id, rid+1]
    elseif rid == n
        phi_tilde *= LE[product_state.id, lid-1]
    else
        phi_tilde *= RE[product_state.id, rid+1] * LE[product_state.id, lid-1]
    end

    yhat = B * phi_tilde
    y = product_state.label
    diff_sq = norm(yhat[] - y)^2
    loss = 0.5 * diff_sq

    return loss
end

function loss_batch(B::ITensor, LE::Matrix, RE::Matrix, pss::Vector{PState}, lid::Int, rid::Int)
    loss_accum = 0
    for ps in pss
        loss_accum += loss_single(B, LE, RE, ps, lid, rid)
    end
    final_loss = loss_accum / length(pss)

    return final_loss
end

function contract_mps_and_product_state(mps::MPS, product_state::PState)
    """Custom function to get the raw overlap for a single sample.
    Returns complex value"""

    ps = product_state.pstate
    res = 1
    num_sites = length(mps)
    for i in num_sites:-1:1
        res *= mps[i] * ps[i]
    end

    return res[]

end

function get_loss_and_acc_dataset(mps::MPS, pss::Vector{PState})
    """Get the loss for an entire mps for a given dataset"""
    loss_accum = 0
    correct_accum = 0
    for ps in pss
        y = ps.label
        yhat = contract_mps_and_product_state(mps, ps)
        diff_sq = (abs(yhat - y))^2
        loss = 0.5 * diff_sq
        loss_accum += loss
        pred = 0
        if abs(yhat) > 0.5
            pred = 1
        end
        if pred == y
            correct_accum += 1
        end

    end

    acc = correct_accum / length(pss)
    loss_final = loss_accum / length(pss)

    return loss_final, acc

end


function loss_flattened_single(B_flat::Vector, B_inds, LE, RE, product_state::PState, lid::Int, rid::Int)
    # loss function for bond tensor as a vector (flattened) for optim
    # first, reconstruct the bond tensor
    B = reconstruct_bond_tensor(B_flat, B_inds)
    loss = loss_single(B, LE, RE, product_state, lid, rid)
    return loss
end

function gradient_flat!(stor, B_flat::Vector, bt_inds, product_state::PState, lid, rid)
    B = reconstruct_bond_tensor(B_flat, bt_inds)
    ps = product_state.pstate
    phi_tilde = ps[lid] * ps[rid] * mps[3] * ps[3] * mps[4] * ps[4]
    yhat = B * phi_tilde
    y = product_state.label
    dP = yhat[] - y
    grad = dP * conj(phi_tilde)
    copyto!(stor, grad)
    return nothing
end


function loss_flattened_batch(B_flat::Vector, B_inds, LE::Matrix, RE::Matrix, pss::Vector{PState}, lid::Int, rid::Int)
    # loss function for bond tensor as a vector (flattened) for optim
    # first, reconstruct the bond tensor
    B = reconstruct_bond_tensor(B_flat, B_inds)
    loss_accum = 0
    for ps in pss
        loss_accum += loss_single(B, LE, RE, ps, lid, rid)
    end

    final_loss = loss_accum / length(pss)

    return final_loss
end

function gradient_flattened_batch!(stor, B_flat::Vector, B_inds, LE::Matrix, RE::Matrix, pss::Vector{PState}, lid, rid)
    B = reconstruct_bond_tensor(B_flat, B_inds)
    grad_accum = ITensor()
    for ps in pss
        prod_state = ps.pstate
        phi_tilde = prod_state[lid] * prod_state[rid]
        n = size(LE, 2)
        if lid == 1
            phi_tilde *= RE[ps.id, rid+1]
        elseif rid == n
            phi_tilde *= LE[ps.id, lid-1]
        else
            phi_tilde *= LE[ps.id, lid-1] * RE[ps.id, rid+1]
        end
        yhat = B * phi_tilde
        y = ps.label
        dP = yhat[] - y
        grad = dP * conj(phi_tilde)
        grad_accum += grad
    end

    grad_mean = grad_accum ./ length(pss)
    copyto!(stor, grad_mean)
    return nothing
end

function fg!(F, G, x, B_inds, LE::Matrix, RE::Matrix, pss::Vector{PState}, lid::Int, rid::Int)
    # common computations
    B = reconstruct_bond_tensor(x, B_inds)
    loss_accum = 0
    grad_accum = ITensor()

    for ps in pss
        prod_state = ps.pstate
        phi_tilde = prod_state[lid] * prod_state[rid]
        n = size(LE, 2) # number of mps sites
        if lid == 1
            phi_tilde *= RE[ps.id, rid+1]
        elseif rid == n
            phi_tilde *= LE[ps.id, lid-1]
        else
            phi_tilde *= LE[ps.id, lid-1] * RE[ps.id, rid+1]
        end
        yhat = B * phi_tilde
        y = ps.label
        dP = yhat[] - y
        diff_sq = norm(dP)^2
        loss_accum += 0.5 * diff_sq
    
        if G !== nothing
            # compute gradient
            grad = dP * conj(phi_tilde)
            grad_accum += grad
        end
    end

    if G !== nothing
        grad_overall = grad_accum ./ length(pss)
        copyto!(G, grad_overall)
    end

    if F !== nothing
        final_loss = loss_accum / length(pss)
        return final_loss
    end
end


function optimise_bond_tensor_batch(B_init::ITensor, LE::Matrix, RE::Matrix, pss::Vector{PState}, lid, rid)
    BT = deepcopy(B_init)
    B_flat, B_flat_inds = flatten_bond_tensor(BT)
    cost = x -> loss_flattened_batch(x, B_flat_inds, LE, RE, pss, lid, rid)
    grad! = (stor, x) -> gradient_flattened_batch!(stor, x, B_flat_inds, LE, RE, pss, lid, rid)
    out = optimize(cost, grad!, B_flat, ConjugateGradient(), Optim.Options(show_trace=true, iterations = 10))
    BT_new = reconstruct_bond_tensor(out.minimizer, B_flat_inds)

    return BT_new
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

function update_caches!(left_site_new::ITensor, right_site_new::ITensor, 
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

end


function basic_sweep(num_sweeps::Int, χ_max::Int=10)
    Random.seed!(47845674)
    s = siteinds("S=1/2", 100)
    mps = randomMPS(ComplexF64, s; linkdims=4)
    #samples, labels = generate_training_data(100; data_pts=20)
    #all_pstates = dataset_to_product_state(samples, labels, s)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = GenerateToyDataset(100, 100)
    scaler = fitScaler(RobustSigmoidTransform, X_train; positive=true);
    X_train_scaled = transformData(scaler, X_train)
    X_test_scaled = transformData(scaler, X_test)

    all_pstates = dataset_to_product_state(X_train_scaled, y_train, s)

    LE, RE = construct_caches(mps, all_pstates; going_left=false)

    init_loss, init_acc = get_loss_and_acc_dataset(mps, all_pstates)
    loss_per_sweep = [init_loss]
    acc_per_sweep = [init_acc]

    for sweep in 1:num_sweeps
        for i = 1:length(mps) - 1
            BT = mps[i] * mps[(i+1)]
            BT_new = optimise_bond_tensor_batch(BT, LE, RE, all_pstates, (i), (i+1))
            left_site_new, right_site_new = decompose_bond_tensor(BT_new, (i); χ_max=χ_max, going_left=false)
            update_caches!(left_site_new, right_site_new, LE, RE, (i), (i+1), all_pstates; going_left=false)
            mps[i] = left_site_new
            mps[(i+1)] = right_site_new
        end

        LE, RE = construct_caches(mps, all_pstates; going_left=true)

        for j = (length(mps)-1):-1:1
            BT = mps[j] * mps[(j+1)]
            BT_new = optimise_bond_tensor_batch(BT, LE, RE, all_pstates, (j), (j+1))
            left_site_new, right_site_new = decompose_bond_tensor(BT_new, (j); χ_max=χ_max, going_left=true)
            update_caches!(left_site_new, right_site_new, LE, RE, (j), (j+1), all_pstates; going_left=true)
            mps[j] = left_site_new
            mps[(j+1)] = right_site_new
        end
        LE, RE = construct_caches(mps, all_pstates; going_left=false)
        loss_sweep, acc_sweep = get_loss_and_acc_dataset(mps, all_pstates)
        push!(loss_per_sweep, loss_sweep)
        push!(acc_per_sweep, acc_sweep)
        println("Sweep $sweep finished. Loss: $loss_sweep | Acc: $acc_sweep.")
    end

    return mps, all_pstates, loss_per_sweep, acc_per_sweep
end



# s = siteinds("S=1/2", 4)
# mps = randomMPS(ComplexF64, s; linkdims=4)
# all_samples, all_labels = generate_training_data(100)
# all_pstates = dataset_to_product_state(all_samples, all_labels, s);
# B = mps[3] * mps[4]
# LE, RE = construct_caches(mps, all_pstates; going_left=true)
# loss_batch(B, LE, RE, all_pstates, 3,4)
# B_flat, B_flat_inds = flatten_bond_tensor(B)
# loss_flattened_batch(B_flat, B_flat_inds, LE, RE, all_pstates, 3, 4)
# B_new = optimise_bond_tensor_batch(B, LE, RE, all_pstates, 3, 4)
# loss_batch(B_new, LE, RE, all_pstates, 3, 4)


# loss_flattened_batch(B_flat, B_flat_inds, mps, all_pstates, 1, 2)

# B_new = optimise_bond_tensor_batch(B, mps, all_pstates, 1, 2)



# loop over all samples
#for ps in all_pstates


#optimise_bond_tensor(B, mps, all_pstates[1], 1, 2)
# loss_single(B, mps, all_pstates[101], 1, 2)
# loss(B, mps, all_pstates[101], 1, 2)
# B_flat, B_flat_inds = flatten_bond_tensor(B)
# loss_flattened_single(B_flat, B_flat_inds, mps, all_pstates[101], 1, 2)
# initial_guess = B_flat
# cost = x -> loss_flattened_single(x, B_flat_inds, mps, all_pstates[101], 1, 2)
# cost(initial_guess)
# grad! = (stor, x) -> gradient_flat!(stor, x, B_flat_inds, all_pstates[101], 1, 2)
# out = optimize(cost, grad!, initial_guess, LBFGS())
# BT_new = reconstruct_bond_tensor(out.minimizer, B_flat_inds)
# loss_single(BT_new, mps, all_pstates[101], 1, 2)
