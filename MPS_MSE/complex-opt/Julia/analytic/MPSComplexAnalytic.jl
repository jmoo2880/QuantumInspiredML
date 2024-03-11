using ITensors
using Random
using Folds
using StatsBase
using Distributions

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
end

function GenerateSine(n, amplitude=1.0, frequency=1.0)
    t = range(0, 2π, n)
    phase = rand(Uniform(0, 2π)) # randomise the phase
    return amplitude .* sin.(frequency .* t .+ phase) .+ 0.2 .* randn(n)
end

function GenerateRandomNoise(n, scale=1)
    return randn(n) .* scale
end

function GenerateToyDataset(n, dataset_size, train_split=0.7, val_split=0.15)
    # calculate size of the splits
    train_size = floor(Int, dataset_size * train_split) # round to an integer
    val_size = floor(Int, dataset_size * val_split) # do the same for the validation set
    test_size = dataset_size - train_size - val_size # whatever remains

    # initialise structures for the datasets
    X_train = zeros(Float64, train_size, n)
    y_train = zeros(Int, train_size)

    X_val = zeros(Float64, val_size, n)
    y_val = zeros(Int, val_size)

    X_test = zeros(Float64, test_size, n)
    y_test = zeros(Int, test_size)

    function insert_data!(X, y, idx, data, label)
        X[idx, :] = data
        y[idx] = label
    end

    for i in 1:train_size
        label = rand(0:1)  # Randomly choose between sine wave (0) and noise (1)
        data = label == 0 ? GenerateSine(n) : GenerateRandomNoise(n)
        insert_data!(X_train, y_train, i, data, label)
    end

    for i in 1:val_size
        label = rand(0:1)
        data = label == 0 ? GenerateSine(n) : GenerateRandomNoise(n)
        insert_data!(X_val, y_val, i, data, label)
    end

    for i in 1:test_size
        label = rand(0:1)
        data = label == 0 ? GenerateSine(n) : GenerateRandomNoise(n)
        insert_data!(X_test, y_test, i, data, label)
    end

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

end

struct PState
    """Define a custom struct for product states"""
    pstate::MPS # product state as a vector of ITenors (MPS)
    label::Int # ground truth class label
    id::Int # identifier for caching
end

function complex_feature_map(x::Float64)
    """Complex feature map with local dimension 2."""

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

function overlaps_dataset(mps::MPS, all_product_states::Vector{PState})
    """Compute the overlaps for an entire dataset"""
    all_overlaps = Vector{Float64}(undef, length(all_product_states))
    for (index, ps) in enumerate(all_product_states)
        contraction_res = contract_mps_and_product_state(mps, ps)
        all_overlaps[index] = abs(contraction_res)
    end

    return all_overlaps
        
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

function compute_loss_and_gradient_single(BT::ITensor, LE::Matrix, RE::Matrix, product_state::PState, 
   lid::Int, rid::Int)

    ps = product_state.pstate
    ps_id = product_state.id
    phi_tilde = ps[lid] * ps[rid]
    num_sites = size(RE, 2)
    if lid == 1
        phi_tilde *= RE[ps_id, (rid+1)]
    elseif rid == num_sites
        phi_tilde *= LE[ps_id, (lid-1)]
    else
        phi_tilde *= RE[ps_id, (rid+1)] * LE[ps_id, (lid-1)]
    end
    yhat = BT * phi_tilde
    y = product_state.label
    diff_sq = (abs(yhat[] - y))^2
    loss = 0.5 * diff_sq

    dP = yhat[] - y
    grad = 0.5 * dP * conj(phi_tilde)

    return [loss, grad]

end

function compute_loss_and_gradient_batch(BT::ITensor, LE::Matrix, RE::Matrix, pss::Vector{PState},
    lid::Int, rid::Int)
    """Computes the loss and gradient for a bond tensor over an entire batch"""
    num_samples = length(pss)
    loss, grad = Folds.reduce(+, compute_loss_and_gradient_single(BT, LE, RE, ps, lid, rid) for ps in pss)
    grad_final = grad ./ num_samples
    loss_final = loss / num_samples

    return loss_final, grad_final
end

function update_bond_tensor(BT_init::ITensor, LE::Matrix, RE::Matrix, lid::Int, rid::Int, 
    pss::Vector{PState}; lr=0.8, max_iters=100)
    """Apply up to max_iters iterations of gradient descent"""
    bond_tensor_name = "S$lid - S$rid"

    B = deepcopy(BT_init)
    for step in 1:max_iters
        loss_step, grad_step = compute_loss_and_gradient_batch(B, LE, RE, pss, lid, rid)
        println("$bond_tensor_name | Step: $step | Loss: $loss_step")
        B = B - lr * grad_step
    end

    #normalize!(B)

    return B
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

function basic_sweep(num_sweeps::Int, lr::Float64, χ_max::Int=10, max_iters=1)
    Random.seed!(47845674)
    s = siteinds("S=1/2", 50)
    mps = randomMPS(ComplexF64, s; linkdims=4)
    #samples, labels = generate_training_data(100; data_pts=200)
    #all_pstates = dataset_to_product_state(samples, labels, s)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = GenerateToyDataset(50, 200)
    scaler = fitScaler(RobustSigmoidTransform, X_train; positive=true);
    X_train_scaled = transformData(scaler, X_train)
    X_test_scaled = transformData(scaler, X_test)

    all_pstates = dataset_to_product_state(X_train_scaled, y_train, s)
    all_test_pstates = dataset_to_product_state(X_test_scaled, y_test, s)

    LE, RE = construct_caches(mps, all_pstates; going_left=false)

    init_loss, init_acc = get_loss_and_acc_dataset(mps, all_pstates)
    loss_per_sweep = [init_loss]
    acc_per_sweep = [init_acc]

    for sweep in 1:num_sweeps
        for i = 1:length(mps) - 1
            BT = mps[i] * mps[(i+1)]
            BT_new = update_bond_tensor(BT, LE, RE, (i), (i+1), all_pstates; lr=lr, max_iters=max_iters)
            left_site_new, right_site_new = decompose_bond_tensor(BT_new, (i); χ_max=χ_max, going_left=false)
            update_caches!(left_site_new, right_site_new, LE, RE, (i), (i+1), all_pstates; going_left=false)
            mps[i] = left_site_new
            mps[(i+1)] = right_site_new
        end

        LE, RE = construct_caches(mps, all_pstates; going_left=true)

        for j = (length(mps)-1):-1:1
            BT = mps[j] * mps[(j+1)]
            BT_new = update_bond_tensor(BT, LE, RE, (j), (j+1), all_pstates; lr=lr, max_iters=max_iters)
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



# s = siteinds("S=1/2", 5)
# mps = randomMPS(ComplexF64, s; linkdims=4)
# samples, labels = generate_training_data(100)
# all_pstates = dataset_to_product_state(samples, labels, s)
# get_loss_and_acc_dataset(mps, all_pstates)
# LE, RE = construct_caches(mps, all_pstates; going_left=false)
# B = mps[1] * mps[2]
# #@show B
# compute_loss_and_gradient_per_sample(B, LE, RE, all_pstates[101], 1, 2)
# B_new = update_bond_tensor(B, LE, RE, 1, 2, all_pstates; lr=0.8)
# #compute_loss_and_gradient_per_sample(B_new, LE, RE, all_pstates[101], 1, 2)