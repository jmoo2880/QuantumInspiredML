using ITensors
using Folds
using Distributions
using OptimKit
using Zygote
using Random

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

    return all_samples, all_labels

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
    @assert dataset_shape[1] > dataset_shape[2] "Ensure time series are in rows"

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

# maybe look at column major order here for the caches...
# also parallelise above a certain threshold...
function construct_caches(mps::MPS, training_product_states::Vector{PState}; going_left=true)
    """Function to pre-allocate tensor contractions between the MPS and the product states.
    LE stores the left environment, i.e. all accumulate contractions from site 1 to site N
    RE stores the right env., all contractions from site N to site 1."""

    # get the number of training samples to pre-allocate caches
    n_train = length(training_product_states)
    n = length(mps)
    # make the caches
    LE = fill(ITensor(), n_train, n) # pre-fill with empty ITensors which will be overwritten
    RE = fill(ITensor(), n_train, n)

    for i in 1:n_train
        # get the product state for the current training sample
        ps = training_product_states[i].pstate

        if going_left
            # initialise the first contraction
            LE[i, 1] = conj(ps[1]) * mps[1]
            for j in 2:n
                LE[i, j] = LE[i, j-1] * conj(ps[j]) * mps[j]
            end

        else
            # going right
            RE[i, n] = conj(ps[n]) * mps[n]
            # accumulate remaining sites
            for j in n-1:-1:1
                RE[i, j] = RE[i, j+1] * conj(ps[j]) * mps[j]
            end
        end
    end

    return LE, RE

end

function contract_mps_and_product_state(mps::MPS, phi::PState)
    """Custom function to get the raw overlap for a single sample.
    Returns complex value"""

    ps = phi.pstate
    res = 1
    for i in eachindex(mps)
        res *= mps[i] * conj(ps[i])
    end

    return res[]

end

function get_loss_single(BT::ITensor, ps::PState, LE::Matrix, RE::Matrix, lid::Int, rid::Int)
    """Function to compute the loss and return whether or not correctly classified"""
    """For a single bond tensor"""

    y = ps.label
    # infer number of sites from product state since only ps is passed in
    num_sites = length(ps.pstate)
    # do bond tensor contraction with its product state sites first, then envs. much faster
    # BT + product state sites gives rank 2 tensor (3 with label) and then contract rank 2 with 2 rank 1s.
    yhat = BT * conj(ps.pstate[lid]) * conj(ps.pstate[rid])
    
    # check if LE or RE exists
    if lid == 1
        # if lid is on the first site, no LE exists...
        yhat *= RE[ps.id, (rid+1)]
    elseif rid == num_sites
        # if rid is on terminal site, no RE exists...
        yhat *= LE[ps.id, (lid-1)]
    else
        yhat *= LE[ps.id, (lid-1)] * RE[ps.id, (rid+1)]
    end

    yhat = abs(yhat[]) # get |<product state | mps >|
    diff_sq = (yhat - y)^2
    loss = 0.5 * diff_sq
    return loss

end

function loss_and_grad_autograd(BT::ITensor, ps::PState, LE::Matrix, RE::Matrix, lid::Int, rid::Int)
    """Apply autograd and only return relevant parts for folding"""
    loss, (grad,) = withgradient(get_loss_single, BT, ps, LE, RE, lid, rid)

    return [loss, grad]

end

function loss_and_grad_bond_tensor(BT::ITensor, pss::Vector{PState}, LE::Matrix, RE::Matrix, lid::Int, rid::Int)
    """Optimise bond tensor for an entire dataset"""
    num_samples = length(pss)
    loss, grad = Folds.reduce(+, loss_and_grad_autograd(BT, ps, LE, RE, lid, rid) for ps in pss)
    
    grad_final = grad ./ num_samples
    loss_final = loss / num_samples
    
    return loss_final, grad_final

end

function update_bond_tensor(BT::ITensor, pss::Vector{PState}, LE::Matrix, RE::Matrix, lid::Int, rid::Int; num_steps=100, lr=0.8)
    """Apply num_steps of gradient descent with learning rate lr"""
    BT_old = BT
    for step in 1:num_steps
        loss, grad = loss_and_grad_bond_tensor(BT_old, pss, LE, RE, lid, rid)
        println("Loss at step $step: $loss")
        new_BT = BT_old - lr * grad
        BT_old = new_BT
    end

    BT_old ./= sqrt(inner(dag(BT_old), BT_old))
    
    return BT_old

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

function UpdateCaches!(left_site_new::ITensor, right_site_new::ITensor, 
    LE::Matrix, RE::Matrix, lid::Int, rid::Int, pss::Vector{PState}; going_left=true)
    """Given a newly updated bond tensor, update the caches."""
    num_train = length(pss)
    num_sites = size(LE)[2]
    if going_left
        for i = 1:num_train
            if rid == num_sites
                RE[i, num_sites] = right_site_new * conj(pss[i].pstate[num_sites])
            else
                RE[i, rid] = RE[i, rid+1] * right_site_new * conj(pss[i].pstate[rid])
            end
        end

    else
        # going right
        for i = 1:num_train
            if lid == 1
                LE[i, 1] = left_site_new * conj(pss[i].pstate[lid])
            else
                LE[i, lid] = LE[i, lid-1] * conj(pss[i].pstate[lid]) * left_site_new
            end
        end
    end

end

Random.seed!(42)
s = siteinds("S=1/2", 4)
mps = randomMPS(ComplexF64, s; linkdims=4)

all_samples, all_labels = generate_training_data(100; data_pts=4)
all_pstates = dataset_to_product_state(all_samples, all_labels, s)

LE, RE = construct_caches(mps, all_pstates; going_left=false)

# make small training loop
nsweeps = 5
for sweep in 1:nsweeps
    for i = 1:length(mps)-1
        BT = mps[i] * mps[i+1]
        BT_new = update_bond_tensor(BT, all_pstates, LE, RE, i, (i+1); num_steps=20)
        left_site_new, right_site_new = decompose_bond_tensor(BT_new, i; χ_max=4, going_left=false)
        UpdateCaches!(left_site_new, right_site_new, LE, RE, i, (i+1), all_pstates; going_left=false)
        mps[i] = left_site_new
        mps[i+1] = right_site_new
    end

    LE, RE = construct_caches(mps, all_pstates; going_left=true)

    for j = (length(mps)-1):-1:1
        BT = mps[j] * mps[j+1]
        BT_new = update_bond_tensor(BT, all_pstates, LE, RE, j, (j+1); num_steps=20)
        left_site_new, right_site_new = decompose_bond_tensor(BT_new, j; χ_max=4, going_left=true)
        UpdateCaches!(left_site_new, right_site_new, LE, RE, j, (j+1), all_pstates; going_left=true)
        mps[j] = left_site_new
        mps[j+1] = right_site_new
    end
end






#get_loss_and_is_correct_single(bt, all_pstates[1], LE, RE)
#f, (grad,) = withgradient(get_loss_and_is_correct_single, bt, all_pstates[1], LE, RE) 



# x, fx, gx, numfg, normgradhistory = optimize(fg, x₀, algorithm; kwargs...)