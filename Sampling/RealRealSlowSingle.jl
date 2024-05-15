using Random
using ITensors
using Zygote
using DelimitedFiles
using Plots, Plot.PlotMeasures
using Folds
using JLD2
using Normalization

struct PState
    """Create a custom structure to store product state objects, 
    along with their associated label and type (i.e, train, test or valid)"""
    pstate::MPS
    label::Int
    id::Int
end

function load_splits_txt(train_set_location::String, val_set_location::String, 
    test_set_location::String)
    """As per typical UCR formatting, assume labels in first column, followed by data"""
    # do checks
    train_data = readdlm(train_set_location)
    val_data = readdlm(val_set_location)
    test_data = readdlm(test_set_location)

    X_train = train_data[:, 2:end]
    y_train = Int.(train_data[:, 1])

    X_val = val_data[:, 2:end]
    y_val = Int.(val_data[:, 1])

    X_test = test_data[:, 2:end]
    y_test = Int.(test_data[:, 1])

    # recombine val and train into train

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

end

function load_jld2_dset(train_loc::String, test_loc::String)
    """Load training and test data from .jld2 files.
    Expect each to contain time series, X, and labels,
    y."""

    # do checks
    train_loaded = JLD2.load(train_loc)
    test_loaded = JLD2.load(test_loc)
    X_train = train_loaded["X_train"]
    y_train = train_loaded["y_train"]
    X_test = test_loaded["X_test"]
    y_test = test_loaded["y_test"]

    return (X_train, y_train), (X_test, y_test)

end

function generate_startingMPS(χ_init::Int, site_indices::Vector{Index{Int64}}; 
    random_state = nothing)

    if random_state !== nothing
        Random.seed!(random_state)
        println("Generating label-less initial weight MPS with bond dimension χ = $χ_init using state $random_state...")
    else
        println("Generating label-less intiial weight MPS with bond dimension χ = $χ_init.")
    end

    W = randomMPS(ComplexF64, site_indices; linkdims = χ_init)

    #l_idx = Index(num_classes, "f(x)")
    #l_tensor = randomITensor(ComplexF64, l_idx)

    #W[1] *= l_tensor
    
    # normalize and orthogonalise the MPS
    normalize!(W)
    # bring into right canonical form for more efficient computations, some overhead but whatever
    orthogonalize!(W, 1)

    return W

end

function feature_map(x::Float64)
    return [exp(1im * (3π/2) * x) * cospi(0.5 * x), exp(-1im * (3π/2) * x) * sinpi(0.5 * x)]
end

function sample_to_product_state(ts::Vector, site_inds::Vector{Index{Int64}})

    product_state = MPS([ITensor(feature_map(ts[i]), site_inds[i]) for i in eachindex(site_inds)])

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

function construct_caches(mps::MPS, training_product_states::Vector{PState}; going_left = true)

    n_train = length(training_product_states)
    n = length(mps)

    # pre-fill with empty ITensors which will be overwritten
    LE = fill(ITensor(), n_train, n)
    RE = fill(ITensor(), n_train, n)

    for i in 1:n_train 
        # get the product state for the current training sample
        ps = training_product_states[i].pstate

        if going_left
            # initialise the first contraction
            LE[i, 1] = mps[1] * conj(ps[1])
            for j in 2:n
                LE[i, j] = LE[i, j-1] * conj(ps[j]) * mps[j]
            end
            
        else
            # going right
            RE[i, n] = conj(ps[n]) * mps[n]
            # accumulate remaining sites
            for j in n-1:-1:1
                RE[i, j] = RE[i, j+1] * conj(ps[j]) * mps[j]
            end
        end
    end

    return LE, RE

end

function compute_loss_per_sample_mps(mps::MPS, product_state::PState)
    # loss without label index
    ground_truth_label = product_state.label # this should just be all ones.
    ps = product_state.pstate
    yhat = 1
    for i in eachindex(mps)
        yhat *= conj(ps[i]) * mps[i]
    end
    # yhat should just be a scalar now
    y = ground_truth_label
    f_ln = yhat * y
    p = abs2.(f_ln[])
    loss = -log(p)

    return loss

end

function compute_loss_for_entire_mps(W::MPS, product_states::Vector{PState})
    # compute the loss for a given MPS

    loss_total = Folds.reduce(+, compute_loss_per_sample_mps(W, ps) for ps in product_states)
    loss_final = loss_total / length(product_states)

    return loss_final 
end

function compute_loss_per_sample(bt::ITensor, LE::Matrix, RE::Matrix, product_state::PState,
    lid::Int, rid::Int)

    ground_truth_label = product_state.label
    ps = product_state.pstate
    ps_id = product_state.id
    phi_tilde = conj(ps[lid]) * conj(ps[rid])

    if lid == 1
        # first site
        phi_tilde *= RE[ps_id, (rid+1)]
    elseif rid == length(ps)
        phi_tilde *= LE[ps_id, (lid-1)]
    else
        phi_tilde *= LE[ps_id, (lid-1)] * RE[ps_id, (rid+1)]
    end

    yhat = bt * phi_tilde
    y = ground_truth_label
    f_ln = first(yhat * y)
    p = abs2.(f_ln[])
    loss = -log(p)

    return loss

end

function compute_loss_per_batch(bt::ITensor, LE::Matrix, RE::Matrix, 
    pss::Vector{PState}, lid::Int, rid::Int)

    loss_total = Folds.reduce(+, compute_loss_per_sample(bt, LE, RE, ps, lid, rid) for ps in pss)
    loss_final = loss_total / length(pss)

    return loss_final

end

function analytic_gradient_per_sample(bt::ITensor, LE::Matrix, RE::Matrix, 
    product_state::PState, lid::Int, rid::Int)

    ground_truth_label = product_state.label
    ps = product_state.pstate
    ps_id = product_state.id
    phi_tilde = conj(ps[lid]) * conj(ps[rid])

    if lid == 1
        # first site
        phi_tilde *= RE[ps_id, (rid+1)]
    elseif rid == length(ps)
        phi_tilde *= LE[ps_id, (lid-1)]
    else
        phi_tilde *= LE[ps_id, (lid-1)] * RE[ps_id, (rid+1)]
    end

    yhat = bt * phi_tilde
    y = ground_truth_label
    f_ln = yhat * y
    gradient = - y * conj(phi_tilde / f_ln[])
    
    return gradient

end

function analytic_gradient_per_batch(bt::ITensor, LE::Matrix, RE::Matrix,
    pss::Vector{PState}, lid::Int, rid::Int)

    total_grad = Folds.reduce(+, analytic_gradient_per_sample(bt, LE, RE, ps, lid, rid) for ps in pss)
    final_grad = total_grad ./ length(pss)

    return final_grad

end

function zygote_gradient_per_sample(bt::ITensor, LE::Matrix, RE::Matrix,
    product_state::PState, lid::Int, rid::Int)

    l = x -> compute_loss_per_sample(x, LE, RE, product_state, lid, rid)
    g, = gradient(l, bt)

    return g

end

function steepest_descent(bt_init::ITensor, LE::Matrix, RE::Matrix, lid::Int, rid::Int,
    pss::Vector{PState}; num_iters = 2, alpha = 0.01, track_cost = false)

    # apply vanilla gradient descent to the bond tensor
    bt_old = bt_init
    for i in 1:num_iters
        # get the gradient
        g = analytic_gradient_per_batch(bt_old, LE, RE, pss, lid, rid)
        #println(g)
        # update the bond tensor
        bt_new = bt_old - alpha * g
        if track_cost
            # get the new loss
            new_loss = compute_loss_per_batch(bt_new, LE, RE, pss, lid, rid)
            println("Loss at step $i: $new_loss")
        end

        bt_old = bt_new

    end

    # rescale
    normalize!(bt_old)

    return bt_old

end

function decompose_bond_tensor(bt::ITensor, lid::Int; chi_max=nothing, going_left=true)
    """Decompose an updated bond tensor back into two tensors using SVD"""
    left_site_index = findindex(bt, "n=$lid")
    #label_index = findindex(bt, "f(x)")
    if going_left
         # need to make sure the label index is transferred to the next site to be updated
         if lid == 1
            U, S, V = svd(bt, (left_site_index); maxdim=chi_max)
        else
            bond_index = findindex(bt, "Link,l=$(lid-1)")
            U, S, V = svd(bt, (left_site_index, bond_index); maxdim=chi_max)
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
            U, S, V = svd(bt, (left_site_index); maxdim=chi_max)
        else
            bond_index = findindex(bt, "Link,l=$(lid-1)")
            U, S, V = svd(bt, (left_site_index, bond_index); maxdim=chi_max)
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

    return LE, RE

end

function do_sweep(X_train::Matrix, y_train::Vector, X_test::Matrix, y_test::Vector,
    random_state::Int = 12; num_grad_iters::Int = 1,
    num_sweeps::Int = 100, chi_max::Int = 15, alpha=0.5)

    @assert !any(i -> i .> 1.0, X_train) & !any(i -> i .< 0.0, X_train) "X_train contains values oustide the expected range [0,1]."
    @assert !any(i -> i .> 1.0, X_test) & !any(i -> i .< 0.0, X_test) "X_test contains values oustide the expected range [0,1]."

    # make the site indices
    s = siteinds("S=1/2", size(X_train, 2))
    # make the mps - already orthogonalised and normalised
    mps = generate_startingMPS(4, s; random_state=random_state)

    training_pstates = dataset_to_product_state(X_train, y_train, s)
    testing_pstates = dataset_to_product_state(X_test, y_test, s)

    LE, RE = construct_caches(mps, training_pstates; going_left=false)
    initial_train_loss = compute_loss_for_entire_mps(mps, training_pstates)
    initial_test_loss = compute_loss_for_entire_mps(mps, testing_pstates)

    println("Initial train loss: $initial_train_loss")
    println("Initial test loss: $initial_test_loss")

    train_loss_per_sweep = [initial_train_loss]
    test_loss_per_sweep = [initial_test_loss]

    for sweep in 1:num_sweeps

        for i in 1:length(mps) - 1

            println("Bond: $i")
            bt = mps[i] * mps[(i+1)]
            bt_new = steepest_descent(bt, LE, RE, (i), (i+1), training_pstates; num_iters=num_grad_iters, alpha=alpha, track_cost=true)
            left_site_new, right_site_new = decompose_bond_tensor(bt_new, (i); chi_max = chi_max, going_left = false)
            LE, RE = update_caches(left_site_new, right_site_new, LE, RE, (i), (i+1), training_pstates; going_left=false)
            mps[i] = left_site_new
            mps[(i+1)] = right_site_new
        end

        println("Finished Forward Pass")
       

        # reset caches after half sweep
        LE, RE = construct_caches(mps, training_pstates; going_left=true)

        for i = (length(mps) - 1):-1:1

            println("Bond: $i")
            bt = mps[i] * mps[(i+1)]
            bt_new = steepest_descent(bt, LE, RE, (i), (i+1), training_pstates; num_iters=num_grad_iters, alpha=alpha, track_cost=true)
            left_site_new, right_site_new = decompose_bond_tensor(bt_new, (i); chi_max = chi_max, going_left = true)
            LE, RE = update_caches(left_site_new, right_site_new, LE, RE, (i), (i+1), training_pstates; going_left=true)
            mps[i] = left_site_new
            mps[(i+1)] = right_site_new

        end

        LE, RE = construct_caches(mps, training_pstates; going_left=false)
        
        train_loss_sweep = compute_loss_for_entire_mps(mps, training_pstates)
        test_loss_sweep = compute_loss_for_entire_mps(mps, testing_pstates)

        push!(train_loss_per_sweep, train_loss_sweep)
        push!(test_loss_per_sweep, test_loss_sweep)
    
        println("Sweep $sweep finished. MPS Norm: $(norm(mps)).")
        println("Train Loss: $train_loss_sweep")
        println("Test Loss: $test_loss_sweep")

    end

    return mps, train_loss_per_sweep, test_loss_per_sweep

end


function train_mps(seed::Int=42, chi_max::Int=15, alpha=0.5, nsweeps=20)

    train_loc = "Sampling/datasets/electricity_hourly/single_test/electricitySingle_train.jld2"
    test_loc = "Sampling/datasets/electricity_hourly/single_test/electricitySingle_test.jld2"
    (X_train, y_train), (X_test, y_test) = load_jld2_dset(train_loc, test_loc)

    # rescale data using RobustSigmoid transform
    scaler = RobustSigmoid(X_train)
    X_train_scaled = scaler(X_train)
    X_test_scaled = scaler(X_test)

    println("Using parameters...")
    println("Initial seed: $seed")
    println("chi max: $chi_max")
    println("alpha: $alpha")
    println("num sweeps: $nsweeps")

    mps, train_loss_per_sweep, test_loss_per_sweep = do_sweep(X_train_scaled, y_train, X_test_scaled, 
        y_test, seed, num_sweeps=nsweeps, chi_max=chi_max, alpha=alpha)

    info = Dict(
        "mps" => mps, 
        "train_loss_per_sweep" => train_loss_per_sweep,
        "test_loss_per_sweep" => test_loss_per_sweep,
        "X_train_scaled" => X_train_scaled,
        "y_train" => y_train,
        "X_test_scaled" => X_test_scaled,
        "y_test" => y_test
    ) 

    return info

end





