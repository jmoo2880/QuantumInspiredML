# no OptimKit, just regular old gradient descent
# going to keep everything on the same script for now
#using Zygote
using Random
using ITensors
using DelimitedFiles
using Plots
using Plots.PlotMeasures
using Folds
using Normalization
using Distributions

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

function load_ecg2000(train_set_location::String, test_set_location::String)
    """As per typical UCR formatting, assume labels in first column, followed by data"""
    # do checks
    train_data = readdlm(train_set_location)
    test_data = readdlm(test_set_location)

    X_train = train_data[:, 2:end]
    y_train = Int.(train_data[:, 1])

    X_test = test_data[:, 2:end]
    y_test = Int.(test_data[:, 1])

    return (X_train, y_train), (X_test, y_test)

end

function label_to_tensor(label, l_idx)
    tensor = onehot(l_idx => label + 1)
    return tensor
end

function generate_startingMPS(χ_init::Int, site_indices::Vector{Index{Int64}}; 
    num_classes = 2, random_state = nothing)

    if random_state !== nothing
        Random.seed!(random_state)
        println("Generating initial weight MPS with bond dimension χ = $χ_init using state $random_state...")
    else
        println("Generating intiial weight MPS with bond dimension χ = $χ_init.")
    end

    W = randomMPS(ComplexF64, site_indices; linkdims = χ_init)

    l_idx = Index(num_classes, "f(x)")
    l_tensor = randomITensor(ComplexF64, l_idx)

    W[1] *= l_tensor
    
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
    # more general function to compute loss for entire MPS
    # exatract the label
    ground_truth_label = product_state.label 
    ps = product_state.pstate
    yhat = 1 # this is the proba amplitude
    for i in eachindex(mps)
        yhat *= conj(ps[i]) * mps[i]
    end
    # find the label index
    label_idx = first(inds(yhat))
    y = onehot(label_idx => (ground_truth_label + 1))
    f_ln = first(yhat * y)
    #orthogonalize!(mps, 1)
    #Z = conj(mps[1]) * mps[1]
    p = abs2.(f_ln) #/ abs(Z[])
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

    label_idx = first(inds(yhat))
    y = onehot(label_idx => (ground_truth_label + 1))
    f_ln = first(yhat * y)
    #orthogonalize!(mps, 1)
    #Z = conj(mps[1]) * mps[1]
    p = abs2.(f_ln) #/ abs(Z[])
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
    label_idx = first(inds(yhat))
    y = onehot(label_idx => (ground_truth_label + 1))
    f_ln = first(yhat * y)
    
    gradient = - y * conj(phi_tilde / f_ln)
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

function zygote_gradient_per_batch(bt::ITensor, LE::Matrix, RE::Matrix,
    pss::Vector{PState}, lid::Int, rid::Int)

    total_grad = Folds.reduce(+, zygote_gradient_per_sample(bt, LE, RE, ps, lid, rid) for ps in pss)
    final_grad = total_grad ./ length(pss)

    return final_grad

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

function compute_raw_overlap(mps::MPS, product_state::PState)
    if !isapprox(norm(mps), 1.0) @warn "MPS is not normalised" end
    ps = product_state.pstate
    res = 1
    for i in eachindex(mps)
        res *= mps[i] * conj(ps[i])
    end
    
    return res

end

function compute_model_probabilities(yhat)
    # converts yhat to a probability distribution over classes 
    # using the "quantum normalisation"
    ps = real(abs2.(yhat)) #/ abs(Z[])
    return ps

end

function get_prediction_single(mps::MPS, product_state::PState)
    # get prediction for a given sample and return probability
    yhat = 1
    ps = product_state.pstate
    for i in eachindex(mps)
        yhat *= mps[i] * conj(ps[i])
    end
    #println(yhat)
    probas = compute_model_probabilities(yhat)
    probas_v = vector(probas)
    prediction = argmax(probas_v) - 1 # convert index back to label in one-hot encoding scheme

    return prediction, maximum(probas)

end

function get_accuracy_dataset(mps::MPS, product_states::Vector{PState})
    # compute the accuracy for an entire dataset
    if !isapprox(norm(mps), 1.0) @warn "WARNING, MPS IS NOT NORMALISED! : norm = $(norm(mps))" end
    # loop through entire dataset and get the accuracies
    # not efficient, I know. 
    correct_count = 0
    all_probas = Vector{Float64}(undef, length(product_states))
    for (index, ps) in enumerate(product_states)
        gt_label = ps.label
        prediction, proba = get_prediction_single(mps, ps)
        if gt_label == prediction
            correct_count += 1
        end
        all_probas[index] = proba
    end

    final_acc = correct_count / length(product_states)

    return final_acc, all_probas

end

function do_sweep(X_train::Matrix, y_train::Vector, X_test::Matrix, y_test::Vector,
    random_state::Int = 12; num_grad_iters::Int = 1,
     num_sweeps::Int = 100, chi_max::Int = 15, alpha=0.5)

    @assert !any(i -> i .> 1.0, X_train) & !any(i -> i .< 0.0, X_train) "X_train contains values oustide the expected range [0,1]."
    @assert !any(i -> i .> 1.0, X_test) & !any(i -> i .< 0.0, X_test) "X_test contains values oustide the expected range [0,1]."

    num_classes = length(unique(y_train)) # infer num classes from y_train
    # make the site indices
    s = siteinds("S=1/2", size(X_train, 2))
    # make the mps - already orthogonalised and normalised
    mps = generate_startingMPS(4, s; random_state=random_state, num_classes=num_classes)

    training_pstates = dataset_to_product_state(X_train, y_train, s)
    testing_pstates = dataset_to_product_state(X_test, y_test, s)

    LE, RE = construct_caches(mps, training_pstates; going_left=false)
    initial_train_loss = compute_loss_for_entire_mps(mps, training_pstates)
    initial_test_loss = compute_loss_for_entire_mps(mps, testing_pstates)
    initial_test_acc, _ = get_accuracy_dataset(mps, testing_pstates)
    initial_train_acc, _ = get_accuracy_dataset(mps, training_pstates)

    println("Initial train loss: $initial_train_loss")
    println("Initial test loss: $initial_test_loss")
    println("Initial test acc: $initial_test_acc")
    println("Initial train acc: $initial_train_acc")

    train_loss_per_sweep = [initial_train_loss]
    test_loss_per_sweep = [initial_test_loss]
    test_acc_per_sweep = [initial_test_acc]
    train_acc_per_sweep = [initial_train_acc]

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
        test_acc_sweep, _ = get_accuracy_dataset(mps, testing_pstates)
        train_acc_sweep, _ = get_accuracy_dataset(mps, training_pstates)

        push!(train_loss_per_sweep, train_loss_sweep)
        push!(test_loss_per_sweep, test_loss_sweep)
        push!(test_acc_per_sweep, test_acc_sweep)
        push!(train_acc_per_sweep, train_acc_sweep)

        println("Sweep $sweep finished. MPS Norm: $(norm(mps)).")
        println("Train Loss: $train_loss_sweep")
        println("Test Loss: $test_loss_sweep")
        println("Test acc: $test_acc_sweep")
        println("Train acc: $train_acc_sweep")

    end


    return mps, train_loss_per_sweep, test_loss_per_sweep, training_pstates, testing_pstates, test_acc_per_sweep, train_acc_per_sweep

end


function generate_toy_timeseries(time_series_length::Int, total_dataset_size::Int, 
    train_split=0.7; random_state=1234, plot_examples=false)
    """Generate two sinusoids of different frequency, and with randomised phase.
    Inject noise with a given amplitude."""
    Random.seed!(random_state)

    train_size = floor(Int, total_dataset_size * train_split)
    test_size = total_dataset_size - train_size

    X_train = zeros(Float64, train_size, time_series_length)
    y_train = zeros(Int, train_size)
    
    X_test = zeros(Float64, test_size, time_series_length)
    y_test = zeros(Int, test_size)

    function generate_sinusoid(length::Int, A::Float64=1.0, 
        f::Float64=1.0, sigma=0.2)
        # sigma is scale of the gaussian noise added to the sinusoid
        t = range(0, 2π, length)
        phase = rand(Uniform(0, 2π)) # randomise the phase

        return A .* sin.(f .*t .+ phase) .+ sigma .* randn(length)

    end

    # generation parameters
    A1, f1, sigma1 = 1.0, 2.0, 0.1 # Class 0
    A2, f2, sigma2 = 1.0, 8.5, 0.1 # Class 1

    for i in 1:train_size
        label = rand(0:1) # choose a label, if 0 use freq f0, if 1 use freq f1. 
        data = label == 0 ? generate_sinusoid(time_series_length, A1, f1, sigma1) : 
            generate_sinusoid(time_series_length, A2, f2, sigma2)
        X_train[i, :] = data
        y_train[i] = label
    end

    for i in 1:test_size
        label = rand(0:1) # choose a label, if 0 use freq f0, if 1 use freq f1. 
        data = label == 0 ? generate_sinusoid(time_series_length, A1, f1, sigma1) : 
            generate_sinusoid(time_series_length, A2, f2, sigma2)
        X_test[i, :] = data
        y_test[i] = label
    end

    # plot some examples
    if plot_examples
        class_0_idxs = findall(x -> x.== 0, y_train)[1:2] # select subset of 5 samples
        class_1_idxs = findall(x -> x.== 1, y_train)[1:2]
        p0 = plot(X_train[class_0_idxs, :]', xlabel="Time", ylabel="x", title="Class 0 Samples (Unscaled)", 
            alpha=0.4, c=:red, label="")
        p1 = plot(X_train[class_1_idxs, :]', xlabel="Time", ylabel="x", title="Class 1 Samples (Unscaled)", 
            alpha=0.4, c=:magenta, label="")
        p = plot(p0, p1, size=(1200, 500), bottom_margin=5mm, left_margin=5mm)
        display(p)
    end

    return (X_train, y_train), (X_test, y_test)

end

function slice_mps_into_label_states(mps::MPS)
    """Gets the label index of the MPS and slices according to the number of classes (dim of the label index)"""
    """Assume one-hot encoding scheme i.e. class 0 = [1, 0], class 1 = [0, 1], etc. """
    dec_index = findindex(mps[1], "f(x)")
    if isnothing(dec_index)
        error("Label index not found on the first site of the MPS!")
    end
    # infer num classes from the dimension of the label index
    n_states = ITensors.dim(dec_index)
    states = []

    for i = 1:n_states
        # make a copy of the MPS so we are protected from any unintentional changes
        state = deepcopy(mps)
        if !isapprox(norm(state), 1.0) @warn "WARNING, MPS NOT NORMALISED!" end
        # create a onehot encoded tensor to slice the MPS
        decision_state = onehot(dec_index => (i))
        println("Class $(i-1) state: $(vector(decision_state))")
        # slice the mps along the dimension i by contracting with the label site
        state[1] *= decision_state

        # normalise the label MPS
        normalize!(state)
        push!(states, state)

    end

    return states

end

function train_mps(seed::Int=42, chi_max::Int=50, alpha=0.5, nsweeps=10)

    (X_train, y_train), (X_test, y_test) = generate_toy_timeseries(200, 625, 0.80; plot_examples = true)
    #(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits_txt("MPS_MSE/datasets/ECG_train.txt", "MPS_MSE/datasets/ECG_val.txt", "MPS_MSE/datasets/ECG_test.txt")
    #(X_train, y_train), (X_test, y_test) = load_ecg2000("/Users/joshua/Documents/QuantumInspiredML/LogLossAlternative/datasets/ecg2000_train.txt", "/Users/joshua/Documents/QuantumInspiredML/LogLossAlternative/datasets/ecg2000_test.txt")
    #X_train = vcat(X_train, X_val)
    #y_train = vcat(y_train, y_val)
    # rescale data using RobustSigmoid transform
    scaler = RobustSigmoid(X_train)
    X_train_scaled = scaler(X_train)
    X_test_scaled = scaler(X_test)

    println("Using parameters...")
    println("Initial seed: $seed")
    println("chi max: $chi_max")
    println("alpha: $alpha")
    println("num sweeps: $nsweeps")

    mps, train_loss_per_sweep, test_loss_per_sweep, _, _, test_acc_per_sweep, 
        train_acc_per_sweep = do_sweep(X_train_scaled, y_train, X_test_scaled, y_test, seed, num_sweeps=nsweeps, chi_max=chi_max, alpha=alpha)

    #data = (X_train_scaled, y_train, X_test_scaled, y_test)
    #losses = (train_loss_per_sweep, test_loss_per_sweep)
    #pstates = (training_pstates, testing_pstates)

    return mps, train_loss_per_sweep, test_loss_per_sweep, test_acc_per_sweep, train_acc_per_sweep, 
        X_train_scaled, y_train, X_test_scaled, y_test

end

function save_mps(mps::MPS, path::String; id::String="W")
    """Saves an MPS as a .h5 file"""
    file = path[end-2:end] == ".h5" ? path[1:end-3] : path
    f = h5open("$file.h5", "w")
    write(f, id, mps)
    close(f)
    println("Succesfully saved mps $id at $file.h5")
end
