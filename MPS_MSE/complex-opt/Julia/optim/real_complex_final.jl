using ITensors
using OptimKit
using Folds
using Plots, Plots.PlotMeasures
using Distributions
using Normalization
using MLBase
include("utils.jl")

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

function compute_yhat_and_derivative(BT::ITensor, LE::Matrix, RE::Matrix, product_state::PState,
    lid::Int, rid::Int)
    """takes in a "real valued" Bond Tensor, extracts the real and imag components and then
    reconstructs the bond tensor"""
    # get the C index
    c_index = findinds(BT, "C")[1]
    BT_real = deepcopy(BT) * onehot(c_index => 1) # 1 gets the real component
    BT_img = deepcopy(BT) * onehot(c_index => 2) # 2 gets the imag component
    # reform the bond tensor
    BT = BT_real + im * BT_img
    ps = product_state.pstate
    ps_id = product_state.id

    d_yhat_dW = ps[lid] * ps[rid] # phi tilde 

    if lid == 1
        d_yhat_dW *= RE[ps_id, (rid+1)]
    elseif rid == length(ps)
        d_yhat_dW *= LE[ps_id, (lid-1)]
    else
        d_yhat_dW *= LE[ps_id, (lid-1)] * RE[ps_id, (rid+1)]
    end

    yhat = BT * d_yhat_dW

    return yhat, d_yhat_dW

end

function compute_loss_and_gradient_per_sample(BT::ITensor, LE::Matrix, RE::Matrix, product_state::PState,
    lid::Int, rid::Int)

    yhat, phi_tilde = compute_yhat_and_derivative(BT, LE, RE, product_state, lid, rid)

    y = product_state.label
    dP = yhat[] - y 
    abs_diff_sq = norm(dP)^2
    loss = 0.5 * abs_diff_sq

    grad = dP * conj(phi_tilde)

    return [loss, grad]

end

# accepts bond tensor with complex/real label index, returns the same
function loss_and_grad_batch(BT::ITensor, LE::Matrix, RE::Matrix, pss::Vector{PState}, 
    lid::Int, rid::Int)

    c_index = findinds(BT, "C")[1]

    loss, grad = Folds.reduce(+, compute_loss_and_gradient_per_sample(BT, LE, RE, prod_state, lid, rid) for
       prod_state in pss)

    loss /= length(pss)
    grad ./= length(pss)

    # needs to return real valued 
    grad_real = real(grad)
    grad_imag = imag(grad)

    C_tensor_real = ITensor([1; 0], c_index)
    grad_real *= C_tensor_real

    C_tensor_imag = ITensor([0; 1], c_index)
    grad_imag *= C_tensor_imag

    grad_combined_real_imag = grad_real + grad_imag

    return loss, grad_combined_real_imag

end

function optimise_bond_tensor(BT_init::ITensor, LE::Matrix, RE::Matrix, lid::Int, rid::Int, 
    pss::Vector{PState}, iters=20)

    # break down the bond tensor to feed into optimkit
    C_index = Index(2, "C")
    bt_real = real(BT_init)
    bt_imag = imag(BT_init)

    bt_real_index_tensor = ITensor([1; 0], C_index)
    bt_real *= bt_real_index_tensor
    bt_imag_index_tensor = ITensor([0; 1], C_index)
    bt_imag *= bt_imag_index_tensor

    # combined
    bt_combined_real_imag = bt_real + bt_imag

    lg = x -> loss_and_grad_batch(x, LE, RE, pss, lid, rid)
    alg = ConjugateGradient(; verbosity=1, maxiter=iters)
    new_BT, fx, _ = optimize(lg, bt_combined_real_imag, alg)

    # reform bond tensor to rescale
    bt_updated_real = deepcopy(new_BT) * onehot(C_index => 1) # 1 gets the real component
    bt_updated_imag = deepcopy(new_BT) * onehot(C_index => 2) # 2 gets the imag component
    # reform the bond tensor
    BT_updated = bt_updated_real + im * bt_updated_imag

    return BT_updated

end

function decompose_bond_tensor(BT::ITensor, lid::Int; χ_max=nothing, going_left=true)
    """Decompose an updated bond tensor back into two tensors using SVD"""
    left_site_index = findindex(BT, "n=$lid")
    #label_index = findindex(BT, "f(x)")
    if going_left
         # need to make sure the label index is transferred to the next site to be updated
         if lid == 1
            U, S, V = svd(BT, (left_site_index); maxdim=χ_max)
        else
            bond_index = findindex(BT, "Link,l=$(lid-1)")
            U, S, V = svd(BT, (left_site_index, bond_index); maxdim=χ_max)
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
            U, S, V = svd(BT, (left_site_index); maxdim=χ_max)
        else
            bond_index = findindex(BT, "Link,l=$(lid-1)")
            U, S, V = svd(BT, (left_site_index, bond_index); maxdim=χ_max)
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

function loss_per_sample(mps::MPS, product_state::PState)
    ps = product_state.pstate
    y = product_state.label # ground truth label

    yhat = 1
    num_sites = length(mps)
    for site = 1:num_sites
        yhat *= mps[site] * ps[site]
    end

    abs_diff_sq = norm(yhat[] - y)^2
    sample_loss = 0.5 * abs_diff_sq

    return sample_loss

end

function loss_batch(mps::MPS, pss::Vector{PState})
    """Compute the loss for entire batch"""
    if !isapprox(norm(mps), 1.0; atol=1e-2) @warn "MPS is un-normalised!" end

    loss_total = Folds.reduce(+, loss_per_sample(mps, ps) for ps in pss)
    batch_loss = loss_total / length(pss)
    
    return batch_loss

end

function generate_toy_timeseries(time_series_length::Int, total_dataset_size::Int, 
    train_split=0.9; random_state=1234, plot_examples=false)
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
        omega::Float64=1.0, sigma=0.2)
        # sigma is scale of the gaussian noise added to the sinusoid
        t = range(0, 2π, length)
        phi = rand(Uniform(0, 2π)) # randomise the phase

        return A .* sin.(omega .*t .+ phi) .+ sigma .* randn(length)

    end

    # generation parameters
    A1, omega1, sigma1 = 1.0, 3.0, 0.05 # Class 0
    A2, omega2, sigma2 = 1.0, 6.0, 0.05 # Class 1

    for i in 1:train_size
        label = rand(0:1)  
        data = label == 0 ? generate_sinusoid(time_series_length, A1, omega1, sigma1) : 
            generate_sinusoid(time_series_length, A2, omega2, sigma2)
        X_train[i, :] = data
        y_train[i] = label
    end

    for i in 1:test_size
        label = rand(0:1)
        data = label == 0 ? generate_sinusoid(time_series_length, A1, omega1, sigma1) : 
            generate_sinusoid(time_series_length, A2, omega2, sigma2)
        X_test[i, :] = data
        y_test[i] = label
    end

    # plot some examples
    if plot_examples
        class_0_idxs = findall(x -> x.== 0, y_train)[1:5] # select subset of 5 samples
        class_1_idxs = findall(x -> x.== 1, y_train)[1:5]
        p0 = plot(X_train[class_0_idxs, :]', xlabel="Time", ylabel="x", title="Class 0 Samples (Unscaled)", 
            alpha=0.4, c=:red, label="")
        p1 = plot(X_train[class_1_idxs, :]', xlabel="Time", ylabel="x", title="Class 1 Samples (Unscaled)", 
            alpha=0.4, c=:magenta, label="")
        p = plot(p0, p1, size=(1200, 500), bottom_margin=5mm, left_margin=5mm)
        display(p)
    end

    return (X_train, y_train), (X_test, y_test)

end

function get_predictions(mps0::MPS, mps1::MPS, pss::Vector{PState})
    # mps0 overlaps with ORIGINAL class 0 and mps1 overlaps with ORIGINAL class 1
    @assert length(mps0) == length(mps1) "MPS lengths do not match!"

    preds = Vector{Int64}(undef, length(pss))
    all_overlaps_mps0 = Vector{Float64}(undef, length(pss))
    all_overlaps_mps1 = Vector{Float64}(undef, length(pss))
    for i in eachindex(pss)
        ps = pss[i].pstate
        overlap_mps0 = 1
        overlap_mps1 = 1
        for j in eachindex(mps0)
            overlap_mps0 *= mps0[j] * ps[j]
            overlap_mps1 *= mps1[j] * ps[j]
        end
        overlap_mps0 = abs(overlap_mps0[])
        overlap_mps1 = abs(overlap_mps1[])
        pred = 0
        if overlap_mps1 > overlap_mps0
            pred = 1
        end
        all_overlaps_mps0[i] = overlap_mps0
        all_overlaps_mps1[i] = overlap_mps1
        preds[i] = pred
    end

    # return overlaps as well for inspection
    return preds, all_overlaps_mps0, all_overlaps_mps1
        
end

function get_overlap(mps::MPS, product_state::PState)
    """Get the overlap of a single product state with a single mps"""
    res = 1
    ps = product_state.pstate
    for i in eachindex(mps)
        res *= mps[i] * ps[i]
    end
    res = abs(res[])

    return res

end

function overlap_confmat(mps0::MPS, mps1::MPS, pstates::Vector{PState}; plot=false)
    """Something like a confusion matrix but for median overlaps.
    Here, mps0 is the mps which overlaps with class 0 and mps1 overlaps w/ class 1"""
    gt_class_0_idxs = [ps.label .== 0 for ps in pstates]
    gt_class_1_idxs = [ps.label .== 1 for ps in pstates]
    # gt class 0, overlap with mps0, we will call this a true negative
    gt_0_mps_0 = [get_overlap(mps0, ps) for ps in pstates[gt_class_0_idxs]]
    # gt class 0, overlaps with mps1, false positive
    gt_0_mps_1 = [get_overlap(mps1, ps) for ps in pstates[gt_class_0_idxs]]
    # gt class 1, overlap with mps0, false negative
    gt_1_mps_0 = [get_overlap(mps0, ps) for ps in pstates[gt_class_1_idxs]]
    # gt class 1, overlaps with mps1, true positive
    gt_1_mps_1 = [get_overlap(mps1, ps) for ps in pstates[gt_class_1_idxs]]

    # get medians
    gt_0_mps_0_median = median(gt_0_mps_0)
    gt_0_mps_1_median = median(gt_0_mps_1)
    gt_1_mps_0_median = median(gt_1_mps_0)
    gt_1_mps_1_median = median(gt_1_mps_1)
    confmat = [gt_0_mps_0_median gt_0_mps_1_median; gt_1_mps_0_median gt_1_mps_1_median]

    # dictionary of stats
    #⟨ps|mps⟩
    stats = Dict(
        "Min/Max ⟨0|0⟩" => (minimum(gt_0_mps_0), maximum(gt_0_mps_0)),
        "Min/Max ⟨1|0⟩" => (minimum(gt_1_mps_0), maximum(gt_1_mps_0)),
        "Min/Max ⟨0|1⟩" => (minimum(gt_0_mps_1), maximum(gt_0_mps_1)),
        "Min/Max ⟨1|1⟩" => (minimum(gt_1_mps_1), maximum(gt_1_mps_1)),
        "MPS States Overlap ⟨1|0⟩" => abs(inner(mps0, mps1))
    )

    if plot
        reversed_confmat = reverse(confmat, dims=1)
        hmap = heatmap(reversed_confmat,
        color=:Blues,
        xticks=(1:size(confmat,2), ["Predicted 0", "Predicted 1"]),
        yticks=(1:size(confmat,1), ["Actual 1", "Actual 0"]),
        xlabel="Predicted class",
        ylabel="Actual class",
        title="Median Overlap Confusion Matrix")

        for (i, row) in enumerate(eachrow(reversed_confmat))
            for (j, value) in enumerate(row)
                
                annotate!(j, i, text(string(value), :center, 10))
            end
        end

        display(hmap)

    end

    return confmat, stats

end

function plot_conf_mat(confmat::Matrix)
    reversed_confmat = reverse(confmat, dims=1)
    hmap = heatmap(reversed_confmat,
        color=:Blues,
        xticks=(1:size(confmat,2), ["Predicted 0", "Predicted 1"]),
        yticks=(1:size(confmat,1), ["Actual 1", "Actual 0"]),
        xlabel="Predicted class",
        ylabel="Actual class",
        title="Confusion Matrix")
        
    for (i, row) in enumerate(eachrow(reversed_confmat))
        for (j, value) in enumerate(row)
            
            annotate!(j, i, text(string(value), :center, 10))
        end
    end

    display(hmap)
end

function get_training_summary(mps0::MPS, mps1::MPS, training_pss::Vector{PState}, testing_pss::Vector{PState})
    # get final traing acc, final training loss
    preds_training, overlaps_mps0_training, overlaps_mps1_training = get_predictions(mps0, mps1, training_pss)
    true_training = [x.label for x in training_pss] # get ground truths
    acc_training = sum(true_training .== preds_training)/length(training_pss)
    println("Training Accuracy: $acc_training")

    # get final testing acc
    preds_testing, overlaps_mps0_testing, overlaps_mps1_testing = get_predictions(mps0, mps1, testing_pss)
    true_testing =  [x.label for x in testing_pss] # get ground truths

    # get overlap between class 0 mps and class 1 mps
    overlap_mps_states = abs(inner(mps0, mps1))
    println("Overlap between state 0 MPS and State 1 MPS ⟨ψ0|ψ1⟩ = $overlap_mps_states")

    # TP, TN, FP, FN FOR TEST SET 
    acc_testing = sum(true_testing .== preds_testing)/length(testing_pss)
    println("Testing Accuracy: $acc_testing")
    r = roc(true_testing, preds_testing)
    prec = precision(r)
    println("Precision: $prec")
    rec = recall(r)
    println("Recall: $rec")
    f1 = f1score(r)
    println("F1 Score: $f1")
    specificity = true_negative(r) / (true_negative(r) + false_positive(r))
    println("Specificity: $specificity")
    sensitivity = true_positive(r) / (true_positive(r) + false_negative(r))
    println("Sensitivity: $sensitivity")
    # balanced acc is arithmetic mean of sensitivy and specificity
    acc_balanced_testing = (sensitivity + specificity) / 2
    confmat = confusmat(2, (true_testing .+ 1), (preds_testing .+ 1)) # need to offset labels becuase function expects labels to start at 1
    println("Confusion Matrix: $confmat")
    # NOTE CONFMAT IS R(i, j) == countnz((gt .== i) & (pred .== j)). So rows (i) are groudn truth and columns (j) are preds

    stats = Dict(
        :train_acc => acc_training,
        :test_acc => acc_testing,
        :test_balanced_acc => acc_balanced_testing,
        :precision => prec,
        :recall => rec,
        :specificity => specificity,
        :f1_score => f1,
        :confmat => confmat
    )

    return stats

end

# train only a single mps to overlap with label 1 class...
# invert labels to overlaps with original class 0
function sweep(X_train::Matrix, y_train::Vector, site_inds::Vector{Index{Int64}}, num_mps_sweeps::Int, χ_max::Int, 
    random_state::Int=1234, num_cgrad_iters::Int=10)

    @assert !any(i -> i .> 1.0, X_train) & !any(i -> i .< 0.0, X_train) "X_train contains values oustide the expected range [0,1]."

    Random.seed!(random_state)
    mps = randomMPS(ComplexF64, site_inds; linkdims=4)
    # orthogonalize mps - make right orthogonal
    orthogonalize!(mps, 1)

    training_pstates = dataset_to_product_state(X_train, y_train, site_inds)
    LE, RE = construct_caches(mps, training_pstates; going_left=false)
    inital_loss = loss_batch(mps, training_pstates)

    loss_per_sweep = [inital_loss]
    norm_per_pass = [norm(mps)]

    for sweep in 1:num_mps_sweeps

        for i = 1:length(mps) - 1

            BT = mps[i] * mps[(i+1)]

            if norm(BT) > 1
                normalize!(BT)
            end

            BT_new = optimise_bond_tensor(BT, LE, RE, (i), (i+1), training_pstates, num_cgrad_iters)
            left_site_new, right_site_new = decompose_bond_tensor(BT_new, (i); χ_max=χ_max, going_left=false)
            LE, RE = update_caches(left_site_new, right_site_new, LE, RE, (i), (i+1), training_pstates; going_left=false)
            mps[i] = left_site_new
            mps[(i+1)] = right_site_new

        end
        println("Finished Forward Pass")

        push!(norm_per_pass, norm(mps))

        # reset caches after half sweep
        LE, RE = construct_caches(mps, training_pstates; going_left=true)

        for i = (length(mps)-1):-1:1

            BT = mps[i] * mps[(i+1)]

            if norm(BT) > 1
                normalize!(BT)
            end

            BT_new = optimise_bond_tensor(BT, LE, RE, (i), (i+1), training_pstates, num_cgrad_iters)
            left_site_new, right_site_new = decompose_bond_tensor(BT_new, (i); χ_max=χ_max, going_left=true)
            LE, RE = update_caches(left_site_new, right_site_new, LE, RE, (i), (i+1), training_pstates; going_left=true)
            mps[i] = left_site_new
            mps[(i+1)] = right_site_new

        end

        push!(norm_per_pass, norm(mps))
    

        LE, RE = construct_caches(mps, training_pstates; going_left=false)
        # compute new loss
        loss_sweep = loss_batch(mps, training_pstates)
        push!(loss_per_sweep, loss_sweep)

        println("Sweep $sweep finished. Loss: $loss_sweep. MPS Norm: $(norm(mps)).")

    end

    return mps, loss_per_sweep, norm_per_pass, training_pstates

end

function train_mps()
    """Rescales training data and calls the sweep function for each class/label MPS"""
    # generate global set of site indices
    seed = 1234
    num_sites = 96
    site_indices = siteinds("S=1/2", num_sites)
    # generate dataset
    #(X_train, y_train), (X_test, y_test) = generate_toy_timeseries(num_sites, 10_000; plot_examples=true)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = LoadSplitsFromTextFile("MPS_MSE/datasets/ECG_train.txt", "MPS_MSE/datasets/ECG_val.txt", "MPS_MSE/datasets/ECG_test.txt")
    #do RobustSigmoid re-scaling
    X_train = vcat(X_train, X_val)
    y_train = vcat(y_train, y_val)


    N = RobustSigmoid(X_train)
    X_train_scaled = N(X_train)
    X_test_scaled = N(X_test)
    #scaler = fitScaler(RobustSigmoidTransform, X_train; positive=true)
    #X_train_scaled = transformData(scaler, X_train)
    #X_test_scaled = transformData(scaler, X_test)
    # sweep
    num_mps_sweeps = 10
    chi_max = 50
    mps1, _, norm_per_pass_mps1, training_pstates1 = sweep(X_train_scaled, y_train, site_indices, num_mps_sweeps, chi_max, seed)

    # remap labels by inverting, train second mps
    invert_classes = Dict(0 => 1, 1 => 0)
    y_train_inverted = [invert_classes[label] for label in y_train]
    # bit confusing with the variable names but mps1 means overlaps with original class 1 labels and mps0 means overlaps with ORIGINAL class 0 labels
    mps0, _, norm_per_pass_mps0, _ = sweep(X_train_scaled, y_train_inverted, site_indices, num_mps_sweeps, chi_max, seed)

    #normalize!(mps0)
    #normalize!(mps1)

    # return test data as well
    testing_pstates = dataset_to_product_state(X_test_scaled, y_test, site_indices)

    # get stats 
    #summary = get_training_summary(mps0, mps1, training_pstates1, testing_pstates)

    return mps0, mps1, X_train_scaled, y_train, X_test_scaled, y_test, testing_pstates, training_pstates1, norm_per_pass_mps0, norm_per_pass_mps1

end

