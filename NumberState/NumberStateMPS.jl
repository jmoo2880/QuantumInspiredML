using ITensors
using Random
using StatsBase
using PyCall
using Plots
using Base.Threads
using DelimitedFiles
pyts = pyimport("pyts.approximation")

struct PState
    """Create a custom structure to store product state objects, 
    along with their associated label and type (i.e, train, test or valid)"""
    pstate::MPS
    label::Int
    type::String
end

function ZScoredTimeSeriesToSAX(time_series::Matrix; n_bins::Int=3, strategy="normal")
    """Function to convert Z-SCORED time series data to a SAX representation.
    Calls on the SAX library in python using pycall."""

    if strategy !== "normal" && strategy !== "quantile" && strategy !== "uniform"
        error("Invalid SAX strategy. Can be either: `normal', `quantil' or `uniform'.")
    end

    # fit the SAX 'model'
    sax = pyts.SymbolicAggregateApproximation(n_bins=n_bins, strategy=strategy)
    sax_fit = sax.fit(time_series)
    X_sax = sax_fit.transform(time_series)

    # return both the model and the transformed data (as pyobject)
    return X_sax, sax_fit

end

function SAXEncodingToProductState(sax_sample, sites::Vector{Index{Int64}}, 
    sax_encoder::PyObject)
    """Function to map the SAX encodings to a product state |ϕ> where the local
    dimension is determined by the alphabet size for SAX."""

    # check that the local Hilbert space dimension and alphabet size match
    if sax_encoder.n_bins !== ITensors.dim(sites[1])
        error("Alphabet size ($(sax_encoder.n_bins)) not equal to the local Hilbert space 
        dimension ($(ITensors.dim(sites[1])))")
    end

    # check the number of site indices and the length of the SAX-encoded sample match.
    if length(sax_sample) !== length(sites)
        error("Length of the SAX-encoded sample ($(length(sax_sample))) does not match
        the number of sites specified by the site indices ($(length(sites)))")
    end

    # dynamically allocate mappings based on the alphabet size
    alphabet_size = sax_encoder.n_bins
    alphabet = 'a' : 'z'

    # use the mapping conveention where 1 maps to a, b to 2, and so on
    mappings = Dict()
    for (i, letter) in enumerate(alphabet[1:alphabet_size])
        mappings[string(letter)] = i
    end

    # create empty product state container
    ϕ = MPS(sites; linkdims=1)

    # loop through each site and fill tensor with fock state
    for s = 1:length(sites)

        T = ITensor(sites[s])
        letter = sax_sample[s]
        T[mappings[letter]] = 1 # one hot encode, so |a> -> |1> -> [1, 0, 0, ..]
        ϕ[s] = T

    end

    return ϕ

end;

function GenerateAllProductStates(X_SAX, y::Vector, type::String, 
        sites::Vector{Index{Int64}}, sax_encoder::PyObject)
    """Convert an entire datset of SAX_encoded time series to a corresponding dataset 
    of product states.
    E.g. convert n × t dataset of n observations and t samples (timepts) into a length n 
    vector where each entry is a product state of t sites"""

    if type == "train"
        println("Initialising training states.")
    elseif type == "test"
        println("Initialising testing states.")
    elseif type == "valid"
        println("Initialising validation states.")
    else
        error("Invalid dataset type. Must be either train, test or valid!")
    end

    num_samples = length(X_SAX)

    # create a vector to store all product states 
    ϕs = Vector{PState}(undef, num_samples)

    Threads.@threads for samp = 1:num_samples
        sample_pstate = SAXEncodingToProductState(X_SAX[samp], sites, sax_encoder)
        sample_label = y[samp]
        ps = PState(sample_pstate, sample_label, type)
        ϕs[samp] = ps
    end

    return ϕs

end

function GenerateStartingMPS(χ_init, site_inds::Vector{Index{Int64}}; random_state=nothing)
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

    W = randomMPS(site_inds, linkdims=χ_init)

    normalize!(W)

    return W

end

function AttachLabelIndex!(W::MPS, num_classes::Int; attach_site::Int=1)
    """
    Function to attach the decision label index to the un-labelled weight MPS at
    the specified site. Dimension is equal to the number of classes."""
    label_idx = Index(num_classes, "f(x)")

    # get the site of interest and copy over the indices
    old_site_idxs = inds(W[attach_site])
    new_site_idxs = old_site_idxs, label_idx
    new_site = randomITensor(new_site_idxs)

    # add the updated site back into the MPS
    W[attach_site] = new_site

    # normalise the MPS again
    normalize!(W)

end

function ConstructCaches(W::MPS, training_pstates::Vector{PState};
    direction::String="forward")
    """Function to pre-compute tensor contractions between the MPS and the product states. """

    # get the number of training samples to pre-allocate a caching matrix
    N_train = length(training_pstates) 
    # get the number of MPS sites
    N = length(W) 

    # pre-allocate left and right environment matrices 
    LE = Matrix{ITensor}(undef, N_train, N)
    RE = Matrix{ITensor}(undef, N_train, N)

    if direction == "forward"

        # initialise the RE with the terminal site
        for i = 1:N_train
            RE[i, N] = training_pstates[i].pstate[N] * W[N]
        end

        # accumulate all other sites working backwards from the terminal site
        for j = (N-1):-1:1
            for i = 1:N_train
                RE[i, j] = RE[i, j+1] * W[j] * training_pstates[i].pstate[j]
            end
        end

    elseif direction == "backward"

        # initialise the LE with the first site
        for i = 1:N_train
            LE[i, 1] = training_pstates[i].pstate[1] * W[1]
        end

        for j = 2:N
            for i = 1:N_train
                LE[i, j] = LE[i, j-1] * training_pstates[i].pstate[j] * W[j]
            end
        end

    else
        error("Invalid direction. Can either be forward or backward.")
    end

    return LE, RE

end

function ContractMPSAndProductState(W::MPS, ϕ::PState)
    """Fucntion to manually contract the weight MPS with a single 
    product state since ITensor `inner' function doesn't like it 
    when there is a label index attached to an MPS. 

    Returns an ITensor."""

    N_sites = length(W)
    res = 1 # store the cumulative contractions
    for i=1:N_sites
        res *= W[i] * ϕ.pstate[i]
    end

    return res 

end

function LossPerSampleAndIsCorrect(W::MPS, ϕ::PState)
    """Evaluate the cost function for a single sample and whether or not the 
    sample was correctly (return 1) or incorrectly (return 0) classified.
    """

    # get the model output/prediction
    yhat = ContractMPSAndProductState(W, ϕ)
    # get the ground truth label
    label_idx = inds(yhat)[1]
    y = onehot(label_idx => (ϕ.label + 1)) # one-hot encode the ground truth label (class 0 -> 1)

    # compute the quadratic cost
    dP = yhat - y
    cost = 0.5 * norm(dP)^2

    correct = 0
    predicted_label = argmax(abs.(Vector(yhat))) - 1 # convert from one-hot back into original labels

    if predicted_label == ϕ.label
        correct = 1
    end
    
    return [cost, correct]

end

function LossAndAccDataset(W::MPS, pstates::Vector{PState})
    """Function to compute the loss and accuracy for an entire dataset (i.e., test/train/validation)"""

    running_loss = Vector{Float64}(undef, length(pstates))
    running_acc = Vector{Float64}(undef, length(pstates))

    for i=1:length(pstates)
        loss, acc = LossPerSampleAndIsCorrect(W, pstates[i])
        running_loss[i] = loss
        running_acc[i] = acc
    end

    loss_total = sum(running_loss)
    acc_total = sum(running_acc)

    return [loss_total/length(pstates), acc_total/length(pstates)]

end

function LossPerBondTensor(B::ITensor, LE::Matrix, RE::Matrix, product_states::Vector{PState};
    id::Int, direction::String="forward")

    N_train = length(product_states)
    N = length(product_states[1].pstate)
    costs = Vector{Float64}(undef, N_train)

    if direction == "forward"
        Threads.@threads for i=1:N_train
            if id == (N-1)
                effective_input = LE[i, id-1] * product_states[i].pstate[id] * product_states[i].pstate[id+1]
            elseif id == 1
                effective_input = RE[i, id+2] * product_states[i].pstate[id] * product_states[i].pstate[id+1]
            else
                effective_input = LE[i, id-1] * RE[i, id+2] * product_states[i].pstate[id] * product_states[i].pstate[id+1]
            end

            # construct decision function output
            P = B * effective_input

            # compute the loss
            label_idx = inds(P)[1]

            y = onehot(label_idx => (product_states[i].label + 1)) # one-hot encode the ground truth label (class 0 -> 1)
            dP = y - P

            costs[i] = 0.5 * norm(dP)^2

        end

    elseif direction == "backward"
        Threads.@threads for i=1:N_train
            if id == N
                effective_input = LE[i, id-2] * product_states[i].pstate[id] * product_states[i].pstate[id-1]
            elseif id == 2
                effective_input = RE[i, id+1] * product_states[i].pstate[id] * product_states[i].pstate[id-1]
            else
                effective_input = LE[i, id-2] * RE[i, id+1] * product_states[i].pstate[id] * product_states[i].pstate[id-1]
            end

            P = B * effective_input

            # compute the loss
            label_idx = inds(P)[1]
            y = onehot(label_idx => (product_states[i].label + 1)) # one-hot encode the ground truth label (class 0 -> 1)
            dP = y - P

            costs[i] = 0.5 * norm(dP)^2
        end
    end

    C = sum(costs)

    return C/N_train

end

function GradientDescent(B::ITensor, LE::Matrix, RE::Matrix, product_states::Vector{PState}; id::Int,
    α::Float64, direction::String)

    """Function to compute the gradient and apply update using the specified step size."""
    B_old = B
    nt = length(product_states)
    N = length(product_states[1].pstate)

    gradient_accumulate = Vector{ITensor}(undef, nt)
    if direction == "forward"
        Threads.@threads for i=1:nt
            if id == (N-1)
                effective_input = LE[i, id-1] * product_states[i].pstate[id] * product_states[i].pstate[id+1]
            elseif id == 1
                effective_input = RE[i, id+2] * product_states[i].pstate[id] * product_states[i].pstate[id+1]
            else
                effective_input = LE[i, id-1] * RE[i, id+2] * product_states[i].pstate[id] * product_states[i].pstate[id+1]
            end

            P = B_old * effective_input

            # compute the loss
            label_idx = inds(P)[1]
            y = onehot(label_idx => (product_states[i].label + 1)) # one-hot encode the ground truth label (class 0 -> 1)
            dP = y - P

            grad = dP * effective_input

            gradient_accumulate[i] = grad
        end

    elseif direction == "backward"
        Threads.@threads for i=1:nt
            if id == N
                effective_input = LE[i, id-2] * product_states[i].pstate[id] * product_states[i].pstate[id-1]
            elseif id == 2
                effective_input = RE[i, id+1] * product_states[i].pstate[id] * product_states[i].pstate[id-1]
            else
                effective_input = LE[i, id-2] * RE[i, id+1] * product_states[i].pstate[id] * product_states[i].pstate[id-1]
            end

            P = B_old * effective_input
            
            # compute the loss
            label_idx = inds(P)[1]
            y = onehot(label_idx => (product_states[i].label + 1)) # one-hot encode the ground truth label (class 0 -> 1)
            dP = y - P

            grad = dP * effective_input

            gradient_accumulate[i] = grad

        end
    end

    ΔB = sum(gradient_accumulate)

    # update the bond tensor
    B_new = B_old + α * ΔB
    
    return B_new

end

function UpdateBondTensor(W::MPS, id::Int, direction::String, product_states::Vector{PState},
    LE::Matrix, RE::Matrix; α::Float64, χ_max=nothing, cutoff=nothing, verbose=true)
    """Function to apply gradient descent to a bond tensor"""

    N_train = length(product_states) # get the number of training samples
    N = length(W) # ge the number of sites

    if direction == "forward"

        left_site = W[id]
        right_site = W[id+1]

        # construct the bond tensor
        B_old = left_site * right_site
        # compute the cost before the update
        cost_before_update = LossPerBondTensor(B_old, LE, RE, product_states; id=id, direction=direction)
        if verbose == true
            println("Bond $id | Cost before optimising: $cost_before_update")
        end

        B_new = GradientDescent(B_old, LE, RE, product_states; id=id, α=α, direction=direction)
        # compute the cost after a single step update
        cost_after_update = LossPerBondTensor(B_new, LE, RE, product_states; id=id, direction=direction)
        if verbose==true
            println("Bond $id | Cost after optimising: $cost_after_update")
        end

        if cost_after_update > cost_before_update
            B_new = B_old
        end

        # SVD back into MPS tensors
        left_site_index = findindex(B_new, "Qudit,Site,n=$id") # retain the left site physical index

        if id == 1
            if χ_max !== nothing && cutoff !== nothing
                U, S, V = svd(B_new, (left_site_index); maxdim=χ_max, cutoff=cutoff)
            elseif χ_max !== nothing
                U, S, V = svd(B_new, (left_site_index); maxdim=χ_max)
            else
                U, S, V = svd(B_new, (left_site_index); cutoff=cutoff)
            end

        else
            bond_index = findindex(B_new, "Link,l=$(id-1)") # retain the bond dimension index
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
        replacetags!(left_site_new, "Link,u", "Link,l=$(id)")
        replacetags!(right_site_new, "Link,u", "Link,l=$(id)")

        # update environments
        for i = 1:N_train
            if id == 1
                LE[i, 1] = left_site_new * product_states[i].pstate[id]
            else
                LE[i, id] = LE[i, id-1] * left_site_new * product_states[i].pstate[id]
            end
        end

    elseif direction == "backward"

        left_site = W[id - 1]
        right_site = W[id]

        B_old = left_site * right_site

        # compute the cost function before the update
        B_old = left_site * right_site
        # compute the cost before the update
        cost_before_update = LossPerBondTensor(B_old, LE, RE, product_states; id=id, direction=direction)
        if verbose == true
            println("Bond $(id-1) | Cost before optimising: $cost_before_update")
        end

        B_new = GradientDescent(B_old, LE, RE, product_states; id=id, α=α, direction=direction)
        # compute the cost after a single step update
        cost_after_update = LossPerBondTensor(B_new, LE, RE, product_states; id=id, direction=direction)
        if verbose==true
            println("Bond $(id-1) | Cost after optimising: $cost_after_update")
        end

        if cost_after_update > cost_before_update
            B_new = B_old
        end

        left_site_index = findindex(B_new, "Qudit,Site,n=$(id-1)")
        label_idx = findindex(B_new, "f(x)")

        if id == 2
            if χ_max !== nothing && cutoff !== nothing
                U, S, V = svd(B_new, (left_site_index, label_idx); maxdim=χ_max, cutoff=cutoff)
            elseif χ_max !== nothing
                U, S, V = svd(B_new, (left_site_index, label_idx); maxdim=χ_max)
            else
                U, S, V = svd(B_new, (left_site_index, label_idx); cutoff=cutoff)
            end
        else
            bond_index = findindex(B_new, "Link,l=$(id-2)")
            if χ_max !== nothing && cutoff !== nothing
                U, S, V = svd(B_new, (left_site_index, bond_index, label_idx); maxdim=χ_max, cutoff=cutoff)
            elseif χ_max !== nothing
                U, S, V = svd(B_new, (left_site_index, bond_index, label_idx); maxdim=χ_max)
            else
                U, S, V = svd(B_new, (left_site_index, bond_index, label_idx); cutoff=cutoff)
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

    end

    return left_site_new, right_site_new, LE, RE 

end

function fitMPS(X_train::Matrix, y_train::Vector, X_val::Matrix, y_val::Vector; num_sax_bins=3, χ_init=5, 
    nsweep=10, α=0.01, χ_max=15, cutoff=nothing, random_state=nothing, sweep_tol=nothing)

    num_mps_sites = size(X_train)[2]
    num_classes = length(unique(y_train))
    sites = siteinds("Qudit", num_mps_sites; dim=num_sax_bins)

    println("Using χ_init = $χ_init | α=$α | nsweep = $nsweep")

    # step one - z score the training data
    zscaler = fit(ZScoreTransform, X_train; dims=1)
    rescaled_data = StatsBase.transform(zscaler, X_train)

    # step two - apply SAX transform to the z-scored data
    println("Applying SAX to the training data. Using $num_sax_bins bins.")
    X_sax, sax_fit = ZScoredTimeSeriesToSAX(rescaled_data; n_bins=num_sax_bins)

    # step three - convert discretised time series to product state encoding
    training_states = GenerateAllProductStates(X_sax, y_train, "train", sites, sax_fit)

    # now encode the validation states
    rescaled_val_data = StatsBase.transform(zscaler, X_val)
    sax_transformed_val_data = sax_fit.transform(rescaled_val_data)
    validation_states = GenerateAllProductStates(sax_transformed_val_data, y_val, "valid", sites, sax_fit)

    # generate the initial MPS
    W = GenerateStartingMPS(χ_init, sites; random_state=random_state)
    AttachLabelIndex!(W, num_classes)

    # construct the initial caches
    LE, RE = ConstructCaches(W, training_states; direction="forward")

    # compute the initial training and validation loss
    init_train_loss, _ = LossAndAccDataset(W, training_states)
    init_valid_loss, _ = LossAndAccDataset(W, validation_states)

    running_train_loss = init_train_loss
    running_valid_loss = init_valid_loss

    for itS = (1:nsweep)
        println("Forward Sweep L -> R ($itS/$nsweep)")

        for j = 1:(length(sites)-1)
            W[j], W[j+1], LE, RE = UpdateBondTensor(W, j, "forward", training_states, LE, RE; α=α, χ_max=χ_max, cutoff=cutoff, verbose=false)
        end

        # finished forward sweep, reset the cache and begin backward sweep
        LE, RE = ConstructCaches(W, training_states; direction="backward")

        println("Backward Sweep R -> L ($itS/$nsweep)")

        for j=(length(sites)):-1:2
            W[j-1], W[j], LE, RE = UpdateBondTensor(W, j, "backward", training_states, LE, RE; α=α, χ_max=χ_max, cutoff=cutoff, verbose=false)
        end

        LE, RE = ConstructCaches(W, training_states, direction="forward")

        # compute new cost
        train_loss, train_acc = LossAndAccDataset(W, training_states)
        valid_loss, valid_acc = LossAndAccDataset(W, validation_states)
        
        println("Validation loss after sweep $itS: $valid_loss | Validation accuracy: $valid_acc")
        println("Training loss after sweep $itS: $train_loss | Training accuracy: $train_acc")

        ΔC_valid = running_valid_loss - valid_loss
        ΔC_train = running_train_loss - train_loss

        println("ΔC train after sweep $itS: $ΔC_train")
        println("ΔC validation after sweep $itS: $ΔC_valid")

        if sweep_tol !== nothing
            if ΔC_valid < sweep_tol
                println("Convergence reached. ΔC Val = $ΔC_valid is less than the threshold $sweep_tol)!")
            end
        end

        running_train_loss = train_loss
        running_valid_loss = valid_loss

    end

    return W, training_states

end

function PlotSaxSample()
    """Function to plot a specified sample before and after being encoded by SAX"""

end



# run test
# raw_data = randn(1000, 10)
# labels = rand([0,1], 1000)

# val_data = randn(1000, 10)
# val_labels = rand([0, 1], 1000)

ecg_dat = readdlm("../ECG200_TRAIN.txt")
X_train = ecg_dat[:, 2:end]
y_train = Int.(ecg_dat[:, 1])
remap = Dict(-1 => 0, 1 => 1)
y_train = [remap[label] for label in y_train];

ecg_dat_test = readdlm("../ECG200_TEST.txt")
X_test = ecg_dat_test[:, 2:end]
y_test = Int.(ecg_dat_test[:, 1])
y_test = [remap[label] for label in y_test]


W, tstates = fitMPS(X_train, y_train, X_test, y_test; num_sax_bins=5, χ_max=15, α=0.1)

# z-score data
#zscaler = fit(ZScoreTransform, raw_data; dims=1)
#rescaled_data = StatsBase.transform(zscaler, raw_data)
#X_sax, sax = ZScoredTimeSeriesToSAX(rescaled_data; n_bins=5)
#s = siteinds("Qudit", 100; dim=5)
#ϕs = GenerateAllProductStates(X_sax, labels, "train", s, sax)
#W = GenerateStartingMPS(5, s; random_state=69)
#AttachLabelIndex!(W, 2)
#LE, RE = ConstructCaches(W, ϕs);
#cost, correct = LossPerSampleAndIsCorrect(W, ϕs[1])
#B = W[1] * W[2]
#GradientDescent(B, LE, RE, ϕs; id=1, α=0.1, direction="forward")
#sax.n_bins