using ITensors
using Random
using StatsBase
using Plots
using PyCall
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
    """Function to manually contract the weight MPS awith a single
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

function ModelOutputToProbability(yhat::ITensor)
    """Function to convert the raw outputs/scores of the MPS classifier to a
    normalised probability distribution over class labels"""
    norm = (yhat * yhat)[] # brackets are workaround to get ITensor to output a julia scalar
    pdist = [yhat[i]^2 / norm for i=1:ITensors.dim(yhat)]

    return pdist
end

function LossPerSampleAndIsCorrect(W::MPS, ϕ::PState)
    """Evaluate the loss function for a single sample and whether or not the sample was
    correctly (return 1) or incorrectly (return 0) classified. 
    Compute for entire MPS (not a single bond tensor)"""

    # get the model output/prediction
    yhat = ContractMPSAndProductState(W, ϕ)
    # get the label index
    label_idx = inds(yhat)[1]
    # one-hot encode the target label
    y = onehot(label_idx => (ϕ.label + 1)) # label 0 maps to onehot encode vector entry 1 (julia)
    # convert the entry of the model output corresponding to the ground-truth label to a probability
    y_yhat = (y * yhat)[]
    yhat_yhat = (yhat * yhat)[] # norm factor
    prob = y_yhat^2 / yhat_yhat

    # compute the loss for a single sample
    log_loss = -log(prob)

    # now get the full probability distribution and argmax to get the predicted label
    correct = 0
    full_probability_dist = ModelOutputToProbability(yhat)
    if (argmax(full_probability_dist) - 1) == ϕ.label # convert the argmax label back to the original labelling scheme
        correct = 1
    end

    return [log_loss, correct]

end

function LossAndAccDataset(W::MPS, pstates::Vector{PState})
    """Function to compute the loss and accuracy for an entire dataset (i.e., test/train/validation)"""

    running_loss = Vector{Float64}(undef, length(pstates))
    running_acc = Vector{Float64}(undef, length(pstates))

    for i=1:length(pstates)
        sample_loss, sample_acc = LossPerSampleAndIsCorrect(W, pstates[i])
        running_loss[i] = sample_loss
        running_acc[i] = sample_acc
    end

    loss_total = sum(running_loss)
    acc_total = sum(running_acc)

    return [loss_total/length(pstates), acc_total/length(pstates)]

end

function LossPerBondTensor(B::ITensor, LE::Matrix, RE::Matrix, product_states::Vector{PState},
    id::Int, direction::String="forward") 
    """Function to compute the loss for a single bond tensor, given that the rest of the tensor
    network remains fixed."""

    N_train = length(product_states)
    N = length(product_states[1].pstate)
    costs = Vector{Float64}(undef, N_train)

    if direction == "forward"
        Threads.@threads for i=1:N_train
            # the effective input here is equivalent to the four-index projected input in Stoudenmire et. al's work.
            # this is the gradient of the decision function output with respect to the bond tensor of interest. 
            if id == (N-1)
                effective_input = LE[i, id-1] * product_states[i].pstate[id] * product_states[i].pstate[id+1]
            elseif id == 1
                effective_input = RE[i, id+2] * product_states[i].pstate[id] * product_states[i].pstate[id+1]
            else
                effective_input = LE[i, id-1] * RE[i, id+2] * product_states[i].pstate[id] * product_states[i].pstate[id+1]
            end

            # now construct the decision function output using the bond tensor and the effective input
            yhat = B * effective_input

            # compute the loss for the bond tensor B output
            label_idx = inds(yhat)[1]
            y = onehot(label_idx => (product_states[i].label + 1))
            # isolate the ground truth index
            y_yhat = (y * yhat)[]
            yhat_yhat = (yhat * yhat)[] # norm factor
            prob = y_yhat^2 / yhat_yhat # prob of the ground truth, given the current bond tensor, so scalar value
            # now compute the log loss
            log_loss = -log(prob)
            costs[i] = log_loss

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

            # same as above

            # now construct the decision function output using the bond tensor and the effective input
            yhat = B * effective_input

            # compute the loss for the bond tensor B output
            label_idx = inds(yhat)[1]
            y = onehot(label_idx => (product_states[i].label + 1))
            # isolate the ground truth index
            y_yhat = (y * yhat)[]
            yhat_yhat = (yhat * yhat)[] # norm factor
            prob = y_yhat^2 / yhat_yhat # prob of the ground truth, given the current bond tensor, so scalar value
            # now compute the log loss
            log_loss = -log(prob)
            costs[i] = log_loss

        end

    end

    # sum the individual losses for each sample and then divide through by number of samples
    C = sum(costs)

    return C/N_train

end

function GetGradient(B::ITensor, LE::Matrix, RE::Matrix, product_states::Vector{PState}; id::Int,
    direction::String)

    """Function to compute the gradient of the bond tensor, B w.r.t the cross entropy loss function."""

    B_old = B
    nt = length(product_states) # number of training states
    N = length(product_states[1].pstate)

    # setup data structure to hold gradient contributions from each parallel worker
    accumulate_gradient = Vector{ITensor}(undef, nt)
    if direction == "forward"
        Threads.@threads for i=1:nt
            # as before, we compute the effective input
            if id == (N-1)
                effective_input = LE[i, id-1] * product_states[i].pstate[id] * product_states[i].pstate[id+1]
            elseif id == 1
                effective_input = RE[i, id+2] * product_states[i].pstate[id] * product_states[i].pstate[id+1]
            else
                effective_input = LE[i, id-1] * RE[i, id+2] * product_states[i].pstate[id] * product_states[i].pstate[id+1]
            end

            yhat = B_old * effective_input
            #println(yhat)
            d_yhat_dW = effective_input

            # get y
            #println(inds(yhat))
            label_idx = inds(yhat)[1]
            #println(label_idx)
            y = onehot(label_idx => (product_states[i].label + 1))
            y_yhat = (y * yhat)[]
            yhat_yhat = (yhat * yhat)[]

            p = (y_yhat)^2 / yhat_yhat

            # split the gradient into two parts and compute separately, then join together
            part_one = y_yhat * (y * d_yhat_dW)
            part_one ./= yhat_yhat

            part_two = y_yhat^2 * (yhat * d_yhat_dW)
            part_two ./= (yhat_yhat)^2

            gradient = -2 * (part_one - part_two)
            gradient ./= p 

            accumulate_gradient[i] = gradient
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

            yhat = B_old * effective_input
            d_yhat_dW = effective_input

            # get y
            label_idx = inds(yhat)[1]
            y = onehot(label_idx => (product_states[i].label + 1))
            y_yhat = (y * yhat)[]
            yhat_yhat = (yhat * yhat)[]

            p = (y_yhat)^2 / yhat_yhat

            # split the gradient into two parts and compute separately, then join together
            part_one = y_yhat * (y * d_yhat_dW)
            part_one ./= yhat_yhat

            part_two = y_yhat^2 * (yhat * d_yhat_dW)
            part_two ./= (yhat_yhat)^2

            gradient = -2 * (part_one - part_two)
            gradient ./= p 

            accumulate_gradient[i] = gradient
        end
    end

    ΔB = sum(accumulate_gradient)
    #print(ΔB)

    return ΔB

end

function UpdateBondTensor(W::MPS, id::Int, direction::String, product_states::Vector{PState},
    LE::Matrix, RE::Matrix; α::Float64, χ_max=nothing, cutoff=nothing, verbose=true)

    """Function to apply a single step of gradient descent to a bond tensor B, 
    using the step size α."""

    N_train = length(product_states) # get the number of training samples
    N = length(W) # get the number of sites

    if direction == "forward"

        left_site = W[id]
        right_site = W[id + 1]

        # construct the bond tensor 
        B_old = left_site * right_site
        # compute the loss before the update
        loss_before_update = LossPerBondTensor(B_old, LE, RE, product_states, id, "forward")
        if verbose == true
            println("Bond $id | Cost before optimising: $loss_before_update")
        end

        # now apply update to the bond tensor using a single step of gradient descent 
        gradient = GetGradient(B_old, LE, RE, product_states; id=id, direction=direction)
        # apply the gradient update
        B_new = B_old + α * gradient
        # now compute the new loss after the update
        loss_after_update = LossPerBondTensor(B_new, LE, RE, product_states, id, "forward")
        if verbose == true
            println("Bond $id | Cost after optimising: $loss_after_update")
        end
        
        # revert to old bond tensor if the loss goes up
        if loss_after_update > loss_before_update
            B_new = B_old
        end

        # now that we have the updated bond tensor, let's SVD back into MPS tensors
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

    # same process as above but for sweeping in backward direction
    elseif direction == "backward"

        left_site = W[id - 1]
        right_site = W[id]
        
        B_old = left_site * right_site
        # compute the loss function before updating
        loss_before_update = LossPerBondTensor(B_old, LE, RE, product_states, id, "backward")
        if verbose == true
            println("Bond $(id-1) | Cost before optimising: $loss_before_update")
        end

        # now apply update to the bond tensor using a single step of gradient descent 
        gradient = GetGradient(B_old, LE, RE, product_states; id=id, direction=direction)
        # apply the gradient update
        B_new = B_old + α * gradient

        # now compute the new loss after the update
        loss_after_update = LossPerBondTensor(B_new, LE, RE, product_states, id, "backward")
        if verbose == true
            println("Bond $(id-1) | Cost after optimising: $loss_after_update")
        end

        # revert to old bond tensor if the loss goes up
        if loss_after_update > loss_before_update
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


dat = randn(100, 50)
labels = rand([0, 1], 100)
X_sax, sax_fit = ZScoredTimeSeriesToSAX(dat)
sites = siteinds("Qudit", 50; dim=3)
ϕs = GenerateAllProductStates(X_sax, labels, "train", sites, sax_fit)
W = GenerateStartingMPS(5, sites)
AttachLabelIndex!(W, 2)
LE, RE = ConstructCaches(W, ϕs; direction="forward");
#B = W[1] * W[2]
L_new, R_new, LE, RE = UpdateBondTensor(W, 1, "forward", ϕs, LE, RE; α=0.01, χ_max=10);
#LossPerBondTensor(B, LE, RE, ϕs, 1, "forward")
#grad = GetGradient(B, LE, RE, ϕs; id=1, direction="forward")
# yhat = ContractMPSAndProductState(W, ϕs[1])
# println(yhat)
# proba = ModelOutputToProbability(yhat)
# println(proba)
# loss, correct = LossPerSampleAndIsCorrect(W, ϕs[3])
# println("Loss: $loss | correct: $correct")
# println(ϕs[3].label)

