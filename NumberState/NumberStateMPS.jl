using ITensors
using Random
using StatsBase
using PyCall
using Plots
using Base.Threads
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
    return X_sax, sax

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

function LossAndAccDataset(W::MPS, pstates::Vector{PStates})
    """Function to compute the loss and accuracy for an entire dataset (i.e., test/train/validation)"""

    running_loss = Vector{Float64}(undef, length(pstates))
    running_acc = Vector{Float64}(undef, length(pstates))

    for i=1:length(pstates)
        loss, acc = LossPerSampleAndIsCorrect(w, pstates[i])
        running_loss[i] = loss
        running_acc[i] = acc
    end

    loss_total = sum(running_loss)
    acc_total = sum(running_acc)

    return [loss_total/length(pstates), acc_total/length(pstates)]

end





# run test
raw_data = randn(10, 100)
labels = rand([0,1], 10)
# z-score data
zscaler = fit(ZScoreTransform, raw_data; dims=1)
rescaled_data = StatsBase.transform(zscaler, raw_data)
X_sax, sax = ZScoredTimeSeriesToSAX(rescaled_data; n_bins=5)
s = siteinds("Qudit", 100; dim=5)
ϕs = GenerateAllProductStates(X_sax, labels, "train", s, sax)
W = GenerateStartingMPS(5, s; random_state=69)
AttachLabelIndex!(W, 2)
LE, RE = ConstructCaches(W, ϕs);
cost, correct = LossPerSampleAndIsCorrect(W, ϕs[1])
