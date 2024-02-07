using ITensors
using Plots
using StatsBase
using Base.Threads
using Random
using DelimitedFiles

struct PState
    """Create a custom structure to store product state objects, 
    along with their associated label and type (i.e, train, test or valid)"""
    pstate::MPS
    label::Int
    type::String
end

function BinariseTimeSeries(time_series::Vector, method="median")

    # check valid method
    if method !== "median" && method !== "mean"
        error("Invalid binarisation method. Can be either `median' or `mean'.")
    end

    # construct empty array for binarised time series
    binarised = zeros(Int, length(time_series))

    if method == "mean"
        thresh = mean(time_series)
    elseif method == "median"
        thresh = median(time_series)
    end

    for (i, val) in enumerate(time_series)
        # 1 for above the mean/median and 0 for below 
        binarised[i] = (val >= thresh) ? 1 : 0
    end
    
    return binarised

end

function BinariseDataset(time_series::Matrix; method="median")
    """Function to binarise an entire dataset of time series"""
    binarised_matrix = zeros(Int, size(time_series)[1], size(time_series)[2])

    for i=1:size(time_series)[1]
        binarised_matrix[i, :] = BinariseTimeSeries(time_series[i, :], method)
    end

    return binarised_matrix

end

function BinaryToProductState(binarised_time_series::Vector, site_indices::Vector{Index{Int64}})
    """convert a binarised time series to a product state where 1 maps to 
    spin down and 0 maps to spin up as per bloch sphere conventions"""

    # intial checks
    if !all(x -> x == 1 || x == 0, binarised_time_series)
        error("Time sereies does not contain binary values.")
    end

    # check that the time series length is equal to the site indices length
    if length(binarised_time_series) !== length(site_indices)
        error("Mismatch betwee number of physical sites ($(length(site_indices))) and length of 
        the binarised time series ($(length(binarised_time_series)))")
    end

    # create empty product state container
    phi = MPS(site_indices; linkdims=1)

    # fill the product state container with tensors corresponding to the states at each site
    for s = 1:length(site_indices)
        T = ITensor(site_indices[s]) # extract the site
        if binarised_time_series[s] == 1
            # spin down for 1 state -> [0; 1]
            T[2] = 1
        else
            # spin up for 0 state -> [1; 0]
            T[1] = 1
        end
        # write new tensor to the product state container
        phi[s] = T
    end

    return phi

end

function GenerateAllProductStates(X_binarised::Matrix, y::Vector, type::String, 
    site_indices::Vector{Index{Int64}})
    """Convert an entire dataset of binarised time series to a corresponding 
    dataset of product states"""

    if type == "train"
        println("Initialising training states.")
    elseif type == "test"
        println("Initialising testing states.")
    elseif type == "valid"
        println("Initialising validation states.")
    else
        error("Invalid dataset type. Must be either train, test or valid!")
    end

    num_samples = size(X_binarised)[1]

    # create a vector of PStates to store all product states
    phis = Vector{PState}(undef, num_samples)

    Threads.@threads for sample = 1:num_samples
        sample_pstate = BinaryToProductState(X_binarised[sample, :], site_indices)
        sample_label = y[sample]
        ps = PState(sample_pstate, sample_label, type)
        phis[sample] = ps
    end

    return phis

end

function GenerateStartingMPS(χ_init, site_indices::Vector{Index{Int64}}; random_state=nothing)
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

    W = randomMPS(site_indices, linkdims=χ_init)

    # canonicalise - bring MPS into canonical form by making all tensors 1,...,j-1 left orthogonal
    # here we assume we start at the right most index
    last_site = length(site_indices)
    orthogonalize!(W, last_site)

    return W

end


function ConstructCaches(W::MPS, training_pstates::Vector{PState};
    going_left=true)
    """Function to pre-compute tensor contractions between the MPS and the product states. """

    # get the number of training samples to pre-allocate the caching matrix
    n_train = length(training_pstates)
    # get number of MPS sites
    num_sites = length(W)

    # pre-allocate left and right environment matrices
    LE = Matrix{ITensor}(undef, n_train, num_sites)
    RE = Matrix{ITensor}(undef, n_train, num_sites)

    if going_left
        # initialise the LE with the first site
        for i = 1:n_train
            LE[i, 1] = training_pstates[i].pstate[1] * W[1]
        end

        # fill the remaining sites cumulatively
        for j = 2:num_sites
            for i = 1:n_train
                LE[i, j] = LE[i, j-1] * training_pstates[i].pstate[j] * W[j]
            end
        end

    else
        # going right
        for i = 1:n_train
            RE[i, num_sites] = training_pstates[i].pstate[num_sites] * W[num_sites]
        end

        # fill remaining sites, starting with site N-1 all the way up to the first site
        for j = (num_sites-1):-1:1
            for i = 1:n_train
                RE[i, j] = RE[i, j+1] * W[j] * training_pstates[i].pstate[j]
            end
        end

    end


    return LE, RE

end

function ComputePsi(W::MPS, phi::PState)
    # result (overlap) can be thought of as the probability amplitude in a QM sense
    """Contract the MPS with the product state representation of the data
    
    PSI :     O-...-O-O-O-...-O
              |     | | |     |
              |     | | |     |
    Data:     O     O O O     O

    Function to manually contract the weight MPS awith a single
    product state since ITensor `inner' function doesn't like it when
    there is a label index attached to the MPS. Returns RAW output
    (prior to conversion to prob. dist). Will return an ITensor
    corresponding to the label index with the raw scores of each class.
    """
    num_sites = length(W)
    res = 1
    for i = 1:num_sites
        res *= W[i] * phi.pstate[i]
    end

    return res

end

function ComputePsiPrime(LE::Matrix, RE::Matrix, product_states::Vector{PState};
    pid::Int, id::Int)
    """Compute the derivative of the MPS with resepct to bond tensor for a single product state.
    
    PSI': O-...-O-      -O-...-O
          |     |        |     |
          |     |  |  |  |     |
    Data: O     O  O  O  O     O

    Resulting in an order-4 tensor (in bulk):

    | | | |
    O O O O

    Uses the caches to access pre-computed tensors.

    id will always index the left-most tensor (of two sites being considered),
    irrespective of direction.

    | | | |
    O O O O
      ^   
      |
      id 

    """
    n_sites = length(product_states[1].pstate)

    # use threading to compute each sample psi prime in parallel
    if id == (n_sites - 1) # id always indexs left-most tensor of the bond tensor
        # last site, no right environment -> order 3 tensor
            """
            MPS:     O-O-...O-
                     | |    | | |
                     | |    | | |
            Data:    O O    O O O

            """
            effective_input = LE[pid, id-1] * product_states[pid].pstate[id] * product_states[pid].pstate[id+1]
    elseif id == 1
        # first site, no left environment -> order 3 tensor 
            """
            MPS:        ...-O-O
                    | |     | |
                    | |     | |
            Data:   O O ....O O
                        RE [2, N]

            Gives

            | | |
            O O O

            """
        effective_input = RE[pid, id+2] * product_states[pid].pstate[id] * product_states[pid].pstate[id+1]
    else
        # both LE and RE exist - we are in the bulk of the MPS
            """
            O-...-O-   -O-...-O 
            |     |     |     |
            |     | | | |     |
            O     O O O O     O

            Gives order 4 tensor:

            | | | |
            O O O O
            """
        effective_input = LE[pid, id-1] * product_states[pid].pstate[id] * product_states[pid].pstate[id+1] * RE[pid, id+2]
            
    end

    return effective_input

end

function ComputeNLL(W::MPS, product_states::Vector{PState})
    """Compute the negative log loss (NLL) of the MPS over a set
    of time series.
    NLL = -(1/Ns) * SUM_{n in Ns} (ln P(n)) = -(1/Ns) * SUM_{n in Ns} (ln(psi(n)**2))/Z
    Using log rules we can rewrite as
    NLL = ln(Z) -(2/Ns) * SUM_{n in Ns} (ln |psi(n)|)
    Where Z is normalisation constant.
    """
    Ns = length(product_states)
    # assume MPS is in canonical form
    Z = inner(dag(W), W)
    lnsum = 0
    for ps in product_states
        lnsum += log(abs(ComputePsi(W, ps)[]))
    end

    # here, log(Z) term is acting to regularise - penalises large values 
    return -2 * (lnsum / Ns) + log(Z) 

end

function ComputeGradient(B::ITensor, W::MPS, LE::Matrix, RE::Matrix, product_states::Vector{PState};
    id::Int)
    """Function to compute the derivative of the NLL w.r.t. the bond tensor B.\
    Update rule:

    dL/dB = Z'/Z - 2/Ns SUM_{n in Ns} psi'(n)/psi(n)

    if MPS is in mixed canonical form, Z' can be simplified to 2A

    
    """
    num_samples = length(product_states)
    num_sites = length(product_states[1].pstate)

    # compute Z - the normalisation factor
    Z = inner(dag(B), B)

    # compute the second term in the equation which has data dependent terms psi'(v)/psi
    # where psi' is the derivative of the bond tensor w.r.t the MPS 
    psi_frac = ITensor()
    for i=1:num_samples
        psi_prime = ComputePsiPrime(LE, RE, product_states; pid=i, id=id)
        psi = ComputePsi(W, product_states[i])[]
        psi_frac_val = psi_prime ./ psi
        psi_frac += psi_frac_val
    end
    #println(psi_frac)
    
    # done, divide through by the number of samples
    psi_frac = psi_frac./num_samples

    #print(psi_frac)

    # gradient
    grad = (B./Z) - psi_frac
    #print(grad)

    return grad

end

function UpdateBondTensor(W::MPS, id::Int, product_states::Vector{PState}, 
    LE::Matrix, RE::Matrix; going_left=true, α::Float64=0.1, χ_max=nothing, cutoff=nothing)     

    num_train = length(product_states)
    num_sites = length(W)

    if going_left

        left_site = W[id]
        right_site = W[id + 1]

        # construct old bond tensor
        B_old = left_site * right_site
        # compute loss before update
        #old_NLL = ComputeNLL(W, product_states)
        #if verbose == true
        #    println("Bond $id | Cost before optimising: $old_NLL")
        #end

        # apply update to the bond tensor using a single step of gradient descent
        gradient = ComputeGradient(B_old, W, LE, RE, product_states; id=id)
        # apply the gradient update
        B_new = B_old - α * gradient
        # rescale the updated bond tensor to keep the MPS normalised
        B_new ./= sqrt(inner(dag(B_new), B_new))

        # now that we have the updated bond tensor, let's SVD back into MPS tensors
        left_site_index = findindex(B_new, "S=1/2,Site,n=$id" )

        if id == 1
            if χ_max !== nothing && cutoff !== nothing
                U, S, V = svd(B_new, (left_site_index); maxdim=χ_max, cutoff=cutoff)
            elseif χ_max !== nothing
                # just max bond dimension has been specified, use default cutoff
                U, S, V = svd(B_new, (left_site_index); maxdim=χ_max)
            else
                # just cutoff specified, use default max bond dimension
                U, S, V = svd(B_new, (left_site_index); cutoff=cutoff)
            end
        else
            # ensure that the U site has the bond id-1 where id is the left site
            bond_index = findindex(B_new, "Link,l=$(id-1)")
            if χ_max !== nothing && cutoff !== nothing
                U, S, V = svd(B_new, (left_site_index, bond_index); maxdim=χ_max, cutoff=cutoff)
            elseif χ_max !== nothing
                U, S, V = svd(B_new, (left_site_index, bond_index); maxdim=χ_max)
            else
                U, S, V = svd(B_new, (left_site_index, bond_index); cutoff=cutoff)
            end
        end

        # to preserve canonicalisation, we need to absorb the singular values into the left site when moving left

        left_site_new = U * S
        right_site_new = V

        replacetags!(left_site_new, "Link,v", "Link,l=$id")
        replacetags!(right_site_new, "Link,v", "Link,l=$id")

        # update environments
        for i = 1:num_train
            # weird indexing because we are always using the left-most tensor as the reference point
            if id == (num_sites - 1)
                # if bond is over sites N-1 (left idx) and N (right idx)
                RE[i, num_sites] = right_site_new * product_states[i].pstate[num_sites]
            else
                #println("ID: $id")
                RE[i, id+1] = RE[i, id+2] * right_site_new * product_states[i].pstate[(id +1)]
            end
        end

    else
        # going right
        left_site = W[id]
        right_site = W[id + 1]

        # construct old bond tensor
        B_old = left_site * right_site

        # compute NLL before update
        #old_NLL = ComputeNLL(W, product_states)
        #if verbose == true
        #    println("Bond $id | Cost before optimising: $old_NLL")
        #end

        # apply update to the bond tensor using a single step of gradient descent
        gradient = ComputeGradient(B_old, W, LE, RE, product_states; id=id)
        # apply the gradient update
        B_new = B_old - α * gradient
        # rescale the updated bond tensor to keep the MPS normalised
        B_new ./= sqrt(inner(dag(B_new), B_new))

        # now that we have the updated bond tensor, let's SVD back into MPS tensors
        left_site_index = findindex(B_new, "S=1/2,Site,n=$id" )

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
        for i = 1:num_train
            if id == 1
                LE[i, 1] = left_site_new * product_states[i].pstate[id]
            else
                LE[i, id] = LE[i, id-1] * left_site_new * product_states[i].pstate[id]
            end
        end

    end

    # get new loss by making a copy
    #W_new = deepcopy(W)
    #W_new[id] = left_site_new
    #W_new[id+1] = right_site_new
    #new_NLL = ComputeNLL(W_new, product_states)

    return left_site_new, right_site_new, LE, RE

end


function fitMPS(X_train::Matrix, y_train::Vector, X_val::Matrix, y_val::Vector; χ_init=5,
    nsweep=10, α=0.01, χ_max=50, cutoff=nothing, random_state=nothing, sweep_tol=nothing)

    num_mps_sites = size(X_train)[2]
    sites = siteinds("S=1/2", num_mps_sites)
    println("Using χ_init = $χ_init | α=$α | nsweep = $nsweep")

    # step one - binarise the data
    binarised_training_data = BinariseDataset(X_train; method="mean")
    #print(binarised_training_data)
    #binarised_time_series = [BinariseTimeSeries(X_train[i, :], "median") for i=1:size(X_train)[1]]
    #binarised_time_series = Matrix(hcat(binarised_time_series...)') # convert to matrix
    # step two - encode as product state
    training_states = GenerateAllProductStates(binarised_training_data, y_train, "train", sites)

    # encode the validation states
    binarised_validation_data = BinariseDataset(X_val; method="mean")
    #binarised_validation_time_series = Matrix(hcat(binarised_validation_time_series...)')
    #println(binarised_validation_time_series)
    validation_states = GenerateAllProductStates(binarised_validation_data, y_val, "valid", sites)

    # generate starting MPS
    W = GenerateStartingMPS(χ_init, sites; random_state=random_state)

    # construct initial caches
    LE, RE = ConstructCaches(W, training_states; going_left=true)

    # compute the initial training and validation loss
    init_train_loss = ComputeNLL(W, training_states)
    init_valid_loss = ComputeNLL(W, validation_states)

    running_train_loss = init_train_loss
    running_valid_loss = init_valid_loss

    println("Initial train loss: $init_train_loss")
    println("Initial validation loss: $init_valid_loss")

    # start the sweep
    for itS = 1:nsweep
        println("Starting backward Sweep: ($itS/$nsweep)")

        for j = (length(sites)-1):-1:1
            lsn, rsn, LE, RE = UpdateBondTensor(W, j, training_states, LE, RE; 
                going_left=true, α=α, χ_max=χ_max, cutoff=cutoff)
            W[j] = lsn
            W[j+1] = rsn
        end

        # finished the backward sweeep
        println("Backward sweep finished.")

        # construct new caches for forward sweep
        LE, RE = ConstructCaches(W, training_states; going_left=false)

        for j = 1:(length(sites) - 1)
            lsn, rsn, LE, RE = UpdateBondTensor(W, j, training_states, LE, RE; 
                going_left=false, α=α, χ_max=χ_max, cutoff=cutoff)
            W[j] = lsn
            W[j+1] = rsn
        end

        println("Finished forward sweep")
        
        # compute new loss
        train_loss = ComputeNLL(W, training_states)
        valid_loss = ComputeNLL(W, validation_states)

        println("Validation loss after sweep $itS: $valid_loss")
        println("Training loss after sweep $itS: $train_loss ")

        if sweep_tol !== nothing
            if ΔC_valid < sweep_tol
                println("Convergence reached. ΔC Val = $ΔC_valid is less than the threshold $sweep_tol)!")
                break
            end
        end

        running_train_loss = train_loss
        running_valid_loss = valid_loss
    end

    return W, training_states

end

# quick test
# Random.seed!(42)
# sites = siteinds("S=1/2", 10)
# x = rand(50, 50)
# x_val = rand(10, 50)
# y = rand([0, 1], 50)
# y_val = rand([0, 1], 10)
ecg_dat = readdlm("../ECG200_TRAIN.txt")
X_train = ecg_dat[:, 2:end]
y_train = Int.(ecg_dat[:, 1])
remap = Dict(-1 => 0, 1 => 1)
y_train = [remap[label] for label in y_train];

ecg_dat_test = readdlm("../ECG200_TEST.txt")
X_test = ecg_dat_test[:, 2:end]
y_test = Int.(ecg_dat_test[:, 1])
y_test = [remap[label] for label in y_test]

W, tstates = fitMPS(X_train, y_train, X_test, y_test; χ_max=25, α=0.0001, nsweep=5, χ_init=50)

#W, training_states = fitMPS(X_train, y_train, X_test, y_test; nsweep=15, χ_max=25)
# W = GenerateStartingMPS(10, sites; random_state=42)
# phis = GenerateAllProductStates(x, y, "train", sites)
# LE, RE = ConstructCaches(W, phis)
#phi_tilde = ComputePsiPrime(LE, RE, phis; pid=1, id=9)
#println(phi_tilde)
#ComputeNLL(W, phis)
#B = W[9] * W[10]
#grad = ComputeGradient(B, LE, RE, phis; id=9)
# lsn, rsn, LE, RE, old_NLL, new_NLL = UpdateBondTensor(W, 9, phis, LE, RE; α=0.1)
