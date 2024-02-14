using Distributions
using ITensors
using Base.Threads
using Folds
using Random

# Highly refined version of all the previous iterations
function OneHotEncodeLabel(label::Int, params::Dict=LocalParameters)
    """Converts a class label to an equivalent one-hot encoded ITensor"""
    label_idx = params[:label_idx]
    label_tensor = ITensor(label_idx)
    label_tensor[label + 1] = 1
    return label_tensor
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

##################################
global LocalParameters = Dict(
    :feature_map => Binariser, 
    :num_mps_sites => 0, # number of physical sites/indices in the MPS (equal to the number of data pts. in the time-series)
    :local_dimension => 2, # local dimension of the Hilbert space
    :χ_init => 5, # initial weight MPS bond dimension
    :random_state => 42, # random seed to generate the initial weight MPS
    :num_classes => 2, # number of classes to distinguish
)
#################################


struct PState
    """ Create a custom structure to store product state objects, along with their associated label and type (either train/test/validation)"""
    pstate::MPS
    label::Int
    type::String
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

function GenerateStartingMPS(params::Dict=LocalParameters; random_state=nothing)
    """Function to generate the initial weight MPS using values sampled from a Gaussian (normal) distribution."""
    if random_state !== nothing
        Random.seed!(random_state)
        println("Generating initial weight MPS with bond dimension χ = $(params[:χ_init]) using random seed $(random_state).")
    else
        println("Generating initial weight MPS with bond dimension χ = $(params[:χ_init]).")
    end

    W = randomMPS(params[:site_inds], linkdims=params[:χ_init])

    return W

end

function AttachLabelIndex(W::MPS; attach_site::Int, params::Dict=LocalParameters)
    """Function to attach the decision label index to the MPS at the specified site"""
    """Dimension is equal to the number of classess"""
    label_idx = Index(params[:num_classes], "decision")
    params[:label_idx] = label_idx

    # get the site of interest and copy over the indicies
    old_site_inds = inds(W[attach_site])
    new_site_inds = old_site_inds, label_idx
    new_site = randomITensor(new_site_inds) # reconstruct the site with the new index attached

    # add the updated site back into the MPS
    W[attach_site] = new_site

    # normalise the MPS again
    normalize!(W)
    
    return W
end

function ConstructCaches(W::MPS, training_pstates::Vector{PState}; direction::String="forward", params::Dict=LocalParameters)
    """Function to pre-compute tensor contractions between the MPS and the product states."""

    N_train = length(training_pstates) # number of training samples
    N = length(W) # number of MPS sites

    LE = Matrix{ITensor}(undef, N_train, N) # Left environment for each training sample 
    RE = Matrix{ITensor}(undef, N_train, N) # Right environment for each training sample

    if direction == "forward"
        
        # intialise the RE with the terminal site and work backwards accumulating contractions site-by-site
        for i = 1:N_train
            RE[i, N] = training_pstates[i].pstate[N] * W[N]
            #normalize!(RE[i, N])
        end

        # accumulate all other sites
        for j = (N-1):-1:1
            for i=1:N_train
                RE[i, j] = RE[i, j+1] * W[j] * training_pstates[i].pstate[j]
                #normalize!(RE[i, j])
            end
        end

    elseif direction == "backward"

        for i = 1:N_train
            # intialise LE with first site and work forward accumularing contractions site-by-site
            LE[i, 1] = training_pstates[i].pstate[1] * W[1]
            #normalize!(LE[i, 1])
        end

        for j=2:N
            for i=1:N_train
                LE[i, j] = LE[i, j - 1] * training_pstates[i].pstate[j] * W[j]
                #normalize!(LE[i, j])
            end
        end

    else
        error("Invalid direction. Can either be forward or backward.")
    end

    return LE, RE
end

function ContractMPSAndProdState(W::MPS, ϕ::PState)
    """ Function to manually contract the weight MPS with a single product. """
    """ Returns an ITensor """

    normalize!(W)
    normalize!(ϕ.pstate)

    N_sites = length(W)
    res = 1 # store the cumulative result of contractions
    for i=1:N_sites
        res *= W[i] * ϕ.pstate[i]
    end

    return res
end

function LossPerSampleAndCorrect(W::MPS, ϕ::PState)
    """ Evaluate the loss function for a single sample and whether the sample was correctly (1) or incorrectly (0) classified"""

    yhat = ContractMPSAndProdState(W, ϕ) # model prediction
    y = OneHotEncodeLabel(ϕ.label) # ground truth label 

    # compute the quadratic cost
    dP = yhat - y
    cost = 0.5 * norm(dP)^2

    correct = 0
    predicted_label = argmax(abs.(Vector(yhat))) - 1

    if predicted_label == ϕ.label
        correct = 1
    end

    return [cost, correct]
end

function LossAndAccDataset(W::MPS, pstates::Vector{PState})
    """Function to compute the loss and accuracy for an entire dataset (i.e. test/train/validation)"""

    running_loss = Vector{Float64}(undef, length(pstates))
    running_acc = Vector{Float64}(undef, length(pstates))

    for i=1:length(pstates)
        loss, acc = LossPerSampleAnCorrect(W, pstates[i])
        running_loss[i] = loss
        running_acc[i] = acc
    end

    loss_total = sum(running_loss)
    acc_total = sum(running_acc)

    return [loss_total/length(pstates), acc_total/length(pstates)]
end

function GradientDescent(B::ITensor, LE::Matrix, RE::Matrix, product_states::Vector{PState}; id::Int, 
    α::Float64, direction::String, params::Dict=LocalParameters, verbose=true)

    """Function computes the gradient and applies update using the step size."""
    B_old = B

    N = params[:num_mps_sites]
    nt = length(product_states) # number of training samples

    gradient_accumulate = Vector{ITensor}(undef, nt)
    # get the gradient
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

            dP = OneHotEncodeLabel(product_states[i].label, params) - P # could be problematic

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
            
            dP = OneHotEncodeLabel(product_states[i].label, params) - P # could be problematic

            grad = dP * effective_input

            gradient_accumulate[i] = grad

        end
    end

    ΔB = sum(gradient_accumulate)
    ΔB ./= nt

    # update the bond tensor
    B_new = B_old + α * ΔB
    
    return B_new
end

function UpdateBondTensor(W::MPS, id::Int, direction::String, product_states::Vector{PState}, 
    LE::Matrix, RE::Matrix, cstore; α::Float64, χ_max=nothing, cutoff=nothing, 
    params::Dict=LocalParameters)

    """Function to apply gradient descent to a bond tensor"""
    d = params[:local_dimension]
    N_train = length(product_states) # number of training samples
    N = length(W) # number of sites

    if direction == "forward"

        left_site = W[id]
        right_site = W[id+1]

        B_old = left_site * right_site
        B_new = GradientDescent(B_old, LE, RE, product_states; id=id, α=α, direction=direction)

        if cost_after > cost_before
            B_new = B_old
            cstore[1, id] = cost_before
        else
            cstore[1, id] = cost_after
        end

        # SVD B_new back into MPS tensors
        
        if d == 2
            left_site_idx = findindex(B_new, "S=1/2,Site,n=$id") # retain left site physical index
        elseif d == 3
            left_site_idx = findindex(B_new, "S=1,Site,n=$id")
        else
            error("Local dimension invalid.")
        end

        if id == 1
            if χ_max !== nothing && cutoff !== nothing
                U, S, V = svd(B_new, (left_site_idx); maxdim=χ_max, cutoff=cutoff)
            elseif χ_max !== nothing
                U, S, V = svd(B_new, (left_site_idx); maxdim=χ_max)
            else
                U, S, V = svd(B_new, (left_site_idx); cutoff=cutoff)
            end
        else
            bond_index = findindex(B_new, "Link,l=$(id-1)") # retain the bond dimension index
            # by convention, any specified indices are reatined on the U tensor
            if χ_max !== nothing && cutoff !== nothing
                U, S, V = svd(B_new, (bond_index, left_site_idx); maxdim=χ_max, cutoff=cutoff)
            elseif χ_max !== nothing
                U, S, V = svd(B_new, (bond_index, left_site_idx); maxdim=χ_max)
            else 
                U, S, V = svd(B_new, (bond_index, left_site_idx); cutoff=cutoff)
            end
        end

        left_site_new = U
        right_site_new = S * V

        # fix tag names
        replacetags!(left_site_new, "Link,u", "Link,l=$(id)")
        replacetags!(right_site_new, "Link,u", "Link,l=$(id)")

        #B_after_SVD = left_site_new * right_site_new
        #cost_after_SVD = LossPerBondTensor(B_after_SVD, LE, RE, product_states; id=id, direction=direction)

        # update environments
        for i=1:N_train
            if id == 1
                LE[i, 1] = left_site_new * product_states[i].pstate[id]
                #normalize!(LE[i,1])
            else
                LE[i, id] = LE[i, id-1] * left_site_new * product_states[i].pstate[id]
                #normalize!(LE[i, id])
            end
        end

    elseif direction == "backward"

        left_site = W[id-1]
        right_site = W[id]

        B_old = left_site * right_site
        B_new = GradientDescent(B_old, LE, RE, product_states; id=id, α=α, direction=direction)

        if cost_after > cost_before
            B_new = B_old
            cstore[1, id] = cost_before
        else
            cstore[1, id] = cost_after
        end

        if d == 2
            left_site_idx = findindex(B_new, "S=1/2,Site,n=$(id-1)")
        elseif d == 3
            left_site_idx = findindex(B_new, "S=1,Site,n=$(id-1)")
        else
            error("Local dimension invalid.")
        end
        label_idx = findindex(B_new, "decision")

        if id == 2
            if χ_max !== nothing && cutoff !== nothing
                U, S, V = svd(B_new, (left_site_idx, label_idx); maxdim=χ_max, cutoff=cutoff)
            elseif χ_max !== nothing
                U, S, V = svd(B_new, (left_site_idx, label_idx); maxdim=χ_max)
            else
                U, S, V = svd(B_new, (left_site_idx, label_idx); cutoff=cutoff)
            end
        else
            bond_index = findindex(B_new, "Link,l=$(id-2)")
            if χ_max !== nothing && cutoff !== nothing
                U, S, V = svd(B_new, (left_site_idx, bond_index, label_idx); maxdim=χ_max, cutoff=cutoff)
            elseif χ_max !== nothing
                U, S, V = svd(B_new, (left_site_idx, bond_index, label_idx); maxdim=χ_max)
            else
                U, S, V = svd(B_new, (left_site_idx, bond_index, label_idx); cutoff=cutoff)
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
                #normalize!(RE[i, N])
            else
                RE[i, id] = RE[i, id+1] * right_site_new * product_states[i].pstate[id]
                #normalize!(RE[i, id])
            end
        end

    else
        error("Invalid direction. Specify either forward or backward.")
    end

    return left_site_new, right_site_new, LE, RE, cstore
end

function fitMPS(X_train, y_train, X_val, y_val, params::Dict=LocalParameters;
    χ_init=5, nsweep=100, α=0.01, χ_max=nothing, cutoff=nothing, verbose=true, feature_map::Function=AngleEncoder,
    random_state=nothing, sweep_tol=nothing)

    params[:num_mps_sites] = size(X_train)[2]
    params[:feature_map] = feature_map
    params[:χ_init] = χ_init
    params[:random_state] = random_state

    num_classes = length(unique(y_train))
    params[:num_classes] = num_classes

    # encode training data

    # step one - binarise the data
    training_data_binarised = BinariseDataset(X_train; method="median")
    # step two - convert to product state
    training_states = GenerateAllProductStates(training_data_binarised, y_train, "train", sites)

    # step three - repeat for validation data
    validation_data_binarised = BinariseDataset(X_val; method="median")
    validation_states = GenerateAllProductStates(validation_data_binarised, y_val, "valid", sites)

    

    # print out parameters 
    if verbose == true
        println("Parameters: ")
        println("Feature map: $feature_map")
        println("Initial bond dimension χ_init: $χ_init")
        if sweep_tol !== nothing
            println("Sweeping tolerance: $(params[:sweep_tol])")
        end
        println("Step size α: $α")
        println("Maximum number of sweeps: $nsweep")
        if χ_max !== nothing
            println("Maximum bond dimension χ_max: $χ_max")
        end
        if cutoff !== nothing
            println("SVD cutoff: $cutoff")
        end
    end

    W = GenerateStartingMPS(; random_state=random_state)
    W = AttachLabelIndex(W; attach_site=1)

    # construct initial environment caches
    LE, RE = ConstructCaches(W, training_states; direction="forward")

    # compute initial training and validation loss
    init_train_loss = ComputeQuadraticCost(W, training_states)
    init_valid_loss = ComputeQuadraticCost(W, validation_states)

    println("Initial train loss: $init_train_loss")
    println("Initial validation loss: $init_valid_loss")

    running_train_loss = init_train_loss
    running_valid_loss = init_valid_loss 


    for itS = (1:nsweep)
        if verbose == true
            println("Forward Sweep L -> R ($itS/$nsweep)")
        end

        for j=1:(length(sites)-1)
            W[j], W[j+1], LE, RE, cstore = UpdateBondTensor(W, j, "forward", training_states, LE, RE, cstore; α=α, χ_max=χ_max, cutoff=cutoff)
        end

        # finished forward sweep, reset cache and begin backward sweep
        LE, RE = ConstructCaches(W, training_states; direction="backward")

        if verbose == true
            println("Backward Sweep R -> L ($itS/$nsweep)")
        end

        for j=(length(sites)):-1:2
            W[j-1], W[j], LE, RE, cstore = UpdateBondTensor(W, j, "backward", training_states, LE, RE, cstore; α=α, χ_max=χ_max, cutoff=cutoff)
        end

        LE, RE = ConstructCaches(W, training_states, direction="forward")

        # compute new cost
        train_loss, train_acc = LossAndAccDataset(W, training_states)
        if verbose == true
            println("Training loss after Sweep $itS: $train_loss | Training Acc: $train_acc")
        end

        ΔC = running_loss - train_loss
        
       
        if isVal
            println("ΔC val after sweep $itS: $ΔC_val")
        end
        println("ΔC train after sweep $itS: $ΔC")


        # check for early stopping if a tolerance is provided
        if sweep_tol !== nothing

            if isVal
                
                # use the validatiopn set to decide early stopping
                if abs(ΔC_val) < sweep_tol && ΔC_val > 0

                    println("Convergence reached. ΔC Val = $ΔC_val is less than the threshold $sweep_tol)!")
                    break

                end
                
            else

                if abs(ΔC) < sweep_tol && ΔC > 0
                    # use the training set to decide early stopping
                    println("Convergence reached. ΔC train = $ΔC is less than the threshold $sweep_tol)!")
                    break
                end


            end
        end

        running_loss = train_loss
        if isVal
            running_val_loss = val_loss
        end

    end

    return W, sites

end

function ScoreMPS(W, X_test, y_test; verbose=true)
    """Return the test accuracy of the final model"""
    test_states = InitialiseProductStates(X_test, y_test, "test")
    test_loss, test_acc = LossAndAccDataset(W, test_states)
    
    if verbose == true
        println("Test loss: $test_loss | Test Accuracy: $test_acc")
    end

    return [test_loss, test_acc]
end

