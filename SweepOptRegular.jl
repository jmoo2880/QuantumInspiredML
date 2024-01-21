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

function AngleEncoder(x::Float64) 
    if x <= 1 && x >= 0
        return [cos(π/2 * x), sin(π/2 * x)]
    else
        println("Data points must be rescaled between 1 and 0 before encoding using the half angle encoder.")
    end
end

function QuditAngleEncoder(x::Float64)
    if x <= 1 && x >= -1

        ϕ = atan(x)
        θ = acos(x)

        dim1 = (cos(θ) * (cos(ϕ) + sin(ϕ)))/(sqrt(2))
        dim2 = (cos(θ) * (cos(ϕ) - sin(ϕ)))/(sqrt(2))
        dim3 = sin(θ)

        return [dim1, dim2, dim3]
    else
        println("Data points must be rescaled between -1 and 1 before encoding using the qudit angle encoder.")
    end
end

function LinearEncoder(x::Float64)
    return [sqrt(1-x), sqrt(x)]
end

##################################
global LocalParameters = Dict(
    :feature_map => AngleEncoder, 
    :num_mps_sites => 0, # number of physical sites/indices in the MPS (equal to the number of data pts. in the time-series)
    :local_dimension => 2, # local dimension of the Hilbert space
    :χ_init => 5, # initial weight MPS bond dimension
    :random_state => 42, # random seed to generate the initial weight MPS
    :num_classes => 2, # number of classes to distinguish
)
#################################

function GenerateProductState(normalised_sample::Vector, params::Dict=LocalParameters)

    """Convert a single normalised sample to a product state with local dimension defined by the feature map."""

    N_sites = params[:num_mps_sites]
    site_inds = params[:site_inds]
    feature_map = params[:feature_map]
    d = params[:local_dimension]

    product_state = MPS(site_inds; linkdims=1)

    if N_sites !== size(normalised_sample)[1]
        error("Number of MPS sites (N = $N_sites) does not match the length of the time-series sample (N = $(size(normalised_sample)[1])).")
    end

    for j=1:N_sites
        T = ITensor(site_inds[j])
        mapped_vals = feature_map(normalised_sample[j])
        if length(mapped_vals) == d
            if length(mapped_vals) == 2
                up_val, down_val = mapped_vals
                T[1] = up_val
                T[2] = down_val
            elseif length(mapped_vals) == 3
                up_val, zero_val, down_val = mapped_vals
                T[1] = up_val
                T[2] = zero_val
                T[3] = down_val
            else
                error("Feature map output dimension incorrect.")
            end
        else
            error("Feature map dimension does match site indices dimension.")
        end
        product_state[j] = T
    end
    return product_state
end

struct PState
    """ Create a custom structure to store product state objects, along with their associated label and type (either train/test/validation)"""
    pstate::MPS
    label::Int
    type::String
end


function InitialiseProductStates(X::Matrix, y::Matrix, type::String)
    """Convert an entire dataset to corresponding product states."""
    if type == "train" 
        println("Initialising training states.")
    elseif type == "test"
        println("Initialising testing states.")
    elseif type == "valid"
        println("Initialising validation states.")
    else
        error("Invalid dataset type. Must be train, test or valid.")
    end

    ns = size(X)[1] # number of samples 
    ϕs = Vector{PState}(undef, ns) # store all product states in a vector

    Threads.@threads for i=1:ns
        sample_pstate = GenerateProductState(X[i, :])
        sample_label = y[i]
        ps = PState(sample_pstate, sample_label, type)
        ϕs[i] = ps
    end

    return ϕs
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

    normalize!(W)

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

function GetYhatRange(W::MPS, ϕs::Vector{PState})
    pstates = map(x -> x, ϕs)

    f = x -> norm(ContractMPSAndProdState(W, x))

    y_hat_min = Folds.minimum(f, collect(pstates))
    y_hat_max = Folds.maximum(f, collect(pstates))

    return y_hat_min, y_hat_max
end

function DivideMPS(W::MPS, factor)
    for t in W
        # for each tensor in MPS
        t = ITensors.tensor(t)
        for i in eachindex(t)
            t[i] = t[i] / factor
        end
    end
end


function RescaleMPS(W::MPS, ϕs::Vector{PState}; verbose=false)
    """Function to check the output of the model and rescale if values are too large or too small"""
    min_yhat, max_yhat = GetYhatRange(W, ϕs)
    rescale = false
    if max_yhat > 1e+5
        # too large
        x = max_yhat / 1e+4
        rescale = true
        if verbose == true
            println("Max yhat:  $max_yhat. Rescaling MPS.")
        end
    elseif min_yhat < 1e-5
        # too small
        x = min_yhat / 1e-5
        rescale = true
        if verbose == true
            println("Min yhat: $min_yhat. Rescaling MPS.")
        end
    else
        if verbose == true
            println("Rescaling not required!")
        end
    end
    if rescale == true
        #l = length(W)
        #x = Float32(x^(1/l))
        DivideMPS(W, x)
    end
end


# each thread can handle its own product state sample
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

function LossPerSampleAnCorrect(W::MPS, ϕ::PState)
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

function LossPerBondTensor(B::ITensor, LE::Matrix, RE::Matrix, product_states::Vector{PState}; 
    id::Int, direction::String="forward", params::Dict=LocalParameters)

    N_train = length(product_states)
    N = params[:num_mps_sites]
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
            dP = OneHotEncodeLabel(product_states[i].label, params) - P
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

            dP = OneHotEncodeLabel(product_states[i].label, params) - P
            costs[i] = 0.5 * norm(dP)^2
        end
    end

    C = sum(costs)

    return C/N_train

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

    # update the bond tensor
    B_new = B_old + α * ΔB
    
    return B_new
end

function UpdateBondTensor(W::MPS, id::Int, direction::String, product_states::Vector{PState}, 
    LE::Matrix, RE::Matrix, cstore; α::Float64, χ_max=nothing, cutoff=nothing, 
    params::Dict=LocalParameters, verbose=false)

    """Function to apply gradient descent to a bond tensor"""
    d = params[:local_dimension]
    N_train = length(product_states) # number of training samples
    N = length(W) # number of sites

    if direction == "forward"

        left_site = W[id]
        right_site = W[id+1]

        B_old = left_site * right_site
        # compute cost function before the update
        cost_before = LossPerBondTensor(B_old, LE, RE, product_states; id=id, direction=direction)
        if verbose == true
            println("Cost before optimising $id: $cost_before")
        end
        B_new = GradientDescent(B_old, LE, RE, product_states; id=id, α=α, direction=direction)
        # compute cost after the update
        # compute loss on the updated bond tensor
        cost_after = LossPerBondTensor(B_new, LE, RE, product_states; id=id, direction=direction)

        if verbose == true
            println("Cost after optimising $id: $cost_after")
        end

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
        # compute cost function before the update
        cost_before = LossPerBondTensor(B_old, LE, RE, product_states; id=id, direction=direction)
        if verbose == true
            println("Cost before optimising: $cost_before")
        end
        B_new = GradientDescent(B_old, LE, RE, product_states; id=id, α=α, direction=direction)
        cost_after = LossPerBondTensor(B_new, LE, RE, product_states; id=id, direction=direction)

        if verbose == true
            println("Cost after optimising: $cost_after")
        end

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

        #B_after_SVD = left_site_new * right_site_new
        #cost_after_SVD = LossPerBondTensor(B_after_SVD, LE, RE, product_states; id=id, direction=direction)

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

function fitMPS(X_train, y_train, X_val=nothing, y_val=nothing, params::Dict=LocalParameters;
    χ_init=5, nsweep=100, α=0.01, χ_max=nothing, cutoff=nothing, verbose=true, feature_map::Function=AngleEncoder,
    random_state=nothing, sweep_tol=nothing)

    isVal = false
    params[:num_mps_sites] = size(X_train)[2]
    params[:feature_map] = feature_map
    params[:χ_init] = χ_init
    params[:random_state] = random_state


    test_pt = feature_map(0.1)
    if length(test_pt) == 2
        sites = siteinds("S=1/2", size(X_train)[2])
        params[:local_dimension] = 2
        params[:site_inds] = sites
    elseif length(test_pt) == 3
        sites = siteinds("S=1", size(X_train)[2])
        params[:local_dimension] = 3
        params[:site_inds] = sites
    else
        error("Invalid feature map. Hilbert space dimension is not 2 or 3.")
    end

    num_classes = length(unique(y_train))
    params[:num_classes] = num_classes

    # encode training data
    training_states = InitialiseProductStates(X_train, y_train, "train")

    if X_val !== nothing

        if y_val !== nothing 

            # validation set exists
            tol_set = "valid"
            isVal = true
            validation_states = InitialiseProductStates(X_val, y_val, "valid")

        else

            error("Validation data provided, but no labels provided.")

        end

    end

    tol_set = "train" # if validation set not provided, use train set ΔC to terminate sweeps early

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

    cstore = zeros(1, length(sites))

    init_loss, _ = LossAndAccDataset(W, training_states)

    running_loss = init_loss

    if isVal
        # get initial validation loss
        init_val_loss, _ = LossAndAccDataset(W, validation_states)
        running_val_loss = init_val_loss
    end

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

        if isVal

            val_loss, val_acc = LossAndAccDataset(W, validation_states)
            println("Val loss after Sweep $itS: $val_loss | Validation Acc: $val_acc")
            ΔC_val = running_val_loss - val_loss

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

        #RescaleMPS(W, training_states)
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



using MLBase:StratifiedKfold
function GetKFold(y::Matrix, n_folds::Int; random_state=nothing)
    """Returns folds in the (train_idx), (val_idx) format."""
    if random_state !== nothing
        Random.seed!(random_state)
    end

    skf = StratifiedKfold(y, n_folds)
    folds = collect(skf)
    all_idxs = 1:length(y)
    all_folds = []

    for i = 1:n_folds
        train_idx = folds[i]
        val_idx = setdiff(all_idxs, train_idx)
        push!(all_folds, (train_idx, val_idx))
    end

    return all_folds

end

function CreateStratifiedKFold(y::Matrix, n_folds::Int; random_state=nothing)
    """ Return the indices for stratified K-fold. Folds are made such that class distributions are preserved. """

    if random_state !== nothing
        rng = MersenneTwister(random_state)
    else
        rng = MersenneTwister()
    end

    folds = [Int[] for _ in 1:n_folds]
    for label in unique(y)
        label_idxs = findall(isequal(label), y[:, 1])
        # shuffle label idxs
        shuffle!(rng, label_idxs)

        # distribute the indices across
        for (idx, fold_idx) in enumerate(label_idxs)
            fold_number = (idx % n_folds) + 1
            push!(folds[fold_number], fold_idx)
        end
    end
    
    return folds
end

function CrossValidate(X::Matrix, y::Matrix, params::Dict, n_folds::Int)
    folds = CreateStratifiedKFold(y, n_folds)
    scores = []

    for i in 1:n_folds
        val_idx = folds[i] # select validation set from the n_folds
        train_idx = vcat(folds[1:i-1]..., folds[i+1:end]...) # use the remaining (n_folds - 1) indices for training 

        X_train, y_train = X[train_idx, :], y[train_idx, :]
        X_val, y_val = X[val_idx, :], y[val_idx, :]

        # train the MPS classifier
        W, sites = fitMPS(X_train, y_train, X_val, y_val; params...)

        # compute validation accuracy 
        # convert validation data to product states
        all_test_states = InitialiseProductStates(X_val, y_val, "valid")
        val_loss, val_acc = LossAndAccDataset(W, all_test_states)
        
        #val_acc = TestAccuracy(W, X_val, y_val, sites, params[:feature_map])

        push!(scores, val_acc)
    end
    println(mean(scores))
    return mean(scores)

end


function GridSearchCV(X::Matrix, y::Matrix, param_dists::Dict, n_folds::Int)

    results = Vector{Tuple{Dict{Symbol,Any}, Float64}}()
   
    subset_dicts = [Dict(:χ_init => χ_init, :nsweep => nsweep, :random_state => random_state, :χ_max => χ_max, :α => α,
        :cutoff => cutoff, :feature_map => feature_map, :sweep_tol => sweep_tol) for χ_init in param_dists[:χ_init] for nsweep in param_dists[:nsweep] 
        for random_state in param_dists[:random_state] for cutoff in param_dists[:cutoff]
            for feature_map in param_dists[:feature_map] for χ_max in param_dists[:χ_max] 
                for sweep_tol in param_dists[:sweep_tol] for α in param_dists[:α]]

    num_combinations = length(subset_dicts)
    for (index, subset) in enumerate(subset_dicts)
        println("Combination: $index/$num_combinations.")
        println(subset)
        acc = CrossValidate(X, y, subset, n_folds)

        push!(results, (subset, acc))
    end

    return results
end

# Additional utilities
function TwoPointCorrelation(W::MPS, class_label::Int, operator::String = "Sz") 

    """Function to return the two point correlation between each MPS site."""

    # get the local hilbert space dimension
    d = ITensors.dim(findindex(W[1], "Site"))

    if d == 2
        if operator == "Sz"
            op = 1/2 .* [1 0; 0 -1]
        elseif operator == "Sx"
            op = 1/2 .* [0 1; 1 0]
        else
            error("Invalid operator. Can either be Sz or Sx.")
        end
    elseif d == 3
        if operator == "Sz"
            op = 1/sqrt(2) .* [0 1 0; 1 0 1; 0 1 0]
        elseif operator == "Sx"
            op = 1/sqrt(2) .* [1 0 0; 0 0 0; 0 0 -1]
        else
            error("Invalid operator. Can either be Sz or Sx.")
        end
    else
        error("Unable to obtain the local Hilbert space dimension. Check the MPS indices.")
    end
    
    normalize!(W)
    ψ = deepcopy(W)
    decision_idx = findindex(ψ[1], "decision") # the label index is assumed to be attached to the first site
    decision_state = onehot(decision_idx => (class_label+1)) # one-hot encoded indexing starts from zero
    ψ[1] *= decision_state # filter out the MPS state corresponding to the target class
    normalize!(ψ)

    corr = correlation_matrix(ψ, op, op)

    return corr
end


function EntanglementEntropyProfile(W::MPS, class_label::Int)

    """Function to compute the entanglement entropy as a function of site/bond index."""
    # isolate the MPS weights for the class of interest
    normalize!(W)
    ψ = deepcopy(W)
    decision_idx = findindex(ψ[1], "decision") # assumes the label index is attached to the first site. 
    decision_state = onehot(decision_idx => (class_label + 1))
    ψ[1] *= decision_state
    normalize!(ψ)
    entropy = Vector{Float64}(undef, (length(W)-2))
    # for MPS of length N we can compute the entanglement entropy of a bipartition of the system into a region "A"
    # which consists of sites 1, 2, ..., b and region B of sites b+1, b+2,...N
    # b tracks the orthogonality center
    for b = (2:length(ψ)-1)
        orthogonalize!(ψ, b) # shift the orthogonality center to the site b of the MPS 
        U, S, V = svd(ψ[b], (linkind(ψ, b-1), siteind(ψ, b)))
        SvN = 0.0
        for n = 1:ITensors.dim(S, 1)
            p = S[n, n]^2
            SvN -= p * log2(p)
        end
        entropy[b-1] = SvN
    end
    return entropy
end


function SliceMPS(W::MPS, class_label::Int)
    """General function to slice the MPS and return the state corresponding to a specific class label."""
    ψ = deepcopy(W)
    normalize!(ψ)
    decision_idx = findindex(ψ[1], "decision")
    decision_state = onehot(decision_idx => (class_label + 1))
    ψ[1] *= decision_idx
    normalize!(ψ)

    return ψ
end

function DifferenceVector(ψ1::MPS, ψ2::MPS)
    """Computes the difference (residual) vector between two state vectors, ψ1 and ψ2"""
    # We subtract the component of class 0 that is parallel to class 1, leaving an orthogonal vector (state)
    # this is the same as projecting state 2 onto state 1 and subtracting to get the residual vector 
    # which component of ψ2 is most different from ψ1

    normalize!(ψ1)
    normalize!(ψ2)

    Δ = ψ2 - (inner(ψ1, ψ2) * ψ1)

    normalize!(Δ)

    return Δ
end

function ComputeOverlapMatrix(X::Matrix, y::Matrix, params::Dict=LocalParameters; return_dmat=true)
    """Compute the overlap between each product state in the dataset after being mapped by the feature map to Hilbert space."""
    """The returned matrix can be interpreted as a dissimilarity matrix (1 - overlap)."""

    feature_map = params[:feature_map]
    d = params[:local_dimension]
    
    if d == 2
        sites = siteinds("S=1/2", size(X)[2])
    elseif d == 3
        sites = siteinds("S=1", size(X)[2])
    else
        error("Local dimension not found. Check the feature map used.")
    end

    overlap_matrix = Matrix{Any}(undef, size(X)[1], size(X)[1])
    product_states = InitialiseProductStates(X, y, "train")
    for i=1:length(product_states)
        for j=1:length(product_states)
        contract = inner(product_states[i].pstate, product_states[j].pstate)
        overlap_matrix[i, j] = contract
        end
    end

    if return_dmat == true
        dmat = gramd2dmat(convert(Matrix{Float64}, overlap_matrix))
        return dmat
    else
        return overlap_matrix
    end
end


function InspectOverlap(X::Matrix, y::Matrix, W::MPS, params::Dict=LocalParameters) 
    """Returns dataframe"""
    n_classes = params[:num_classes]
    product_states = InitialiseProductStates(X, y, "test")

    contractions = Matrix{Union{Float64, Int}}(undef, size(X, 1), n_classes+2) 
    
    for i=1:size(X, 1)
        overlap = ContractMPSAndProdState(W, product_states[i])
        contractions[i, 1:n_classes] = overlap

        # get prediction based on overlaps
        contractions[i, n_classes+1] = argmax(abs.(overlap)) - 1
        # include ground truth label in last column
        contractions[i, end] = product_states[i].label 
    end

    # convert matrix to dataframe
    col_names = [Symbol("class_$i") for i=0:n_classes + 1] # double-check the indexing
    # add the prediction and ground truth label column names
    push!(col_names, :prediction, :ground_truth_label)
    df = DataFrame(contractions, col_names)

    return df
end


#sites = siteinds("S=1/2", 10)
#LocalParameters[:num_mps_sites] = 10
#LocalParameters[:site_inds] = sites
#W = GenerateStartingMPS()
#W = AttachLabelIndex(W; attach_site=1)


#Random.seed!(73)
#X = rand(Uniform(0,1), 100, 10)
#y = rand(0:1, 100, 1) # assuming binary labels 0 and 1 for simplicity

#X_test = rand(Uniform(0,1), 100, 10)
#y_test = rand(0:1, 100, 1)
#dataset_type = "train"
#ϕs = InitialiseProductStates(X, y, "trai")

#W, sites = fitMPS(X, y, X_test, y_test; α=0.01, χ_max=20, feature_map=AngleEncoder)
#loss, acc = ScoreMPS(W, X_test_normalised, y_test)

#χ_vals = [5, 10, 15, 20, 25, 30, 35, 40, 45]
#accs_all = []
#for χ in χ_vals
#    W, sites = fitMPS(X_train_normalised, y_train; sweep_tol=1E-3, α=0.01, χ_max=χ)
#    loss, acc = ScoreMPS(W, X_test_normalised, y_test)
#    push!(accs_all, acc)
#end

#p = plot(χ_vals, accs_all)
#display(p)
#print(accs_all)

#LE, RE = ConstructCaches(W, ϕs)
#B = W[1] * W[2]

#println(GradientDescent(B, LE, RE, 1, ϕs; α=0.1, direction="forward")) 

#cstore = zeros(1, length(sites))
#left_site_new, right_site_new, LE, RE, cstore = UpdateBondTensor(W, 1,"forward", ϕs, LE, RE, cstore; α=0.1)
#println(left_site_new)
#println(right_site_new)
#println(cstore)

#println(RE[4, 5])










#