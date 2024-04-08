using ITensors
using OptimKit
using Random
using Zygote
include("utils.jl")
using Normalization

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

    d_yhat_dW = conj(ps[lid]) * conj(ps[rid]) # phi tilde 

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

function loss_and_grad_per_sample(BT::ITensor, LE::Matrix, RE::Matrix, product_state::PState, 
    lid::Int, rid::Int)
    
    # where yhat is raw overlap
    yhat, phi_tilde = compute_yhat_and_derivative(BT, LE, RE, product_state, lid, rid)

    y = product_state.label 
    predicted_proba = norm(yhat[])^2

    loss = - [y * log(predicted_proba) + (1-y) * log(1 - predicted_proba)]

    return loss

end

s = siteinds("S=1/2", 10)
mps = randomMPS(ComplexF64, s; linkdims=5)
training_samples = rand(10, 10)
labels = rand([0, 1], 10)
ps = dataset_to_product_state(training_samples, labels, s)
LE, RE = construct_caches(mps, ps; going_left=false)
BT = mps[1] * mps[2]

C_index = Index(2, "C")
bt_real = real(BT)
bt_imag = imag(BT)

bt_real_index_tensor = ITensor([1; 0], C_index)
bt_real *= bt_real_index_tensor
bt_imag_index_tensor = ITensor([0; 1], C_index)
bt_imag *= bt_imag_index_tensor

# combined
bt_combined_real_imag = bt_real + bt_imag

yhat, deriv = compute_yhat_and_derivative(bt_combined_real_imag, LE, RE, ps[1], 1, 2)
loss_and_grad_per_sample(bt_combined_real_imag, LE, RE, ps[1], 1, 2)

l = x -> loss_and_grad_per_sample(x, LE, RE, ps[1], 1, 2)