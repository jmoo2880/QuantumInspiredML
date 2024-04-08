using ITensors


struct PState
    pstate::MPS
    label::Int
    id::Int
end

const PCache = Matrix{ITensor}
const PCacheCol = SubArray{ITensor, 1, PCache, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}
const Maybe{T} = Union{T, Nothing}

const timeSeriesIterable = Vector{PState}

function contract_mps_and_product_state(mps::MPS, phi::PState)
    res = 1
    for i in eachindex(mps)
        res *= mps[i] * conj(phi[i].pstate)
    end

    return res

end

function compute_loss_per_sample_and_is_correct(mps::MPS, phi::PState)
    """For a given smaple, compute the log loss and whether or not the 
    prediction is correct"""
    label = phi.label
    label_idx = findindex(mps[end], "f(x)")
    if isnothing(label_idx)
        label_idx = findindex(mps[1], "f(x)")
        y = onehot(label_idx => label + 1)
    else
        y = onehot(label_idx => label + 1)
    end

    yhat = contract_mps_and_product_state(mps, phi)

    f_n_l = yhat * y 
    loss = - log(abs2(first(f_n_l)))

    # now get the predicted label
    correct = 0

    if (argmax(abs.(vector(yhat))) - 1) == phi.label
        correct = 1
    end

    return [loss, correct]

end

function angle_encoder(x::Float64)
    """Function to convert normalised time series to an angle encoding"""
    @assert x <= 1.0 && x >= 0.0 "Data points must be rescaled between 1 and 0 before using the angle encoding"
    s1 = exp(1im * (3π/2) * x) * cospi(0.5 * x)
    s2 = exp(-1im * (3π/2) * x) * sinpi(0.5 * x)
    return [s1, s2]
end

function normalised_data_to_product_state(sample::Vector, site_indices::Vector{Index{Int64}})
    @assert length(sample) == length(site_indices)
    product_state = MPS([ITensor(angle_encoder(sample[i]), site_indices[i]) for i in eachindex(site_indices)])
    return product_state
end

function generate_all_product_states(X_normalised::Matrix, y::Vector{Int64},
    site_indices::Vector{Index{Int64}})

    @assert all((0 .<= X_normalised) .& (X_normalised .<= 1))
    num_samples = size(X_normalised)[1]
    all_product_states = timeSeriesIterable(undef, num_samples)

    for i in eachindex(all_product_states)
        sample_pstate = normalised_data_to_product_state(X_normalised[i, :], site_indices)
        sample_label = y[i]
        product_state = PState(sample_pstate, sample_label, i)
        all_product_states[i] = product_state
    end

    return all_product_states

end





