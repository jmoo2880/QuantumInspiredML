using Random
using Plots
using DelimitedFiles

struct PState
    """Create a custom structure to store product state objects, 
    along with their associated label and type (i.e, train, test or valid)"""
    pstate::MPS
    label::Int
    id::Int
end

# type aliases
const PCache = Matrix{ITensor}
const PCacheCol = SubArray{ITensor, 1, PCache, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true} # for view mapping shenanigans
const Maybe{T} = Union{T,Nothing} 

const timeSeriesIterable = Vector{PState}

function angle_encoder(x::Float64) 
    """Function to convert normalised time series to an angle encoding."""
    @assert x <= 1.0 && x >= 0.0 "Data points must be rescaled between 1 and 0 before encoding using the angle encoder."
    s1 = exp(1im * (3π/2) * x) * cospi(0.5 * x)
    s2 = exp(-1im * (3π/2) * x) * sinpi(0.5 * x)
    return [s1, s2]
 
end

function normalised_data_to_product_state(sample::Vector, site_indices::Vector{Index{Int64}})
    """Function to convert a single normalised sample to a product state
    with local dimension 2, as specified by the feature map."""

    @assert length(sample) == length(site_indices) "Mismatch between number of sites and sample length."

    product_state = MPS([ITensor(angle_encoder(sample[i]), site_indices[i]) for i in eachindex(site_indices)])

    return product_state

end

function dataset_to_product_state(ts_dataset::Matrix, ts_labels::Vector{Int}, site_inds::Vector{Index{Int64}})
    """Convert ALL time series (ts) in a dataset to a vector of
    PStates"""
    dataset_shape = size(ts_dataset)

    all_product_states = Vector{PState}(undef, dataset_shape[1])
    for p in 1:length(all_product_states)
        time_series_as_product_state = normalised_data_to_product_state(ts_dataset[p, :], site_inds)
        time_series_label = ts_labels[p]
        product_state = PState(time_series_as_product_state, time_series_label, p)
        all_product_states[p] = product_state
    end

    return all_product_states
end

function generate_toy_timeseries(time_series_length::Int, total_dataset_size::Int, 
    train_split=0.7; random_state=42, plot_examples=false)
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
    A1, f1, sigma1 = 1.0, 1.0, 0.0 # Class 0
    A2, f2, sigma2 = 1.0, 6.0, 0.0 # Class 1

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
        class_0_idxs = findall(x -> x.== 0, y_train)[1:3] # select subset of 5 samples
        class_1_idxs = findall(x -> x.== 1, y_train)[1:3]
        p0 = plot(X_train[class_0_idxs, :]', xlabel="Time", ylabel="x", title="Class 0 Samples (Unscaled)", 
            alpha=0.4, c=:red, label="")
        p1 = plot(X_train[class_1_idxs, :]', xlabel="Time", ylabel="x", title="Class 1 Samples (Unscaled)", 
            alpha=0.4, c=:magenta, label="")
        p = plot(p0, p1, size=(1200, 500), bottom_margin=5mm, left_margin=5mm)
        display(p)
    end

    return (X_train, y_train), (X_test, y_test)

end




