using StatsBase
using Random
using Plots
using Plots.PlotMeasures
using DelimitedFiles
using HDF5





function load_splits_txt(train_set_location::String, val_set_location::String, 
    test_set_location::String)
    """As per typical UCR formatting, assume labels in first column, followed by data"""
    # do checks
    train_data = readdlm(train_set_location)
    val_data = readdlm(val_set_location)
    test_data = readdlm(test_set_location)

    X_train = train_data[:, 2:end]
    y_train = Int.(train_data[:, 1])

    X_val = val_data[:, 2:end]
    y_val = Int.(val_data[:, 1])

    X_test = test_data[:, 2:end]
    y_test = Int.(test_data[:, 1])

    # recombine val and train into train

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

end

function generate_training_data(samples_per_class::Int, data_pts::Int=5)

    class_A_samples = zeros(samples_per_class, data_pts)
    class_B_samples = ones(samples_per_class, data_pts)
    all_samples = vcat(class_A_samples, class_B_samples)
    all_labels = Int.(vcat(zeros(size(class_A_samples)[1]), ones(size(class_B_samples)[1])))

    shuffle_idxs = shuffle(1:samples_per_class*2)


    return all_samples[shuffle_idxs, :], all_labels[shuffle_idxs]

end

function generate_sine(n, amplitude=1.0, frequency=1.0)
    t = range(0, 2π, n)
    phase = rand(Uniform(0, 2π)) # randomise the phase
    #amplitude = rand(Uniform(0.1, 1.0))
    return amplitude .* sin.(frequency .* t .+ phase) .+ 0.2 .* randn(n)
end

function generate_gnoise(n, scale=1)
    return randn(n) .* scale
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

function plot_training_summary(info::Dict)
    """Takes in the dictionary of training information 
    and summary information"""
    # extract the keys
    training_loss = info["train_loss"]
    num_sweeps = length(training_loss) - 1
    time_per_sweep = info["time_taken"]

    train_accuracy = info["train_acc"]
    test_accuracy = info["test_acc"]

    train_loss = info["train_loss"]
    test_loss = info["test_loss"]

    # compute the mean time per sweep
    mean_sweep_time = mean(time_per_sweep)
    println("Mean sweep time: $mean_sweep_time (s)")

    # compute the maximum accuracy acheived across any sweep
    max_acc_sweep = argmax(test_accuracy)
    # subtract one because initial test accuracy before training included at index 1
    println("Maximum test accuracy: $(test_accuracy[max_acc_sweep]) achieved on sweep $(max_acc_sweep-1)")

    # create curves
    sweep_range = collect(0:num_sweeps)
    p1 = plot(sweep_range, train_loss, label="train loss", alpha=0.4, c=palette(:default)[1])
    scatter!(sweep_range, train_loss, alpha=0.4, label="", c=palette(:default)[1])
    plot!(sweep_range, test_loss, label="test loss", alpha=0.4, c=palette(:default)[3])
    scatter!(sweep_range, test_loss, alpha=0.4, label="", c=palette(:default)[3])
    xlabel!("Sweep")
    ylabel!("Loss")

    p2 = plot(sweep_range, train_accuracy, label="train acc", c=palette(:default)[1], alpha=0.4)
    scatter!(sweep_range, train_accuracy, label="", c=palette(:default)[1], alpha=0.4)
    plot!(sweep_range, test_accuracy, label="test acc", c=palette(:default)[3], alpha=0.4)
    scatter!(sweep_range, test_accuracy, label="", c=palette(:default)[3], alpha=0.4)
    xlabel!("Sweep")
    ylabel!("Accuracy")

    p3 = bar(collect(1:length(time_per_sweep)), time_per_sweep, label="", color=:skyblue,
        xlabel="Sweep", ylabel="Time taken (s)", title="Training time per sweep")
    
    ps = [p1, p2, p3]

    p = plot(ps..., size=(1000, 500), left_margin=5mm, bottom_margin=5mm)
    display(p)

end

struct RobustSigmoidTransform{T<:Real} <: AbstractDataTransform
    median::T
    iqr::T
    k::T
end

function robust_sigmoid(x::Real, median::Real, iqr::Real, k::Real)
    xhat = 1.0 / (1.0 + exp(-(x - median) / (iqr / k)))
    return xhat
end

function fit_scaler(::Type{RobustSigmoidTransform}, X::Matrix; k::Real=1.35)
    medianX = median(X)
    iqrX = iqr(X)
    #enforce all of these having the same type
    medianX, iqrX, k = promote(medianX, iqrX, k)
    return RobustSigmoidTransform(medianX, iqrX, k)
end

function transform_data(t::RobustSigmoidTransform, X::Matrix; range=range, minmax_output=true)
    Xt = map(x -> robust_sigmoid(x, t.median, t.iqr, t.k), X)

    if minmax_output
        Xt .-= minimum(Xt)
        Xt ./= maximum(Xt)
    end

    a,b = range
    @. Xt = (b-a) *Xt + a
    return Xt
end

function transform_data(X::Matrix; range=range, minmax_output=true)
    Xt = copy(X)

    if minmax_output
        Xt .-= minimum(Xt)
        Xt ./= maximum(Xt)
    end

    a,b = range
    @. Xt = (b-a) *Xt + a
    return Xt
end


function find_label(W::MPS; lstr="f(x)")
    l_W = lastindex(ITensors.data(W))
    posvec = [l_W, 1:(l_W-1)...]

    for pos in posvec
        label_idx = findindex(W[pos], lstr)
        if !isnothing(label_idx)
            return pos, label_idx
        end
    end
    @warn "find_label did not find a label index!"
    return nothing, nothing
end

function expand_label_index(mps::MPS; lstr="f(x)")
    "Returns a vector of MPS's, each with a different value set for the label index"

    weights_by_class = []
    pos, l_ind = find_label(mps, lstr=lstr)

    for iv in eachindval(l_ind)
        mpsc = deepcopy(mps)
        mpsc[pos] = mpsc[pos] * onehot(iv)
        push!(weights_by_class, mpsc)
    end
    
    return Vector{MPS}(weights_by_class), l_ind
end




function saveMPS(mps::MPS, path::String; id::String="W")
    """Saves an MPS as a .h5 file"""
    file = path[end-2:end] == ".h5" ? path[1:end-3] : path
    f = h5open("$file.h5", "w")
    write(f, id, mps)
    close(f)
    println("Succesfully saved mps $id at $file.h5")
end

function loadMPS(path::String; id::String="W")
"""Loads an MPS from a .h5 file. Returns and ITensor MPS."""
    file = path[end-2:end] != ".h5" ? path * ".h5" : path
    f = h5open("$file","r")
    mps = read(f, "$id", MPS)
    close(f)
    return mps
end

function get_siteinds(W::MPS)
    W1 = deepcopy(W)
    pos, label_idx = find_label(W1)
    W1[pos] *= onehot(label_idx => 1) # eliminate label index

    return siteinds(W1)
end

function loadMPS_tests(path::String; id::String="W", opts::Options=Options())

    W = loadMPS(path;id=id)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits_txt("MPS_MSE/datasets/ECG_train.txt", 
   "MPS_MSE/datasets/ECG_val.txt", "MPS_MSE/datasets/ECG_test.txt")
    X_train = vcat(X_train, X_val)
    y_train = vcat(y_train, y_val)

    sites = get_siteinds(W)

    # now let's handle the training/testing data
    # rescale using a robust sigmoid transform
    scaler = fit_scaler(RobustSigmoidTransform, X_train);
    X_train_scaled = transform_data(scaler, X_train)
    X_test_scaled = transform_data(scaler, X_test)

    # generate product states using rescaled data
    
    training_states = encode_dataset(X_train_scaled, y_train, "train", sites; opts=opts)
    testing_states = encode_dataset(X_test_scaled, y_test, "test", sites; opts=opts)


    return W, training_states, testing_states
end