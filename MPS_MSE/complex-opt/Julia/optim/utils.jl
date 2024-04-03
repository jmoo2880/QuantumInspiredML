using StatsBase
using Random
using Plots
using ITensors
using DelimitedFiles
using HDF5

function LoadSplitsFromTextFile(train_set_location::String, val_set_location::String, 
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

function GenerateSine(n, amplitude=1.0, frequency=1.0)
    t = range(0, 2π, n)
    phase = rand(Uniform(0, 2π)) # randomise the phase
    #amplitude = rand(Uniform(0.1, 1.0))
    return amplitude .* sin.(frequency .* t .+ phase) .+ 0.2 .* randn(n)
end

function GenerateRandomNoise(n, scale=1)
    return randn(n) .* scale
end

function GenerateToyDataset(n, dataset_size, train_split=0.7)
    # calculate size of the splits
    train_size = floor(Int, dataset_size * train_split) # round to an integer
    #val_size = floor(Int, dataset_size * val_split) # do the same for the validation set
    test_size = dataset_size - train_size

    # initialise structures for the datasets
    X_train = zeros(Float64, train_size, n)
    y_train = zeros(Int, train_size)

    #X_val = zeros(Float64, val_size, n)
    #y_val = zeros(Int, val_size)

    X_test = zeros(Float64, test_size, n)
    y_test = zeros(Int, test_size)

    function insert_data!(X, y, idx, data, label)
        X[idx, :] = data
        y[idx] = label
    end

    for i in 1:train_size
        label = rand(0:1)  # Randomly choose between sine wave (0) and noise (1)
        data = label == 0 ? GenerateSine(n, 1.0, 2.0) : GenerateSine(n, 1.0, 5.0)
        insert_data!(X_train, y_train, i, data, label)
    end

    # for i in 1:val_size
    #     label = rand(0:1)
    #     data = label == 0 ? GenerateSine(n) : GenerateRandomNoise(n)
    #     insert_data!(X_val, y_val, i, data, label)
    # end

    for i in 1:test_size
        label = rand(0:1)
        data = label == 0 ? GenerateSine(n, 1.0, 2.0) : GenerateSine(n, 1.0, 5.0)
        insert_data!(X_test, y_test, i, data, label)
    end

    return (X_train, y_train), (X_test, y_test)

end

using Plots.PlotMeasures
function PlotTrainingSummary(info::Dict)
    """Takes in the dictionary of training information 
    and summary information"""
    # extract the keys
    training_loss = info["train_loss"]
    num_sweeps = length(training_loss) - 1
    time_per_sweep = info["time_taken"]

    train_accuracy = info["train_acc"]
    test_accuracy = info["test_acc"]
    validation_accuracy = info["val_acc"]

    train_loss = info["train_loss"]
    test_loss = info["test_loss"]
    validation_loss = info["val_loss"]

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
    plot!(sweep_range, validation_loss, label="valid loss", alpha=0.4, c=palette(:default)[2])
    scatter!(sweep_range, validation_loss, alpha=0.4, label="", c=palette(:default)[2])
    plot!(sweep_range, test_loss, label="test loss", alpha=0.4, c=palette(:default)[3])
    scatter!(sweep_range, test_loss, alpha=0.4, label="", c=palette(:default)[3])
    xlabel!("Sweep")
    ylabel!("Loss")

    p2 = plot(sweep_range, train_accuracy, label="train acc", c=palette(:default)[1], alpha=0.4)
    scatter!(sweep_range, train_accuracy, label="", c=palette(:default)[1], alpha=0.4)
    plot!(sweep_range, validation_accuracy, label="valid acc", c=palette(:default)[2], alpha=0.4)
    scatter!(sweep_range, validation_accuracy, label="", c=palette(:default)[2], alpha=0.4)
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
    positive::Bool

    function RobustSigmoidTransform(median::T, iqr::T, k::T, positive=true) where T<:Real
        new{T}(median, iqr, k, positive)
    end
end

function robust_sigmoid(x::Real, median::Real, iqr::Real, k::Real, positive::Bool)
    xhat = 1.0 / (1.0 + exp(-(x - median) / (iqr / k)))
    if !positive
        xhat = 2*xhat - 1
    end
    return xhat
end

function fitScaler(::Type{RobustSigmoidTransform}, X::Matrix; k::Real=1.35, positive::Bool=true)
    medianX = median(X)
    iqrX = iqr(X)
    return RobustSigmoidTransform(medianX, iqrX, k, positive)
end

function transformData(t::RobustSigmoidTransform, X::Matrix)
    return map(x -> robust_sigmoid(x, t.median, t.iqr, t.k, t.positive), X)
end

# New SigmoidTransform
struct SigmoidTransform <: AbstractDataTransform
    positive::Bool
end

function sigmoid(x::Real, positive::Bool)
    xhat = 1.0 / (1.0 + exp(-x))
    if !positive
        xhat = 2*xhat - 1
    end
    return xhat
end

function fitScaler(::Type{SigmoidTransform}, X::Matrix; positive::Bool=true)
    return SigmoidTransform(positive)
end

function transformData(t::SigmoidTransform, X::Matrix)
    return map(x -> sigmoid(x, t.positive), X)
end;

function save_mps_as_h5(mps::MPS, id::String, out::String)
    """Saves an MPS as a .h5 file"""
    f = h5open("$out.h5", "w")
    write(f, id, mps)
    close(f)
    println("Succesfully saved mps $id at $out.")
end

function load_mps_from_h5(file::String, id::String)
    """Loads an MPS from a .h5 file. Returns and ITensor MPS."""
    f = h5open("$file","r")
    mps_loaded = read(f, "$id", MPS)
    return mps_loaded
end

function feature_map(x::Float64)
    s1 = exp(1im * (3π/2) * x) * cospi(0.5 * x)
    s2 = exp(-1im * (2π/2) * x) * sinpi(0.5 * x)
    return [s1, s2]
end

function generate_sample(mps_original::MPS; dx=0.1)
    mps = deepcopy(mps_original)
    s = siteinds(mps)
    xs = 0.0:dx:1.0

    x_samples = Vector{Float64}(undef, length(mps))
    for i in eachindex(mps)
        orthogonalize!(mps, i)
        ρ = prime(mps[i], s[i]) * dag(mps[i])
        # check properties
        if !isapprox(real(tr(ρ)), 1.0; atol=1E-3) @warn "Trace of RDM ρ at site $i not equal to 1 ($(abs(tr(ρ))))." end
        if !isequal(ρ.tensor, adjoint(ρ).tensor) @warn "RDM at site $i not Hermitian." end
        ρ_m = matrix(ρ)
        probs = [real(feature_map(x)' * ρ_m * feature_map(x)) for x in xs];
        probs_normed = probs ./ sum(probs)
        cdf = cumsum(probs_normed)
        r = rand()
        cdf_selected_index = findfirst(x -> x > r, cdf)
        selected_x = xs[cdf_selected_index]
        x_samples[i] = selected_x
        selected_state = feature_map(selected_x)
        site_measured_state = ITensor(selected_state, s[i])
        m = MPS(1)
        m[1] = site_measured_state
        # make into a projector
        site_projector = projector(m)
        # make into projector operator
        site_projector_operator = op(matrix(site_projector[1]), s[i])
        mps[i] *= site_projector_operator
        noprime!(mps[i])
        normalize!(mps)

    end

    return x_samples

end

function interpolate_sample(mps::MPS, sample::Vector, start_site::Int; dx=0.1)
    """Assumes forward sequential interpolation for now, i.e., 
    is sample corresponds to sites 1:50, then interpolate sites 51 to 100.
    Start site is the starting point IN THE MPS (last site in sample + 1).
    Return a new mps conditioned on the sample."""
    # check whether the length of the mps is > sample
    @assert length(mps) > length(sample) "Sample is longer than MPS."
    s = siteinds(mps)
    # check mps is normalised
    @assert isapprox(norm(mps), 1.0; atol=1E-3) "MPS is not normalised!"
    for i in 1:(start_site-1)
        # condition each site in the mps on the sample values
        # start by getting the state corresponding to the site
        site_state = ITensor(feature_map(sample[i]), s[i])
        # construct projector, need to use 1 site mps to make one site projector 
        m = MPS(1)
        m[1] = site_state
        site_projector = projector(m)
        # turn projector into a local MPO
        site_projector_operator = op(matrix(site_projector[1]), s[i])
        # check properties are valid for operator
        # if !isapprox(abs(tr(site_projector_operator)), 0.0; atol=1E-3) 
        #     @warn "Projector at site $i does not have tr(ρ) ≈ 1"
        # end
        # if !isequal(adjoint(site_projector_operator).tensor, site_projector_operator.tensor)
        #     @warn "Projector at site $i not hermitian."
        # end
        # apply to the site of interest
        orthogonalize!(mps, i)
        mps[i] *= site_projector_operator
        noprime!(mps[i])
        # normalise 
        normalize!(mps)
    end

    # now generate the remaining sites by sampling from the conditional distribution 
    xs = 0.0:dx:1.0
    samples = []
    for i in start_site:length(mps)
        orthogonalize!(mps, i)
        # get reduced density matrix
        ρ = prime(mps[i], s[i]) * dag(mps[i])
        # check ρ properties
        if !isapprox(real(tr(ρ)), 1.0; atol=1E-3) @warn "Trace of RDM ρ at site $i not equal to 1 ($(abs(tr(ρ))))." end
        if !isequal(ρ.tensor, adjoint(ρ).tensor) @warn "RDM at site $i not Hermitian." end
        ρ_m = matrix(ρ)
        # compute probability of state x at site i
        probs = [real(feature_map(x)' * ρ_m * feature_map(x)) for x in xs];
        probs_normed = probs ./ sum(probs)
        cdf = cumsum(probs_normed)
        r = rand()
        cdf_selected_index = findfirst(x -> x > r, cdf)
        selected_x = xs[cdf_selected_index]
        push!(samples, selected_x)

        # now condition the MPS on the sampled state at site i
        selected_site_state = ITensor(feature_map(selected_x), s[i])
        m = MPS(1)
        m[1] = selected_site_state
        site_projector = projector(m)
        # turn projector into a local MPO
        site_projector_operator = op(matrix(site_projector[1]), s[i])
        mps[i] *= site_projector_operator
        noprime!(mps[i])
        normalize!(mps)
    end

    return samples

end

function interpolate_between_sites(mps::MPS, sample::Vector, interpolate_sites::Tuple; dx=0.1)
    """Takes in an MPS and time series sample,
    and conditions on all known values, then interpolates the missing parts.
    The variable interpolate_sites is a tuple (start, end) inclusive."""
    @assert length(mps) > length(sample) "MPS is shorter than the time series sample!"
    # check that interpolation sites within range
    start_interp_site, end_interp_site = interpolate_sites[1], interpolate_sites[2]
    

end
