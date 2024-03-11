using ITensors
using OptimKit
using Random
using Distributions
using DelimitedFiles
using Folds
using JLD2
using StatsBase
using Plots



function GenerateSine(n, amplitude=1.0, frequency=1.0)
    t = range(0, 2π, n)
    phase = rand(Uniform(0, 2π)) # randomise the phase
    return amplitude .* sin.(frequency .* t .+ phase) .+ 0.2 .* randn(n)
end

function GenerateRandomNoise(n, scale=1)
    return randn(n) .* scale
end

function GenerateToyDataset(n, dataset_size, train_split=0.7, val_split=0.15)
    # calculate size of the splits
    train_size = floor(Int, dataset_size * train_split) # round to an integer
    val_size = floor(Int, dataset_size * val_split) # do the same for the validation set
    test_size = dataset_size - train_size - val_size # whatever remains

    # initialise structures for the datasets
    X_train = zeros(Float64, train_size, n)
    y_train = zeros(Int, train_size)

    X_val = zeros(Float64, val_size, n)
    y_val = zeros(Int, val_size)

    X_test = zeros(Float64, test_size, n)
    y_test = zeros(Int, test_size)

    function insert_data!(X, y, idx, data, label)
        X[idx, :] = data
        y[idx] = label
    end

    for i in 1:train_size
        label = rand(0:1)  # Randomly choose between sine wave (0) and noise (1)
        data = label == 0 ? GenerateSine(n) : GenerateRandomNoise(n)
        insert_data!(X_train, y_train, i, data, label)
    end

    for i in 1:val_size
        label = rand(0:1)
        data = label == 0 ? GenerateSine(n) : GenerateRandomNoise(n)
        insert_data!(X_val, y_val, i, data, label)
    end

    for i in 1:test_size
        label = rand(0:1)
        data = label == 0 ? GenerateSine(n) : GenerateRandomNoise(n)
        insert_data!(X_test, y_test, i, data, label)
    end

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

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

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

end

function SliceMPS(W::MPS)
    """Gets the label index of the MPS and slices according to the number of classes (dim of the label index)"""
    """Assume one-hot encoding scheme i.e. class 0 = [1, 0], class 1 = [0, 1], etc. """
    dec_index = findindex(W[end], "f(x)")
    if dec_index == nothing
        error("Label index not found on the first site of the MPS!")
    end

    n_states = ITensors.dim(dec_index)
    states = []
    for i=1:n_states
        state = deepcopy(W)
        if !isapprox(norm(state), 0)
            normalize!(state)
        end
        decision_state = onehot(dec_index => (i))
        println("Class $(i-1) state: $(vector(decision_state))")
        state[end] *= decision_state
        normalize!(state)
        push!(states, state)
    end

    return states
end;

function entropy_von_neumann(ψ::MPS, b::Int)
    psi = deepcopy(ψ)
    s = siteinds(psi)
    orthogonalize!(psi, b) # change orthogonality center to site B
    #print(norm(psi))
    if b == 1
        _, S = svd(psi[b], (siteind(psi, b),))
    else
        _, S = svd(psi[b], (linkind(psi, b-1), s[b]))
    end
    SvN = 0.0
    for n in 1:ITensors.dim(S, 1)
        p = S[n, n]^2
        if p > 1E-12
            SvN -= p * log(p)
        end
    end

    return SvN
end;

function LoadMPS(fname::String)
    """Function to load a trained MPS"""
    f = h5open(fname, "r")
    W = read(f, "mps", MPS)

    return W

end

function SampleMPS(W::MPS; num_discrete_pts=100)
    """Function to sample from a MPS.
    Takes in an MPS corresponding to a state that overlaps with a given
    class (no label index attached)."""

    # grid the angle range 
    theta_range = collect(range(0, stop=pi/2, length=num_discrete_pts));
    states = [[cos(theta), sin(theta)] for theta in theta_range];

    mps = deepcopy(W)
    sites = siteinds(mps)
    num_sites = length(W)
    sampled_angles = Vector{Float64}(undef, num_sites)

    for i=1:num_sites
        orthogonalize!(mps, i) # shift orthogonality center to site i
        ρ = prime(mps[i], sites[i]) * dag(mps[i])
        ρ = matrix(ρ)
        # checks on the rdm
        if !isapprox(tr(ρ), 1)
            error("Reduced density matrix at site $i does not have a tr(ρ) ≈ 1")
        end

        probas = Vector{Float64}(undef, num_discrete_pts)
        for j=1:length(states)
            psi = states[j]
            expect = psi' * ρ * psi
            probas[j] = expect
        end

        # normalise the probabilities
        probas_normalised = probas ./ sum(probas)
        # ensure normalised
        if !isapprox(sum(probas_normalised), 1)
            error("Probabilities not normalised!")
        end

        # create a categorical distribution with the normalised probabilities
        dist = Categorical(probas_normalised)
        sampled_index = rand(dist)
    
        sampled_angles[i] = theta_range[sampled_index]

        # now make the measurement at the site
        sampled_state = states[sampled_index]

        # make the projector |x><x|
        site_projector_matrix = sampled_state * sampled_state'
        site_projector_operator = op(site_projector_matrix, sites[i])
        site_before_measure = deepcopy(mps[i])

        # apply single site mpo
        site_after_measure = site_before_measure * site_projector_operator
        noprime!(site_after_measure)

        # add measured site back into the mps
        mps[i] = site_after_measure
        normalize!(mps)
    end

    # convert angles back to time-series values

    sampled_time_series_values = (2 .* sampled_angles) ./ π

    return sampled_time_series_values

end

function MPSMonteCarlo(W::MPS, num_trials::Int; num_discrete_pts::Int=100)

    all_samples = Matrix{Float64}(undef, num_trials, length(W))

    # each threads gets its own local RNG to ensure no correlation between the RN in each thread
    #rngs = [MersenneTwister(i) for i in 1: Threads.nthreads()];
    
    Threads.@threads for trial=1:num_trials
        
        s = SampleMPS(W; num_discrete_pts=num_discrete_pts)
        all_samples[trial, :] = s
    end

    mean_values = mean(all_samples, dims=1)
    std_values = std(all_samples, dims=1)

    return mean_values, std_values
        
end