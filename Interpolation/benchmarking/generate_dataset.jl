using Plots
using Random, Distributions
using JLD2

function generate_nonstat_periodic(T::Int, A::Float64, f::Float64, sl::Float64, 
    sigma::Float64, rand_phase::Bool=true)
    """Generate nonstationary periodic time series
    with linear trend. Randomised phase offset by default. 
    
    Params:
    T: time series length (number of samples)
    A: amplitude
    f: frequency (π)
    sl: slope coefficient for linear trend
    sigma: amplitude of i.i.d. gaussian noise. 
    rand_phase: enable randomised phase offset in range ϕ ∈ [0, 2π]
    """
    phase = 0
    if rand_phase
        phase = rand(Uniform(0, 2π))
    end
    trend = sl .* collect(1:1:T)
    noise = sigma .* rand(T)
    sinusoid = A .* sin.(2 .* range(0, f*π, length=T) .+ phase)
    return sinusoid + noise + trend
end

function generate_nonstat_periodic_with_linear_noise(T::Int, A::Float64, f::Float64, sl::Float64, 
    low_var=0.0001, high_var=0.15, rand_phase::Bool=true)
    """Linearly interpolates between low variance and high variance.
    Generates a sinuosid with linear trend + linearly increasing noise."""
    phase = 0
    if rand_phase
        phase = rand(Uniform(0, 2π))
    end
    trend = sl .* collect(1:1:T)
    # generate noise with linearly increasing variance 
    noise_variance = low_var .+ (high_var - low_var) * (collect(1:1:T) .- collect(1:1:T)[1]) / (collect(1:1:T)[end] - collect(1:1:T)[1])
    noise = rand.(Normal.(0, sqrt.(noise_variance)))
    sinusoid = A .* sin.(2 .* range(0, f*π, length=T) .+ phase)

    return sinusoid + noise + trend
end

function generate_datasets(total_samples::Int, train_ratio::Float64; T = 100)
    train_size = Int.(total_samples * train_ratio)
    samples_per_class = Int(total_samples/2)
    # linear increasing 
    tr_class_0 = Matrix{Float64}(undef, samples_per_class, T)
    tr_class_1 = Matrix{Float64}(undef, samples_per_class, T)

    for i in 1:samples_per_class
        tr_class_0[i, :] = generate_nonstat_periodic_with_linear_noise(T, 1.0, 10.0, 0.05, 0.001, 0.15)
        tr_class_1[i, :] = generate_nonstat_periodic_with_linear_noise(T, 1.0, 10.0, -0.05, 0.001, 0.15)
    end

    X_all = vcat(tr_class_0, tr_class_1)
    y_all = Int.(vcat(zeros(samples_per_class), ones(samples_per_class)))
    shuffled_idxs = shuffle(1:size(X_all, 1))
    X_all = X_all[shuffled_idxs, :]
    y_all = y_all[shuffled_idxs]
    # split into train and test set
    X_train = X_all[1:train_size, :]
    y_train = y_all[1:train_size]
    X_test = X_all[(train_size + 1):end, :]
    y_test = y_all[(train_size + 1):end]
    

    return (X_train, y_train), (X_test, y_test)

end

