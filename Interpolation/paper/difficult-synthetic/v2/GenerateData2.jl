using Plots, Plots.PlotMeasures
using Random, Distributions
using StatsBase
using JLD2

function generate_noisy_sinusoid(T::Int, noise_std::Union{Nothing, Float64},
    trend::Union{Nothing, Float64}, frequency::Union{Nothing, Float64}, 
    phase::Union{Nothing, Float64})
    # if nothing is passed in for a keyword, then random value will be used
    #rng = MersenneTwister(random_state)

    # dists over parameters of interest
    phase_dist = Uniform(0, 2π)
    noise_dist = Uniform(0.1, 0.5)
    trend_dist = Uniform(-0.3, 0.3)
    frequency_dist = Uniform(5, 10)

    # draw parameter values if nothing passed, else use fixed value
    phase_val = isnothing(phase) ? rand(phase_dist) : phase
    noise_std_val = isnothing(noise_std) ? rand(noise_dist) : noise_std
    trend_val = isnothing(trend) ? rand(trend_dist) : trend
    freq_val = isnothing(frequency) ? rand(frequency_dist) : frequency

    # log the values for inspection
    params = Dict(
        :phase => phase_val, 
        :noise => noise_std_val,
        :trend => trend_val,
        :freq => freq_val
    )

    # generate signal
    x = collect(range(0, freq_val*pi, T))
    signal = 3 .* sin.(2 * x .+ phase_val)
    signal_trend = trend_val .* x
    noise = rand(Normal(0, noise_std_val), T)

    noisy_signal_trend = signal .+ noise .+ signal_trend

    return noisy_signal_trend, params

end


function generate_dataset(tpoints::Int, num_train::Int, num_test::Int,
    noise_std::Union{Nothing, Float64}, trend::Union{Nothing, Float64}, 
    frequency::Union{Nothing, Float64}, phase::Union{Nothing, Float64}, 
    save_file::Union{Nothing, String})

    train_mat = Matrix{Float64}(undef, num_train, tpoints)
    test_mat = Matrix{Float64}(undef, num_test, tpoints)
    train_labels = Int.(zeros(num_train))
    test_labels = Int.(zeros(num_test))
    params_all_train = Vector{Dict}(undef, num_train)
    params_all_test = Vector{Dict}(undef, num_test)
    for i in 1:num_train
        signal, params = generate_noisy_sinusoid(tpoints, noise_std, trend, frequency, phase)
        train_mat[i, :] = signal
        params_all_train[i] = params
    end
    for i in 1:num_test
        signal, params = generate_noisy_sinusoid(tpoints, noise_std, trend, frequency, phase)
        test_mat[i, :] = signal
        params_all_test[i] = params
    end

    X_train = train_mat
    y_train = train_labels

    X_test = test_mat
    y_test = test_labels

    if !isnothing(save_file)
        #save as jld2
        JLD2.@save "$save_file.jld2" X_train y_train params_all_train X_test y_test params_all_test
    end

    return X_train, y_train, params_all_train, X_test, y_test, params_all_test
end

function plot_param_dists!(params::Vector{Dict})

    phase_dist = [params[i][:phase] for i in 1:length(params)]
    noise_dist = [params[i][:noise] for i in 1:length(params)]
    freq_dist = [params[i][:freq] for i in 1:length(params)]
    trend_dist = [params[i][:trend] for i in 1:length(params)]
    pal = palette(:tab10)
    h1 = histogram(phase_dist, bins=30, xlabel="Phase", title="Phase Dist", c=pal[1], label="")
    h2 = histogram(noise_dist, bins=30, xlabel="Noise", title="Noise Dist", c=pal[2], label="")
    h3 = histogram(freq_dist, bins=30, xlabel="Freq.", title="Freq Dist", c=pal[3], label="")
    h4 = histogram(trend_dist, bins=30, xlabel="m", title="Trend Dist", c=pal[4], label="")
    p = plot(h1, h2, h3, h4)
    display(p)
end

function plot_example_ts!(samples::Matrix{Float64})
    """Plot 20 random example time series"""
    plot_idxs = sample(1:size(samples, 1), 20; replace=false)
    ps = []
    for (i, idx) in enumerate(plot_idxs)
        p = plot(samples[idx, :], title="Sample $i", xlabel="Time (samples)", ylabel="x", label="")
        push!(ps, p)
    end
    p_final = plot(ps..., size=(2000, 2000), layout=(5, 4), bottom_margin=5mm, left_margin=5mm)
    display(p_final)
end

T = 100;
num_train = 1000
num_test = 200
phase = nothing

# case one - fixed noise, fixed freq, variable trend
noise_std = 0.3
trend = nothing
frequency = 5.0
save_file = "/Users/joshua/Desktop/QTNML_paper/QuantumInspiredML/Interpolation/paper/difficult-synthetic/v2/datasets/variable_trend"
X_train, y_train, params_all_train, X_test, y_test, params_all_test = generate_dataset(T, num_train, num_test, noise_std, trend, frequency, phase, save_file)
plot_param_dists!(params_all_train)
plot_example_ts!(X_train)

# case two - fixed freq, fixed trend, variable noise
noise_std = nothing
trend = 0.2
frequency = 5.0
save_file = "/Users/joshua/Desktop/QTNML_paper/QuantumInspiredML/Interpolation/paper/difficult-synthetic/v2/datasets/variable_noise"
X_train, y_train, params_all_train, X_test, y_test, params_all_test = generate_dataset(T, num_train, num_test, noise_std, trend, frequency, phase, save_file)
plot_param_dists!(params_all_train)
plot_example_ts!(X_train)

# case three - fixed noise, fixed trend, variable freq
noise_std = 0.3
trend = 0.2
frequency = nothing
save_file = "/Users/joshua/Desktop/QTNML_paper/QuantumInspiredML/Interpolation/paper/difficult-synthetic/v2/datasets/variable_freq"
X_train, y_train, params_all_train, X_test, y_test, params_all_test = generate_dataset(T, num_train, num_test, noise_std, trend, frequency, phase, save_file)
plot_param_dists!(params_all_train)
plot_example_ts!(X_train)

# case four - all variable 
noise_std = nothing
trend = nothing
frequency = nothing
save_file = "/Users/joshua/Desktop/QTNML_paper/QuantumInspiredML/Interpolation/paper/difficult-synthetic/v2/datasets/all_variable"
X_train, y_train, params_all_train, X_test, y_test, params_all_test = generate_dataset(T, num_train, num_test, noise_std, trend, frequency, phase, save_file)
plot_param_dists!(params_all_train)
plot_example_ts!(X_train)