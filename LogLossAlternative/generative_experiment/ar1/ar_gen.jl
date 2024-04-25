using Plots
using Random, Distributions

function simulate_single_ar1(phi::Float64, n::Int)
    """Generate one realisation of an AR(1) process.
    n time points, and AR coefficient, phi."""
    dist = Normal()
    y = [0.0 for i in 1:n]
    noise = rand(dist, n)
    
    for i in 1:(n-1)
        y[(i + 1)] = phi .* y[i] .+ noise[i]
    end
    
    return y
end

function generate_ar1_dataset(train_samps_per_class::Int, test_samps_per_class::Int)
    """Generate a dataset of P AR(1) processess
    with fixed length, N."""
    # try ϕ = 0.2, 0.8
    # generate train data

    class_0_train_samples = Matrix{Float64}(undef, train_samps_per_class, 100)
    class_1_train_samples = Matrix{Float64}(undef, train_samps_per_class, 100)
    y_train = Int.(vcat(zeros(train_samps_per_class), ones(train_samps_per_class)))
    
    for i in 1:train_samps_per_class
        class_0_train_samples[i, :] = simulate_single_ar1(0.1, 100)
    end

    for i in 1:train_samps_per_class
        class_1_train_samples[i, :] = simulate_single_ar1(0.9, 100)
    end

    X_train = vcat(class_0_train_samples, class_1_train_samples)
    # generate test data
    class_0_test_samples = Matrix{Float64}(undef, test_samps_per_class, 100)
    class_1_test_samples = Matrix{Float64}(undef, test_samps_per_class, 100)
    y_test = Int.(vcat(zeros(test_samps_per_class), ones(test_samps_per_class)))

    for i in 1:test_samps_per_class
        class_0_test_samples[i, :] = simulate_single_ar1(0.2, 100)
    end

    for i in 1:test_samps_per_class
        class_1_test_samples[i, :] = simulate_single_ar1(0.8, 100)
    end

    X_test = vcat(class_0_test_samples, class_1_test_samples)

    return (X_train, y_train), (X_test, y_test)

end