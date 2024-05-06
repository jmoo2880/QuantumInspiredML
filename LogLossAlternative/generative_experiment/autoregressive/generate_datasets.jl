using Plots
using Random
using Distributions
using JLD2

const ϕ1 = 0.95
const ϕ2 = 0.1
#const ϕ2_1 = 1.3
#const ϕ2_2 = -0.7
const σ = 1.0
const num_series_per_class = 5000

function AR1(ϕ, σ, T)
    x = zeros(T)
    ϵ = rand(Normal(0, σ), T)
    x[1] = ϵ[1]
    for t in 2:T
        x[t] = ϕ * x[t-1] + ϵ[t]
    end
    return x
end

function AR2(ϕ1, ϕ2, σ, T)
    x = zeros(T)
    ϵ = rand(Normal(0, σ), T)
    x[1] = ϵ[1]
    x[2] = ϕ1 * x[1] + ϵ[2]
    for t in 3:T
        x[t] = ϕ1 * x[t-1] + ϕ2 * x[t-2] + ϵ[t]
    end

    return x
end

class0_data = Matrix{Float64}(undef, num_series_per_class, 100)
class0_labels = Vector{Int}(undef, num_series_per_class)
for i in 1:num_series_per_class
    class0_data[i, :] = AR1(ϕ1, σ, 100)
    class0_labels[i] = 0
end

class1_data = Matrix{Float64}(undef, num_series_per_class, 100)
class1_labels = Vector{Int}(undef, num_series_per_class)
for i in 1:num_series_per_class
    class1_data[i, :] = AR1(ϕ2, σ, 100)
    class1_labels[i] = 1
end

X_train_c0 = class0_data[1:4000, :]
y_train_c0 = class0_labels[1:4000]
X_train_c1 = class1_data[1:4000, :]
y_train_c1 = class1_labels[1:4000]

X_train = vcat(X_train_c0, X_train_c1)
y_train = vcat(y_train_c0, y_train_c1)

shuffled_idxs = shuffle(1:8000)
X_train_final = X_train[shuffled_idxs, :]
y_train_final = y_train[shuffled_idxs]

X_test_c0 = class0_data[4001:end, :]
y_test_c0 = class0_labels[4001:end]
X_test_c1 = class1_data[4001:end, :]
y_test_c1 = class1_labels[4001:end, :]

X_test = vcat(X_test_c0, X_test_c1)
y_test = vcat(y_test_c0, y_test_c1)

shuffled_idxs = shuffle(1:2000)
X_test_final = X_test[shuffled_idxs, :]
y_test_final = y_test[shuffled_idxs]

@save "/Users/joshua/Documents/QuantumInspiredML/LogLossAlternative/generative_experiment/autoregressive/AR_train_big.jld2" X_train_final y_train_final
@save "/Users/joshua/Documents/QuantumInspiredML/LogLossAlternative/generative_experiment/autoregressive/AR_test_big.jld2" X_test_final y_test_final
