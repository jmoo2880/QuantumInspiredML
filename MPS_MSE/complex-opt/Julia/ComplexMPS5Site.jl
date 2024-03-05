using ITensors
using Folds
using Distributions
using OptimKit
using Zygote
using Plots 


struct PState
    """Custom struct for product states"""
    pstate::MPS
    label::Int
end

function complex_feature_map(x::Float64)
    s1 = exp(1im * (3π/2) * x) * cospi(0.5 * x)
    s2 = exp(-1im * (2π/2) * x) * sinpi(0.5 * x)
    return [s1, s2]
end

function generate_training_data(samples_per_class::Int; data_pts::Int=2)

    class_A_samples = zeros(samples_per_class, data_pts)
    class_B_samples = ones(samples_per_class, data_pts)
    all_samples = vcat(class_A_samples, class_B_samples)
    all_labels = Int.(vcat(zeros(size(class_A_samples)[1]), ones(size(class_B_samples)[1])))

    return all_samples, all_labels

end






