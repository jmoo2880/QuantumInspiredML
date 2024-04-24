using Plots
using Random, Distributions

function simulate_single_ar1(phi::Float64, n::Int)
    """Generate one realisation of an AR(1) process.
    n time points, and AR coefficient, phi."""
    dist = Normal()
    y = [0.0 for i = 1:n]

    noise = rand(dist, n)

    for i in 1:(n-1)
        y[i + 1] = phi * y + noise[i]
    end

    return y

end

function generate_ar1_dataset()
    """Generate a dataset of P AR(1) processess
    with fixed length, N. """
end