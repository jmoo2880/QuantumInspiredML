using Plots

# GD with/without momentum
# original function is y = 0.3x^4 - 0.1x^3 - 2x^2 - 0.8x
# grad is 1.2x^3 - 0.3x^2 - 4x - 0.8
function func(x)
    return 0.3 .* x.^4 - 0.1 .* x.^3 -2x.^2 - 0.8 .* x
end

function grad(x)
    return 1.2*x^3 - 0.3*x^2 - 4*x - 0.8
end

function gd_no_momentum(x0, steps, gamma)
    x = x0
    xs = [x0]
    for step in 1:steps
        x_new = x - gamma * grad(x)
        push!(xs, x_new)
        x = x_new
    end
    return xs
end

function gd_momentum(x0, steps, gamma, mu)
    x = x0
    xs = [x0]
    b = 0
    for step in 1:steps
        b_new = mu * b + grad(x)
        new_x = x - gamma * b_new
        b = b_new
        println(b)
        push!(xs, new_x)
        x = new_x
    end
    return xs

end

# plot 100 iterations of both
starting_pt = -2.8
μ = 0.7
lr = 0.05
steps = 100
xs_no_momentum = gd_no_momentum(starting_pt, steps, lr)
xs_momentum = gd_momentum(starting_pt, steps, lr, μ)
xs = LinRange(-3, 3, 1000)
plot(xs, func(xs), lw=2, label="", xlabel="x", ylabel="y", title="100 Iterations")
scatter!(xs_no_momentum, func(xs_no_momentum), label="GD", alpha=0.3)
scatter!(xs_momentum, func(xs_momentum), label="GD + momentum", alpha=0.3)