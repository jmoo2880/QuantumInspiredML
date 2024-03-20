#
# Computes the norm of the projected gradient
# One method is defined for each possible scenario 
# of bounds (all, only lower, only upper, none). This is
# needed such that the lower and upper bounds vectors can 
# be of type Nothing if those bounds are not defined
#
# Both bounds
function pr_gradnorm(g, x, lower, upper)
    gnorm = zero(eltype(x))
    for i in eachindex(x)
        z = max(lower[i], min(upper[i], x[i] - g[i])) - x[i]
        gnorm = max(gnorm, abs(z))
    end
    return gnorm
end
# Only lower
function pr_gradnorm(g, x, lower, upper::Nothing)
    gnorm = zero(eltype(x))
    for i in eachindex(x)
        z = max(lower[i], x[i] - g[i]) - x[i]
        gnorm = max(gnorm, abs(z))
    end
    return gnorm
end
# Only upper
function pr_gradnorm(g, x, lower::Nothing, upper)
    gnorm = zero(eltype(x))
    for i in eachindex(x)
        z = min(upper[i], x[i] - g[i]) - x[i]
        gnorm = max(gnorm, abs(z))
    end
    return gnorm
end
# No bounds
pr_gradnorm(g, x, lower::Nothing, upper::Nothing) = maximum(abs, g)

#
# Compute a trial point
#
function compute_xn!(xn, x, t, g, lower, upper)
    for i in eachindex(x)
        xn[i] = x[i] - t * g[i]
        xn[i] = max(xn[i], lower[i])
        xn[i] = min(xn[i], upper[i])
    end
end
function compute_xn!(xn, x, t, g, lower, upper::Nothing)
    for i in eachindex(x)
        xn[i] = x[i] - t * g[i]
        xn[i] = max(xn[i], lower[i])
    end
end
function compute_xn!(xn, x, t, g, lower::Nothing, upper)
    for i in eachindex(x)
        xn[i] = x[i] - t * g[i]
        xn[i] = min(xn[i], upper[i])
    end
end
compute_xn!(xn, x, t, g, lower::Nothing, upper::Nothing) = @. xn = x - t * g
