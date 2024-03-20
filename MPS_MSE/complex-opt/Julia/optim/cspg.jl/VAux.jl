#
# Structure to store auxiliary arrays
#
struct VAux{T,F}
    g::T
    xn::T
    gn::T
    fprev::Vector{F}
end
function VAux(x, fx; m=10)
    g = similar(x)
    xn = similar(x)
    gn = similar(x)
    fprev = typeof(fx)[]
    for i in 1:m
        push!(fprev, zero(fx))
    end
    return VAux{typeof(g),typeof(fx)}(g, xn, gn, fprev)
end