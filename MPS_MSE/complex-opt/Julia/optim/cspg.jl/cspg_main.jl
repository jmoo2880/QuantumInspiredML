# Implementation of CSPG from Zhang & Zia
# Adapted from the package SPGBOX, by J. M. Martínez (IMECC - UNICAMP)
#  which implements the algorithm of:
#
# NONMONOTONE SPECTRAL PROJECTED GRADIENT METHODS ON CONVEX SETS
# ERNESTO G. BIRGIN, JOSÉ MARIO MARTÍNEZ, AND MARCOS RAYDAN
# SIAM J. O. PTIM. Vol. 10, No. 4, pp. 1196-1211


function cspg! end

# Accepted input types
const GenVec = Union{AbstractArray, ITensor}

# main function

function cspg!(
    fg!::FG,
    x::GenVec;
    callback::H=nothing,
    func_only::FO=nothing,
    manif::Manifold=Flat();
    eps=oneunit(T) / 100_000,
    nitmax::Int=100,
    nfevalmax::Int=1000,
    m::Int=10,
    vaux::VAux=VAux(x, (isnothing(func_only) ? fg!(similar(x), x) : func_only(x)), m=m),
    iprint::Int=0,
    project_x0::Bool=true,
    step_nc=100
) where {
    FG<:Function,
    FO<:Union{<:Function,Nothing},
    H<:Union{<:Function,Nothing},
} where {T}
    # Adimentional variation of T (base Number type)
    adT = typeof(one(T))

    # Number of variables
    n = length(x)

    # Auxiliary arrays (associate names and check dimensions)
    g = vaux.g
    xn = vaux.xn
    gn = vaux.gn
    fprev = vaux.fprev
    length(g) == n || throw(DimensionMismatch("Auxiliar gradient vector `g` must be of the same length as `x`"))
    length(xn) == n || throw(DimensionMismatch("Auxiliar vector `xn` must be of the same length as `x`"))
    length(gn) == n || throw(DimensionMismatch("Auxiliar vector `gn` must be of the same length as `x`"))
    length(fprev) == m || throw(DimensionMismatch("Auxiliar vector `fprev` must be of length `m`"))

    # Project the initial point onto the manifold
    if project_x0
        retract!(manif,x)
    elseif !isapprox(retract(manif,x), x )
        throw(ArgumentError(" Initial value of x is not on the manifold, and `project_x0` is set to `false`. ", ))
    end

    # Iteration counter
    nit = 1

    # Objective function and gradient at initial point
    nfeval = 1
    fcurrent = fg!(g, x)
    gnorm = maximum(abs, project_tangent!(manif, g,x)) #mapr_gradnorm(g, x, lower, upper)
    gnorm <= eps && return CSPGResult(x, fcurrent, gnorm, nit, nfeval, 0, false)

    # Do a consertive initial step
    small = T(sqrt(Base.eps(T)))
    tspg = small / max(T(1), gnorm)

    # Initialize array of previous function values
    # Allow slight nonmonotonicity
    for i in eachindex(fprev)
        fprev[i] = fcurrent + abs(fcurrent) / 10
    end

    while nit < nitmax

        if iprint > 0
            println("----------------------------------------------------------- ")
            println(" Iteration: ", nit)
            println(" x = ", x[begin], " ... ", x[end])
            println(" Objective function value = ", fcurrent)
            println(" ")
            println(" Norm of the projected gradient = ", gnorm)
            println(" Number of function evaluations = ", nfeval)
        end

        fref = maximum(fprev)
        if iprint > 2
            println(" fref = ", fref)
            println(" fprev = ", fprev)
            println(" t = ", t)
        end

        # compute_xn!(xn, x, tspg, g, lower, upper)
        xn = retract(manif, x - tspg*project_tangent!(manif,g,x))

        lsfeval, fn =
            safequad_ls!(xn, gn, x, g, fcurrent, tspg, fref, nfevalmax - nfeval, iprint, func_only, fg!, lower, upper)

        if lsfeval < 0
            return CSPGResult(x, fcurrent, gnorm, nit, nfeval - lsfeval, 2, false)
        else
            nfeval += lsfeval
        end

        # Trial point accepted
        num = zero(T)
        den = zero(T)
        for i in eachindex(xn, x, gn, g)
            num = num + (xn[i] - x[i])^2 / oneunit(T)
            den = den + (xn[i] - x[i]) * (gn[i] - g[i]) / oneunit(T)
        end
        if den <= zero(T)
            tspg = adT(step_nc)
        else
            tspg = max(min(adT(1.0e30), num / den), adT(1.0e-30))
        end
        fcurrent = fn
        for i in eachindex(x, xn, g, gn)
            x[i] = xn[i]
            g[i] = gn[i]
        end
        for i = firstindex(fprev):lastindex(fprev)-1
            fprev[i] = fprev[i+1]
        end
        fprev[end] = fcurrent
        nit = nit + 1

        # Compute projected gradient norm
        gnorm = maximum(abs, project_tangent!(manif, g,x))

        # Call callback function
        if !isnothing(callback)
            if callback(CSPGResult(x, fcurrent, gnorm, nit, nfeval, 0, false))
                return CSPGResult(x, fcurrent, gnorm, nit, nfeval, 0, true)
            end
        end

        # Check convergence
        gnorm <= eps && return CSPGResult(x, fcurrent, gnorm, nit, nfeval, 0, false)
    end

    # Maximum number of iterations achieved
    return CSPGResult(x, fcurrent, gnorm, nit, nfeval, 1, false)
end

"Perform a safeguarded quadratic line search"
function safequad_ls!(
    xn::AbstractVecOrMat{T},
    gn::AbstractVecOrMat{T},
    x::AbstractVecOrMat{T},
    g::AbstractVecOrMat{T},
    fcurrent::Number,
    tspg::Number,
    fref::Number,
    nfevalmax::Int,
    iprint::Int,
    func_only::Union{Nothing,Function},
    fg!::Function
) where {T}
    one_T = one(T)
    # Armijo parameter
    gamma = one_T / 10_000

    gtd = zero(fref)
    for i in eachindex(xn, x, g)
        gtd += (xn[i] - x[i]) * g[i]
    end

    # Compute function values at initial point
    fn = fg!(gn, xn)
    nfeval = 1
    nfeval > nfevalmax && return -nfeval, fn

    alpha, trials = one_T, 0
    while true
        trials += 1
        if trials > 1
            # Compute a new trial point and its function values
            compute_xn!(xn, x, alpha * tspg, g, lower, upper)
            gtd = zero(T)
            for i in eachindex(xn, x, g)
                gtd += (xn[i] - x[i]) * g[i]
            end
            if iprint > 2
                println(" xn = ", xn[begin], " ... ", xn[end])
                println(" f[end] = ", fn, " fref = ", fref)
            end
            if !isnothing(func_only)
                fn = func_only(xn)
            else
                fn = fg!(gn, xn)
            end
            nfeval += 1
            nfeval > nfevalmax && return -nfeval, fn
        end

        # If the point is not acceptable
        if fn >= fref + gamma * gtd
            # Perform a safeguarded quadratic interpolation step
            if alpha <= one_T / 10
                alpha /= 2
            else
                atemp = -gtd * alpha^2 / (2 * (fn - fcurrent - alpha * gtd))
                if atemp <= one_T / 10 || atemp >= 9 * one_T / 10 * alpha
                    atemp = alpha / 2
                end
                alpha = atemp
            end
            # If the point is acceptable
        else
            # Update the gradient at the accepted point, if necessary
            if trials > 1 && !isnothing(func_only)
                fn = fg!(gn, xn)
                nfeval += 1
                nfeval > nfevalmax && return -nfeval, fn
            end
            break
        end
    end

    return nfeval, fn
end


#
# This method converts a call that provides explicit function and gradient functions,
# to a call where the same function computes the function and the gradient. The `func_only`
# parameter assumes the value of the objective function to compute the output type
#
function cspg!(f::F, g!::G, x::AbstractVecOrMat{T}; callback::H=nothing, kargs...) where {F<:Function,G<:Function,H<:Union{<:Function,Nothing},T}
    function fg!(g,x)
        g!(g, x)
        return f(x)
    end
    cspg!(fg!, x; callback=callback, func_only=f, kargs...)
end


"""
    cspg(f, g!, x::AbstractVecOrMat; lower=..., upper=..., options...)`

See `cspg!` for additional help.

Minimizes function `f` starting from initial point `x`, given the function to compute the gradient, `g!`. 
`f` must be of the form `f(x)`, and `g!` of the form `g!(g,x)`, where `g` is the gradient vector to be modified. 

Optional lower and upper box bounds can be provided using optional keyword arguments `lower` and `upper`.

    cspg(fg!, x::AbstractVecOrMat; lower=..., upper=..., options...)`

Given a single function `fg!(g,x)` that updates a gradient vector `g` and returns the function value, minimizes the function.  

These functions return a structure of type `CSPGResult`, containing the best solution found in `x` and the final objective function in `f`.

These functions *do not* mutate the `x` vector, instead it will create a (deep)copy of it (see `cspg!` for the in-place alternative). 

"""
function cspg(
    f::F, g!::G, x::AbstractVecOrMat{T};
    callback::H=nothing, kargs...
) where {F<:Function,G<:Function,H<:Union{<:Function,Nothing},T}
    x0 = copy(x)
    return cspg!(f, g!, x0; callback=callback, kargs...)
end
# With a single function to compute the function and the gradient
function cspg(fg::FG, x::AbstractVecOrMat{T};
    callback::H=nothing, kargs...
) where {FG<:Function,H<:Union{<:Function,Nothing},T}
    x0 = copy(x)
    return cspg!(fg, x0; callback=callback, kargs...)
end