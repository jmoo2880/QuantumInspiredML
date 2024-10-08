# Sets of Basis Functions

function angle_encode(x::Float64, d::Int; periods=1/4)
    @assert d == 2 "Stoudenmire Angle encoding only supports d = 2!"
    return angle_encode(x;periods=periods)
end

function angle_encode(x::Float64; periods=1/4) 
    """Function to convert normalised time series to an angle encoding."""
    @assert x <= 1.0 && x >= 0.0 "Data points must be rescaled between 1 and 0 before encoding using the angle encoder."
    s1 = cispi( 3*x/2) * cospi(2*periods* x)
    s2 = cispi(-3*x/2) * sinpi(2*periods * x)
    return [s1, s2]
 
end


function fourier(x::Float64, i::Integer, d::Integer)
    return cispi.(i*x) / sqrt(d)
end

function get_fourier_freqs(d)
    bound = (d-1.)/2.
    # if d-1 is odd, then select the positive term first

    hbound = ceil(Integer, bound)
    return vcat(0, [[i,-i] for i in 1:hbound]...)[1:d]
end

function fourier_encode(x::Float64, d::Integer;)
    bounds = get_fourier_freqs(d)

    return [fourier(x,i,d) for i in bounds]

end

function fourier_encode(x::Float64, nds::Integer, ds::AbstractVector{Integer})
    return [fourier(x, d, nds) for d in ds]
end

fourier_encode(x::Float64, nds::Integer, ti::Integer, ds::AbstractVector{<:AbstractVector{<:Integer}}) = fourier_encode(x, nds, ds[ti])


function sahand(x::Float64, i::Integer,d::Integer)
    dx = 2/d # width of one interval
    interval = ceil(i/2)
    startx = (interval-1) * dx
    if startx <= x <= interval*dx
        if isodd(i)
            s = cispi(3*x/2/dx) * cospi(0.5 * (x - startx)/dx )
        else
            s = cispi(-3*x/2/dx) * sinpi(0.5 * (x - startx)/dx )
        end
    else
        s = complex(0.)
    end

    return s
end

function sahand_encode(x::Float64, d::Int)
    @assert iseven(d) "Sahand encoding only supports even dimension"

    return [sahand(x,i,d) for i in 1:d]
end


function legendre(x::Float64, i::Int, d::Int)
    return Pl(x, i; norm = Val(:normalized))
end

function legendre_encode(x::Float64, d::Int; norm = true)
    ls = [legendre(x,i,d) for i in 0:(d-1)] 
    
    if norm # this makes 
        # make sure that |ls|^2 <= 1
        ls /= sqrt(Pl(1,d; norm = Val(:normalized)) * d)
    end

    return ls
end

function legendre_encode(x::Float64, nds::Integer, ds::AbstractVector{<:Integer}; norm = true)
    ls = [legendre(x,d,nds) for d in ds] 
    
    if norm # this makes 
        # make sure that |ls|^2 <= 1
        d = maximum(ds)
        ls /= sqrt(Pl(1,d; norm = Val(:normalized)) * d)
    end

    return ls
end

legendre_encode(x::Float64, nds::Integer, ti::Integer, ds::AbstractVector{<:AbstractVector{<:Integer}}; norm=true) = legendre_encode(x, nds, ds[ti]; norm=norm)
legendre_encode_no_norm(args...; kwargs...) = legendre_encode(args...; kwargs..., norm=false) # praise be to overriding keywords

function uniform_encode(x::Float64, d::Int) # please don't use this unless it's auxilliary to some kind of splitting method
    return [1 for _ in 1:d] / d
end


#### Projection Initialisers
function series_expand(basis::AbstractVector{<:Function}, xs::AbstractVector{T}, ys::AbstractVector{U}, d::Integer) where {T<: Real, U <: Number}
    coeffs = []
    for f in basis
        bs = f.(xs)
        problem = SampledIntegralProblem(ys .* conj.(bs), xs)
        method = TrapezoidalRule()
        push!(coeffs, solve(problem, method).u)
    end
    return partialsortperm(abs2.(coeffs), 1:d; rev=true)
end

series_expand(f::Function, xs::AbstractVector{T}, ys::AbstractVector{U}, d::Integer; series_terms::AbstractVector{Integer}) where {T<: Real, U <: Number} = series_expand([x->f(x,n) for n in series_terms], xs, ys, d) 
series_expand(f::Function, xs::AbstractVector{T}, ys::AbstractVector{U}, d::Integer; max_series_terms::Integer=10*d) where {T<: Real, U <: Number} = series_expand(f, xs, ys, d; series_terms=0:(max_series_terms-1)) 


function project_fourier_time_independent(Xs::Matrix{T}, d::Integer; kwargs...) where {T <: Real}

    return project_fourier(mean(Xs; dims=2), d::Integer; kwargs...)
end

function project_fourier(Xs::Matrix{T}, d::Integer; kwargs...) where {T <: Real}

    return [[project_fourier(xs, d; kwargs...) for xs in eachrow(Xs)]]
end

function project_fourier(xs::AbstractVector{T}, d; max_series_terms=10*d, max_samples=200, bandwidth=0.8, kwargs...) where {T <: Real}
    kdense = kde(xs; bandwidth=bandwidth) 
    xs_samp = range(-1,1,max_samples) # sample the KDE more often than xs does, this helps with the frequency limits on the series expansion
    ys = pdf(kdense, xs_samp)

    wf = sqrt.(ys);
    basis = [x -> cispi(n * x) for n in get_fourier_freqs(max_series_terms)]
    return series_expand(basis, xs_samp, wf, d)
end





function project_legendre_time_independent(Xs::AbstractMatrix{T}, d::Integer; kwargs...) where {T <: Real}

    return project_legendre(mean(Xs; dims=2), d; kwargs...)
end


function project_legendre(Xs::AbstractMatrix{T}, d::Integer; kwargs...) where {T <: Real}

    return [[project_legendre(xs, d; kwargs...) for xs in eachrow(Xs)]]
end

function project_legendre(xs::AbstractVector{T}, d::Integer; max_series_terms::Integer=7*d, max_samples=200, bandwidth=0.8, kwargs...) where {T <: Real}
    kdense = kde(xs; bandwidth=bandwidth) 
    xs_samp = range(-1,1,max_samples) # sample the KDE more often than xs does, this helps with the frequency limits on the series expansion
    ys = pdf(kdense, xs_samp)

    wf = sqrt.(ys);
    basis= [x -> Pl(x,l; norm = Val(:normalized)) for l in 0:(max_series_terms-1)]
    return series_expand(basis, xs_samp, wf, d)
end

project_legendre(Xs::AbstractMatrix{<:Real}, ys::AbstractVector{<:Integer}; opts, kwargs...) = project_legendre(Xs, opts.d; kwargs...)

include("splitbases.jl")