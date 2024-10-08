################## Splitting Initialisers

function unif_split_init(X_norm::AbstractMatrix, y::AbstractVector{Int}; opts::Options)
    nbins = get_nbins_safely(opts)

    bins = opts.encoding.splitmethod(X_norm, nbins, opts.encoding.range...)
    split_args = [bins, opts.aux_basis_dim, opts.encoding.basis]

    basis_args = isnothing(opts.encoding.basis.init) ? [] : opts.encoding.basis.init(X_norm, y; opts=opts)

    return [basis_args, split_args]
end

function hist_split_init(X_norm::AbstractMatrix, y::AbstractVector{Int}; opts::Options)
    
    nbins = get_nbins_safely(opts)
    bins = opts.encoding.splitmethod(X_norm, nbins, opts.encoding.range...)
    split_args = [bins, opts.aux_basis_dim, opts.encoding.basis]

    if isnothing(opts.encoding.basis.init)
        basis_args = [[] for b in bins]
    end

    basis_args = isnothing(opts.encoding.basis.init) ? [] : opts.encoding.basis.init(X_norm, y; opts=opts)
    
    return [basis_args, split_args]
end

function get_nbins_safely(opts)
    nbins = opts.d / opts.aux_basis_dim
    try
        convert(Int, nbins)
    catch e
        if e isa InexactError
            error("The auxilliary basis dimension ($(opts.aux_basis_dim)) must evenly divide the total feature dimension ($(opts.d))")
        else
            throw(e)
        end
    end

    return convert(Int, nbins) # try blocks have their own scope
end

################## Splitting encoding helpers
function rect(x)
    # helper used to construct the split basis. It is important that rect(0.5) returns 0.5 because if an encoded point lies exactly on a bin boundary we want enc(x) = (0,...,0, 0.5, 0.5, 0,...0)
    # (Having two 1s instead of two 0.5s would violate our normalised encoding assumption)
    return  abs(x) == 0.5 ? 0.5 : 1. * float(-0.5 <= x <= 0.5)
end

function project_onto_unif_bins(x::Float64, d::Int, basis_args::AbstractVector, split_args::AbstractVector; norm=true)
    bins::AbstractVector{Float64}, aux_dim::Int, basis::Basis = split_args
    widths = diff(bins)

    encoding = []
    for i = 1:Int(d/aux_dim)
        auxvec = xx -> basis.encode(xx, aux_dim, basis_args...)
        #auxvec = xx -> ones(aux_dim)
        
        dx = widths[i]
        y = norm ? 1. : 1/widths[i]

        select = y * rect((x - bins[i])/dx - 0.5)
        aux_enc = select == 0 ? zeros(aux_dim) : select .* auxvec((x-bins[i])/dx) # necessary so that we don't evaluate aux basis function out of its domain
        push!(encoding, aux_enc)
    end


    return vcat(encoding...)
end


function project_onto_unif_bins(x::Float64, d::Int, ti::Int, basis_args::AbstractVector, split_args::AbstractVector; norm=true)
    # time dep version in case the aux basis is time dependent
    bins::AbstractVector{Float64}, aux_dim::Int, basis::Basis = split_args
    widths = diff(bins)

    encoding = []
    for i = 1:Int(d/aux_dim)
        auxvec = xx -> basis.encode(xx, aux_dim, ti, basis_args...) # we know it's time dependent

        dx = widths[i]
        y = norm ? 1. : 1/widths[i]
        select = y * rect((x - bins[i])/dx - 0.5)
        aux_enc = select == 0 ? zeros(aux_dim) : select .* auxvec((x-bins[i])/dx) # necessary so that we don't evaluate aux basis function out of its domain
        push!(encoding, aux_enc)
    end


    return vcat(encoding...)
end


function project_onto_hist_bins(x::Float64, d::Int, ti::Int, basis_args::AbstractVector, split_args::AbstractVector; norm=true)
    all_bins::AbstractVector{<:AbstractVector{Float64}}, aux_dim::Int, basis::Basis = split_args
    bins = all_bins[ti]
    widths = diff(bins)

    encoding = []
    for i = 1:Int(d/aux_dim)
        if basis.istimedependent
            auxvec = xx -> basis.encode(xx, aux_dim, ti, basis_args...)
        else
            auxvec = xx -> basis.encode(xx, aux_dim, basis_args...)
        end

        dx = widths[i]
        y = norm ? 1. : 1/widths[i]
        select = y * rect((x - bins[i])/dx - 0.5)
        aux_enc = select == 0 ? zeros(aux_dim) : select .* auxvec((x-bins[i])/dx) # necessary so that we don't evaluate aux basis function out of its domain
        push!(encoding, aux_enc)
    end


    return vcat(encoding...)
end


##################### Splitting methods
function unif_split(data::AbstractMatrix, nbins::Integer, a::Real, b::Real)
    dx = 1/nbins# width of one interval
    return collect(a:dx:b)
end

function hist_split(samples::AbstractVector, nbins::Integer, a::Real, b::Real) # samples should be a vector of timepoints at a single time (NOT a timeseries) for all sensible use cases
    npts = length(samples)
    bin_pts = Int(round(npts/nbins))

    if bin_pts == 0
        @warn("Less than one data point per bin! Putting the extra bins at x=1 and hoping for the best")
        bin_pts = 1
    end

    bins = fill(convert(eltype(samples), a), nbins+1) # look I'm not happy about this syntax either. Why does zeros() take a type, but not fill()?
    
    #bins[1] = a # lower bound
    j = 2
    ds = sort(samples)
    for (i,x) in enumerate(ds)
        if i % bin_pts == 0 && i < length(samples)
            if j == nbins + 1
                # This can happen if bin_pts is very small due to a small dataset, e.g. npts = 18, nbins = 8, then we can get as high as j = 10 and IndexError!
                #@warn("Only $bin_pts data point(s) per bin! This may seriously bias the encoding/lead to per performance (last bin contains $(npts - i) extra points)")
                break
            end
            bins[j] = (x + ds[i+1])/2
            j += 1
        end
    end
    if j <=  nbins
        bins[bins .== a] .= b 
        bins[1] = a
    end

    bins[end] = b # upper bound
    return bins
end

function hist_split(X_norm::AbstractMatrix, nbins::Integer, a::Real, b::Real)
    return [hist_split(samples,nbins, a, b) for samples in eachrow(X_norm)]
end