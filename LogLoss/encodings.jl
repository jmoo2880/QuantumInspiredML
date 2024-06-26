using LegendrePolynomials
using StatsBase: countmap, sample
using PrettyTables

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

function fourier_encode(x::Float64, nds::Integer, ds::Vector{Integer})
    return [fourier(x, d, nds) for d in ds]
end

fourier_encode(x::Float64, nds::Integer, ti::Integer, ds::Vector{Vector{Integer}}) = fourier_encode(x, nds, ds[ti])






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

function legendre_encode(x::Float64, nds::Integer, ds::Vector{Integer}; norm = true)
    ls = [legendre(x,d,nds) for d in ds] 
    
    if norm # this makes 
        # make sure that |ls|^2 <= 1
        d = maximum(ds)
        ls /= sqrt(Pl(1,d; norm = Val(:normalized)) * d)
    end

    return ls
end

legendre_encode(x::Float64, nds::Integer, ti::Integer, ds::Vector{Vector{Integer}}; norm=true) = legendre_encode(x, nds, ds[ti]; norm=norm)


function uniform_encode(x::Float64, d::Int) # please don't use this unless it's auxilliary to some kind of splitting method
    return [1 for _ in 1:d] / d
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

#### Projection Initialisers
function series_expand(basis::AbstractVector{Function}, xs::AbstractVector{T}, ys::AbstractVector{U}, d::Integer) where {T<: Real, U <: Number}
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

    return [project_fourier(xs, d; kwargs...) for xs in eachrow(Xs)]
end

function project_fourier(xs::AbstractVector{T}, d; max_series_terms=10*d, max_samples=200, bandwidth=0.8) where {T <: Real}
    kdense = kde(xs; bandwidth=bandwidth) 
    xs_samp = range(-1,1,max_samples) # sample the KDE more often than xs does, this helps with the frequency limits on the series expansion
    ys = pdf(kdense, xs_samp)

    wf = sqrt.(ys);
    basis = [x -> cispi(n * x) for n in get_fourier_freqs(max_series_terms)]
    return series_expand(basis, xs_samp, wf, d)
end





function project_legendre_time_independent(Xs::AbstractMatrix{T}, d; kwargs...) where {T <: Real}

    return project_legendre(mean(Xs; dims=2), d; kwargs...)
end

function project_legendre(Xs::AbstractMatrix{T}, d; kwargs...) where {T <: Real}

    return [project_legendre(xs, d; kwargs...) for xs in eachrow(Xs)]
end

function project_legendre(xs::AbstractVector{T}, d; max_series_terms=10*d, max_samples=200, bandwidth=0.8) where {T <: Real}
    kdense = kde(xs; bandwidth=bandwidth) 
    xs_samp = range(-1,1,max_samples) # sample the KDE more often than xs does, this helps with the frequency limits on the series expansion
    ys = pdf(kdense, xs_samp)

    wf = sqrt.(ys);
    basis= [x -> Pl(x,l; norm = Val(:normalized)) for n in 0:(max_series_terms-1)]
    return series_expand(basis, xs_samp, wf, d)
end



################## Splitting Initialisers

function unif_split_init(X_norm::AbstractMatrix, y::Vector{Int}; opts::Options)
    nbins = get_nbins_safely(opts)

    bins = opts.encoding.splitmethod(X_norm, nbins, opts.encoding.range...)
    split_args = [bins, opts.aux_basis_dim, opts.encoding.basis]

    basis_args = isnothing(opts.encoding.basis.init) ? [] : opts.encoding.basis.init(X_norm, y; opts=opts)

    return [basis_args, split_args]
end

function hist_split_init(X_norm::AbstractMatrix, y::Vector{Int}; opts::Options)
    
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
    bins::Vector{Float64}, aux_dim::Int, basis::Basis = split_args
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
    bins::Vector{Float64}, aux_dim::Int, basis::Basis = split_args
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
    all_bins::Vector{Vector{Float64}}, aux_dim::Int, basis::Basis = split_args
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



###################################################################################
function encode_TS(sample::Vector, site_indices::Vector{Index{Int64}}, encoding_args::AbstractVector; opts::Options=Options())
    """Function to convert a single normalised sample to a product state
    with local dimension 2, as specified by the feature map."""

    n_sites = length(site_indices) # number of mps sites
    product_state = MPS(opts.dtype,site_indices; linkdims=1)
    
    
    # check that the number of sites matches the length of the time series
    if n_sites !== length(sample)
        error("Number of MPS sites: $n_sites does not match the time series length: $(length(sample))")
    end

    for j=1:n_sites

        if opts.encoding.istimedependent
            states = opts.encoding.encode(sample[j], opts.d, j, encoding_args...)
        else
            states = opts.encoding.encode(sample[j], opts.d, encoding_args...)
        end

        product_state[j] = ITensor(opts.dtype, states, site_indices[j])
    end

    return product_state

end

function encode_dataset(ES::EncodeSeparate, X_norm::AbstractMatrix, y::Vector{Int}, args...; kwargs...)
    # sort the arrays by class. This will provide a speedup if classes are trained/encoded separately
    # the loss grad function assumes the timeseries are sorted! Removing the sorting now breaks the algorithm
    order = sortperm(y)

    return encode_safe_dataset(ES, X_norm[:,order], y[order], args...; kwargs...)
end



function encode_safe_dataset(::EncodeSeparate{true}, X_norm::AbstractMatrix, y::Vector{Int}, type::String, site_indices::Vector{Index{Int64}}; kwargs...)
    # X_norm has dimension num_elements * numtimeseries

    classes = unique(y)
    states = Vector{PState}(undef, length(y))

    enc_args = []

    for c in classes
        cis = findall(y .== c)
        ets, enc_as = encode_safe_dataset(EncodeSeparate{false}(), X_norm[:, cis], y[cis], type * " Sep Class", site_indices; kwargs...)
        states[cis] .= ets.timeseries
        push!(enc_args, enc_as)
    end
    class_map = countmap(y)
    class_distribution = collect(values(class_map))[sortperm(collect(keys(class_map)))]  # return the number of occurances in each class sorted in order of class index
    return EncodedTimeseriesSet(states, class_distribution), enc_args
end

function encode_safe_dataset(::EncodeSeparate{false}, X_norm::AbstractMatrix, y::Vector{Int}, type::String, 
    site_indices::Vector{Index{Int64}}; opts::Options=Options(), balance_classes=opts.encoding.isbalanced, 
    rng=MersenneTwister(1234), class_keys::Dict{T, I}) where {T, I<:Integer}
    """"Convert an entire dataset of normalised time series to a corresponding 
    dataset of product states"""
    verbosity = opts.verbosity
    # pre-allocate
    spl = String.(split(type; limit=2))
    type = spl[1]

    num_ts = size(X_norm)[2] 

    types = ["train", "test", "valid"]
    if type in types
        if length(spl) > 1
            verbosity > - 1 && println("Initialising $type states for class $(first(y)).")
        else
            verbosity > - 1 && println("Initialising $type states.")
        end
    else
        error("Invalid dataset type. Must be train, test, or valid.")
    end

    # check data is in the expected range first
    a,b = opts.encoding.range
    name = opts.encoding.name
    if all((a .<= X_norm) .& (X_norm .<= b)) == false
        error("Data must be rescaled between $a and $b before a $name encoding.")
    end

    # check class balance
    cm = countmap(y)
    balanced = all(i-> i == first(values(cm)), values(cm))
    if !balanced
        verbosity > - 1 && println("Classes are not Balanced:")
        verbosity > - 1 && pretty_table(cm, header=["Class", "Count"])
    end

    # handle the encoding initialisation
    if isnothing(opts.encoding.init)
        encoding_args = []
    elseif !balanced && balance_classes
        throw(ErrorException("Not Implemented Correctly yet!"))
        min_s = minimum(values(cm))
        verbosity > - 1 && println("Balancing Encoding initialisation by cutting to $min_s samples in each class!\n")
        Xns = []
        ys = []
        for k in keys(cm)
            Xn = X_norm[:, y .== k]
            Xn = Xn[:, sample(rng, 1:end, min_s; replace=false)] # randomly select min_s samples from this class
            push!(Xns, Xn)
            push!(ys, fill(k, min_s))
        end
        X_balanced = vcat(Xns...) #hcat?
        ys = vcat(ys...)

        encoding_args = opts.encoding.init(X_balanced, ys; opts=opts)

    else
        encoding_args = opts.encoding.init(X_norm, y; opts=opts)
    end

    all_product_states = TimeseriesIterable(undef, num_ts)
    for i=1:num_ts
        sample_pstate = encode_TS(X_norm[:, i], site_indices, encoding_args; opts=opts)
        sample_label = y[i]
        label_idx = class_keys[sample_label]
        product_state = PState(sample_pstate, sample_label, label_idx)
        all_product_states[i] = product_state
    end
       
    

    class_map = countmap(y)
    class_distribution = collect(values(class_map))[sortperm(collect(keys(class_map)))] # return the number of occurances in each class sorted in order of class index
    
    return EncodedTimeseriesSet(all_product_states, class_distribution), encoding_args

end


function encoding_test(::EncodeSeparate{true}, X_norm::AbstractMatrix, y::Vector{Int}, site_indices::Vector{Index{Int64}};  kwargs...)

    classes = sort(unique(y))
    states = Vector{Matrix{Vector{opts.dtype}}}(undef, length(classes)) # Only the best, most pristine types used here

    for (i,c) in enumerate(classes)
        cis = findall(y .== c)
        ets = encoding_test(EncodeSeparate{false}(), X_norm[:, cis], y[cis], site_indices; kwargs...)
        states[i] = first(ets) # for type stability reasons
    end

    return states

end

function encoding_test(::EncodeSeparate{false}, X_norm::AbstractMatrix, y::Vector{Int}, site_indices::Vector{Index{Int64}}; 
    opts::Options=Options(), 
    num_ts=size(X_norm,2)
    )
    # a test encoding used to plot the basis being used if "test_run"=true

    # handle the encoding initialisation
    if isnothing(opts.encoding.init)
        encoding_args = []
    else
        encoding_args = opts.encoding.init(X_norm, y; opts=opts)
    end

    a,b = opts.encoding.range
    #num_ts = 1000 # increase resolution for plotting
    stp = (b-a)/(num_ts-1)
    X_norm = Matrix{Float64}(undef, size(X_norm,1), num_ts)
    for row in eachrow(X_norm)
        row[:] = collect(a:stp:b)
    end


   
    all_product_states = Matrix{Vector{opts.dtype}}(undef, size(X_norm))

    for i=1:num_ts
        sample_pstate = encode_TS(X_norm[:, i], site_indices, encoding_args; opts=opts)
        all_product_states[:,i] .= Vector.(sample_pstate)
    end

    return [all_product_states] # one element vector is for type stability reasons

end
