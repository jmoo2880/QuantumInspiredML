using LegendrePolynomials

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


function fourier(x::Float64, i::Int,d::Int)
    return cispi.(i*x) / sqrt(d)
end

function fourier_encode(x::Float64, d::Int; exclude_DC::Bool=true)
    if exclude_DC
        return [fourier(x,i,d) for i in 1:d]
    else
        return [fourier(x,i,d) for i in 0:(d-1)]
    end
end


function sahand(x::Float64, i::Int,d::Int)
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
        s = 0
    end

    return s
end

function sahand_encode(x::Float64, d::Int)
    @assert iseven(d) "Sahand encoding only supports even dimension"

    return [sahand(x,i,d) for i in 1:d]
end


function legendre(x::Float64,i::Int, d::Int)
    return Pl(x, i; norm = Val(:normalized))
end

function legendre_encode(x::Float64, d::Int)
    return [legendre(x,i,d) for i in 1:d]
end


function uniform_encode(x::Float64, d::Int) # please don't use this unless it's auxilliary to some kind of splitting method
    return [1 for _ in 1:d]
end


##################### Splitting methods
function unif_split(data::Matrix, nbins::Integer, a::Float64, b::Float64)
    dx = 1/nbins# width of one interval
    return collect(a:dx:b)
end

function hist_split(samples::Vector, nbins::Integer, a::Float64, b::Float64) # samples should be a vector of timepoints at a single time (NOT a timeseries) for all sensible use cases
    npts = length(samples)
    bin_pts = Int(round(npts/nbins))

    bins = Vector{eltype(samples)}(undef, nbins+1)
    
    bins[1] = a # lower bound
    j = 2
    ds = sort(samples)
    for (i,x) in enumerate(ds)
        if i % bin_pts == 0 && i < length(samples)
            bins[j] = (x + ds[i+1])/2
            j += 1
        end
    end
    bins[end] = b # upper bound
    return bins
end

function hist_split(X_norm::Matrix, nbins::Integer)
    return [hist_split(samples,nbins) for samples in eachrow(X_norm)]
end




################## Splitting Initialisers
function unif_split_init(X_norm::Matrix, y::Vector{Int}; opts::Options)
    nbins = get_nbins_safely(opts)

    bins = opts.encoding.splitmethod(X_norm, nbins, opts.encoding.range...)
    split_args = [bins, opts.aux_basis_dim, opts.encoding.basis]

    basis_args = isnothing(opts.encoding.basis.init) ? [] : opts.encoding.basis.init(X_norm::Matrix, y::Vector{Int}; opts::Options)

    return [basis_args, split_args]
end

function hist_split_init(X_norm::Matrix, y::Vector{Int}; opts::Options)
    
    nbins = get_nbins_safely(opts)
    bins = opts.encoding.splitmethod(X_norm, nbins, opts.encoding.range...)
    split_args = [bins, opts.aux_basis_dim, opts.encoding.basis]

    basis_args = isnothing(opts.encoding.basis.init) ? [] : opts.encoding.basis.init(X_norm::Matrix, y::Vector{Int}; opts::Options)
    
    return [basis_args, split_args]
end

function get_nbins_safely(opts)
    nbins = opts.d / opts.aux_basis_dim
    try
        nbins = convert(Int, nbins)
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
    return  abs(x) == 1/2 ? 1/2 : float(-0.5 < x < 0.5)
end

function project_onto_unif_bins(x::Float64, d::Int, basis_args::Vector, split_args::Vector;)
    bins::Vector{Float64}, aux_dim::Int, basis::Basis = split_args
    widths = diff(bins)

    encoding = []
    for i = 1:(d/aux_dim)
        auxvec = xx -> basis.encode(xx, opts.d, basis_args...)
        
        dx = widths[i]
        aux_enc = rect((x - bins[i])/dx - 0.5)/dx .* auxvec((x-bins[i])/dx)
        push!(encoding, aux_enc)
    end


    return vcat(encoding)
end


function project_onto_unif_bins(x::Float64, d::Int, ti::Int, basis_args::Vector, split_args;::Vector)
    # time dep version in case the aux basis is time dep
    bins::Vector{Float64}, aux_dim::Int, basis::Basis = split_args
    widths = diff(bins)

    encoding = []
    for i = 1:(d/aux_dim)
        auxvec = xx -> basis.encode(xx, opts.d, ti, basis_args...) # we know it's time dependent


        dx = widths[i]
        aux_enc = rect((x - bins[i])/dx - 0.5)/dx .* auxvec((x-bins[i])/dx)
        push!(encoding, aux_enc)
    end


    return vcat(encoding)
end


function project_onto_hist_bins(x::Float64, d::Int, ti::Int, basis_args, split_args;)
    all_bins::Vector{Vector{Float64}}, aux_dim::Int, basis::Basis = split_args
    bins = all_bins[ti]
    widths = diff(bins)

    encoding = []
    for i = 1:(d/aux_dim)
        if basis.istimedependent
            auxvec = xx -> basis.encode(xx, opts.d, ti, basis_args...)
        else
            auxvec = xx -> basis.encode(xx, opts.d, basis_args...)
        end

        dx = widths[i]
        aux_enc = rect((x - bins[i])/dx - 0.5)/dx .* auxvec((x-bins[i])/dx)
        push!(encoding, aux_enc)
    end


    return vcat(encoding)
end



###################################################################################
function encode_TS(sample::Vector, site_indices::Vector{Index{Int64}}, encoding_args::Vector; opts::Options=Options())
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

function encode_dataset(X_norm::Matrix, y::Vector{Int}, type::String, 
    site_indices::Vector{Index{Int64}}; opts::Options=Options())
    """"Convert an entire dataset of normalised time series to a corresponding 
    dataset of product states"""

    types = ["train", "test", "valid"]
    if type in types
        println("Initialising $type states.")
    else
        error("Invalid dataset type. Must be train, test, or valid.")
    end

    # check data is in the expected range first
    a,b = opts.encoding.range
    name = opts.encoding.name
    if all((a .<= X_norm) .& (X_norm .<= b)) == false
        error("Data must be rescaled between $a and $b before a $name encoding.")
    end

    # handle the encoding initialisation
    if isnothing(opts.encoding.init)
        encoding_args = []
    else
        encoding_args = opts.encoding.init(X_norm, y; opts=opts)
    end


    num_samples = size(X_norm)[1]
    # pre-allocate
    all_product_states = timeSeriesIterable(undef, num_samples)

    for i=1:num_samples
        sample_pstate = encode_TS(X_norm[i, :], site_indices, encoding_args; opts=opts)
        sample_label = y[i]
        product_state = PState(sample_pstate, sample_label, type)
        all_product_states[i] = product_state
    end

    return all_product_states

end;