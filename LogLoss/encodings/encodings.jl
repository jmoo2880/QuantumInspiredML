using LegendrePolynomials
using StatsBase: countmap, sample
using PrettyTables
using KernelDensity
using Integrals

include("bases.jl")
include("splitbases.jl")

function encode_TS(sample::AbstractVector, site_indices::AbstractVector{Index{Int64}}, encoding_args::AbstractVector; opts::Options=Options())
    """Function to convert a single normalised sample to a product state
    with local dimension as specified by the feature map."""

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

function encode_dataset(ES::EncodeSeparate, X_norm::AbstractMatrix, y::AbstractVector, args...; kwargs...)
    """Convert an entire dataset of normalised time series to a corresponding 
    dataset of product states"""
    # sort the arrays by class. This will provide a speedup if classes are trained/encoded separately
    # the loss grad function assumes the timeseries are sorted! Removing the sorting now breaks the algorithm
    if size(X_norm, 2) == 0
        encoding_args = []
        return EncodedTimeseriesSet(eltype(y)), encoding_args
    end

    order = sortperm(y)

    return encode_safe_dataset(ES, X_norm[:,order], y[order], args...; kwargs...)
end



function encode_safe_dataset(::EncodeSeparate{true}, X_norm::AbstractMatrix, y::AbstractVector, type::String, site_indices::AbstractVector{Index{Int64}}; kwargs...)
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

function encode_safe_dataset(::EncodeSeparate{false}, X_norm::AbstractMatrix, y::AbstractVector, type::String, 
    site_indices::AbstractVector{Index{Int64}}; opts::Options=Options(), balance_classes=opts.encoding.isbalanced, 
    rng=MersenneTwister(1234), class_keys::Dict{T, I}) where {T, I<:Integer}
    """"Convert an entire dataset of normalised time series to a corresponding 
    dataset of product states, assumes that inout dataset is sorted by class"""
    verbosity = opts.verbosity
    # pre-allocate
    spl = String.(split(type; limit=2))
    type = spl[1]

    num_ts = size(X_norm)[2] 

    types = ["train", "test", "valid"]
    if type in types
        if verbosity > 0
            if length(spl) > 1
                println("Initialising $type states for class $(y[1]).")
            else
                println("Initialising $type states.")
            end
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
    if verbosity > 1 && !balanced
        println("Classes are not Balanced:")
        pretty_table(cm, header=["Class", "Count"])
    end

    # handle the encoding initialisation
    if isnothing(opts.encoding.init)
        encoding_args = []
    elseif !balanced && balance_classes
        error("balance_classes is not implemented correctly yet!")
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


function encoding_test(::EncodeSeparate{true}, X_norm::AbstractMatrix, y::AbstractVector{Int}, site_indices::AbstractVector{Index{Int64}};  kwargs...)

    classes = sort(unique(y))
    states = Vector{Matrix{Vector{opts.dtype}}}(undef, length(classes)) # Only the best, most pristine types used here

    for (i,c) in enumerate(classes)
        cis = findall(y .== c)
        ets = encoding_test(EncodeSeparate{false}(), X_norm[:, cis], y[cis], site_indices; kwargs...)
        states[i] = ets[1] # for type stability reasons
    end

    return states

end

function encoding_test(::EncodeSeparate{false}, X_norm::AbstractMatrix, y::AbstractVector{Int}, site_indices::AbstractVector{Index{Int64}}; 
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
