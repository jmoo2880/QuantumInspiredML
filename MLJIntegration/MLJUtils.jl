
function MPSOptions(m::MPSClassifier; verbosity::Int=m.reformat_verbosity)
    properties = propertynames(m)
    properties = filter(s -> !(s in [:reformat_verbosity]), properties)

    # this is actually cool syntax I have to say
    opts = MPSOptions(; [field => getfield(m,field) for field in properties]..., verbosity=verbosity)
    return opts

end

function encoderows(sites::AbstractVector{<:Index{<:Integer}}, opts::Options, Xs::AbstractMatrix, ys::AbstractVector)
    @assert size(Xs, 2) == size(ys, 1) "Size of training dataset and number of training labels are different!"
    range = opts.encoding.range
    if opts.sigmoid_transform
        # rescale with a sigmoid prior to minmaxing
        scaler = fit(RobustSigmoid, Xs);
        Xs_scaled = transform_data(scaler, Xs; range=range, minmax_output=opts.minmax)

    else
        Xs_scaled = transform_data(Xs; range=range, minmax_output=opts.minmax)

    end

    # generate product states using rescaled data
    if opts.encoding.iscomplex
        if opts.dtype <: Real
            error("Using a complex valued encoding but the MPS is real")
        end

    elseif !(opts.dtype <: Real)
        @warn "Using a complex valued MPS but the encoding is real"
    end

    @assert !(opts.encode_classes_separately && opts.encoding.isbalanced) "Attempting to balance classes while encoding separately is ambiguous"

    # generate the starting MPS with uniform bond dimension chi_init and random values (with seed if provided)
    classes = unique(ys)
    num_classes = length(classes)

    @assert eltype(classes) <: Integer "Classes must be integers" #TODO fix PState so this is unnecessary
    sort!(classes)
    class_keys = Dict(zip(classes, 1:num_classes))


    s = EncodeSeparate{opts.encode_classes_separately}()
    training_states, enc_args_tr = encode_dataset(s, Xs_scaled, ys, "train", sites; opts=opts, class_keys=class_keys)
    return training_states
end

encoderows(sites::AbstractVector{<:Index{<:Integer}}, opts::Options, Xs::AbstractMatrix) = encoderows(sites, opts, Xs, -ones(Int, size(Xs,2)))
encoderows(opts::Options, Xs::AbstractMatrix, args...; kwargs...) = encoderows( siteinds(opts.d, size(Xs, 1)), opts, Xs, args...; kwargs...)

function MPSpredict(W::MPS, X::EncodedTimeseriesSet)
    pss = X.timeseries
    yhat::Vector{UInt} = [argmax(abs.(vector(contractMPS(W, ps)))) for ps in pss]

    return yhat
end