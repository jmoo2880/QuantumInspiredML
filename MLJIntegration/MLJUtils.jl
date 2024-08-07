

function model_encoding(s::Symbol)
    if s == :Legendre_No_Norm
        enc = legendre_no_norm()
    elseif s == :Legendre
        enc = legendre()
    elseif s == :Stoudenmire 
        enc = stoudenmire()
    elseif s == :Fourier
        enc = fourier()
    elseif s == :Sahand
        s = sahand()
    end
    return enc
end

function model_bbopt(s::Symbol)
    if s == :GD
        opt = BBOpt("CustomGD")
    elseif s == :TSGO
        opt = BBOpt("CustomGD", "TSGO")
    elseif s == :Optim
        opt = BBOpt("Optim")
    elseif s == :OptimKit
        opt = BBopt("OptimKit")
    end

    return opt
end

function model_loss_func(s::Symbol)
    if s == :KLD
        lf = loss_grad_KLD
    elseif s == :MSE
        lf = loss_grad_MSE
    elseif s == :Mixed
        lf = loss_grad_mixed
    end
    return lf
end

function Options(m::MPSClassifier; verbosity::Int=m.reformat_verbosity)
    return Options(
        m.nsweeps, 
        m.chi_max, 
        m.cutoff, 
        m.update_iters, 
        verbosity, 
        m.dtype, 
        model_loss_func(m.loss_grad), 
        model_bbopt(m.bbopt), 
        m.track_cost, 
        m.eta, 
        m.rescale, 
        m.d, 
        m.aux_basis_dim, 
        model_encoding(m.encoding), 
        m.train_classes_separately, 
        m.encode_classes_separately, 
        m.return_encoding_meta_info, 
        m.minmax, 
        m.exit_early, 
        m.sigmoid_transform
    )
end


function encoderows(sites::AbstractVector{<:Index{<:Integer}}, opts::Options, Xs::AbstractMatrix, ys::AbstractVector)
    @assert size(Xs, 2) == size(ys, 1) "Size of training dataset and number of training labels are different!"
    range = opts.encoding.range
    if opts.sigmoid_transform
        # rescale with a sigmoid prior to minmaxing
        scaler = fit_scaler(RobustSigmoidTransform, Xs);
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