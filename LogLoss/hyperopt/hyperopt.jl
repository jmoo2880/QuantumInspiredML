# gridsearch hyperparameter opt
using Base.Threads
using LineSearches
using Optim

include("../RealRealHighDimension.jl")


using MLJBase: train_test_pairs, StratifiedCV




include("hyperUtils.jl")
include("methods/GridSearch.jl")
include("methods/NearestNeighbourSearch.jl")

""" #TODO Docstring """
function hyperopt end

# default to a gridsearch
hyperopt(Xs::AbstractMatrix, args...; kwargs...) = hyperopt(GridSearch(encoding=legendre()), enc, args...; kwargs...) 


function hyperopt(search::SearchMethod, Xs::AbstractMatrix, ys::AbstractVector; 
    chi_init::Integer=4,
    mps_seed::Real=4567,
    kfoldseed::Real=1234567890, # overridden by the rng parameter
    foldrng::AbstractRNG=MersenneTwister(kfoldseed),
    update_iters::Integer=1,
    verbosity::Real=-1,
    dtype::Type = first(search.encodings).iscomplex ? ComplexF64 : Float64,
    loss_grad::Function=loss_grad_KLD,
    bbopt::BBOpt=BBOpt("CustomGD", "TSGO"),
    track_cost::Bool=false,
    rescale::Tuple{Bool,Bool}=(false, true),
    aux_basis_dim::Integer=2,
    encode_classes_separately::Bool=false,
    train_classes_separately::Bool=false,
    minmax::Bool=true,
    cutoff::Number=1e-10,
    force_overwrite::Bool=false,
    always_abort::Bool=false,
    dir::String="LogLoss/hyperopt/",
    exit_early::Bool=true,
    sigmoid_transform::Bool=false,
    kwargs...
    )

    if force_overwrite && always_abort 
        error("You can't force_overwrite and always_abort that doesn't make any sense")
    end


    ########## Sanity checks ################
    if length(search.encodings) > 1
        error("More than one encoding at a time is not yet supported by any hyperopt algorithm (sorry)")
    end
    encoding = first(search.encodings)

    if encoding.iscomplex
        if dtype <: Real
            error("Using a complex valued encoding but the MPS is real")
        end

    elseif !(dtype <: Real)
        @warn "Using a complex valued MPS but the encoding is real"
    end

   


    @assert issorted(search.ds) "Hyperparamater vector \"ds\" is not sorted"


    # data is _input_ in python canonical (row major) format
    @assert size(Xs, 1) == size(ys, 1) "Size of training dataset and number of training labels are different!"

    


    ############### Data structures aand definitions ########################
    println("Allocating initial Arrays and checking for existing files")
    masteropts = Options(; nsweeps=max_sweeps, chi_max=1, d=1, eta=1, cutoff=cutoff, update_iters=update_iters, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad,
        bbopt=bbopt, track_cost=track_cost, rescale = rescale, aux_basis_dim=aux_basis_dim, encoding=encoding, encode_classes_separately=encode_classes_separately,
        train_classes_separately=train_classes_separately, minmax=minmax, exit_early=exit_early, sigmoid_transform=sigmoid_transform)

    

    # Output files
    sig_str = sigmoid_transform ? "ST_" : ""
    e_str = exit_early ? "EE_" : "NEE_"
    tstring = encoding.istimedependent ? "TD_" : "NTD_"
    vstring = train_classes_separately ? "Split_train_" : ""
    pstr = e_str * tstring * encoding.name * "_" * vstring * sig_str* "r$(mps_seed)_" * search.instance_name


    path = dir* pstr *"/"
    logpath = path # could change to a new folder if we wanted

    logfile = logpath * "log.txt"
    resfile = logpath  * "results.jld2"
    finfile = logpath * "finished.jld2"

    #################### Check if hyperopt already exists?  ################
    # resume if possible, after running this function there will always be a results file at "resfile"
    load_hyperopt!(search, logpath; force_overwrite=force_overwrite, always_abort=always_abort, logfile=logfile, resfile=resfile, finfile=finfile)

    ################### Definitions continued ##############################

    num_mps_sites = size(Xs, 2)
    classes = unique(ys)
    num_classes = length(classes)


    # A few more checks
    @assert eltype(classes) <: Integer "Classes must be integers"
    sort!(classes)
    class_keys = Dict(zip(classes, 1:num_classes)) # Assign each class a 'Key' from 1 to n
    

    Xs_train_enc = Matrix{EncodedTimeseriesSet}(undef, length(search.ds), search.nfolds) # Training data for each dimension and each fold
    Xs_val_enc = Matrix{EncodedTimeseriesSet}(undef, length(search.ds), search.nfolds) # Validation data for each dimension and each fold


    sites = Array{Vector{Index{Int64}}}(undef, num_mps_sites, length(search.ds)) # Array to hold the siteindices for each dimension
    Ws = Vector{MPS}(undef, length(search.ds)) # stores an MPS for each encoding dimension



    ############### generate the starting MPS for each d ####################
    println("Generating $(length(search.ds)) initial MPSs")
    #TODO parallelise
    for (di,d) in enumerate(search.ds)
        opts = _set_options(masteropts; d=d, verbosity=-1)

        sites[di] = siteinds(d, num_mps_sites)
           # generate the starting MPS with uniform bond dimension chi_init and random values (with seed if provided)
        Ws[di] = generate_startingMPS(chi_init, sites[di]; num_classes=num_classes, random_state=mps_seed, opts=opts)
    end


    
    #############  initialise the folds  ####################

    scv = StratifiedCV(;nfolds=search.nfolds, rng=foldrng)
    fold_inds = train_test_pairs(scv, eachindex(ys), ys)


    ########### Perform the encoding step for all d and f ############
    #TODO can encode more efficiently for certain encoding types if not time / order dependent. This will save a LOT of memory
    println("Encoding $(search.nfolds) folds with $(length(search.ds)) different encoding dimensions")
    @sync for f in 1:search.nfolds, (di,d) in enumerate(search.ds)
        if (isodd(d) && titlecase(encoding.name) == "Sahand") || (d != 2 && titlecase(encoding.name) == "Stoudenmire" )
            continue
        end

        @spawn begin
            opts= _set_options(masteropts;  d=d, verbosity=-1)

            tr_inds, val_inds = fold_inds[f]
            local f_Xs_tr = Xs[tr_inds, :]
            local f_Xs_val = Xs[val_inds, :]

            local f_ys_tr = ys[tr_inds]
            local f_ys_val = ys[val_inds]


            # Encode
            range = opts.encoding.range
            scaler = fit_scaler(RobustSigmoidTransform, f_Xs_tr);
            Xs_train_scaled = permutedims(transform_data(scaler, f_Xs_tr; range=range, minmax_output=minmax))
            Xs_val_scaled = permutedims(transform_data(scaler, f_Xs_val; range=range, minmax_output=minmax))


            ########### ATTENTION: due to permutedims, data has been transformed to column major order (each timeseries is a column) ############


            s = EncodeSeparate{opts.encode_classes_separately}()
            training_states, enc_args_tr = encode_dataset(s, Xs_train_scaled, f_ys_tr, "train", sites[di]; opts=opts, class_keys=class_keys)
            validation_states, enc_args_val = encode_dataset(s, Xs_val_scaled, f_ys_val, "valid", sites[di]; opts=opts, class_keys=class_keys)
            
            Xs_train_enc[di, f] = training_states
            Xs_val_enc[di,f] = validation_states
        end

    end

    # generated: masteropts, Ws, Xs_train_enc, Xs_val_enc, path, logfile, resfile, finfile 

    
    return search_parameter_space(search, Ws, masteropts, Xs_train_enc, Xs_val_enc; path=path, logfile=logfile, resfile=resfile, finfile=finfile, kwargs...)
end


