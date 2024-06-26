# gridsearch hyperparameter opt
using Base.Threads

include("../RealRealHighDimension.jl")
include("hyperUtils.jl")


function hyperopt(encoding::Encoding, Xs_train::AbstractMatrix, ys_train::AbstractVector, Xs_val::AbstractMatrix, ys_val::AbstractVector; 
    method="GridSearch", 
    etas::Vector{<:Number}, 
    max_sweeps::Integer, 
    ds::Vector{<:Integer}, 
    chi_maxs::Vector{<:Integer}, 
    chi_init::Integer=4,
    nfolds::Integer=1,
    mps_seed::Real=456,
    kfoldseed::Real=1234567890, # overridden by the rng parameter
    rng::AbstractRNG=MersenneTwister(kfoldseed),
    update_iters::Integer=1,
    verbosity::Real=-1,
    dtype::Type = encoding.iscomplex ? ComplexF64 : Float64,
    loss_grad::Function=loss_grad_KLD,
    bbopt::BBOpt=BBOpt("CustomGD", "TSGO"),
    track_cost::Bool=false,
    rescale::Tuple{Bool,Bool}=(false, true),
    aux_basis_dim::Integer=2,
    encode_classes_separately::Bool=false,
    train_classes_separately::Bool=false,
    minmax::Bool=true,
    cutoff::Number=1e-10,
    force_overwrite=false,
    always_abort=false,
    dir="LogLoss/hyperopt/"
    )

    if force_overwrite && always_abort 
        error("You can't force_overwrite and always_abort that doesn't make any sense")
    end

    ########## Sanity checks ################
    if encoding.iscomplex
        if dtype <: Real
            error("Using a complex valued encoding but the MPS is real")
        end

    elseif !(dtype <: Real)
        @warn "Using a complex valued MPS but the encoding is real"
    end


    @assert issorted(ds) "Hyperparamater vector \"ds\" is not sorted"
    @assert issorted(etas) "Hyperparamater vector \"etas\" is not sorted"
    @assert issorted(chi_maxs) "Hyperparamater vector \"chi_maxs\" is not sorted"


    # data is _input_ in python canonical (row major) format
    @assert size(Xs_train, 1) == size(ys_train, 1) "Size of training dataset and number of training labels are different!"
    @assert size(Xs_val, 1) == size(ys_val, 1) "Size of validation dataset and number of validation labels are different!"

    


    ############### Data structures aand definitions ########################
    masteropts = Options(; nsweeps=max_sweeps, chi_max=1, d=1, eta=1, cutoff=cutoff, update_iters=update_iters, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad,
        bbopt=bbopt, track_cost=track_cost, rescale = rescale, aux_basis_dim=aux_basis_dim, encoding=encoding, encode_classes_separately=encode_classes_separately,
        train_classes_separately=train_classes_separately, minmax=minmax)

    
    # Output files
    function repr_vec(v::AbstractVector) 
        c = minimum(abs.(diff(v)), init=1.)
        midstr = c == 1. ? "" : "$(c):"
        "$(first(v)):$(midstr)$(last(v))"
    end
    
    vstring = train_classes_separately ? "Split_train_" : ""
    pstr = encoding.name * "_" * vstring * "$(nfolds)fold_r$(mps_seed)_eta$(repr_vec(etas))_ns$(max_sweeps)_chis$(repr_vec(chi_maxs))_ds$(repr_vec(ds))"


    path = dir* pstr *"/"
    svfol = path*"data/"
    logpath = path # could change to a new folder if we wanted

    logfile = logpath * "log.txt"
    resfile = logpath  * "results.jld2"
    finfile = logpath * "finished.jld2"
    encodings = [encoding] # backwards compatibility reasons (/future feature)

    ######## Check if hyperopt already exists?  ################
    # resume if possible

if isdir(path) && !isempty(readdir(path))
    files = sort(readdir(path))
    safe_dir = all(files == sort(["log.txt", "results.jld2"])) || all(files == sort(["log.txt", "results.jld2", "finished.jld2"]))

    if !safe_dir
        error("Unknown (or missing) files in \"$path\". Move your data or it could get deleted!")
    end

    if isfile(finfile)
        if always_abort
            error("Aborting to conserve existing data")
        elseif !force_overwrite

            while true
                print("A hyperopt with these parameters already exists, overwrite the contents of \"$path\"? [y/n]: ")
                input = lowercase(readline())
                if input == "y"
                    break
                elseif input == "n"
                    error("Aborting to conserve existing data")
                end
            end
        end
        # Remove the saved files
        # the length is for safety so we can never recursively remove something terrible like "/" (shout out to the steam linux runtime)
        if isdir(path) && length(path) >=3 
            rm(logfile)
            rm(resfile)
            rm(finfile)
            rm(path; recursive=false ) 
        end
        results = Array{Union{Result,Missing}}(missing, nfolds,  max_sweeps+1, length(etas), length(ds), length(chi_maxs), length(encodings)) # Somewhere to save the results for no sweeps up to max_sweeps

    elseif isfile(resfile)
        resume = check_status(resfile, nfolds, etas, chi_maxs, ds, encodings)
        if resume
            results, fold_r, nfolds_r, max_sweeps_r, eta_r, etas_r, chi_r, chi_maxs_r, d_r, ds_r, e_r, encodings_r = load_result(resfile) 
            done = Int(sum((!ismissing).(results)) / (max_sweeps+1))
            todo = Int(prod(size(results)) / (max_sweeps+1))
            println("Found interrupted benchmark with $(done)/$(todo) trains complete, resuming")

        else
            error("??? A status file exists but the parameters don't match!\nnfolds=$(nfolds_r)\netas=$(repr_vec(etas_r))\nns=$(max_sweeps_r)\nchis=$(repr_vec(chi_maxs_r))\nds=$(repr_vec(ds_r))")
        end
    else
        error("A non benchmark folder with the name\n$path\nAlready exists")
    end
else
    results = Array{Union{Result,Missing}}(missing, nfolds,  max_sweeps+1, length(etas), length(ds), length(chi_maxs), length(encodings)) # Somewhere to save the results for no sweeps up to max_sweeps
end

# make the folders and output file if they dont already exist
if !isdir(path) 
    mkdir(path)
    save_results(resfile, results, -1, nfolds, max_sweeps, -1., etas, -1, chi_maxs, -1, ds, first(encodings), encodings) 

    f = open(logfile, "w")
    close(f)
end



################### Definitions continued ##########################


    # all data concatenated for folding purposes
    Xs = [Xs_train ; Xs_val]
    ys = [ys_train ; ys_val]

    ntrs = length(ys_train)
    nvals = length(ys_val)

    num_mps_sites = size(Xs_train, 2)
    classes = unique(ys)
    num_classes = length(classes)


    # A few more checks
    @assert num_mps_sites == size(Xs_val, 2) "The number of sites supported by the training and validation data do not match! "
    @assert eltype(classes) <: Integer "Classes must be integers"
    sort!(classes)
    class_keys = Dict(zip(classes, 1:num_classes)) # Assign each class a 'Key' from 1 to n
    

    fold_inds = Array{Vector{Integer}}(undef, 2, nfolds) # Stores the indices that map the validation folds to the original data

    Xs_train_enc = Matrix{EncodedTimeseriesSet}(undef, length(ds), nfolds) # Training data for each dimension and each fold
    Xs_val_enc = Matrix{EncodedTimeseriesSet}(undef, length(ds), nfolds) # Validation data for each dimension and each fold


    sites = Array{Vector{Index{Int64}}}(undef, num_mps_sites, length(ds)) # Array to hold the siteindices for each dimension
    Ws = Vector{MPS}(undef, length(ds)) # stores an MPS for each encoding dimension



    ############### generate the starting MPS for each d ####################

    #TODO parallelise
    for (di,d) in enumerate(ds)
        opts = _set_options(masteropts; d=d, verbosity=-1)

        sites[di] = siteinds(d, num_mps_sites)
           # generate the starting MPS with uniform bond dimension chi_init and random values (with seed if provided)
        Ws[di] = generate_startingMPS(chi_init, sites[di]; num_classes=num_classes, random_state=mps_seed, opts=opts)
    end


    
    #############  initialise the fold indices  ####################
    fold_inds[:,1] = [collect(1:ntrs), collect((ntrs+1):(ntrs+nvals))]

    for i in 2:nfolds
        inds = randperm(rng, ntrs + nvals)
        fold_inds[:,i] = [inds[1:ntrs], inds[(ntrs+1):(ntrs+nvals)]]
    end



    ########### Perform the encoding step for all d and f ############
    #TODO Check if it is correct to scale with Xs or just with Xs_train; 
    scaler = fit_scaler(RobustSigmoidTransform, Xs);


    #TODO can encode more efficiently for certain encoding types if not time / order dependent. This will save a LOT of memory
    #TODO parallelise
    for f in 1:nfolds, (di,d) in enumerate(ds)
        opts= _set_options(masteropts;  d=d, verbosity=-1)

        tr_inds = fold_inds[1, f]
        val_inds = fold_inds[2, f]
        local f_Xs_tr = Xs[tr_inds, :]
        local f_Xs_val = Xs[val_inds, :]

        local f_ys_tr = ys[tr_inds]
        local f_ys_val = ys[val_inds]


        range = opts.encoding.range
        Xs_train_scaled = permutedims(transform_data(scaler, f_Xs_tr; range=range, minmax_output=minmax))
        Xs_val_scaled = permutedims(transform_data(scaler, f_Xs_val; range=range, minmax_output=minmax))


        ########### ATTENTION: due to permutedims, data has been transformed to column major order (each timeseries is a column) #########


        s = EncodeSeparate{opts.encode_classes_separately}()
        training_states, enc_args_tr = encode_dataset(s, Xs_train_scaled, f_ys_tr, "train", sites[di]; opts=opts, class_keys=class_keys)
        validation_states, enc_args_val = encode_dataset(s, Xs_val_scaled, f_ys_val, "valid", sites[di]; opts=opts, class_keys=class_keys)
        
        # enc_args = vcat(enc_args_tr, enc_args_val) 
        Xs_train_enc[di, f] = training_states
        Xs_val_enc[di,f] = training_states

    end



    #TODO maybe introduce some artificial balancing on the threads, or use a library like transducers
    writelock = ReentrantLock()
    # the loop order here is: changes execution time the least -> changes execution time the most
    @sync for f in 1:nfolds, (etai, eta) in enumerate(etas), (ei, e) in enumerate(encodings), (di,d) in enumerate(ds), (chmi, chi_max) in enumerate(chi_maxs)
        # if the loop will be continued instantly don't bother with the overhead of spawning a task
        !ismissing(results[f, 1, etai, di, chmi]) && continue
        isodd(d) && titlecase(e.name) == "Sahand" && continue
        d != 2 && titlecase(e.name) == "Stoudenmire" && continue

        @spawn begin
            opts = _set_options(masteropts;  d=d, encoding=e, eta=eta, chi_max=chi_max)
            W_init = Ws[di]
            local f_training_states_meta = Xs_train_enc[di, f]
            local f_validation_states_meta = Xs_val_enc[di, f]

            _, info, _,_ = fitMPS(W_init, f_training_states_meta, f_validation_states_meta; opts=opts)


            results[f, :, etai, di, chmi, ei] = Result(info)
            lock(writelock)
            try
                save_results(resfile, results, f, nfolds, max_sweeps, eta, etas, chi_max, chi_maxs, d, ds, e, encodings) 
            finally
                unlock(writelock)
            end
        end
    end
    save_status(finfile, nfolds, nfolds, last(etas), etas, last(chi_maxs), chi_maxs, last(ds), ds, last(encodings), encodings)

    return results
end
