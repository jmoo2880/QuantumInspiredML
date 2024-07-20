# gridsearch hyperparameter opt
using Base.Threads
using LineSearches

include("../RealRealHighDimension.jl")
include("hyperUtils.jl")

using MLJBase: train_test_pairs, StratifiedCV

abstract type SearchMethod end

struct GridSearch <: SearchMethod end
struct HGradientDescent <: SearchMethod end

hyperopt(enc::Encoding, args...; kwargs...) = hyperopt(GridSearch(), enc, args...; kwargs...) # default to a gridsearch

function hyperopt(::GridSearch, encoding::Encoding, Xs::AbstractMatrix, ys::AbstractVector; 
    etas::AbstractVector{<:Number}, 
    max_sweeps::Integer, 
    ds::AbstractVector{<:Integer}, 
    chi_maxs::AbstractVector{<:Integer}, 
    chi_init::Integer=4,
    train_ratio=0.9,
    force_complete_crossval::Bool=true, # overrides train_ratio
    nfolds::Integer= force_complete_crossval ? round(Int, ceil(1 / (1-train_ratio); digits=5)) : 1, # you can use this to override the number of folds you _should_ use, but don't. You need the round for floating point reasons
    mps_seed::Real=4567,
    kfoldseed::Real=1234567890, # overridden by the rng parameter
    foldrng::AbstractRNG=MersenneTwister(kfoldseed),
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
    force_overwrite::Bool=false,
    always_abort::Bool=false,
    dir::String="LogLoss/hyperopt/",
    distribute::Bool=true, # whether to destroy my ram or not
    exit_early::Bool=true,
    skip_low_chi::Bool=true,
    sigmoid_transform::Bool=false
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
    @assert size(Xs, 1) == size(ys, 1) "Size of training dataset and number of training labels are different!"

    


    ############### Data structures aand definitions ########################
    println("Allocating initial Arrays and checking for existing files")
    masteropts = Options(; nsweeps=max_sweeps, chi_max=1, d=1, eta=1, cutoff=cutoff, update_iters=update_iters, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad,
        bbopt=bbopt, track_cost=track_cost, rescale = rescale, aux_basis_dim=aux_basis_dim, encoding=encoding, encode_classes_separately=encode_classes_separately,
        train_classes_separately=train_classes_separately, minmax=minmax, exit_early=exit_early, sigmoid_transform=sigmoid_transform)

    
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
            rm(path; recursive=false) # safe because it will only remove empty directories 
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
            results, fold_r, nfolds_r, max_sweeps_r, eta_r, etas_r, chi_r, chi_maxs_r, d_r, ds_r, e_r, encodings_r = load_result(resfile) 
            error("??? A status file exists but the parameters don't match!\nnfolds=$(nfolds_r)\netas=$(etas_r)\nns=$(max_sweeps_r)\nchis=$(chi_maxs_r)\nds=$(ds_r)")
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
    # ntrs = round(Int, length(ys) *train_ratio)
    # nvals = length(ys) - ntrs

    num_mps_sites = size(Xs, 2)
    classes = unique(ys)
    num_classes = length(classes)


    # A few more checks
    @assert eltype(classes) <: Integer "Classes must be integers"
    sort!(classes)
    class_keys = Dict(zip(classes, 1:num_classes)) # Assign each class a 'Key' from 1 to n
    

    Xs_train_enc = Matrix{EncodedTimeseriesSet}(undef, length(ds), nfolds) # Training data for each dimension and each fold
    Xs_val_enc = Matrix{EncodedTimeseriesSet}(undef, length(ds), nfolds) # Validation data for each dimension and each fold


    sites = Array{Vector{Index{Int64}}}(undef, num_mps_sites, length(ds)) # Array to hold the siteindices for each dimension
    Ws = Vector{MPS}(undef, length(ds)) # stores an MPS for each encoding dimension



    ############### generate the starting MPS for each d ####################
    println("Generating $(length(ds)) initial MPSs")
    #TODO parallelise
    for (di,d) in enumerate(ds)
        opts = _set_options(masteropts; d=d, verbosity=-1)

        sites[di] = siteinds(d, num_mps_sites)
           # generate the starting MPS with uniform bond dimension chi_init and random values (with seed if provided)
        Ws[di] = generate_startingMPS(chi_init, sites[di]; num_classes=num_classes, random_state=mps_seed, opts=opts)
    end


    
    #############  initialise the folds  ####################

    nvirt_folds = max(round(Int, ceil(1 / (1-train_ratio); digits=5)), nfolds)
    scv = StratifiedCV(;nfolds=nvirt_folds, rng=foldrng)
    fold_inds = train_test_pairs(scv, eachindex(ys), ys)




    ########### Perform the encoding step for all d and f ############
    #TODO can encode more efficiently for certain encoding types if not time / order dependent. This will save a LOT of memory
    #TODO parallelise
    println("Encoding $nfolds folds with $(length(ds)) different encoding dimensions")
    @sync for f in 1:nfolds, (di,d) in enumerate(ds)
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


            range = opts.encoding.range
            scaler = fit_scaler(RobustSigmoidTransform, f_Xs_tr);
            Xs_train_scaled = permutedims(transform_data(scaler, f_Xs_tr; range=range, minmax_output=minmax))
            Xs_val_scaled = permutedims(transform_data(scaler, f_Xs_val; range=range, minmax_output=minmax))


            ########### ATTENTION: due to permutedims, data has been transformed to column major order (each timeseries is a column) #########


            s = EncodeSeparate{opts.encode_classes_separately}()
            training_states, enc_args_tr = encode_dataset(s, Xs_train_scaled, f_ys_tr, "train", sites[di]; opts=opts, class_keys=class_keys)
            validation_states, enc_args_val = encode_dataset(s, Xs_val_scaled, f_ys_val, "valid", sites[di]; opts=opts, class_keys=class_keys)
            
            # enc_args = vcat(enc_args_tr, enc_args_val) 
            Xs_train_enc[di, f] = training_states
            Xs_val_enc[di,f] = validation_states
        end

    end



    #TODO maybe introduce some artificial balancing on the threads, or use a library like transducers
    writelock = ReentrantLock()
    done = Int(sum((!ismissing).(results)) / (max_sweeps+1))
    todo = Int(prod(size(results)) / (max_sweeps+1))
    tstart = time()
    println("Analysing a $todo size parameter grid")
    if distribute
        # the loop order here is: changes execution time the least -> changes execution time the most
        @sync for f in 1:nfolds, (etai, eta) in enumerate(etas), (ei, e) in enumerate(encodings), (di,d) in enumerate(ds), (chmi, chi_max) in enumerate(chi_maxs)
            !ismissing(results[f, 1, etai, di, chmi]) && continue
            isodd(d) && titlecase(e.name) == "Sahand" && continue
            d != 2 && titlecase(e.name) == "Stoudenmire" && continue

            @spawn begin
                opts = _set_options(masteropts;  d=d, encoding=e, eta=eta, chi_max=chi_max)
                W_init = deepcopy(Ws[di])
                local f_training_states_meta = Xs_train_enc[di, f]
                local f_validation_states_meta = Xs_val_enc[di, f]

                _, info, _, _ = fitMPS(W_init, f_training_states_meta, f_validation_states_meta; opts=opts)


                res_by_sweep = Result(info)
                results[f, :, etai, di, chmi, ei] = [res_by_sweep; [res_by_sweep[end] for _ in 1:(max_sweeps+1-length(res_by_sweep))]] # if the training exits early (as it should) then repeat the final value
                lock(writelock)
                try
                    done +=1
                    println("Finished $done/$todo in $(length(res_by_sweep)-1) sweeps at t=$(time() - tstart)")
                    save_results(resfile, results, f, nfolds, max_sweeps, eta, etas, chi_max, chi_maxs, d, ds, e, encodings) 
                finally
                    unlock(writelock)
                end
            end
        end
    else
        for f in 1:nfolds, (etai, eta) in enumerate(etas), (ei, e) in enumerate(encodings), (di,d) in enumerate(ds), (chmi, chi_max) in enumerate(chi_maxs)
            !ismissing(results[f, 1, etai, di, chmi]) && continue
            isodd(d) && titlecase(e.name) == "Sahand" && continue
            d != 2 && titlecase(e.name) == "Stoudenmire" && continue
            if skip_low_chi && chi_max < 5 * d
                continue
            end

            
            opts = _set_options(masteropts;  d=d, encoding=e, eta=eta, chi_max=chi_max)
            W_init = deepcopy(Ws[di])
            local f_training_states_meta = Xs_train_enc[di, f]
            local f_validation_states_meta = Xs_val_enc[di, f]

            _, info, _, _ = fitMPS(W_init, f_training_states_meta, f_validation_states_meta; opts=opts)

            res_by_sweep = Result(info)
            results[f, :, etai, di, chmi, ei] = [res_by_sweep; [res_by_sweep[end] for _ in 1:(max_sweeps+1-length(res_by_sweep))]] # if the training exits early (as it should) then repeat the final value

            done +=1
            println("Finished $done/$todo in $(length(res_by_sweep)-1) sweeps at t=$(time() - tstart)")
            save_results(resfile, results, f, nfolds, max_sweeps, eta, etas, chi_max, chi_maxs, d, ds, e, encodings) 

        end
    end
    save_status(finfile, nfolds, nfolds, last(etas), etas, last(chi_maxs), chi_maxs, last(ds), ds, last(encodings), encodings)

    return results
end





function hyperopt(::HGradientDescent, encoding::Encoding, Xs::AbstractMatrix, ys::AbstractVector; 
    eta_init::Real=0.01, # if you want eta to be a complex number, fix the indexing for complex numbers in hyperUtils.eta_to_index()
    eta_range::AbstractBounds{<:Real} = (0.001,0.5),
    deta_perc::Real=0.1, # step in eta used to calculate the derivative as a percentage
    min_eta_eta::Real=0.0005, # minimum step in eta
    eta_eta_init::Real=1.,
    min_eta_step::Real = eta_range[1]/1000, # servers aas 'epsilon' for eta. Used to make saving and reading to a dictionary indexed by eta consistent
    d_init::Number=2, 
    d_range::AbstractBounds{<:Integer}=(2,20),
    d_step::Integer=1,
    chi_max_init::Number=25, 
    chi_max_range::AbstractBounds{<:Integer}=(15,80),
    chi_step::Integer=1,
    max_sweeps::Integer=10, 
    max_grad_steps::Number=Inf,
    chi_init::Integer=4,
    train_ratio=0.9,
    force_complete_crossval::Bool=true, # overrides train_ratio
    nfolds::Integer= force_complete_crossval ? round(Int, ceil(1 / (1-train_ratio); digits=5)) : 1, # you can use this to override the number of folds you _should_ use, but don't. You need the round for floating point reasons
    mps_seed::Real=4567,
    kfoldseed::Real=1234567890, # overridden by the rng parameter
    foldrng::AbstractRNG=MersenneTwister(kfoldseed),
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
    force_overwrite::Bool=false,
    always_abort::Bool=false,
    dir::String="LogLoss/hyperopt/",
    exit_early::Bool=true,
    use_backtracking_linesearch::Bool=true
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


    @assert issorted(d_range) "Min dimension is less than max dimension"
    @assert issorted(eta_range) "Min learning rate is less than max learning rate"
    @assert issorted(chi_max_range) "Min chi_max is less than max chi_max"


    # data is _input_ in python canonical (row major) format
    @assert size(Xs, 1) == size(ys, 1) "Size of training dataset and number of training labels are different!"

    


    ############### Data structures aand definitions ########################
    println("Allocating initial Arrays and checking for existing files")
    masteropts = Options(; nsweeps=max_sweeps, chi_max=1, d=1, eta=1, cutoff=cutoff, update_iters=update_iters, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad,
        bbopt=bbopt, track_cost=track_cost, rescale = rescale, aux_basis_dim=aux_basis_dim, encoding=encoding, encode_classes_separately=encode_classes_separately,
        train_classes_separately=train_classes_separately, minmax=minmax, exit_early=exit_early)

    
    # Output files
    function repr_vec(v::AbstractVector) 
        c = minimum(abs.(diff(v)), init=1.)
        midstr = c == 1. ? "" : "$(c):"
        "$(first(v)):$(midstr)$(last(v))"
    end
    
    vstring = train_classes_separately ? "Split_train_" : ""
    pstr = encoding.name * "_" * vstring * "GD_$(nfolds)fold_r$(mps_seed)_eta$(eta_range)_ns$(max_sweeps)_chis$(chi_max_range)_ds$(d_range)"


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
            rm(path; recursive=false) # safe because it will only remove empty directories 
        end
        results = ResDict() # Somewhere to save the results for no sweeps up to max_sweeps

    elseif isfile(resfile)
        resume = check_status(resfile, nfolds, eta_range, chi_max_range, d_range, encodings)
        if resume
            results, fold_r, nfolds_r, max_sweeps_r, eta_r, eta_range_r, chi_r, chi_max_range_r, d_r, d_range_r, e_r, encodings_r = load_result(resfile) 
            println("Found interrupted benchmark, resuming")

            eta_init, chi_max_init, d_init = eta_r, chi_r, d_r

        else
            results, fold_r, nfolds_r, max_sweeps_r, eta_r, eta_range_r, chi_r, chi_max_range_r, d_r, d_range_r, e_r, encodings_r = load_result(resfile) 
            error("??? A status file exists but the parameters don't match!\nnfolds=$(nfolds_r)\netas=$(etas_r)\nns=$(max_sweeps_r)\nchis=$(chi_maxs_r)\nds=$(ds_r)")
        end
    else
        error("A non benchmark folder with the name\n$path\nAlready exists")
    end
else
    results = ResDict() # Somewhere to save the results for no sweeps up to max_sweeps
end

# make the folders and output file if they dont already exist
if !isdir(path) 
    mkdir(path)
    save_results(resfile, results, -1, nfolds, max_sweeps, eta_init, eta_range, chi_max_init, chi_max_range, d_init, d_range, first(encodings), encodings) 

    f = open(logfile, "w")
    close(f)
end



################### Definitions continued ##########################


    # all data concatenated for folding purposes
    # ntrs = round(Int, length(ys) *train_ratio)
    # nvals = length(ys) - ntrs
    ds = range(d_range...) |> collect

    num_mps_sites = size(Xs, 2)
    classes = unique(ys)
    num_classes = length(classes)


    # A few more checks
    @assert eltype(classes) <: Integer "Classes must be integers"
    sort!(classes)
    class_keys = Dict(zip(classes, 1:num_classes)) # Assign each class a 'Key' from 1 to n
    

    Xs_train_enc = Matrix{EncodedTimeseriesSet}(undef, length(ds), nfolds) # Training data for each dimension and each fold
    Xs_val_enc = Matrix{EncodedTimeseriesSet}(undef, length(ds), nfolds) # Validation data for each dimension and each fold


    sites = Array{Vector{Index{Int64}}}(undef, num_mps_sites, length(ds)) # Array to hold the siteindices for each dimension
    Ws = Vector{MPS}(undef, length(ds)) # stores an MPS for each encoding dimension



    ############### generate the starting MPS for each d ####################
    println("Generating $(length(ds)) initial MPSs")
    #TODO parallelise
    for (di,d) in enumerate(ds)
        opts = _set_options(masteropts; d=d, verbosity=-1)

        sites[di] = siteinds(d, num_mps_sites)
           # generate the starting MPS with uniform bond dimension chi_init and random values (with seed if provided)
        Ws[di] = generate_startingMPS(chi_init, sites[di]; num_classes=num_classes, random_state=mps_seed, opts=opts)
    end


    
    #############  initialise the folds  ####################

    nvirt_folds = max(round(Int, ceil(1 / (1-train_ratio); digits=5)), nfolds)
    scv = StratifiedCV(;nfolds=nvirt_folds, rng=foldrng)
    fold_inds = train_test_pairs(scv, eachindex(ys), ys)




    ########### Perform the encoding step for all d and f ############
    #TODO can encode more efficiently for certain encoding types if not time / order dependent. This will save a LOT of memory
    #TODO parallelise
    println("Encoding $nfolds folds with $(length(ds)) different encoding dimensions")
    @sync for f in 1:nfolds, (di,d) in enumerate(ds)
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


            range = opts.encoding.range
            scaler = fit_scaler(RobustSigmoidTransform, f_Xs_tr);
            Xs_train_scaled = permutedims(transform_data(scaler, f_Xs_tr; range=range, minmax_output=minmax))
            Xs_val_scaled = permutedims(transform_data(scaler, f_Xs_val; range=range, minmax_output=minmax))


            ########### ATTENTION: due to permutedims, data has been transformed to column major order (each timeseries is a column) #########


            s = EncodeSeparate{opts.encode_classes_separately}()
            training_states, enc_args_tr = encode_dataset(s, Xs_train_scaled, f_ys_tr, "train", sites[di]; opts=opts, class_keys=class_keys)
            validation_states, enc_args_val = encode_dataset(s, Xs_val_scaled, f_ys_val, "valid", sites[di]; opts=opts, class_keys=class_keys)
            
            # enc_args = vcat(enc_args_tr, enc_args_val) 
            Xs_train_enc[di, f] = training_states
            Xs_val_enc[di,f] = validation_states
        end

    end


    # Begin!
    global tstart = time()




    function folds_acc(res::Matrix{Result})
        return mean(getfield.(res[:,end], :acc))
    end

    function folds_acc(::Missing)
        return missing
    end

    function eval_gridpoint!(results, res_keys, nfolds, max_sweeps, eta_f, chi_max, d, e)

        # deal with floating point tomfoolery
        local eta_ind = eta_to_index(eta_f, min_eta_step)
        local eta = index_to_eta(eta_ind, min_eta_step)

        res_keys = res_keys |> collect
        key = findfirst(k -> k == (eta_ind, chi_max, d, e), res_keys)
    

        if !isnothing(key)
            return folds_acc(results[res_keys[key]])
        end

        if (isodd(d) && titlecase(e.name) == "Sahand") || (d != 2 && titlecase(e.name) == "Stoudenmire") 
            return results[eta, chi_max, d, e] = missing
        end


        local di = findfirst(d .== ds)
        local resmat = Matrix{Result}(undef, nfolds, max_sweeps+1)
        print("t=$(round(time() - tstart,digits=2))s: Evaluating a $nfolds fold mps with eta=$eta, chi_max=$(chi_max), d=$d... ")
        for f in 1:nfolds
            local opts = _set_options(masteropts;  d=d, encoding=e, eta=eta, chi_max=chi_max)
            local W_init = deepcopy(Ws[di])
            local f_training_states_meta = Xs_train_enc[di, f]
            local f_validation_states_meta = Xs_val_enc[di, f]

            _, info, _, _ = fitMPS(W_init, f_training_states_meta, f_validation_states_meta; opts=opts)

            res_by_sweep = Result(info)
            resmat[f,:] = [res_by_sweep; [res_by_sweep[end] for _ in 1:(max_sweeps+1-length(res_by_sweep))]] # if the training exits early (as it should) then repeat the final value. Its helpful to know the final sweep value is always the result irrespective of how many sweeps the training actually took

        end
        results[eta_ind, chi_max, d, e]  = resmat
        local acc = folds_acc(resmat)
        println("acc = $acc")
        return acc
    end

    function step_bound(i::Number, a::Number, b::Number, step::Number=1)
        if i - step < a 
            if i + step > b
                return (i)
            else
                return (i,i+step)
            end

        elseif i + step > b 
            return (i-step,i)

        else
            return (i-step,i,i+step)
        end
    end

    function acc_step!(alpha::Number, results, nfolds, max_sweeps, eta::Number, g_approx::Number, chi_max, d, e)
        local eta_new = eta+alpha*g_approx
        if eta_new < eta_range[1] || eta_new > eta_range[2]
            return 0

        else
            return eval_gridpoint!(results, keys(results), nfolds, max_sweeps, eta_new, chi_max, d, e)
        end
        
    end


    
    eta, chi_max, d, e = eta_init, chi_max_init, d_init, encoding
    acc_prev = eval_gridpoint!(results, keys(results), nfolds, max_sweeps, eta, chi_max, d, e)
    acc = acc_prev
    
    finished = false

    #TODO maybe introduce some artificial balancing on the threads, or use a library like transducers
    
    println("Beginning search")
    nsteps = 0
    linesearch = BackTracking(order=3) # may be unused
    while !finished && nsteps < max_grad_steps
        @assert !ismissing(acc_prev) "Current accuracy cannot be \"missing\"! Check the initial points are valid: acc(eta=$eta, chi_max=$chi_max, d=$d, enc=$(encoding.name) = missing)"
        nsteps += 1

        println("t=$(round(time() - tstart,digits=2))s: Beginning chi/d step opt for step $nsteps in , current accuracy is $acc with eta=$eta, chi_max=$(chi_max), d=$d")

        # get neighbours
        chi_neighbours = step_bound(chi_max, chi_max_range..., chi_step)
        d_neighbours = step_bound(d, d_range..., d_step)

        neighbours = (Iterators.product(chi_neighbours, d_neighbours) |> collect)[:]

        # step in the d, chi direction
        accs = Vector{Float64}(undef, length(neighbours))
        for (i, n) in enumerate(neighbours)
            accs[i] = eval_gridpoint!(results, keys(results), nfolds, max_sweeps, eta, n..., e) # overhead on duplicates is negligible because of caching
        end


        acc, acc_ind = findmax(accs)

        if acc == acc_prev
            println("chi, d step did nothing!")
            # think about incrementing chi_step/d_step
            chi_d_step = false
        else
            chi_max, d = neighbours[acc_ind]
            chi_d_step = true
            acc_prev = acc
        end

        save_results(resfile, results, nfolds, nfolds, max_sweeps, eta, eta_range, chi_max, chi_max_range, d, d_range, e, encodings) 

        println("t=$(round(time() - tstart,digits=2))s: Finished chi/d step opt for step $nsteps, current accuracy is $acc with eta=$eta, chi_max=$(chi_max), d=$d")

        eta_step = deta_perc/100 * eta
        eta_stepped = false
        acc_forward = eval_gridpoint!(results, keys(results), nfolds, max_sweeps, eta + eta_step, chi_max, d, e)

        eta_eta = eta_eta_init
        g_approx = (acc_forward - acc_prev) / (eta_step) # love me a forward derivative
    

        if use_backtracking_linesearch
            ls_f = alpha -> acc_step!(alpha, results, nfolds, max_sweeps, eta, g_approx, chi_max, d, e)
            eta_eta, acc_new = linesearch(ls_f, eta_eta, acc_prev, g_approx )

            eta_new = eta+eta_eta*g_approx
            if eta_new < eta_range[1] || eta_new > eta_range[2] && eta_eta >= min_eta_eta
                println("Eta step outside of acceptable bounds!")
                println("eta_new = $eta_new, eta_eta = $eta_eta")

            elseif acc_new > acc_prev
                eta_stepped = true
                eta = eta_new
                acc = acc_new
            end
        else
            eta_new = eta + eta_eta*g_approx

            while eta_new < eta_range[1] || eta_new > eta_range[2] && eta_eta >= min_eta_eta
                eta_eta /= 2
                eta_new = eta + eta_eta*g_approx
                if eta_eta < min_eta_eta
                    println("No linesearch done because eta steps out of bounds")
                end
            end

            while eta_eta >= min_eta_eta && abs(eta - eta_new) >= min_eta_step
                
                acc = eval_gridpoint!(results, keys(results), nfolds, max_sweeps, eta_new, chi_max, d, e)
                if acc <= acc_prev
                    eta_eta /= 2
                    eta_new = eta + eta_eta*g_approx
                    #println("eta step did nothing!")
                    # think about incrementing chi_step/d_step
                else
                    eta_stepped = true
                    acc_prev = acc
                    eta = eta_new
                    break
                end

            end
        end
        finished = !chi_d_step && !eta_stepped

        save_results(resfile, results, nfolds, nfolds, max_sweeps, eta, eta_range, chi_max, chi_max_range, d, d_range, e, encodings) 

        println("t=$(round(time() - tstart,digits=2))s: Finished step $nsteps current accuracy is $acc with eta=$eta, chi_max=$(chi_max), d=$d")
        acc_prev = acc
    end
    
    save_status(finfile, nfolds, nfolds, eta, eta_range, chi_max, chi_max_range, d, d_range, last(encodings), encodings)

    return results
end

