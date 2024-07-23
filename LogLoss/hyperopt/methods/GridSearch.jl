
struct GridSearch <: SearchMethod 
    encodings::AbstractVector{<:Encoding}
    nfolds::Integer
    etas::AbstractVector{<:Number}
    ds::AbstractVector{<:Integer}
    chi_maxs::AbstractVector{<:Integer}
    max_sweeps::Integer
    instance_name::String
end

function GridSearch(;
    encodings::Union{Nothing, AbstractVector{<:Encoding}}=nothing,
    encoding::Union{Nothing, Encoding}=nothing,
    nfolds::Integer=10,
    etas::AbstractVector{<:Number}=[0.1], 
    ds::AbstractVector{<:Integer}=[2,3,4], 
    chi_maxs::AbstractVector{<:Integer}=[15,20,25,30,35], 
    max_sweeps::Integer=10,
    instance_name::String="GridSearch($(nfolds)fold_ns$(max_sweeps)_eta$(repr_vec(etas))_chis$(repr_vec(chi_maxs))_ds$(repr_vec(ds)))")

    enc = configure_encodings(encodings, encoding)

    return GridSearch(enc, nfolds,etas, ds, chi_maxs, max_sweeps, instance_name)
end



""" Safely handle checking for previous GridSearch hyperoptimisations with the same canonical name. Handles the options to load/overwrite/abort"""
function load_hyperopt!(GS::GridSearch, path::String;
    force_overwrite::Bool,
    always_abort::Bool,
    logfile::String=path*"log.txt", 
    resfile::String=path*"results.jld2", 
    finfile::String=path*"finished.jld2")
    # Searches for a saved hyperopt under path "path". If it exists, then offer to load/overwrite/abort.
    # If it doesn't exist, create it.

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
            results = Array{Union{Result,Missing}}(missing, GS.nfolds,  GS.max_sweeps+1, length(GS.etas), length(GS.ds), length(GS.chi_maxs), length(GS.encodings)) # Somewhere to save the results for no sweeps up to max_sweeps
    
        elseif isfile(resfile)
            resume = check_status(resfile, GS.nfolds, GS.etas, GS.chi_maxs, GS.ds, GS.encodings)
            if resume
                results, fold_r, nfolds_r, max_sweeps_r, eta_r, etas_r, chi_r, chi_maxs_r, d_r, ds_r, e_r, encodings_r = load_result(resfile) 
                done = Int(sum((!ismissing).(results)) / (GS.max_sweeps+1))
                todo = Int(prod(size(results)) / (GS.max_sweeps+1))
                println("Found interrupted benchmark with $(done)/$(todo) trains complete, resuming")
    
            else
                results, fold_r, nfolds_r, max_sweeps_r, eta_r, etas_r, chi_r, chi_maxs_r, d_r, ds_r, e_r, encodings_r = load_result(resfile) 
                error("??? A status file exists but the parameters don't match!\nnfolds=$(nfolds_r)\netas=$(etas_r)\nns=$(max_sweeps_r)\nchis=$(chi_maxs_r)\nds=$(ds_r)")
            end
        else
            error("A non benchmark folder with the name\n$path\nAlready exists")
        end
    else
        results = Array{Union{Result,Missing}}(missing, GS.nfolds,  GS.max_sweeps+1, length(GS.etas), length(GS.ds), length(GS.chi_maxs), length(GS.encodings)) # Somewhere to save the results for no sweeps up to max_sweeps
    end

    # make the folders and output file if they dont already exist
    if !isdir(path) 
        mkdir(path)
        save_results(resfile, results, -1, GS.nfolds, GS.max_sweeps, -1., GS.etas, -1, GS.chi_maxs, -1, GS.ds, first(GS.encodings), GS.encodings) 
    
        f = open(logfile, "w")
        close(f)
    end

end


function search_parameter_space(
    GS::GridSearch, 
    Ws::AbstractVector{MPS},
    masteropts::Options,
    Xs_train_enc::Matrix{EncodedTimeseriesSet}, 
    Xs_val_enc::Matrix{EncodedTimeseriesSet}; 
    path::String, 
    logfile::String, 
    resfile::String, 
    finfile::String,
    distribute::Bool=true, # whether to destroy my ram or not
    skip_low_chi::Bool=true, # whether to skip chi <= 5*d 
    )


    @assert issorted(GS.ds) "Hyperparamater vector \"ds\" is not sorted"
    @assert issorted(GS.etas) "Hyperparamater vector \"etas\" is not sorted"
    @assert issorted(GS.chi_maxs) "Hyperparamater vector \"chi_maxs\" is not sorted"

    results, _... = load_result(resfile) 

    #TODO maybe introduce some artificial balancing on the threads, or use a library like transducers
    writelock = ReentrantLock()
    done = Int(sum((!ismissing).(results)) / (max_sweeps+1))
    todo = Int(prod(size(results)) / (max_sweeps+1))
    tstart = time()
    println("Analysing a $todo size parameter grid")
    if distribute
        # the loop order here is: changes execution time the least -> changes execution time the most
        @sync for f in 1:GS.nfolds, (etai, eta) in enumerate(GS.etas), (ei, e) in enumerate(GS.encodings), (di,d) in enumerate(GS.ds), (chmi, chi_max) in enumerate(GS.chi_maxs)
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
                results[f, :, etai, di, chmi, ei] = [res_by_sweep; [res_by_sweep[end] for _ in 1:(GS.max_sweeps+1-length(res_by_sweep))]] # if the training exits early (as it should) then repeat the final value
                lock(writelock)
                try
                    done +=1
                    println("Finished $done/$todo in $(length(res_by_sweep)-1) sweeps at t=$(time() - tstart)")
                    save_results(resfile, results, f, Gs.nfolds, Gs.max_sweeps, eta, Gs.etas, chi_max, Gs.chi_maxs, d, Gs.ds, e, Gs.encodings) 
                finally
                    unlock(writelock)
                end
            end
        end
    else
        for f in 1:GS.nfolds, (etai, eta) in enumerate(GS.etas), (ei, e) in enumerate(GS.encodings), (di,d) in enumerate(GS.ds), (chmi, chi_max) in enumerate(GS.chi_maxs)
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
            results[f, :, etai, di, chmi, ei] = [res_by_sweep; [res_by_sweep[end] for _ in 1:(GS.max_sweeps+1-length(res_by_sweep))]] # if the training exits early (as it should) then repeat the final value

            done +=1
            println("Finished $done/$todo in $(length(res_by_sweep)-1) sweeps at t=$(time() - tstart)")
            save_results(resfile, results, f, GS.nfolds, GS.max_sweeps, eta, GS.etas, chi_max, GS.chi_maxs, d, ds, e, GS.encodings) 

        end
    end
    save_status(finfile, GS.nfolds, GS.nfolds, last(GS.etas), GS.etas, last(GS.chi_maxs), GS.chi_maxs, last(GS.ds), ds, last(GS.encodings), GS.encodings)

    return results
end