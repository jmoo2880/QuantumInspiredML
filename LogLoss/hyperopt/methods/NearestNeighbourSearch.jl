
struct EtaGDChiDNearestNeighbour <: SearchMethod 
    encodings::AbstractVector{<:Encoding}
    nfolds::Integer
    eta_init::Real # if you want eta to be a complex number, fix the indexing for complex numbers in hyperUtils.eta_to_index()
    eta_range::AbstractBounds{<:Real}
    deta_perc::Real # step in eta used to calculate the derivative as a percentage
    min_eta_eta::Real # minimum step in eta
    eta_eta_init::Real
    min_eta_step::Real # servers aas 'epsilon' for eta. Used to make saving and reading to a dictionary indexed by eta consistent
    d_init::Number
    d_range::AbstractBounds{<:Integer}
    d_step::Integer
    chi_max_init::Number
    chi_max_range::AbstractBounds{<:Integer}
    chi_step::Integer
    max_sweeps::Integer 
    max_search_steps::Number
    use_backtracking_linesearch::Bool
    ds::AbstractVector{<:Integer} # Only used to pregenerate the folds/ W_init
    instance_name::String
    max_neighbours::Integer
end

function EtaGDChiDNearestNeighbour(; 
    encodings::Union{Nothing, AbstractVector{<:Encoding}}=nothing,
    encoding::Union{Nothing, Encoding}=nothing,
    nfolds::Integer=10,
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
    max_search_steps::Number=Inf,
    use_backtracking_linesearch::Bool=true,
    instance_name::String="EtaGradDesc($(nfolds)fold_ns$(max_sweeps)_eta$(repr_vec(eta_range))_chis$(repr_vec(chi_max_range))_ds$(repr_vec(d_range)))",
    max_neighbours::Integer=3)

    enc = configure_encodings(encodings, encoding)
    @warn("EtaGDChiDNearestNeighbour is deprecated! Use EtaOptimChiDNearestNeighbour instead (it's just better)")

    return EtaGDChiDNearestNeighbour(enc, nfolds, eta_init, eta_range, deta_perc, min_eta_eta, eta_eta_init, min_eta_step, d_init, d_range, d_step,chi_max_init, chi_max_range, chi_step, max_sweeps, max_search_steps, use_backtracking_linesearch, range(d_range...), instance_name, max_neighbours)
end

struct EtaOptimChiDNearestNeighbour <: SearchMethod 
    encodings::AbstractVector{<:Encoding}
    nfolds::Integer
    eta_init::Real # if you want eta to be a complex number, fix the indexing for complex numbers in hyperUtils.eta_to_index()
    eta_range::AbstractBounds{<:Real}
    min_eta_step::Real
    d_init::Number 
    d_range::AbstractBounds{<:Integer}
    d_step::Integer
    chi_max_init::Number
    chi_max_range::AbstractBounds{<:Integer}
    chi_step::Integer
    max_sweeps::Integer
    max_search_steps::Number
    max_eta_steps::Number
    ds::AbstractVector{<:Integer} # Only used to pregenerate the folds/ W_init
    instance_name::String
    max_neighbours::Integer
    method::String
    immediate_ret::Bool
end

function EtaOptimChiDNearestNeighbour(;
    encodings::Union{Nothing, AbstractVector{<:Encoding}}=nothing,
    encoding::Union{Nothing, Encoding}=nothing,
    nfolds::Integer=10,
    eta_init::Real=0.01, # if you want eta to be a complex number, fix the indexing for complex numbers in hyperUtils.eta_to_index()
    eta_range::AbstractBounds{<:Real} = (0.001,0.5),
    min_eta_step::Real=1e-7, # used only for the purposes of saving the data to a dict
    d_init::Number=2, 
    d_range::AbstractBounds{<:Integer}=(2,20),
    d_step::Integer=1,
    chi_max_init::Number=25, 
    chi_max_range::AbstractBounds{<:Integer}=(15,80),
    chi_step::Integer=1,
    max_sweeps::Integer=10, 
    max_search_steps::Number=Inf,
    max_eta_steps::Number=10,
    instance_name::String="EtaOptimChiDNearestNeighbour($(nfolds)fold_ns$(max_sweeps)_eta$(repr_vec(eta_range))_chis$(repr_vec(chi_max_range))_ds$(repr_vec(d_range)))",
    max_neighbours::Integer=3,
    method::String="Alternate",
    immediate_ret::Bool=false)

    enc = configure_encodings(encodings, encoding)

    return EtaOptimChiDNearestNeighbour(enc, nfolds, eta_init, eta_range, min_eta_step, d_init, d_range, d_step, chi_max_init, chi_max_range, chi_step, max_sweeps, max_search_steps, max_eta_steps, range(d_range...), instance_name,max_neighbours,titlecase(method), immediate_ret)
end


# find all the values "step" away from "i" that are in the range [a,b]
function step_bound(i::Number, a::Number, b::Number, step::Number=1)
    neighbours = i .+ (-step:step)

    return filter(x-> (a <= x <= b), neighbours)
end


function search_parameter_space(EOP::EtaOptimChiDNearestNeighbour, 
    Ws::AbstractVector{MPS},
    masteropts::Options,
    Xs_train_enc::Matrix{EncodedTimeseriesSet}, 
    Xs_val_enc::Matrix{EncodedTimeseriesSet}; 
    path::String, 
    logfile::String, 
    resfile::String, 
    finfile::String,
    rel_tol::Real=0.01 # smallest step in % of eta that univariate optim solver is allowed to take
    )


    @assert issorted(EOP.d_range) "Min dimension is less than max dimension"
    @assert issorted(EOP.eta_range) "Min learning rate is less than max learning rate"
    @assert issorted(EOP.chi_max_range) "Min chi_max is less than max chi_max"


    results, _, _, _, eta_init, _, chi_max_init, _, d_init, _... = load_result(resfile) 
    encoding = first(EOP.encodings)

    function folds_acc(res::Matrix{Result})
        return mean(getfield.(res[:,end], :acc))
    end

    function folds_acc(::Missing)
        return missing
    end

    function eval_gridpoint!(results, res_keys, nfolds, max_sweeps, eta_f, chi_max, d, e)

        # deal with floating point tomfoolery
        local eta_ind = eta_to_index(eta_f, EOP.min_eta_step)
        local eta = index_to_eta(eta_ind, EOP.min_eta_step)

        res_keys = res_keys |> collect
        key = findfirst(k -> k == (eta_ind, chi_max, d, e), res_keys)
    

        if !isnothing(key)
            return folds_acc(results[res_keys[key]])
        end

        if (isodd(d) && titlecase(e.name) == "Sahand") || (d != 2 && titlecase(e.name) == "Stoudenmire") 
            return results[eta, chi_max, d, e] = missing
        end


        local di = findfirst(d .== EOP.ds)
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

    
    # Begin!
    tstart = time()

    eta, chi_max, d, e = eta_init, chi_max_init, d_init, encoding
    acc = eval_gridpoint!(results, keys(results), EOP.nfolds, EOP.max_sweeps, eta, chi_max, d, e)
    EOP.immediate_ret && return results
    finished = false

    # EOP parameters. I prefer doing this explicitly rather than using @unpack 
    nfolds, max_sweeps = EOP.nfolds, EOP.max_sweeps
    eta_range, d_range, chi_max_range = EOP.eta_range, EOP.d_range, EOP.chi_max_range
    chi_step, d_step = EOP.chi_step, EOP.d_step
    method = EOP.method

    println("Beginning search")
    nsteps = 0
    eta_stepped = method == "Alternate" ? true : false
    chi_d_stepped=true
    n_neighbours = 1

    while !finished && nsteps < EOP.max_search_steps
        @assert !ismissing(acc) "Current accuracy cannot be \"missing\"! Check the initial points are valid: acc(eta=$eta, chi_max=$chi_max, d=$d, enc=$(encoding.name) = missing)"
        nsteps += 1

        println("t=$(round(time() - tstart,digits=2))s: Beginning chi/d step opt for step $nsteps, current accuracy is $acc with eta=$eta, chi_max=$(chi_max), d=$d")

        if !chi_d_stepped && !eta_stepped
            n_neighbours += 1
            println("Got stuck, Expanding neighbourhood to n=$n_neighbours")
        else
            n_neighbours = 1
        end
        # get neighbours
        chi_neighbours = step_bound(chi_max, chi_max_range..., n_neighbours)
        d_neighbours = step_bound(d, d_range..., n_neighbours)

        neighbours = (Iterators.product(chi_neighbours, d_neighbours) |> collect)[:]

        # step in the d, chi direction
        accs_new = Vector{Float64}(undef, length(neighbours))
        etas =  Vector{Float64}(undef, length(neighbours))
        for (i, n) in enumerate(neighbours)
            if method == "Alternate"
                accs_new[i] = eval_gridpoint!(results, keys(results), nfolds, max_sweeps, eta, n..., e) # overhead on duplicates is negligible because of caching

            elseif method == "Best_Eta"
                acc_eta = eta_var -> -eval_gridpoint!(results, keys(results), nfolds, max_sweeps, eta_var, n..., e)
                res = Optim.optimize(acc_eta, max(eta_range[1], eta/2), min(eta_range[2], 2*eta ), rel_tol=rel_tol, abs_tol=EOP.min_eta_step, iterations=EOP.max_eta_steps)
                accs_new[i] = -Optim.minimum(res)
                etas[i] = Optim.minimizer(res)
                save_results(resfile, results, nfolds, nfolds, max_sweeps, eta, eta_range, chi_max, chi_max_range, d, d_range, e, EOP.encodings) 

            else
                error("Unknown method $method")
            end
        end


        acc_new, acc_ind = findmax(accs_new)

        if acc == acc_new
            println("chi, d step did nothing!")
            # think about incrementing chi_step/d_step
            chi_d_stepped = false
        else
            chi_max, d = neighbours[acc_ind]
            chi_d_stepped = true
            acc = acc_new
            if method == "Best_Eta"
                eta = etas[acc_ind]
            end
        end

        save_results(resfile, results, nfolds, nfolds, max_sweeps, eta, eta_range, chi_max, chi_max_range, d, d_range, e, EOP.encodings) 

        println("t=$(round(time() - tstart,digits=2))s: Finished chi/d step opt for step $nsteps, current accuracy is $acc with eta=$eta, chi_max=$(chi_max), d=$d")

        if method != "Best_Eta"
            acc_eta = eta_var -> -eval_gridpoint!(results, keys(results), nfolds, max_sweeps, eta_var, chi_max, d, e)

            res = Optim.optimize(acc_eta, max(eta_range[1], eta/2), min(eta_range[2], 2*eta ), rel_tol=rel_tol, abs_tol=EOP.min_eta_step, iterations=EOP.max_eta_steps)
            eta_new = Optim.minimizer(res)
            acc_new = -Optim.minimum(res)

            if acc_new > acc
                eta_stepped = true
                eta = eta_new
                acc = acc_new
            else
                println("eta step did nothing!")
                eta_stepped = false
            end
        

            save_results(resfile, results, nfolds, nfolds, max_sweeps, eta, eta_range, chi_max, chi_max_range, d, d_range, e, EOP.encodings) 

            println("t=$(round(time() - tstart,digits=2))s: Finished step $nsteps current accuracy is $acc with eta=$eta, chi_max=$(chi_max), d=$d")
        end
        finished = !chi_d_stepped && !eta_stepped && n_neighbours == EOP.max_neighbours

    end
    
    save_status(finfile, nfolds, nfolds, eta, eta_range, chi_max, chi_max_range, d, d_range, last(EOP.encodings), EOP.encodings)

    return results
end


function search_parameter_space(EGD::EtaGDChiDNearestNeighbour, 
    Ws::AbstractVector{MPS},
    masteropts::Options,
    Xs_train_enc::Matrix{EncodedTimeseriesSet}, 
    Xs_val_enc::Matrix{EncodedTimeseriesSet}; 
    path::String, 
    logfile::String, 
    resfile::String, 
    finfile::String)



    @assert issorted(EGD.d_range) "Min dimension is less than max dimension"
    @assert issorted(EGD.eta_range) "Min learning rate is less than max learning rate"
    @assert issorted(EGD.chi_max_range) "Min chi_max is less than max chi_max"


    results, _, _, _, eta_init, _, chi_max_init, _, d_init, _... = load_result(resfile) 
    encoding = first(EGD.encodings)

    function folds_acc(res::Matrix{Result})
        return mean(getfield.(res[:,end], :acc))
    end

    function folds_acc(::Missing)
        return missing
    end

    function eval_gridpoint!(results, res_keys, nfolds, max_sweeps, eta_f, chi_max, d, e)

        # deal with floating point tomfoolery
        local eta_ind = eta_to_index(eta_f, EGD.min_eta_step)
        local eta = index_to_eta(eta_ind, EGD.min_eta_step)

        res_keys = res_keys |> collect
        key = findfirst(k -> k == (eta_ind, chi_max, d, e), res_keys)
    

        if !isnothing(key)
            return folds_acc(results[res_keys[key]])
        end

        if (isodd(d) && titlecase(e.name) == "Sahand") || (d != 2 && titlecase(e.name) == "Stoudenmire") 
            return results[eta, chi_max, d, e] = missing
        end


        local di = findfirst(d .== EGD.ds)
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



    function acc_step!(alpha::Number, results, nfolds, max_sweeps, eta::Number, g_approx::Number, chi_max, d, e)
        local eta_new = eta+alpha*g_approx
        if eta_new < EGD.eta_range[1] || eta_new > EGD.eta_range[2]
            return 0

        else
            return eval_gridpoint!(results, keys(results), nfolds, max_sweeps, eta_new, chi_max, d, e)
        end
        
    end


    
    # Begin!
    tstart = time()

    eta, chi_max, d, e = eta_init, chi_max_init, d_init, encoding
    acc = eval_gridpoint!(results, keys(results), EGD.nfolds, EGD.max_sweeps, eta, chi_max, d, e)
    
    finished = false

    # EGD parameters. I prefer doing this explicitly rather than using @unpack 
    nfolds, max_sweeps = EGD.nfolds, EGD.max_sweeps
    eta_range, d_range, chi_max_range = EGD.eta_range, EGD.d_range, EGD.chi_max_range
    chi_step, d_step = EGD.chi_step, EGD.d_step
    deta_perc, min_eta_eta, min_eta_step, eta_eta_init = EGD.deta_perc, EGD.min_eta_eta, EGD.min_eta_step, EGD.eta_eta_init

    println("Beginning search")
    nsteps = 0
    linesearch = BackTracking(order=3) # may be unused
    chi_d_stepped=true
    eta_stepped=true
    n_neighbours = 1

    while !finished && nsteps < EGD.max_search_steps
        @assert !ismissing(acc) "Current accuracy cannot be \"missing\"! Check the initial points are valid: acc(eta=$eta, chi_max=$chi_max, d=$d, enc=$(encoding.name) = missing)"
        nsteps += 1

        println("t=$(round(time() - tstart,digits=2))s: Beginning chi/d step opt for step $nsteps, current accuracy is $acc with eta=$eta, chi_max=$(chi_max), d=$d")

        if !chi_d_stepped && !eta_stepped
            n_neighbours += 1
        else
            n_neighbours = 1
        end

        # get neighbours
        chi_neighbours = step_bound(chi_max, chi_max_range..., n_neighbours)
        d_neighbours = step_bound(d, d_range..., n_neighbours)

        neighbours = (Iterators.product(chi_neighbours, d_neighbours) |> collect)[:]

        # step in the d, chi direction
        accs_new = Vector{Float64}(undef, length(neighbours))
        for (i, n) in enumerate(neighbours)
            accs_new[i] = eval_gridpoint!(results, keys(results), nfolds, max_sweeps, eta, n..., e) # overhead on duplicates is negligible because of caching
        end


        acc_new, acc_ind = findmax(accs_new)

        if acc_new == acc
            println("chi, d step did nothing!")
            # think about incrementing chi_step/d_step
            chi_d_stepped = false
        else
            chi_max, d = neighbours[acc_ind]
            chi_d_stepped = true
            acc = acc_new
        end

        save_results(resfile, results, nfolds, nfolds, max_sweeps, eta, eta_range, chi_max, chi_max_range, d, d_range, e, EGD.encodings) 

        println("t=$(round(time() - tstart,digits=2))s: Finished chi/d step opt for step $nsteps, current accuracy is $acc with eta=$eta, chi_max=$(chi_max), d=$d")

        # Calc Forward Derivative
        eta_step = deta_perc/100 * eta
        eta_stepped = false
        acc_forward = eval_gridpoint!(results, keys(results), nfolds, max_sweeps, eta + eta_step, chi_max, d, e)

        eta_eta = eta_eta_init
        g_approx = (acc_forward - acc) / (eta_step) # love me a forward derivative
    

        # Linesearch
        if EGD.use_backtracking_linesearch
            ls_f = alpha -> acc_step!(alpha, results, nfolds, max_sweeps, eta, g_approx, chi_max, d, e)
            eta_eta, acc_new = linesearch(ls_f, eta_eta, acc, g_approx )

            eta_new = eta+eta_eta*g_approx
            if eta_new < eta_range[1] || eta_new > eta_range[2] && eta_eta >= min_eta_eta
                println("Eta step outside of acceptable bounds!")
                println("eta_new = $eta_new, eta_eta = $eta_eta")

            elseif acc_new > acc
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
                
                acc_new = eval_gridpoint!(results, keys(results), nfolds, max_sweeps, eta_new, chi_max, d, e)
                if acc <= acc_new
                    eta_eta /= 2
                    eta_new = eta + eta_eta*g_approx
                    #println("eta step did nothing!")
                    # think about incrementing chi_step/d_step
                else
                    eta_stepped = true
                    acc = acc_new
                    eta = eta_new
                    break
                end

            end
        end
        finished = !chi_d_stepped && !eta_stepped && n_neighbours == EGD.max_neighbours

        save_results(resfile, results, nfolds, nfolds, max_sweeps, eta, eta_range, chi_max, chi_max_range, d, d_range, e, EGD.encodings) 

        println("t=$(round(time() - tstart,digits=2))s: Finished step $nsteps current accuracy is $acc with eta=$eta, chi_max=$(chi_max), d=$d")
    end
    
    save_status(finfile, nfolds, nfolds, eta, eta_range, chi_max, chi_max_range, d, d_range, last(EGD.encodings), EGD.encodings)

    return results
end



""" Safely handle checking for previous NearestNeighbourSearch hyperoptimisations with the same canonical name. Handles the options to load/overwrite/abort"""
function load_hyperopt!(ChiDNN::Union{EtaGDChiDNearestNeighbour,EtaOptimChiDNearestNeighbour}, 
    path::String;
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
            results = ResDict() # Somewhere to save the results for no sweeps up to max_sweeps
    
        elseif isfile(resfile)
            resume = check_status(resfile, ChiDNN.nfolds, ChiDNN.eta_range, ChiDNN.chi_max_range, ChiDNN.d_range, ChiDNN.encodings)
            if resume
                results, fold_r, nfolds_r, max_sweeps_r, eta_r, eta_range_r, chi_r, chi_max_range_r, d_r, d_range_r, e_r, encodings_r = load_result(resfile) 
                println("Found interrupted benchmark, resuming")
            
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
        save_results(resfile, results, -1, ChiDNN.nfolds, ChiDNN.max_sweeps, ChiDNN.eta_init, ChiDNN.eta_range, ChiDNN.chi_max_init, ChiDNN.chi_max_range, ChiDNN.d_init, ChiDNN.d_range, first(ChiDNN.encodings), ChiDNN.encodings) 
    
        f = open(logfile, "w")
        close(f)
    end

end
