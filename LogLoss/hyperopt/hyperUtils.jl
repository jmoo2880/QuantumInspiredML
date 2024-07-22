using Printf
using Plots
using StatsBase
using Measures

include("result.jl") # Results struct and various crimes that should not see the light of day
include("vishypertrain.jl")

const AbstractBounds{T} = Union{AbstractVector{T}, NTuple{2,T}}
const ResDict = Dict{Tuple{Integer, Integer, Integer, Encoding}, Union{Matrix{Result}, Missing}}



function eta_to_index(eta::Real, eta_min::Real)
    return round(Int, 2*eta/eta_min)
end

function index_to_eta(index, eta_min::Real)
    return index * eta_min/2
end


function save_status(path::String, fold::N1, nfolds::N1, eta::C, etas::AbstractVector{C}, chi::N2, chi_maxs::AbstractVector{N2}, d::N3, ds::AbstractVector{N3}, e::T, encodings::AbstractVector{T}; append=false) where {N1 <: Integer, N2 <: Integer, N3 <: Integer, C <: Number, T <: Encoding}
    flag = append ? "a" :  "w"

    disable_sigint() do
        f = jldopen(path, flag)
        write(f, "fold", fold)
        write(f, "nfolds", nfolds)

        write(f, "eta", eta)
        write(f, "etas", etas)

        write(f, "chi", chi)
        write(f, "chi_maxs", chi_maxs)

        write(f, "d", d)
        write(f, "ds", ds)

        write(f, "e", e)
        write(f, "encodings", encodings)
        close(f)
    end
end

function save_status(path::String, fold::N1, nfolds::N1, eta::C, eta_range::AbstractBounds{C}, chi::N2, chi_max_range::AbstractBounds{N2}, d::N3, d_range::AbstractBounds{N3}, e::T, encodings::AbstractVector{T}; append=false) where {N1 <: Integer, N2 <: Integer, N3 <: Integer, C <: Number, T <: Encoding}
    flag = append ? "a" :  "w"

    disable_sigint() do
        f = jldopen(path, flag)
        write(f, "fold", fold)
        write(f, "nfolds", nfolds)

        write(f, "eta", eta)
        write(f, "eta_range", eta_range)

        write(f, "chi", chi)
        write(f, "chi_max_range", chi_max_range)

        write(f, "d", d)
        write(f, "d_range", d_range)

        write(f, "e", e)
        write(f, "encodings", encodings)
        close(f)
    end
end


function read_status(path::String)
    f = jldopen(path, "r")

    if "ds" in keys(f)
        fold = read(f, "fold")
        nfolds = read(f, "nfolds")

        eta = read(f, "eta")
        etas = read(f, "etas")

        chi = read(f, "chi")
        chi_maxs = read(f, "chi_maxs")

        d = read(f, "d")
        ds = read(f, "ds")

        e = read(f, "e")
        encodings = read(f, "encodings")
        close(f)

        return fold, nfolds, eta, etas, chi, chi_maxs, d, ds, e, encodings
    else
        fold = read(f, "fold")
        nfolds = read(f, "nfolds")

        eta = read(f, "eta")
        eta_range = read(f, "eta_range")

        chi = read(f, "chi")
        chi_max_range = read(f, "chi_max_range")

        d = read(f, "d")
        d_range = read(f, "d_range")

        e = read(f, "e")
        encodings = read(f, "encodings")
        close(f)

        return fold, nfolds, eta, eta_range, chi, chi_max_range, d, d_range, e, encodings
    end

end


function check_status(path::String, nfolds::N1, etas::AbstractVector{C}, chi_maxs::AbstractVector{N2}, ds::AbstractVector{N3}, encodings::AbstractVector{T}) where {N1 <: Integer, N2 <: Integer, N3 <: Integer, C <: Number, T <: Encoding}
    fold_r, nfolds_r, eta_r, etas_r, chi_r, chi_maxs_r, d_r, ds_r, e_r, encodings_r = read_status(path)

    return nfolds_r == nfolds && etas_r == etas && chi_maxs_r == chi_maxs && ds_r == ds && encodings_r == encodings

end

function check_status(path::String, nfolds::N1, eta_range::AbstractBounds{C}, chi_max_range::AbstractBounds{N2}, d_range::AbstractBounds{N3}, encodings::AbstractVector{T}) where {N1 <: Integer, N2 <: Integer, N3 <: Integer, C <: Number, T <: Encoding}
    fold_r, nfolds_r, eta_r, eta_range_r, chi_r, chi_max_range_r, d_r, d_range_r, e_r, encodings_r = read_status(path)

    return nfolds_r == nfolds && eta_range_r == eta_range && chi_max_range_r == chi_max_range && d_range_r == d_range && encodings_r == encodings
end

function save_results(resfile::String, results::AbstractArray{Union{Result, Missing}, 6}, fold::N1, nfolds::N1, max_sweeps::N2, eta::C, etas::AbstractVector{C}, chi::N3, chi_maxs::AbstractVector{N3}, d::N4, ds::AbstractVector{N4}, e::T, encodings::AbstractVector{T}) where {N1 <: Integer, N2 <: Integer, N3 <: Integer, N4 <: Integer, C <: Number, T <: Encoding}
    disable_sigint() do
        f = jldopen(resfile, "w")
            write(f, "results", results)
            write(f, "max_sweeps", max_sweeps)
        close(f)
    end
    save_status(resfile, fold, nfolds, eta, etas, chi, chi_maxs, d, ds, e, encodings; append=true)
end

function save_results(resfile::String, results::Dict, fold::N1, nfolds::N1, max_sweeps::N2, eta::C, eta_range::AbstractBounds{C}, chi::N3, chi_max_range::AbstractBounds{N3}, d::N4, d_range::AbstractBounds{N4}, e::T, encodings::AbstractVector{T}) where {N1 <: Integer, N2 <: Integer, N3 <: Integer, N4 <: Integer, C <: Number, T <: Encoding}
    disable_sigint() do
        f = jldopen(resfile, "w")
            write(f, "results", results)
            write(f, "max_sweeps", max_sweeps)
        close(f)
    end
    save_status(resfile, fold, nfolds, eta, eta_range, chi, chi_max_range, d, d_range, e, encodings; append=true)
end


function load_result(resfile::String)
    fold, nfolds, status... = read_status(resfile)
    f = jldopen(resfile,"r")
        results = f["results"]
        max_sweeps = f["max_sweeps"]
    close(f)

    return results, fold, nfolds, max_sweeps, status...
end


function logdata(fpath::String, fold::Integer, nfolds::Integer, W::MPS, info::Dict, train_states::Union{TimeseriesIterable, Nothing}, test_states::Union{TimeseriesIterable, Nothing}, opts::Options; 
    err::Bool=false, err_str::String="")
    disable_sigint() do
        f = open(fpath, "a")

        println("###############")
        println(f, "Fold $fold/$nfolds")
        println("###############\n")

        print_opts(opts; io=f)
        if !err
            stats = get_training_summary(W, train_states, test_states; print_stats=true, io=f);

            sweep_summary(info; io=f)
        else
            print(f, "Simulation aborted due to Error: $err_str")
            stats = []
        end
        print(f, "\n\n/=======================================================================================================================================================\\ \n\n")
        close(f)
    end
    return stats
end




function format_result(r::Union{Result, Missing}, i::Int, j::Int; conf=true, fancy_conf=false, conf_titles=true, data)

    summary = true
    isrange = false
    nrows = size(data,1)
    ncols = size(data,2)
    # check if we should be printing a summary
    if i > nrows
        # we should summarise the column
        if j > ncols
            # we're in the corner
            return "--"
        end

        isrange, ismean, isstd = (i .== (nrows +1, nrows + 2, nrows + 3) )

        
        #std/mean cols
        col = Vector{Union{Result, Missing}}(undef, size(data,1))
        col .= data[:,j]
        if all(ismissing.(col))
            return "."
        end

        # yes, I do regret using the nothing type rather than the missing type, why do you ask?
        col[col .== nothing] .= missing
        col = skipmissing(col)


        cacc = map(x->x.acc, col)
        cKLD = map(x->x.KLD, col)
        cMSE = map(x->x.MSE, col)
        cconf = map(x->x.conf, col)

        cmacc = map(x->x.maxacc, col)
        cminKLD = map(x->x.minKLD, col)

        if isrange
            cf = Matrix{Tuple{Float64, Float64}}(undef, 2, 2)
            
            for i in eachindex(cf)
                collapsed = [cmat[i] for cmat in cconf]
                cf[i] =  (minimum(collapsed), maximum(collapsed) )
            end

            acc = maximum(cacc)
            mse = minimum(cMSE)
            kld = minimum(cKLD)

            macc = @sprintf("(%.3f,%.1f)", maximum(cmacc)...)
            minKLD = @sprintf("(%.3f,%.1f)", minimum(cminKLD)...)

        elseif ismean
            # mean
            cf =  mean(cconf)
            acc = mean(cacc)
            mse = mean(cMSE)
            kld = mean(cKLD)

            macc = @sprintf("(%.3f,%.1f)", mean([ma[1] for ma in cmacc]), mean([ma[2] for ma in cmacc]))
            minKLD = @sprintf("(%.1f,%.1f)", mean([mKLD[1] for mKLD in cminKLD]), mean([mKLD[2] for mKLD in cminKLD]))

        else
            # std dev
            cf =  std(cconf)
            acc = std(cacc)
            mse = std(cMSE)
            kld = std(cKLD)

            macc = @sprintf("(%.3f,%.1f)", std([ma[1] for ma in cmacc]), std([ma[2] for ma in cmacc]))
            minKLD = @sprintf("(%.1f,%.1f)", std([mKLD[1] for mKLD in cminKLD]), std([mKLD[2] for mKLD in cminKLD]))
        end
        

    elseif j > ncols
        # we should summarise the row
        isrange, ismean, isstd = (j .== (ncols +1, ncols + 2, ncols + 3) )

        row = Vector{Union{Result, Missing}}(undef, size(data,2))
        row .= data[i,:]
        if all(isnothing.(row))
            return "."
        end

        row = skipmissing(row)


        racc = map(x->x.acc, row)
        rKLD = map(x->x.KLD, row)
        rMSE = map(x->x.MSE, row)
        rconf = map(x->x.conf, row)

        rmacc = map(x->x.maxacc, row)
        rminKLD = map(x->x.minKLD, row)

        if isrange
            cf = Matrix{Tuple{Float64, Float64}}(undef, 2, 2)
            
            for i in eachindex(cf)
                collapsed = [cmat[i] for cmat in rconf]
                cf[i] =  (minimum(collapsed), maximum(collapsed) )
            end

            acc = maximum(racc)
            mse = minimum(rMSE)
            kld = minimum(rKLD)

            macc = @sprintf("(%.3f,%.1f)", maximum(rmacc)...)
            minKLD = @sprintf("(%.1f,%.1f)", minimum(rminKLD)...)
            

        elseif ismean
            # mean
            cf =  mean(rconf)
            acc = mean(racc)
            mse = mean(rMSE)
            kld = mean(rKLD)

            macc = @sprintf("(%.3f, %.1f)", mean([ma[1] for ma in rmacc]), mean([ma[2] for ma in rmacc]) )
            minKLD = @sprintf("(%.1f, %.1f)", mean([mKLD[1] for mKLD in rminKLD]), mean([mKLD[2] for mKLD in rminKLD]))

        else #isstd
            # std dev
            cf =  std(rconf)
            acc = std(racc)
            mse = std(rMSE)
            kld = std(rKLD)

            macc = @sprintf("(%.3f,%.1f)", std([ma[1] for ma in rmacc]), std([ma[2] for ma in rmacc]) )
            minKLD = @sprintf("(%.1f,%.1f)", std([mKLD[1] for mKLD in rminKLD]), std([mKLD[2] for mKLD in rminKLD]))

            
        end


    elseif !ismissing(r)
        # a standard entry
        summary = false
        cf = r.conf
        acc = r.acc
        mse = r.MSE
        kld = r.KLD

        macc = @sprintf("(%.3f, %d)", r.maxacc...)
        minKLD = @sprintf("(%.1f, %d)", r.minKLD...)
    else
        # the entry is not supposed to be a summary statistic and it is nothing
        return "."
    end

    nclasses = size(cf,1)


    if fancy_conf
        
        if isrange
            # d is a tuple of floats
            fmt = (d,_,__) -> @sprintf("(%.2f, %.2f)", d...)

        elseif summary
            # d is a float
            fmt =  (d,_,__) -> @sprintf("%.2f", d) 

        else
            # d is an int
            fmt = (d,_,__) -> string(d)
        end

        if conf_titles
            header = ["Pred. |$n⟩" for n in 0:(nclasses-1)]
            row_labels = ["True |$n⟩" for n in 0:(nclasses-1)]

            cf = pretty_table(String,cf;
            hlines=:all,
            compact_printing=true,
            header=header,
            row_labels=row_labels,
            highlighters = Highlighter(f = (data, i, j) -> (i == j), crayon = crayon"bold green"),
            formatters=fmt)
        else
            cf = pretty_table(String,cf;
            hlines=:all,
            compact_printing=true,
            show_header=false,
            highlighters = Highlighter(f = (data, i, j) -> (i == j), crayon = crayon"bold green"),
            formatters=fmt)
        end
    else

        cf = string.(cf) .* "|"
        cf = hcat(["c$(n): |" for n in 0:(nclasses-1)], cf)
        cf = reduce(*,cf; dims=2)
    end

    if conf
        return @sprintf("%s\nAcc: %.3f; KLD: %.2f; MSE: %.2f\nMaxAcc: %s; MinKLD: %s;",cf, acc, kld, mse, macc, minKLD)
    else
        return @sprintf("Acc: %.3f; KLD: %.2f; MSE: %.2f\nMaxAcc: %s; MinKLD: %s;", acc, kld, mse, macc, minKLD)
    end
end

# format_result(::Nothing, args...; kwargs...) = nothing




function tab_results(results::AbstractArray{Union{Result, Missing},6}, nfolds::Integer, etas::AbstractVector{Number}, max_sweeps, chi_maxs::AbstractVector{Integer}, ds::AbstractVector{Integer}, encodings::AbstractVector{T};
        io::IO=stdin, fancy_conf=false, conf_titles=true, conf=true, etai=1, swi=max_sweeps) where {T <: Encoding}


    h1 = Highlighter((data, i, j) -> j < length(header) && data[i, j] == maximum(data[i,1:(end-1)]),
    bold       = true,
    foreground = :red )

    h2 = Highlighter((data, i, j) -> j < length(header) && data[i, j] == minimum(data[i,1:(end-1)]),
    bold       = true,
    foreground = :blue )

    for (ei,e) in enumerate(encodings)
        res = mean(results, dims=1)[swi, etai,:,:, ei]

        # some extra whitespace
        print(io, "\n\n\n")

        res_with_sum = Array{Union{Result, Missing, String},2}(missing, (size(res) .+ 3)...)
        # the last two rows and columns should be the mean 
        res_with_sum[1:end-3, 1:end-3] = res
        # dim2 = hcat(mean_and_std(res,2))
        # if all(isnothing.(dim2))
        #     res_with_sum[1:end-2, end-1:end] .= nothing 
        # else
        #     res_with_sum[1:end-2, end-1:end] = dim2
        # end
        # # res_with_sum[end-1:end, 1:end-2] = hcat(mean_and_std(res,1))
        # res_with_sum[end-1:end, end-1:end] .= nothing

        pretty_table(io,res_with_sum;
                    title=e.name * " Encoding, sweep $(swi)/$(max_sweeps), eta=$(etas[etai])",
                    title_alignment=:c,
                    title_same_width_as_table=true,
                    header = vcat(["χmax = $n" for n in chi_maxs]..., "Max Acc/Min Loss of Row", "Mean of Row", "SD of Row"),
                    row_labels = vcat(["d = $n" for n in ds]..., "Max Acc/Min Loss of Col", "Mean of Col", "SD of Col"),
                    alignment=:c,
                    hlines=:all,
                    linebreaks=true,
                    #highlighters = (h1,h2),
                    formatters = (args...) -> format_result(args...; conf=conf,fancy_conf=fancy_conf, conf_titles=conf_titles, data=res))


    end
end

function tab_results(path::String; kwargs...)
    results, fold, nfolds, max_sweeps, eta, etas, chi, chi_maxs, d, ds, e, encodings = load_result(path) 
    tab_results(results, nfolds, max_sweeps, etas, chi_maxs, ds, encodings; kwargs...)
end



function results_summary(results::AbstractArray{Union{Result, Missing},6}, nfolds::Integer, max_sweeps, etas::AbstractVector{Number}, chi_maxs::AbstractVector{Integer}, ds::AbstractVector{Integer}, encodings::AbstractVector{T};
    io::IO=stdin, fancy_conf=false, conf_titles=true, etai=1, swi=max_sweeps) where {T <: Encoding}
    for (ei,e) in enumerate(encodings)
        res = mean(results, dims=1)[swi,etai,:,:, ei]
        all(isnothing.(res)) && continue
        res_exp, ds_exp, chi_maxs_exp = expand_dataset(res, ds, chi_maxs)

        accs = get_resfield.(res_exp,:acc)
        klds = get_resfield.(res_exp,:KLD)
        mses = get_resfield.(res_exp,:MSE)

        klds_tr = nothing
        do_tr = false
        try 
            klds_tr = get_resfield.(res_exp, :KLD_tr)
            do_tr = true
        catch e
            if e isa ErrorException && e.msg == "type Result has no field KLD_tr"
                println("Training KLDS not saved, skipping")
            else
                throw(e)
            end
        end

        if all(isnothing.(klds_tr) .|| ismissing.(klds_tr))
            println("Training KLDS not saved, skipping")
            do_tr = false
        end

    end
end

function results_summary(path::String; kwargs...)
    results, fold, nfolds, max_sweeps, eta, etas, chi, chi_maxs, d, ds, e, encodings = load_result(path) 
    return results_summary(results, nfolds, max_sweeps, etas, chi_maxs, ds, encodings; kwargs...)
end



function get_baseKLD(chi_max::Integer, d::Integer, e::T;) where {T <: Encoding}
    KLDmap2C = load("LogLoss/benchmarking/KLDmap.jld2", "KLDmap2C")

    if (chi_max, d, e) in keys(KLDmap2C)
        KLD = KLDmap2C[(chi_max, d, e)]
    else
        println("KLD($chi_max, $d, $e) not cached, generating")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits_txt("LogLoss/datasets/ECG_train.txt", "LogLoss/datasets/ECG_val.txt", "LogLoss/datasets/ECG_test.txt")
        X_train = vcat(X_train, X_val)
        y_train = vcat(y_train, y_val)

        dtype = e.iscomplex ? ComplexF64 : Float64
        opts = Options(d=d, dtype=dtype, verbosity=-1, encoding=e, chi_max=chi_max)
        W, _, _, test_states, _ = fitMPS(X_train, y_train, X_test, y_test; random_state=456, chi_init=chi_max, opts=opts, test_run=true)         
        KLD = KL_div(W, test_states)
        KLDmap2C[(chi_max, d, e)] = KLD
        jldsave("LogLoss/benchmarking/KLDmap.jld2"; KLDmap2C)
    end

    return KLD
end


function load_hyperopt!(::GridSearch, path::String; 
    # Searches for a saved hyperopt under path "path". If it exists, then offer to load/overwrite/abort.
    # If it doesn't exist, create it.
    logfile::String=path*"log.txt", 
    resfile::String=path*"results.jld2", 
    finfile::String=path*"finished.jld2")

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

end


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


function load_hyperopt!(EGD::EtaGradientDescent, path::String;
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
            resume = check_status(resfile, EGD.nfolds, EGD.eta_range, EGD.chi_max_range, EGD.d_range, EGD.encodings)
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
        save_results(resfile, results, -1, EGD.nfolds, EGD.max_sweeps, EGD.eta_init, EGD.eta_range, EGD.chi_max_init, EGD.chi_max_range, EGD.d_init, EGD.d_range, first(EGD.encodings), EGD.encodings) 
    
        f = open(logfile, "w")
        close(f)
    end

end