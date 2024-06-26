using Printf
using Plots

include("result.jl") # Results struct and various crimes that should not see the light of day


function save_status(path::String, fold::N1, nfolds::N1, eta::C, etas::Vector{C}, chi::N2, chi_maxs::Vector{N2}, d::N3, ds::Vector{N3}, e::T, encodings::Vector{T}; append=false) where {N1 <: Integer, N2 <: Integer, N3 <: Integer, C <: Number, T <: Encoding}
    flag = append ? "a" :  "w"

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

function read_status(path::String)
    f = jldopen(path, "r")
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
end


function check_status(path::String, nfolds::N1, etas::Vector{C}, chi_maxs::Vector{N2}, ds::Vector{N3},encodings::Vector{T}) where {N1 <: Integer, N2 <: Integer, N3 <: Integer, C <: Number, T <: Encoding}
    fold_r, nfolds_r, eta_r, etas_r, chi_r, chi_maxs_r, d_r, ds_r, e_r, encodings_r = read_status(path)

    return nfolds_r == nfolds && etas_r == etas && chi_maxs_r == chi_maxs && ds_r == ds && encodings_r == encodings

end


function save_results(resfile::String, results::Array{Union{Result, Missing}, 6}, fold::N1, nfolds::N1, max_sweeps::N2, eta::C, etas::Vector{C}, chi::N3, chi_maxs::Vector{N3}, d::N4, ds::Vector{N4}, e::T, encodings::Vector{T}) where {N1 <: Integer, N2 <: Integer, N3 <: Integer, N4 <: Integer, C <: Number, T <: Encoding}
    f = jldopen(resfile, "w")
        write(f, "results", results)
        write(f, "max_sweeps", max_sweeps)
    close(f)
    save_status(resfile, fold, nfolds, eta, etas, chi, chi_maxs, d, ds, e, encodings; append=true)
end


function load_result(resfile::String)
    fold, nfolds, eta, etas, chi, chi_maxs, d, ds, e, encodings = read_status(resfile)
    f = jldopen(resfile,"r")
        results = f["results"]
        max_sweeps = f["max_sweeps"]
    close(f)

    return results, fold, nfolds, max_sweeps, eta, etas, chi, chi_maxs, d, ds, e, encodings
end


function logdata(fpath::String, fold::Integer, nfolds::Integer, W::MPS, info::Dict, train_states::Union{TimeseriesIterable, Nothing}, test_states::Union{TimeseriesIterable, Nothing}, opts::Options; 
    err::Bool=false, err_str::String="")
    
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




function tab_results(results::Array{Union{Result, Missing},6}, nfolds::Integer, etas::Vector{Number}, max_sweeps, chi_maxs::Vector{Integer}, ds::Vector{Integer}, encodings::Vector{T};
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



function results_summary(results::Array{Union{Result, Missing},6}, nfolds::Integer, max_sweeps, etas::Vector{Number}, chi_maxs::Vector{Integer}, ds::Vector{Integer}, encodings::Vector{T};
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


function expand_dataset(out::Matrix{Union{Result, Missing}}, ds, chi_maxs)
    ds_d = minimum(abs.(diff(ds)))
    chi_maxs_d = minimum(abs.(diff(chi_maxs)))

    ds_exp = collect(minimum(ds):ds_d:maximum(ds))
    chi_maxs_exp = collect(minimum(chi_maxs):chi_maxs_d:maximum(chi_maxs))

    out_exp = Matrix{Union{Result, Missing}}(missing, length(ds_exp), length(chi_maxs_exp))

    for i in 1:size(out,1), j in 1:size(out,2)
        ie = findfirst(d -> d == ds[i], ds_exp)
        je = findfirst(chi -> chi == chi_maxs[j], chi_maxs_exp)
        out_exp[ie, je] = out[i,j]
    end

    return out_exp, ds_exp, chi_maxs_exp
end


function get_resfield(res::Union{Result,Missing},s::Symbol)
    if ismissing(res)
        return missing
    else
        return getfield(res,s)
    end
end


function minmax_colourbar(out::AbstractArray{Union{Result, Missing}}, field::Symbol; threshold::Real=0.8)

    data = first.(skipmissing(get_resfield.(out, field))) # the first handles the case of a maxacc tuple ECG_test

    clims = (max(threshold, minimum(data)), maximum(data))
    nomiss = collect(data)
    cticks = sort(unique(round.(Int, nomiss[nomiss.>= threshold] .* 100)) )

    return clims, cticks
end

function bench_heatmap(results::Array{Union{Result, Missing},6}, nfolds::Integer, max_sweeps::Integer, chi_maxs::Vector{Int}, ds::Vector{Int}, encodings::Vector{T}; balance_klds=false, etai=1, swi=max_sweeps) where {T <: Encoding}
    
    acc_plots = []
    max_acc_plots = []
    kld_plots = []
    min_kld_plots = []
    mse_plots = []
    kld_tr_plots = []
    overfit_plots = []

    #colourbar tomfoolery

    clims_acc, cticks_acc = minmax_colourbar(results, :acc)
    clims_macc, cticks_macc = minmax_colourbar(results, :maxacc)


    for (ei,e) in enumerate(encodings)
        res = mean(results, dims=1)[swi, etai, :,:, ei]
        all(isnothing.(res)) && continue
        res_exp, ds_exp, chi_maxs_exp = expand_dataset(res, ds, chi_maxs)

        accs = get_resfield.(res_exp,:acc)
        klds = get_resfield.(res_exp,:KLD)
        mses = get_resfield.(res_exp,:MSE)

        mfirst(x) = ismissing(x) ? missing : first(x) 
        max_accs = mfirst.(get_resfield.(res_exp, :maxacc))
        min_klds = mfirst.(get_resfield.(res_exp,:minKLD))


        klds_tr = missing
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
        if balance_klds
            for (i, d) in enumerate(ds_exp), (j,chi) in enumerate(chi_maxs_exp)
                KLD = klds[i,j]
                ismissing(KLD) && continue
                KLD_rand = get_baseKLD(chi, d, e)

                klds[i,j] = KLD_rand - KLD
                if do_tr 
                    klds_tr[i,j] = KLD_rand - klds_tr[i,j]
                end
            end
        end
        # println(ds)
        # println(chi_maxs_exp)

 

        pt = heatmap(chi_maxs_exp,ds_exp, accs ; 
        xlabel="χmax",
        ylabel="Dimension",
        colorbar_title="Accuracy",
        clims=clims_acc,
        cmap = palette([:red, :blue], 2*(length(cticks_acc))),
        colourbar_ticks=cticks_acc[2:end] .- 0.5, # the 0.5 makes the colourbarticks line up at the centre of the colours
        colourbar_tick_labels=string.(cticks_acc),
        title=e.name * " Encoding, sweep $(swi)/$(max_sweeps), eta=$(etas[etai])")
        push!(acc_plots, pt)


        pt = heatmap(chi_maxs_exp,ds_exp, max_accs;
        xlabel="χmax",
        ylabel="Dimension",
        colorbar_title="Max Accuracy",
        clims=clims_macc,
        cmap = palette([:red, :blue], 2*(length(cticks_macc)-1)),
        colourbar_ticks=cticks_macc,
        title=e.name * " Encoding, sweep $(swi)/$(max_sweeps), eta=$(etas[etai])")
        push!(max_acc_plots, pt)

        pt = heatmap(chi_maxs_exp,ds_exp, klds;
        xlabel="χmax",
        ylabel="Dimension",
        colorbar_title="KL Div.",
        title=e.name * " Encoding, sweep $(swi)/$(max_sweeps), eta=$(etas[etai])")
        push!(kld_plots, pt)

        pt = heatmap(chi_maxs_exp,ds_exp, min_klds;
        xlabel="χmax",
        ylabel="Dimension",
        colorbar_title="Min KL Div.",
        title=e.name * " Encoding, sweep $(swi)/$(max_sweeps), eta=$(etas[etai])")
        push!(min_kld_plots, pt)

        pt = heatmap(chi_maxs_exp,ds_exp, mses;
        xlabel="χmax",
        ylabel="Dimension",
        colorbar_title="MSE",
        title=e.name * " Encoding, sweep $(swi)/$(max_sweeps), eta=$(etas[etai])")
        push!(mse_plots, pt)

        if do_tr
            pt = heatmap(chi_maxs_exp,ds_exp, klds_tr;
            xlabel="χmax",
            ylabel="Dimension",
            colorbar_title="KL Div. train",
            title=e.name * " Encoding, sweep $(swi)/$(max_sweeps), eta=$(etas[etai])")
            push!(kld_tr_plots, pt)
        end


        pt = heatmap(chi_maxs_exp,ds_exp, klds - min_klds;
        xlabel="χmax",
        ylabel="Dimension",
        colorbar_title="KLD Overfit",
        title=e.name * " Encoding, sweep $(swi)/$(max_sweeps), eta=$(etas[etai])")
        push!(overfit_plots, pt)
    end


    return acc_plots, max_acc_plots, kld_plots, min_kld_plots, mse_plots, kld_tr_plots, overfit_plots
end

function bench_heatmap(path::String; kwargs...)
    results, fold, nfolds, max_sweeps, eta, etas, chi, chi_maxs, d, ds, e, encodings = load_result(path) 
    return bench_heatmap(results, nfolds, max_sweeps, chi_maxs, ds, encodings; kwargs...)
end



function parse_log(s::String)
    opts = []
    training_info = []
    count = 0
    f = open(s, "r")
    b_init = 1
    was_empty = true
    for (i, ln) in enumerate(eachline(f))
        if first(ln) == '/'
            # a delimation point
            if was_empty
                break # reached the end of the simulation data
            end
            opt, tinfo = parse_block(f, b_init, i)
            push!(opts, opt)
            push!(training_info, tinfo)

            # reset block 
            b_init = i
            was_empty = true
        elseif !isempty(ln)
            was_empty &= true
        end
    end
    close(f)
end

function parse_block(f::IO, bi, be)
    training_information = Dict(
        "train_loss" => Float64[],
        "train_acc" => Float64[],
        "test_loss" => Float64[],
        "test_acc" => Float64[],
        "time_taken" => Float64[], # sweep duration
        "train_KL_div" => Float64[],
        "test_KL_div" => Float64[],
    )
    error("Not Implemented Yet!")
    return nothing, nothing
    
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