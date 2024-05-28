using Printf
using Plots
# import Base: /, *, ^, +, -
# import StatsBase: mean_and_std

struct Result
    acc::Float64
    maxacc::Union{Tuple{Float64, Integer}, Missing} # oopsie
    conf::Matrix{Float64}
    KLD::Float64
    minKLD::Union{Tuple{Float64, Integer}, Missing} # oopsie
    KLD_tr::Union{Float64,Missing} # oopsie
    MSE::Float64
end

# dummy example for debugging
Result() = Result(1.,(1., 2), ones(2,2),1., (2., 4), 1., 1.)

# julia crimes below -----------------------------------
# for (fname) in [:/,:*,:-,:+,:^,]
#     @eval begin
#         function $fname(r::Result, d::Number)
#             acc = $fname(r.acc, d)
#             maxacc = ismissing(r.maxacc) ? missing : ($fname(r.maxacc[1], d), r.maxacc[2])
#             conf = $fname.(r.conf, d)
#             KLD = $fname(r.KLD, d)
#             minKLD = ismissing(r.minKLD) ? missing : ($fname(r.minKLD[1], d), r.minKLD[2])
#             KLD_tr = ismissing(r.KLD_tr) ? missing : $fname(r.KLD_tr,d)
#             MSE = $fname(r.MSE, d)
    
#             return Result(acc, maxacc, conf, KLD, minKLD, KLD_tr, MSE)
#         end

#         $fname(d::Number, r::Result) = $fname(r::Result, d::Number)

#         function $fname(r1::Result, r2::Result)
#             acc = $fname(r1.acc, r2.acc)
#             maxacc = ismissing(r1.maxacc) ? missing : $fname.(r1.maxacc, r2.maxacc)
#             conf = $fname.(r1.conf, r2.conf)
#             KLD = $fname(r1.KLD, r2.KLD)
#             minKLD = ismissing(r1.minKLD) ? missing : $fname.(r1.minKLD, r2.minKLD)
#             KLD_tr = ismissing(r1.KLD_tr) ? missing : $fname(r1.KLD_tr,r2.KLD_tr)
#             MSE = $fname(r1.MSE, r2.MSE)
    
#             return Result(acc, maxacc, conf, KLD, minKLD, KLD_tr, MSE)
#         end
#     end
# end
# for (fname) in [:-,:(Base.abs), :(Base.abs2), :(Base.sqrt), :(Base.real), :(Base.conj), :(Base.imag)]
#     @eval begin
#         function $fname(r::Result)
#             acc = $fname(r.acc)
#             maxacc = ismissing(r.maxacc) ? missing : ($fname(r.maxacc[1]), r.maxacc[2])
#             conf = $fname.(r.conf)
#             KLD = $fname(r.KLD)
#             minKLD = ismissing(r.minKLD) ? missing : ($fname(r.minKLD[1]), r.minKLD[2])
#             KLD_tr = ismissing(r.KLD_tr) ? missing : $fname(r.KLD_tr)
#             MSE = $fname(r.MSE)
    
#             return Result(acc, maxacc, conf, KLD, minKLD, KLD_tr, MSE)
#         end

#     end
# end

# Base.length(::Result) = 1
# Base.iterate(r::Result) = (r, nothing)
# Base.iterate(::Result, ::Nothing) = nothing


# function Result(d::Dict{String,Vector{Float64}},s::Dict{Symbol, Any})

#     acc = d["test_acc"][end]
#     conf = s[:confmat]
#     KLD = d["test_KL_div"][end]
#     KLD_tr = d["train_KL_div"][end]
#     MSE = d["test_loss"][end]

#     maxacc = findmax(d["test_acc"])
#     minKLD = findmin(d["test_KL_div"])
#     return Result(acc, maxacc, conf, KLD, minKLD,KLD_tr,MSE)
# end

# function mean_and_std(res::AbstractArray{Union{Result,Nothing}}, dim::Integer)
#     @show res, dim
#     res_clean = Matrix{Result}(undef, size(res)...)
#     res_clean[res .!== nothing] =  res[res .!== nothing]
#     res_clean[res .!== nothing] .=  Result(0.,(0., 0), zeros(2,2),0., (0., 0), 0., 0.)


#     @show ndims(res_clean)
#     if ndims(res_clean) < dim
#         return nothing
#     else
#         return mean_and_std(res_clean, dim)
#     end
# end

# function mean_and_std(res::Matrix{Result}, dim::Integer)

#     if dim == 1
#         return [mean_and_std(c) for c in eachcol(res)]
#     else
#         return [mean_and_std(r) for r  in eachrow(res)]
#     end

# end


# Julia crimes have ended -------------------------


function save_status(path::String,chi::Int,d::Int,e::Encoding, chis::Vector{Int},ds::Vector{Int},encodings::Vector{T}; append=false) where {T <: Encoding}
    flag = append ? "a" :  "w"

    f = jldopen(path, flag)
    write(f, "chi", chi)
    write(f, "chis", chis)

    write(f, "d", d)
    write(f, "ds", ds)

    write(f, "e", e.name)
    write(f, "encodings", [enc.name for enc in encodings])
    close(f)
end

function read_status(path::String)
    f = jldopen(path, "r")
    chi = read(f, "chi")
    chis = read(f, "chis")

    d = read(f, "d")
    ds = read(f, "ds")

    e = Encoding(read(f, "e"))
    encodings = Encoding.(read(f, "encodings"))
    close(f)

    return chi, chis, d, ds, e, encodings
end


function check_status(path::String,chis::Vector{Int},ds::Vector{Int},encodings::Vector{T}) where {T <: Encoding}

    chi_r, chis_r, d_r, ds_r, e_r, encodings_r = read_status(path)

    if chis_r == chis && ds_r == ds && encodings_r == encodings
        return true, [chi_r,d_r,e_r]

    else
        return false, [chis_r,ds_r,encodings_r]
    end
end

function logdata(fpath::String, W::MPS, info::Dict, train_states::Union{timeSeriesIterable, Nothing}, test_states::Union{timeSeriesIterable, Nothing}, opts::Options; 
    err::Bool=false, err_str::String="")
    
    f = open(fpath, "a")
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

function load_result(resfile::String)
    chi, chis, d, ds, e, encodings = read_status(resfile)
    f = jldopen(resfile,"r")
        results = f["output"]
    close(f)

    if first(results) isa JLD2.ReconstructedMutable
        
        res_mut = results
        results = Array{Union{Result,Nothing}}(undef, size(results)...)

        for i in eachindex(results)
            res = res_mut[i]
            if isnothing(res)
                results[i] = nothing
            elseif res isa JLD2.ReconstructedMutable{:Result, (:acc, :conf, :KLD, :KLD_tr, :MSE), Tuple{Float64, Any, Float64, Any, Float64}}
                results[i] = Result(res.acc, missing, res.conf, res.KLD, missing, res.KLD_tr, res.MSE)
            else
                results[i] = Result(res.acc, missing, res.conf, res.KLD, missing, missing, res.MSE)

            end
        end
    end

    return results, chi, chis, d, ds, e, encodings
end



function format_result(r::Union{Result, Nothing}, i::Int, j::Int; conf=true, fancy_conf=false, conf_titles=true, data)

    summary = true
    # check if we should be printing a summary
    if i > size(data,1)
        # we should summarise the column
        if j > size(data, 2)
            # we're in the corner
            return "--"
        end

        col = Vector{Union{Result, Nothing, Missing}}(undef, size(data,1))
        col .= data[:,j]
        if all(isnothing.(col))
            return "nothing"
        end

        # yes, I do regret using the nothing type rather than the missing type, why do you ask?
        col[col .== nothing] .= missing
        col = skipmissing(col)


        cacc = map(x->x.acc, col)
        cKLD = map(x->x.KLD, col)
        cMSE = map(x->x.MSE, col)
        cconf = map(x->x.conf, col)

        if i == size(data,1) + 1
            # mean
            cf = mean(cconf)
            acc = mean(cacc)
            mse = mean(cMSE)
            kld = mean(cKLD)

        else
            # std dev
            cf = std(cconf)
            acc = std(cacc)
            mse = std(cMSE)
            kld = std(cKLD)

        end

    elseif j > size(data,2)
        # we should summarise the row
        row = Vector{Union{Result, Nothing, Missing}}(undef, size(data,2))
        row .= data[i,:]
        if all(isnothing.(row))
            return "nothing"
        end

        row[row .== nothing] .= missing
        row = skipmissing(row)


        racc = map(x->x.acc, row)
        rKLD = map(x->x.KLD, row)
        rMSE = map(x->x.MSE, row)
        rconf = map(x->x.conf, row)

        if j == size(data,2) + 1
            # mean
            cf = mean(rconf)
            acc = mean(racc)
            mse = mean(rMSE)
            kld = mean(rKLD)

        else
            # std dev
            cf = std(rconf)
            acc = std(racc)
            mse = std(rMSE)
            kld = std(rKLD)
        end

    elseif !isnothing(r)
        # a standard entry
        summary = false
        cf = r.conf
        acc = r.acc
        mse = r.MSE
        kld = r.KLD
    else
        # the entry is not supposed to be a summary statistic and it is nothing
        return "nothing"
    end

    nclasses = size(cf,1)


    if fancy_conf
        fmt = summary ? (d,_,__) -> @sprintf("%.2f", d) : (d,_,__) -> string(d)
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
        cf = string.(Int.(cf)) .* "|"
        cf = hcat(["c$(n): |" for n in 0:(nclasses-1)], cf)
        cf = reduce(*,cf; dims=2)
    end

    if conf
        return @sprintf("%s\nAcc: %.3f; KLD: %.4f; MSE: %.4f",cf, acc, kld, mse)
    else
        return @sprintf("Acc: %.3f; KLD: %.4f; MSE: %.2e", acc, kld, mse)
    end
end

# format_result(::Nothing, args...; kwargs...) = nothing




function tab_results(results::Array{Union{Result, Nothing},3}, chis::Vector{Int}, ds::Vector{Int}, encodings::Vector{T};
        io::IO=stdin, fancy_conf=false, conf_titles=true, conf=true) where {T <: Encoding}


    h1 = Highlighter((data, i, j) -> j < length(header) && data[i, j] == maximum(data[i,1:(end-1)]),
    bold       = true,
    foreground = :red )

    h2 = Highlighter((data, i, j) -> j < length(header) && data[i, j] == minimum(data[i,1:(end-1)]),
    bold       = true,
    foreground = :blue )

    for (ei,e) in enumerate(encodings)
        res = results[ei,:,:]

        # some extra whitespace
        print(io, "\n\n\n")

        res_with_sum = Array{Union{Result, Nothing, String},2}(nothing, (size(res) .+ 2)...)
        # the last two rows and columns should be the mean 
        res_with_sum[1:end-2, 1:end-2] = res
        # dim2 = hcat(mean_and_std(res,2))
        # if all(isnothing.(dim2))
        #     res_with_sum[1:end-2, end-1:end] .= nothing 
        # else
        #     res_with_sum[1:end-2, end-1:end] = dim2
        # end
        # # res_with_sum[end-1:end, 1:end-2] = hcat(mean_and_std(res,1))
        # res_with_sum[end-1:end, end-1:end] .= nothing

        pretty_table(io,res_with_sum;
                    title=e.name * " Encoding",
                    title_alignment=:c,
                    title_same_width_as_table=true,
                    header = vcat(["χmax = $n" for n in chis]..., "Mean of row", "SD of row"),
                    row_labels = vcat(["d = $n" for n in ds]..., "Mean of col", "SD of col"),
                    alignment=:c,
                    hlines=:all,
                    linebreaks=true,
                    #highlighters = (h1,h2),
                    formatters = (args...) -> format_result(args...; conf=conf,fancy_conf=fancy_conf, conf_titles=conf_titles, data=res))


    end
end

function tab_results(path::String; io::IO=stdin, fancy_conf=true, conf_titles=true, conf=true)
    results, chi, chis, d, ds, e, encodings = load_result(path) 
    tab_results(results, chis, ds, encodings; io=io, fancy_conf=fancy_conf, conf_titles=conf_titles, conf=conf)
end



function results_summary(results::Array{Union{Result, Nothing},3}, chis::Vector{Int}, ds::Vector{Int}, encodings::Vector{T};
    io::IO=stdin, fancy_conf=false, conf_titles=true) where {T <: Encoding}
    for (ei,e) in enumerate(encodings)
        res = results[ei,:,:]
        all(isnothing.(res)) && continue
        res_exp, ds_exp, chis_exp = expand_dataset(res, ds, chis)

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

function results_summary(path::String; io::IO=stdin, fancy_conf=true, conf_titles=true)
    results, chi, chis, d, ds, e, encodings = load_result(path) 
    return results_summary(results, chis, ds, encodings; io=io, fancy_conf=fancy_conf, conf_titles=conf_titles)
end



function get_resfield(res::Union{Result,Nothing},s::Symbol)
    if isnothing(res)
        return nothing
    else
        return getfield(res,s)
    end
end

function expand_dataset(out::Matrix{Union{Result, Nothing}}, ds, chis)
    ds_d = minimum(abs.(diff(ds)))
    chis_d = minimum(abs.(diff(chis)))

    ds_exp = collect(minimum(ds):ds_d:maximum(ds))
    chis_exp = collect(minimum(chis):chis_d:maximum(chis))

    out_exp = Matrix{Union{Result, Nothing}}(nothing, length(ds_exp), length(chis_exp))

    for i in 1:size(out,1), j in 1:size(out,2)
        ie = findfirst(d -> d == ds[i], ds_exp)
        je = findfirst(chi -> chi == chis[j], chis_exp)
        out_exp[ie, je] = out[i,j]
    end

    return out_exp, ds_exp, chis_exp
end

function bench_heatmap(results::Array{Union{Result, Nothing},3}, chis::Vector{Int}, ds::Vector{Int}, encodings::Vector{T}; balance_klds=false) where {T <: Encoding}
    
    acc_plots = []
    kld_plots = []
    mse_plots = []
    kld_tr_plots = []
    for (ei,e) in enumerate(encodings)
        res = results[ei,:,:]
        all(isnothing.(res)) && continue
        res_exp, ds_exp, chis_exp = expand_dataset(res, ds, chis)

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
        if balance_klds
            for (i, d) in enumerate(ds_exp), (j,chi) in enumerate(chis_exp)
                KLD = klds[i,j]
                isnothing(KLD) && continue
                KLD_rand = get_baseKLD(chi, d, e)

                klds[i,j] = KLD_rand - KLD
                if do_tr 
                    klds_tr[i,j] = KLD_rand - klds_tr[i,j]
                end
            end
        end
        # println(ds)
        # println(chis_exp)
        pt = heatmap(chis_exp,ds_exp, accs;
        xlabel="χmax",
        ylabel="Dimension",
        colorbar_title="Accuracy",
        clims=(0.89, 1),
        cmap = palette([:purple, :green], 11),
        title=e.name * " Encoding")

        push!(acc_plots, pt)

        pt = heatmap(chis_exp,ds_exp, klds;
        xlabel="χmax",
        ylabel="Dimension",
        colorbar_title="KL Div.",
        title=e.name * " Encoding")
        push!(kld_plots, pt)

        pt = heatmap(chis_exp,ds_exp, mses;
        xlabel="χmax",
        ylabel="Dimension",
        colorbar_title="MSE",
        title=e.name * " Encoding ")
        push!(mse_plots, pt)

        if do_tr
            pt = heatmap(chis_exp,ds_exp, klds_tr;
            xlabel="χmax",
            ylabel="Dimension",
            colorbar_title="KL Div. train",
            title=e.name * " Encoding")
            push!(kld_tr_plots, pt)
        end
    end


    return acc_plots, kld_plots, mse_plots, kld_tr_plots
end

function bench_heatmap(path::String; balance_klds::Bool=false)
    results, chi, chis, d, ds, e, encodings = load_result(path) 
    return bench_heatmap(results, chis, ds, encodings; balance_klds=balance_klds)
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
        "val_loss" => Float64[],
        "val_acc" => Float64[],
        "test_loss" => Float64[],
        "test_acc" => Float64[],
        "time_taken" => Float64[], # sweep duration
        "train_KL_div" => Float64[],
        "test_KL_div" => Float64[],
        "val_KL_div" => Float64[]
    )
    @error("Not Implemented Yet!")
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
        W, _, _, test_states, _ = fitMPS(X_train, y_train, X_val, y_val, X_test, y_test; random_state=456, chi_init=chi_max, opts=opts, test_run=true)         
        KLD = KL_div(W, test_states)
        KLDmap2C[(chi_max, d, e)] = KLD
        jldsave("LogLoss/benchmarking/KLDmap.jld2"; KLDmap2C)
    end

    return KLD
end