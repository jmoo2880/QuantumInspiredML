using Printf
using Plots

struct Result
    acc::Float64
    conf::Matrix{Float64}
    KLD::Float64
    MSE::Float64
end

function Result(d::Dict{String,Vector{Float64}},s::Dict{Symbol, Any})

    acc = d["test_acc"][end]
    conf = s[:confmat]
    KLD = d["test_KL_div"][end]
    MSE = d["test_loss"][end]
    return Result(acc, conf, KLD, MSE)
end



function save_status(path::String,chi::Int,d::Int,e::Encoding, chis::Vector{Int},ds::Vector{Int},encodings::Vector{Encoding}; append=false)
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


function check_status(path::String,chis::Vector{Int},ds::Vector{Int},encodings::Vector{Encoding})

    chi_r, chis_r, d_r, ds_r, e_r, encodings_r = read_status(path)

    if chis_r == chis && ds_r == ds && encodings_r == encodings
        return true, [chi_r,d_r,e_r]

    else
        return false, [chis_r,ds_r,encodings_r]
    end
end

function logdata(fpath::String, W::MPS, info::Dict, train_states::timeSeriesIterable, test_states::timeSeriesIterable, opts::Options)
    f = open(fpath, "a")
    print_opts(opts; io=f)
    stats = get_training_summary(W, train_states, test_states; print_stats=true, io=f);

    sweep_summary(info; io=f)
    print(f, "\n\n/=======================================================================================================================================================\\ \n\n")
    close(f)
    return stats
end

function load_result(resfile::String)
    chi, chis, d, ds, e, encodings = read_status(resfile)
    f = jldopen(resfile,"r")
        results = f["output"]
    close(f)

    return results, chi, chis, d, ds, e, encodings
end


function format_result(r::Result, i::Int, j::Int; fancy_conf=false, conf_titles=true)

    cf = r.conf
    acc = r.acc
    mse = r.MSE
    kld = r.KLD

    nclasses = size(cf,1)


    if fancy_conf
        if conf_titles
            header = ["Pred. |$n⟩" for n in 0:(nclasses-1)]
            row_labels = ["True |$n⟩" for n in 0:(nclasses-1)]

            cf = pretty_table(String,cf;
            hlines=:all,
            compact_printing=true,
            header=header,
            row_labels=row_labels,
            highlighters = Highlighter(f = (data, i, j) -> (i == j), crayon = crayon"bold green"))
        else
            cf = pretty_table(String,cf;
            hlines=:all,
            compact_printing=true,
            show_header=false,
            highlighters = Highlighter(f = (data, i, j) -> (i == j), crayon = crayon"bold green"))
        end
    else
        cf = string.(Int.(cf)) .* "|"
        cf = hcat(["c$(n): |" for n in 0:(nclasses-1)], cf)
        cf = reduce(*,cf; dims=2)
    end

    return @sprintf("%s\nAcc: %.3f; KLD: %.4f; MSE: %.4f",cf, acc, kld, mse)
end

format_result(::Nothing, args...; kwargs...) = nothing


function tab_results(results::Array{Union{Result, Nothing},3}, chis::Vector{Int}, ds::Vector{Int}, encodings::Vector{Encoding};
        io::IO=stdin, fancy_conf=false, conf_titles=true)


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
        pretty_table(io,res;
                    title=e.name * " Encoding",
                    title_alignment=:c,
                    title_same_width_as_table=true,
                    header = ["χmax = $n" for n in chis],
                    row_labels = ["d = $n" for n in ds],
                    alignment=:c,
                    hlines=:all,
                    linebreaks=true,
                    #highlighters = (h1,h2),
                    formatters = (args...) -> format_result(args...; fancy_conf=fancy_conf, conf_titles=conf_titles))


    end
end

function tab_results(path::String; io::IO=stdin, fancy_conf=true, conf_titles=true)
    results, chi, chis, d, ds, e, encodings = load_result(path) 
    tab_results(results, chis, ds, encodings; io=io, fancy_conf=fancy_conf, conf_titles=conf_titles)
end

function get_resfield(res::Union{Result,Nothing},s::Symbol)
    if isnothing(res)
        return nothing
    else
        return getfield(res,s)
    end
end

function bench_heatmap(results::Array{Union{Result, Nothing},3}, chis::Vector{Int}, ds::Vector{Int}, encodings::Vector{Encoding})
    
    acc_plots = []
    kld_plots = []
    mse_plots = []
    for (ei,e) in enumerate(encodings)
        res = results[ei,:,:]
        all(isnothing.(res)) && continue
        res_exp, ds_exp, chis_exp = expand_dataset(res, ds, chis)

        accs = get_resfield.(res_exp,:acc)
        klds = get_resfield.(res_exp,:KLD)
        mses = get_resfield.(res_exp,:MSE)
        # println(ds)
        # println(chis_exp)
        pt = heatmap(chis_exp,ds_exp, accs;
        xlabel="χmax",
        ylabel="Dimension",
        colorbar_title="Accuracy",
        clims=(minimum((x) -> isnothing(x) ? 1 : x,accs), 1),
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
    end


    return acc_plots, kld_plots, mse_plots
end

function bench_heatmap(path::String)
    results, chi, chis, d, ds, e, encodings = load_result(path) 
    return bench_heatmap(results, chis, ds, encodings)
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