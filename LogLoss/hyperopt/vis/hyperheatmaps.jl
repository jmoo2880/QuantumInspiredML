# included in vishypertrain.jl

function expand_dataset(out::AbstractMatrix{Union{Result, Missing}}, ys::AbstractVector, xs::AbstractVector)
    # take two vectors of 'axes' and the results vector and allocate a grid and two vectors that align the results

    ys_d = unique(abs.(diff(ys)))
    xs_d = unique(abs.(diff(xs)))

    ys_exp = vcat([minimum(ys):yd:maximum(ys) for yd in ys_d]...) |>  unique |> sort
    xs_exp = vcat([minimum(xs):xd:maximum(xs) for xd in xs_d]...) |>  unique |> sort
    out_exp = Matrix{Union{Result, Missing}}(missing, length(ys_exp), length(xs_exp))

    
    for i in axes(out,1), j in axes(out,2)
        ie = findfirst(d -> d == ys[i], ys_exp)
        je = findfirst(chi -> chi == xs[j], xs_exp)
        out_exp[ie, je] = out[i,j]
    end

    return out_exp, ys_exp, xs_exp
end



function minmax_colourbar(out::AbstractArray{Union{Result, Missing}}, field::Symbol; threshold::Real=0.8)

    data = first.(skipmissing(get_resfield.(out, field))) # the first handles the case of a maxacc tuple ECG_test

    clims = (max(threshold, minimum(data)), maximum(data))
    nomiss = collect(data)
    mindiff = minimum(abs.(diff(sort(unique(nomiss)))))
    cticks = clims[1]:mindiff:clims[2]

    return clims, cticks
end


function bench_heatmap(results::AbstractArray{Union{Result, Missing},6}, 
    ax1::Symbol, 
    ax2::Symbol, 
    ax3::Symbol,
    cax::Symbol, 
    nfolds::Integer, 
    max_sweeps::Integer, 
    etas::AbstractVector{<:Number},
    chi_maxs::AbstractVector{<:Integer}, 
    ds::AbstractVector{<:Integer}, 
    encodings::AbstractVector{T}; 
    balance_klds=false, 
    eta_ind=1,
    sweep_ind=max_sweeps+1,
    d_ind=length(ds),
    chi_ind = length(chi_maxs),
    enc_ind=1
    ) where {T <: Encoding}
    
    plots = []

    axisdict = Dict(
        :sweeps => collect(1:max_sweeps+1),
        :etas => etas,
        :chi_maxs => chi_maxs,
        :ds => ds,
        :encodings => encodings
    )

    indsdict = Dict(
        :folds=> 1,
        :sweeps => sweep_ind,
        :etas => eta_ind,
        :chi_maxs => chi_ind,
        :ds => d_ind,
        :encodings => enc_ind
    )

    labels = Dict(
        :sweeps => "Sweep Number",
        :etas => "Learning Rate",
        :chi_maxs => "χmax",
        :ds => "Encoding Dim.",
        :encodings => "Encoding",
        :acc=>"Accuracy",
        :maxacc=>"Max. Accuracy",
        :KLD=>"Val. KL Div.",
        :KLD_tr=>"Train KL Div.",
        :minKLD=>"Min. Val. KL Div.",
        :MSE=>"Mean Sq. Error"
    )

    xs = axisdict[ax1]
    ys = axisdict[ax2]

    axis_symbols = [:folds, :sweeps, :etas, :ds, :chi_maxs, :encodings]
    ax1ind = findfirst(ax1 .== axis_symbols)
    ax2ind = findfirst(ax2 .== axis_symbols)
    ax3ind = findfirst(ax3 .== axis_symbols)

    inds = (1, sweep_ind, eta_ind, d_ind, chi_ind, enc_ind)

    static = [1,2,3,4,5,6]
    filter!(x -> !(x in [ax1ind, ax2ind, ax3ind]), static)     # only the static inds remain


    # @show static
    # @show [ax1ind, ax2ind, ax3ind]
    i = 0

    if last(encodings).name !== "Stoudenmire"
        while ismissing(results[end-i,end,end,end,end,end])
            i += 1
        end

    else
        while ismissing(results[end-i,end,end,1,end,end])
            i += 1
        end
    end

    if i > 0
        @warn("Dropping $i folds of missing values!")

    end


    resmean = mean(results[1:(end-i),:,:,:,:,:], dims=1)
    resslice = eachslice(resmean, dims=Tuple(static), drop=true) # I wish "selectdims" took multiple inputs. Maybe theres a way to select where to put a ':' with CartesianIndex or something

    sinds = [inds[sind] for sind in static]
    res3d = resslice[sinds...] # down from 6d to 3d!

    ax3ind -= sum(static .< ax3ind) # if there have been indices dropped from below ax3ind, account for it

    # make a string that says what the static indices are set to
    staticsymbs = [axis_symbols[sind] for sind in static[2:end]]
    staticlabels = [labels[symb] for symb in staticsymbs]
    staticvals = [axisdict[symb][indsdict[symb]] for symb in staticsymbs]

    valstring = [", " * staticlabels[i] *"=" * string(staticvals[i]) for i in eachindex(sinds[2:end])]
    # @show size(resmean)
    # @show size(resslice)
    # @show size(res3d)

    

    #colourbar tomfoolery
    if cax in [:acc, :maxacc]
        # clims, cticks = minmax_colourbar(results, cax)
        accs = first.(skipmissing(get_resfield.(res3d, cax))) 
        clims = (minimum(accs),maximum(accs))
        # cmap = palette([:red, :blue], 4*length(cticks))
        # colourbar_ticks = cticks # the 0.5 makes the colourbarticks line up at the centre of the colours
        # colourbar_tick_labels = string.(cticks)
        cbarargs = (:clims=>clims,)#, :cmap=>cmap)
    else
        cbarargs = ()
    end

    for (i,ax3val) in enumerate(axisdict[ax3])

        res = selectdim(res3d, ax3ind, i) # down to 2d!
        all(ismissing.(res)) && continue

        
        if size(res,1) !== length(ys)
            res = permutedims(res)
        end

        res_exp, ys_exp, xs_exp = expand_dataset(res, ys, xs)


        cvals = get_resfield.(res_exp,cax) 

        # using first in case cax is :maxacc or :minKLD, in which case get_resfield will return a tuple (which we want the first entry of)
        cvals = map(x-> ismissing(x) ? x : first(x), cvals)


        
        pt = heatmap(xs_exp, ys_exp, cvals; 
        xlabel=labels[ax1],
        ylabel=labels[ax2],
        colorbar_title=labels[cax],
        title= String(ax3)[1:(end-1)] *"=$ax3val" * valstring[1] * "\n"* valstring[2][3:end],
        margin=5mm,
        cbarargs...)
        push!(plots, pt)

    end
    return plots
end

function bench_heatmap(path::String, ax1::Symbol, ax2::Symbol, ax3::Symbol, cax::Symbol; kwargs...)
    results, fold, nfolds, max_sweeps, eta, etas, chi, chi_maxs, d, ds, e, encodings = load_result(path) 
    return bench_heatmap(results, ax1, ax2, ax3, cax, nfolds, max_sweeps, etas, chi_maxs, ds, encodings; kwargs...)
end





function KLD_heatmaps(results::AbstractArray{Union{Result, Missing},6}, nfolds::Integer, max_sweeps::Integer, chi_maxs::AbstractVector{Int}, ds::AbstractVector{Int}, encodings::AbstractVector{T}; balance_klds=false, etai=1, swi=max_sweeps) where {T <: Encoding}
    
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
        all(ismissing.(res)) && continue
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

function KLD_heatmaps(path::String; kwargs...)
    results, fold, nfolds, max_sweeps, eta, etas, chi, chi_maxs, d, ds, e, encodings = load_result(path) 
    return KLD_heatmaps(results, nfolds, max_sweeps, chi_maxs, ds, encodings; kwargs...)
end

