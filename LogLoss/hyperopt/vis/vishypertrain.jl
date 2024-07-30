# included in hyperUtils.jl
include("hyperheatmaps.jl")

function get_resfield(res::Union{Result,Missing},s::Symbol)
    if ismissing(res)
        return missing
    else
        return getfield(res,s)
    end
end


function training_map(results::AbstractArray{Union{Result, Missing},6}, 
    nfolds::Integer, 
    max_sweeps::Integer, 
    etas::AbstractVector{<:Number},
    chi_maxs::AbstractVector{<:Integer}, 
    ds::AbstractVector{<:Integer}, 
    encodings::AbstractVector{T}; 
    d_ind=length(ds),
    chi_ind = length(chi_maxs),
    chi_max::Union{Nothing, Integer}=nothing,
    d::Union{Nothing, Integer}=nothing,
    enc_ind=1
    ) where {T <: Encoding}

    if !isnothing(chi_max)
        chi_ind = findfirst(chi_maxs .== chi_max)
    end

    if !isnothing(d)
        d_ind = findfirst(ds .== d)
    end
    plots = []

    for (eta_ind,eta) in enumerate(etas)
        max_fold = nfolds
        while max_fold > 1 && (mean(results[1:max_fold,:,:,:,:,:], dims=1) .|> ismissing |> all)
            max_fold -=1
        end
        res = mean(results[1:max_fold,:,:,:,:,:], dims=1)[1, :, eta_ind, d_ind,chi_ind, enc_ind] #avg over the folds

        all(ismissing.(res)) && continue
        res_exp = collect(skipmissing(res))

        te_accs = get_resfield.(res_exp,:acc)
        tr_accs = get_resfield.(res_exp,:acc_tr)

        te_klds = get_resfield.(res_exp,:KLD)
        tr_klds = get_resfield.(res_exp,:KLD_tr)


        
        plot(1:max_sweeps+1, tr_klds; 
        xlabel="Sweep",
        ylabel="KL. Div.",
        title= "eta=$eta, d=$(ds[d_ind]), chi_max=$(chi_maxs[chi_ind])",
        label="Train KLD",
        colour=:red,
        legend=:bottomright
        )

        plot!( 1:max_sweeps+1, te_klds; 
        label="Test KLD",
        colour=:green
        )
        ax = twinx()

        # pt = plot!( ax, [1:max_sweeps+1,1:max_sweeps+1],  [tr_accs, te_accs]; 
        # ylabel="Acc.",
        # label=["Train Acc"; "Test Acc."],
        # legend=:right
        # )
        plot!( ax, 1:max_sweeps+1, tr_accs; 
        ylabel="Acc.",
        label="Train Acc.",
        legend=:right)

        pt = plot!( ax, 1:max_sweeps+1, te_accs; 
        label="Test Acc.")


        push!(plots, pt)

    end
    return plots
end

function training_map(path::String; kwargs...)
    results, fold, nfolds, max_sweeps, eta, etas, chi, chi_maxs, d, ds, e, encodings = load_result(path) 
    return training_map(results, nfolds, max_sweeps, etas, chi_maxs, ds, encodings; kwargs...)
end



function vis_train_end(results::AbstractArray{Union{Result, Missing},6}, 
    ax1::Symbol, 
    ax3::Symbol,
    nfolds::Integer, 
    max_sweeps::Integer, 
    etas::AbstractVector{<:Number},
    chi_maxs::AbstractVector{<:Integer}, 
    ds::AbstractVector{<:Integer}, 
    encodings::AbstractVector{T}; 
    balance_klds=false, 
    eta_ind::Integer=1,
    d_ind::Integer=length(ds),
    chi_ind::Integer = length(chi_maxs),
    enc_ind::Integer=1,
    KLD_ratio::Real=0.01
    ) where {T <: Encoding}
    
    plots = []
    ax2= :sweeps
    cax = :acc

    axisdict = Dict(
        :sweeps => collect(1:max_sweeps+1),
        :etas => etas,
        :chi_maxs => chi_maxs,
        :ds => ds,
        :encodings => encodings
    )

    indsdict = Dict(
        :folds=> 1,
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

    inds = (1, 1, eta_ind, d_ind, chi_ind, enc_ind)

    static = [1,2,3,4,5,6]
    filter!(x -> !(x in [ax1ind, ax2ind, ax3ind]), static)     # only the static inds remain


    # @show static
    # @show [ax1ind, ax2ind, ax3ind]

    resmean = mean(results, dims=1)
    resslice = eachslice(resmean, dims=Tuple(static), drop=true) # I wish "selectdims" took multiple inputs. Maybe theres a way to select where to put a ':' with CartesianIndex or something

    sinds = [inds[sind] for sind in static]
    res3d = resslice[sinds...] # down from 6d to 3d!

    ax3ind -= sum(static .< ax3ind) # if there have been indices dropped from below ax3ind, account for it

    # make a string that says what the static indices are set to. Exclude sweep number because it's not truly static
    staticsymbs = [axis_symbols[sind] for sind in static[2:end]]
    staticsymbs = filter(x-> !(x in [:sweeps, :encodings]), staticsymbs) 

    staticlabels = [labels[symb] for symb in staticsymbs]
    staticvals = [axisdict[symb][indsdict[symb]] for symb in staticsymbs]

    valstring = prod([", " * staticlabels[i] *"=" * string(staticvals[i]) for i in eachindex(staticsymbs)])
    # @show size(resmean)
    # @show size(resslice)
    # @show size(res3d)



    for (i,ax3val) in enumerate(axisdict[ax3])

        res = selectdim(res3d, ax3ind, i) # down to 2d!
        all(ismissing.(res)) && continue

               
        if size(res,1) !== length(ys)
            res = permutedims(res)
        end


        te_accs = get_resfield.(res,:acc)
        tr_accs = get_resfield.(res,:acc_tr)

        te_klds = get_resfield.(res,:KLD)
        tr_klds = get_resfield.(res,:KLD_tr)

        KL_diff = Vector{Union{Missing, Float64}}(missing, length(xs))
        fivesweep = Vector{Union{Missing, Float64}}(missing, length(xs))
        tracc_one = Vector{Union{Missing, Float64}}(missing, length(xs))


    
        for (x_ind, sweepvec) in enumerate(eachcol(res))
            tr_accs = get_resfield.(sweepvec,:acc_tr)
            tr_KLDs = get_resfield.(sweepvec,:KLD_tr)
            accs = get_resfield.(sweepvec,:acc)


            # fivesweep 
            if ax3 == :etas
                eta = ax3val
            elseif ax1 == :etas
                eta = etas[x_ind]
            else
                eta = etas[eta_ind]
            end

            fivesweep_ind = round(Int, max(5, min(5 * 0.1/eta, max_sweeps+1)))
            fivesweep[x_ind] = accs[fivesweep_ind]


            # KLD diff < KLD ratio 
            KLDiffs = abs.(diff(tr_KLDs)) ./ tr_KLDs[1:end-1]
            kl_ind = findfirst(x-> x <= KLD_ratio, KLDiffs)

            KL_diff[x_ind] =  isnothing(kl_ind) ? accs[max_sweeps+1] : accs[kl_ind]

            # Tr acc hits one
            tracc_ind = findfirst(x-> x==1, tr_accs)

            tracc_one[x_ind] = isnothing(tracc_ind) ? accs[max_sweeps+1] : accs[tracc_ind]

        end

 
        
        pt = plot(xs, tracc_one; 
        label="Tr acc 1",        
        title= String(ax3)[1:(end-1)] *"=$ax3val" * valstring,
        )
        plot!(pt, xs, KL_diff,
        label="KLD dif < $(round(KLD_ratio; digits=3))%")

        plot!(pt, xs, fivesweep,
        label="5 Sw. equi.")

        push!(plots, pt)

    end
    return plots
end

function vis_train_end(path::String, ax1::Symbol, ax3::Symbol; kwargs...)
    results, fold, nfolds, max_sweeps, eta, etas, chi, chi_maxs, d, ds, e, encodings = load_result(path) 
    return vis_train_end(results, ax1, ax3, nfolds, max_sweeps, etas, chi_maxs, ds, encodings; kwargs...)
end


function vis_train_avgs(results::AbstractArray{Union{Result, Missing},6}, 
    nfolds::Integer, 
    max_sweeps::Integer, 
    etas::AbstractVector{<:Number},
    chi_maxs::AbstractVector{<:Integer}, 
    ds::AbstractVector{<:Integer}, 
    encodings::AbstractVector{T}; 
    chi_max_range=[first(chi_maxs), last(chi_maxs)],
    d_range=[first(ds), last(ds)],
    enc_ind=1
    ) where {T <: Encoding}

    d_inds = [findfirst(ds .>= d_range[1]), findlast(ds .<= d_range[2])]
    chi_max_inds = [findfirst(chi_maxs .>= chi_max_range[1]), findlast(chi_maxs .<= chi_max_range[2])]

    
    plots = []

    for (eta_ind,eta) in enumerate(etas)

        ys_res = Vector{Result}(undef, max_sweeps+1)
        for sw in 1:max_sweeps+1
            res = results[:,sw,eta_ind,range(d_inds...),range(chi_max_inds...),:][:]
            ys_res[sw] = mean(skipmissing(res))
        end

        res_exp = collect(skipmissing(ys_res))

        te_accs = get_resfield.(res_exp,:acc)
        tr_accs = get_resfield.(res_exp,:acc_tr)

        te_klds = get_resfield.(res_exp,:KLD)
        tr_klds = get_resfield.(res_exp,:KLD_tr)


        
        plot(0:max_sweeps, tr_klds; 
        xlabel="Sweep",
        ylabel="KL. Div.",
        title= "eta=$eta",
        label="Train KLD",
        colour=:red,
        legend=:bottomright
        )

        plot!( 0:max_sweeps, te_klds; 
        label="Test KLD",
        colour=:green
        )
        ax = twinx()

        # pt = plot!( ax, [1:max_sweeps+1,1:max_sweeps+1],  [tr_accs, te_accs]; 
        # ylabel="Acc.",
        # label=["Train Acc"; "Test Acc."],
        # legend=:right
        # )
        plot!(ax, 0:max_sweeps, tr_accs; 
        ylabel="Acc.",
        label="Train Acc.",
        legend=:right)

        pt = plot!(ax,  0:max_sweeps, te_accs; 
        label="Test Acc.")


        push!(plots, pt)

    end
    return plots
end




function vis_train_avgs(path::String; kwargs...)
    results, fold, nfolds, max_sweeps, eta, etas, chi, chi_maxs, d, ds, e, encodings = load_result(path) 
    i = 0
    while ismissing(results[end-i,end,end,1,end,end])
        i += 1
    end

    if i > 0
        @warn("Dropping $i folds of missing values!")
    end

    res = results[1:(end-i),:,:,:,:,:]
    return vis_train_avgs(res, nfolds, max_sweeps, etas, chi_maxs, ds, encodings; kwargs...)
end