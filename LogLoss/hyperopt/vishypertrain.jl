function training_map(results::AbstractArray{Union{Result, Missing},6}, 
    nfolds::Integer, 
    max_sweeps::Integer, 
    etas::AbstractVector{<:Number},
    chi_maxs::AbstractVector{<:Integer}, 
    ds::AbstractVector{<:Integer}, 
    encodings::AbstractVector{T}; 
    d_ind=length(ds),
    chi_ind = length(chi_maxs),
    enc_ind=1
    ) where {T <: Encoding}
    
    plots = []

    for (eta_ind,eta) in enumerate(etas)
        res = mean(results, dims=1)[:, eta_ind, d_ind,chi_ind, enc_ind] #avg over the folds

        all(ismissing.(res)) && continue

        res_exp = collect(skipmissing(res))
        cvals = first.(get_resfield.(res_exp,cax)) # using first in case cax is :maxacc or :minKLD, in which case get_resfield will return a tuple (which we want the first entry of)

        te_accs = get_resfield.(res_exp,:acc)
        tr_accs = get_resfield.(res_exp,:acc_tr)

        te_klds = get_resfield.(res_exp,:KLD)
        tr_klds = get_resfield.(res_exp,:KLD_tr)


        
        plot(1:max_sweeps, tr_klds; 
        xlabel="Sweep",
        ylabel="KL. Div.",
        title= "eta=$eta, d=$(ds[d_ind]), chi_max=$(chi_maxs[chi_ind])",
        label="Train KLD",
        legend=:bottomright
        )

        plot!( 1:max_sweeps, te_klds; 
        label="Test KLD"
        )

        pt = plot!( twinx(), 1:max_sweeps,  [tr_accs, te_accs]; 
        ylabel="Acc.",
        label=["Train Acc", "Test Acc."],
        colour=[:orange, :green]
        )


        push!(plots, pt)

    end
    return plots
end

function training_map(path::String; kwargs...)
    results, fold, nfolds, max_sweeps, eta, etas, chi, chi_maxs, d, ds, e, encodings = load_result(path) 
    return training_map(results, nfolds, max_sweeps, etas, chi_maxs, ds, encodings; kwargs...)
end


