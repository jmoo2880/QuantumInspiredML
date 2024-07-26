using Printf
using Plots
using StatsBase
using Measures

include("result.jl") # Results struct and various crimes that should not see the light of day
include("vis/vishypertrain.jl")

const AbstractBounds{T} = Union{AbstractVector{T}, NTuple{2,T}}
const ResDict = Dict{Tuple{Integer, Integer, Integer, Encoding}, Union{Matrix{Result}, Missing}}
abstract type SearchMethod end


"""Prints a vector in range format min_val:min_step:max_val"""
function repr_vec(v::AbstractVector) 
    c = minimum(abs.(diff(v)))
    n = length(string(float(first(v)))) - 2
    fstr = Printf.Format("%.$(n)d:")
    midstr = c == 1. ? "" : Printf.format(fstr,c)
    "$(first(v)):$(midstr)$(last(v))"
end

function repr_vec(t::Tuple)
    return repr_vec([t...])
end

"""Converts a real valued eta into a unique integer, used to safely index dictionaries by value"""
function eta_to_index(eta::Real, eta_min::Real)
    return round(Int, 2*eta/eta_min)
end

"""Converts a unique integer into a real valued eta, inverse of eta_to_index and used to safely index dictionaries by value"""

function index_to_eta(index, eta_min::Real)
    return index * eta_min/2
end

"""Saves the status of a GridSearch Style hyperoptimisation"""
function save_status(path::String, fold::N1, nfolds::N1, eta::C, etas_or_range::AbstractBounds{C}, chi::N2, chi_maxs::AbstractVector{N2}, d::N3, ds::AbstractVector{N3}, e::T, encodings::AbstractVector{T}; append=false) where {N1 <: Integer, N2 <: Integer, N3 <: Integer, C <: Number, T <: Encoding}
    flag = append ? "a" :  "w"

    disable_sigint() do
        f = jldopen(path, flag)
        write(f, "fold", fold)
        write(f, "nfolds", nfolds)

        write(f, "eta", eta)
        write(f, "etas_or_range", etas_or_range)

        write(f, "chi", chi)
        write(f, "chi_maxs", chi_maxs)

        write(f, "d", d)
        write(f, "ds", ds)

        write(f, "e", e)
        write(f, "encodings", encodings)
        close(f)
    end
end

"""Saves the status of a NNSearch Style hyperoptimisation"""

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

"""Reads the status of a hyperoptimisation"""

function read_status(path::String)
    f = jldopen(path, "r")

    if "ds" in keys(f)
        fold = read(f, "fold")
        nfolds = read(f, "nfolds")

        eta = read(f, "eta")
        etas_or_range = nothing
        try # legacy reasons
            etas_or_range = read(f, "etas_or_range")
        catch KeyError
            etas_or_range = read(f, "etas")
        end

        chi = read(f, "chi")
        chi_maxs = read(f, "chi_maxs")

        d = read(f, "d")
        ds = read(f, "ds")

        e = read(f, "e")
        encodings = read(f, "encodings")
        close(f)

        return fold, nfolds, eta, etas_or_range, chi, chi_maxs, d, ds, e, encodings
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


"""Checks if savefile "path" is the same GridSearch Style hyperoptimisation as is currently being run"""

function check_status(path::String, nfolds::N1, etas_or_range::AbstractBounds{C}, chi_maxs::AbstractVector{N2}, ds::AbstractVector{N3}, encodings::AbstractVector{T}) where {N1 <: Integer, N2 <: Integer, N3 <: Integer, C <: Number, T <: Encoding}
    fold_r, nfolds_r, eta_r, etas_or_range_r, chi_r, chi_maxs_r, d_r, ds_r, e_r, encodings_r = read_status(path)

    return nfolds_r == nfolds && etas_or_range_r == etas_or_range && chi_maxs_r == chi_maxs && ds_r == ds && encodings_r == encodings

end

"""Checks if savefile "path" is the same NNSearch Style hyperoptimisation as is currently being run"""

function check_status(path::String, nfolds::N1, eta_range::AbstractBounds{C}, chi_max_range::AbstractBounds{N2}, d_range::AbstractBounds{N3}, encodings::AbstractVector{T}) where {N1 <: Integer, N2 <: Integer, N3 <: Integer, C <: Number, T <: Encoding}
    fold_r, nfolds_r, eta_r, eta_range_r, chi_r, chi_max_range_r, d_r, d_range_r, e_r, encodings_r = read_status(path)

    return nfolds_r == nfolds && eta_range_r == eta_range && chi_max_range_r == chi_max_range && d_range_r == d_range && encodings_r == encodings
end

""" Saves the results vector of a GridSearch Style hyperoptimisation"""
function save_results(resfile::String, results::AbstractArray{Union{Result, Missing}, 6}, fold::N1, nfolds::N1, max_sweeps::N2, eta::C, etas_or_range::AbstractBounds{C}, chi::N3, chi_maxs::AbstractVector{N3}, d::N4, ds::AbstractVector{N4}, e::T, encodings::AbstractVector{T}) where {N1 <: Integer, N2 <: Integer, N3 <: Integer, N4 <: Integer, C <: Number, T <: Encoding}
    disable_sigint() do
        f = jldopen(resfile, "w")
            write(f, "results", results)
            write(f, "max_sweeps", max_sweeps)
        close(f)
    end
    save_status(resfile, fold, nfolds, eta, etas_or_range, chi, chi_maxs, d, ds, e, encodings; append=true)
end

""" Saves the results vector of a NNSearch Style hyperoptimisation"""
function save_results(resfile::String, results::Dict, fold::N1, nfolds::N1, max_sweeps::N2, eta::C, eta_range::AbstractBounds{C}, chi::N3, chi_max_range::AbstractBounds{N3}, d::N4, d_range::AbstractBounds{N4}, e::T, encodings::AbstractVector{T}) where {N1 <: Integer, N2 <: Integer, N3 <: Integer, N4 <: Integer, C <: Number, T <: Encoding}
    disable_sigint() do
        f = jldopen(resfile, "w")
            write(f, "results", results)
            write(f, "max_sweeps", max_sweeps)
        close(f)
    end
    save_status(resfile, fold, nfolds, eta, eta_range, chi, chi_max_range, d, d_range, e, encodings; append=true)
end


"""Loads the results of and current status of a previously run (but not necessarily complete) hyperoptimisation"""
function load_result(resfile::String)
    fold, nfolds, status... = read_status(resfile)
    f = jldopen(resfile,"r")
        results = f["results"]
        max_sweeps = f["max_sweeps"]
    close(f)

    return results, fold, nfolds, max_sweeps, status...
end

""" Simple helper function that lets a user define an optimiser through either the "encodings" or the "encoding" keyword"""
function configure_encodings(
    encodings::Union{Nothing, AbstractVector{<:Encoding}}=nothing,
    encoding::Union{Nothing, Encoding}=nothing
    )

    if isnothing(encoding)
        if isnothing(encodings)
            throw(ArgumentError("Must define either encoding or encodings!"))
        else
            enc = encodings
        end
    elseif !isnothing(encodings)
        if [encoding] == encodings
            enc = encodings
        else
            throw(ArgumentError("Define either \"encoding\" or  \"encodings\"! (or make them agree)"))
        end
    else
        enc = [encoding]
    end
return enc
end

getmissingproperty(f, s::Symbol) = ismissing(f) ? -1 : getproperty(f,s)

function get_exemplar(results::Array{Union{Missing,Result}}, nfolds, max_sweeps, etas, chi_maxs, ds, encodings; num=1, fix_sweep = false)
    unfolded = mean(results; dims=1)

    val_accs = getmissingproperty.(unfolded, :acc)
    sweep_indep = val_accs[1,end,:,:,:,:]
    inds = partialsortperm(sweep_indep[:], 1:num; rev=true)
    # acc, ind = findmax(val_accs)

    outs = []
    for (i,ind) in enumerate(inds)
        acc = sweep_indep[ind]
        etai, di, chmi, ei = Tuple(CartesianIndices(sweep_indep)[ind])

        if fix_sweep
            swi = size(results,2)
        else
            swi = findfirst(val_accs[1, :, etai, di, chmi, ei] .== acc) # make extra extra sure findmax wasnt confused by the sweep format

        end
        if size(results, 3) > 1
            # GS eta
            println("Ex $i: Acc $(round(acc, digits=3)) occured at:\n\tsweep=$(swi)\n\td=$(ds[di])\n\tchi_max=$(chi_maxs[chmi]))\n\teta=$(etas[etai])")
        else
            #optim eta
            println("Ex $i: Acc $(round(acc, digits=3)) occured at:\n\tsweep=$(swi)\n\td=$(ds[di])\n\tchi_max=$(chi_maxs[chmi]))\n\teta_range=$etas")
        end
        push!(outs, (max_sweeps, etas[etai], chi_maxs[chmi], ds[di], encodings[ei]))

    end

    return outs

end

function get_exemplar(results::ResDict, nfolds, max_sweeps, etas, chi_maxs, ds, encodings; num=1, fix_sweep = false)

    vals = values(results) |> collect
    k = keys(results) |> collect
    unfolded = mean.(vals; dims=1)
    accs = map(m->m[1,max_sweeps+1].acc, unfolded)

    inds = partialsortperm(accs, 1:num; rev=true)

    outs = []
    for (i,ind) in enumerate(inds)
        acc = accs[ind]

        key = k[ind]
        etai, chi_max, d, e = key

        eta = index_to_eta(etai, 1e-7)

        swi = size(vals[ind],2)

  
        # GS eta
        println("Ex $i: Acc $(round(acc, digits=3)) occured at:\n\tsweep=$(swi)\n\td=$(d)\n\tchi_max=$(chi_max))\n\teta=$(eta) in $etas (probably)")
        push!(outs, (max_sweeps, eta, chi_max, d, e))
    end

    return outs

end


function get_exemplar(path::String; kwargs...)
    results, fold, nfolds, max_sweeps, eta, etas, chi, chi_maxs, d, ds, e, encodings = load_result(path) 
    i = 0
    while ismissing(results[end-i,end,end,1,end,end])
        i += 1
    end
    
    if i > 0
        @warn("Dropping $i folds of missing values!")
    end
    return get_exemplar(results[1:end-i,:,:,:,:,:], nfolds, max_sweeps, etas, chi_maxs, ds, encodings; kwargs...)
end


function test_exemplars(results::Array{Union{Missing,Result}}, Xs, ys, nfolds, max_sweeps, etas, chi_maxs, ds, encodings; num=1, fix_sweep = false, ntestfolds=10, early_exit=false, sigmoid_transform=true)
    params = get_exemplar(results, nfolds, max_sweeps, etas, chi_maxs, ds, encodings; num=num, fix_sweep=fix_sweep)

    for (i,paramset) in enumerate(params)
        max_sweeps, eta, chi_max, d, e = paramset
        gd = EtaOptimChiDNearestNeighbour(;encoding=e,     
            eta_init=eta, # if you want eta to be a complex number, fix the indexing for complex numbers in hyperUtils.eta_to_index()
            eta_range = [eta*0.9, eta*1.1],
            d_init = d, 
            d_range = [d,d+1],
            chi_max_init = chi_max, 
            chi_max_range = [chi_max, chi_max+1],
            max_sweeps=max_sweeps,
            max_search_steps=20,
            nfolds=ntestfolds,
            immediate_ret=true)

        results = hyperopt(gd, Xs, ys; dir="LogLoss/hyperopt/Benchmarks/valid/",exit_early=early_exit, sigmoid_transform=sigmoid_transform, hverbosity=-1)
        acc = mean(first(values(results)); dims=1)[1,end].acc
        println("Ex $i: Acc $(round(acc, digits=3)) occured at:\n\tsweep=$(max_sweeps)\n\td=$(d)\n\tchi_max=$(chi_max))\n\teta=$(eta)")
    end

end


function test_exemplars(path::String, Xs, ys; kwargs...)
    results, fold, nfolds, max_sweeps, eta, etas, chi, chi_maxs, d, ds, e, encodings = load_result(path) 
    i = 0
    while ismissing(results[end-i,end,end,1,end,end])
        i += 1
    end
    
    if i > 0
        @warn("Dropping $i folds of missing values!")
    end
    return test_exemplars(results[1:end-i,:,:,:,:,:], Xs, ys, nfolds, max_sweeps, etas, chi_maxs, ds, encodings; kwargs...)
end

  


function artificial_early_exit(results::Array{Union{Missing,Result}})
    tr_accs = getmissingproperty.(results, :acc_tr)

    res = Array{Union{Missing,Result}}(undef, size(results)...)

    for sw_slice in eachslice(tr_accs;dims=2)
        for bench_ind in eachindex(sw_slice[1])
            sw = 1
            f, etai, di, chmi, ei = bench_ind |> Tuple
            while sw < length(sw_slice)
                tracc = sw_slice[sw][bench_ind]
                if !ismissing(tracc) && tracc == 1.
                    res[f,sw:end, etai, di, chmi, ei] = results[f,sw:end,etai,di,chmi,ei] # early exit
                    break
                end

                res[f,sw, etai, di, chmi, ei] = results[f,sw,etai,di,chmi,ei]
                sw += 1
            end

        end

    end

end

