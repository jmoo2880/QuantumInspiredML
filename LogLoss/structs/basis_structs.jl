# timeseries encoding shell
abstract type Encoding end

function Encoding(s::String)  # probably should not be called directly
    if occursin("Split", titlecase(s))
        return SplitBasis(s)
    else
        return Basis(s)
    end
end


struct Basis <: Encoding # probably should not be called directly
    name::String
    init::Union{Function,Nothing}
    encode::Function
    iscomplex::Bool
    istimedependent::Bool
    isbalanced::Bool
    range::Tuple{Real, Real}
    project::Bool
    # Basis(s::String,init::Union{Function,Nothing}, enc::Function, isc::Bool, istd::Bool, isb, range::Tuple{Real, Real}) = begin
    #     if !(titlecase(s) in ["Stoud", "Stoudenmire", "Fourier", "Sahand", "Legendre", "Uniform", "Legendre_No_Norm"]) 
    #         error("""Unknown Basis "$s", options are [\"Stoud\", \"Stoudenmire\", \"Fourier\", \"Sahand\", \"Legendre\", "Uniform", "Legendre_No_Norm"]""")
    #     end
    #     new(s, init, enc, isc, istd, isb, range)
    # end
end


function Basis(s::AbstractString)
    @warn("Calling Basis(basis_name::String) is deprecated and may lead to unexpected results. Call the function $(lowercase(s))() instead.")
    sl = titlecase(s)
    init = nothing
    project = false
    if sl == "Stoud" || sl == "Stoudenmire"
        sl = "Stoudenmire" 
        enc = angle_encode
        iscomplex=true
        istimedependent=false
        isbalanced=false
        range = (0,1)
    elseif sl == "Fourier"
        enc = fourier_encode
        iscomplex=true
        istimedependent=false
        isbalanced=false
        range = (-1,1)
    elseif sl == "Sahand"
        enc = sahand_encode
        iscomplex=true
        istimedependent=false
        isbalanced=false
        range = (0,1)
    elseif sl in ["Legendre", "Lengendre_Norm"]
        sl = "Legendre_Norm"
        enc = legendre_encode
        iscomplex = false
        istimedependent=false
        isbalanced=false
        range = (-1,1)
    elseif sl == "Legendre_No_Norm"
        enc = (args...) -> legendre_encode(args...; norm=false)
        iscomplex = false
        istimedependent=false
        isbalanced=false
        range = (-1,1)
    elseif sl =="Uniform"
        enc = uniform_encode
        iscomplex = false
        istimedependent=false
        isbalanced=false
        range = (0,1)
    else
        error("Unknown Basis name!")
    end
    return Basis(sl, init, enc, iscomplex, istimedependent,isbalanced, range, project)
end



function Base.show(io::IO, E::Basis)
    print(io,"Basis($(E.name))")
end

# Splitting up into a time dependent histogram
struct SplitBasis <: Encoding
    name::String
    init::Union{Function,Nothing}
    splitmethod::Function
    basis::Basis
    encode::Function
    iscomplex::Bool
    istimedependent::Bool
    isbalanced::Bool
    range::Tuple{Real, Real}
    SplitBasis(s::String, init::Union{Function,Nothing}, spm::Function, basis::Basis, enc::Function, isc::Bool, istd::Bool, isb, range::Tuple{Real, Real}) = begin
        # spname = replace(s, Regex(" "*basis.name*"\$")=>"") # strip the basis name from the end
        # if !(titlecase(spname) in ["Hist Split", "Histogram Split", "Hist Split Balanced", "Histogram Split Balanced", "Uniform Split Balanced", "Uniform Split"])
        #     error("""Unkown split type "$spname", options are ["Hist Split", "Histogram Split", "Hist Split Balanced", "Histogram Split Balanced", "Uniform Split Balanced", "Uniform Split"]""")
        # end

        if basis.iscomplex != isc
            error("The SplitBasis and its auxilliary basis must agree on whether they are complex!")
        end

        if basis.range != range #TODO This is probably not actually necessary, likely could be handled in encode_TS?
            error("The SplitBasis and its auxilliary basis must agree on the normalised timeseries range!")
        end
        new(s, init, spm, basis, enc, isc, istd, isb, range)
    end
end

function SplitBasis(s::AbstractString)
    return SplitBasis(String.(rsplit(s; limit=2))...)
end

function SplitBasis(s::AbstractString, bn::AbstractString)
    basis = Basis(bn)
    isc = basis.iscomplex
    range = basis.range

    spname = replace(s, Regex(" "*basis.name*"\$")=>"")

    spl = String.(rsplit(spname; limit=2))

    if titlecase(spl[2]) == "Balanced"
        st = spl[1]
        isb = true
    else
        st = spname
        isb = false
    end

    st = titlecase(st)

    if st in ["Hist Split", "Histogram Split"]
        st = "Histogram Split"
        init = hist_split_init
        splitmethod = hist_split
        istd = true
        enc = project_onto_hist_bins

    elseif st == "Uniform Split"
        init = unif_split_init
        splitmethod = unif_split
        istd = basis.istimedependent # if the aux. basis _is_ time dependent we have to treat the entire split as such
        enc = project_onto_unif_bins
    else
        error("Unknown split type \"$st\"")
    end

    return SplitBasis(titlecase(spname)*" "*basis.name,init, splitmethod, basis, enc, isc, istd, isb, range)

end

function Base.show(io::IO, E::SplitBasis)
    print(io,"SplitBasis($(E.name))")
end


##############################################

function stoudenmire()
    project = false
    sl = "Stoudenmire" 
    enc = angle_encode
    iscomplex=true
    istimedependent=false
    isbalanced=false
    range = (0,1)
    init = nothing

    return Basis(sl, init, enc, iscomplex, istimedependent, isbalanced, range, project)
end


function fourier(; project=false)
    sl = "Fourier"
    enc = fourier_encode
    iscomplex=true
    istimedependent=project
    isbalanced=false
    range = (-1,1)
    init = project ? project_fourier : nothing

    return Basis(sl, init, enc, iscomplex, istimedependent, isbalanced, range, project)
end

function legendre(; norm=false, project=false)
    sl = norm ? "Legendre_Norm" : "Legendre_No_Norm"
    enc = norm ? legendre_encode : legendre_encode_no_norm
    iscomplex = false
    istimedependent=project
    isbalanced=false
    range = (-1,1)
    init = project ? project_legendre : nothing

    return Basis(sl, init, enc, iscomplex, istimedependent, isbalanced, range, project)
end

legendre_no_norm(; project=false) = legendre(; norm=false, project) 

function sahand_legendre(istimedependent::Bool=true)
    sl = "Sahand-Legendre" * (istimedependent ? " Time Dependent" : " Time Independent")
    enc = sahand_legendre_encode
    iscomplex = false
    istimedependent=istimedependent
    isbalanced=false
    range = (-1,1)
    init = istimedependent ?  init_sahand_legendre_time_dependent : init_sahand_legendre
    project=false

    return Basis(sl, init, enc, iscomplex, istimedependent, isbalanced, range, project)
end

function sahand()
    project = false
    sl = "Sahand"
    enc = sahand_encode
    iscomplex=true
    istimedependent=false
    isbalanced=false
    range = (0,1)
    init = nothing

    return Basis(sl, init, enc, iscomplex, istimedependent, isbalanced, range, project)
end

function _uniform()
    project = false
    sl = "Uniform"
    enc = uniform_encode
    iscomplex = false
    istimedependent=false
    isbalanced=false
    range = (0,1)
    init = nothing

    return Basis(sl, init, enc, iscomplex, istimedependent, isbalanced, range, project)
end


function histsplit(basis::Basis; balanced=false)
    isc = basis.iscomplex
    range = basis.range

    bstr = balanced ? "Balanced " : ""
    name = bstr * "Hist. Split $(basis.name)" 

    init = hist_split_init
    splitmethod = hist_split
    istd = true
    enc = project_onto_hist_bins

    return SplitBasis(name, init, splitmethod, basis, enc, isc, istd, balanced, range)

end

function unifsplit(basis::Basis; balanced=false)
    isc = basis.iscomplex
    range = basis.range

    bstr = balanced ? "Balanced " : ""
    name = bstr * "Unif. Split $(basis.name)" 

    init = unif_split_init
    splitmethod = unif_split
    istd = basis.istimedependent # if the aux. basis _is_ time dependent we have to treat the entire split as such
    enc = project_onto_unif_bins

    return SplitBasis(name, init, splitmethod, basis, enc, isc, istd, balanced, range)

end

histsplit(; balanced=false) = histsplit(_uniform(); balanced=balanced)
unifsplit(; balanced=false) = unifsplit(_uniform(); balanced=balanced)

