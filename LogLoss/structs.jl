using Parameters
using ITensors

import Base.convert
# type aliases
const PCache = Matrix{ITensor}
const PCacheCol = SubArray{ITensor, 1, PCache, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true} # for view mapping shenanigans
const Maybe{T} = Union{T,Nothing} 





# data structures
struct PState
    """Create a custom structure to store product state objects, 
    along with their associated label and type (i.e, train, test or valid)"""
    pstate::MPS
    label::Int
    type::String
end

const timeSeriesIterable = Vector{PState}


# Black box optimiser shell
struct BBOpt 
    name::String
    fl::String
    BBOpt(s::String, fl::String) = begin
        if !(lowercase(s) in ["optim", "optimkit", "customgd"]) 
            error("Unknown Black Box Optimiser $s, options are [CustomGD, Optim, OptimKit]")
        end
        new(s,fl)
    end
end

function BBOpt(s::String)
    sl = lowercase(s)
    if sl == "customgd"
        return BBOpt(s, "GD")
    else
        return BBOpt(s, "CGD")
    end
end
function Base.show(io::IO, O::BBOpt)
    print(io,O.name," with ", O.fl)
end
# timeseries encoding shell

struct Basis
    name::String
    encode::Function
    iscomplex::Bool
    range::Tuple{Real, Real}
    Basis(s::String, enc::Function, isc::Bool, range::Tuple{Real, Real}) = begin
        if !(lowercase(s) in ["stoud", "stoudenmire", "fourier", "sahand", "legendre"]) 
            error("Unknown Basis $s, options are [\"Stoud\", \"Stoudenmire\", \"Fourier\", \"Sahand\", \"Legendre\"]")
        end
        new(s,enc,isc, range)
    end
end


function Basis(s::String)
    
    sl = titlecase(s)
    
    if sl == "Stoud" || sl == "Stoudenmire"
        sl = "Stoudenmire" 
        enc = angle_encode
        iscomplex=true
        range = (0,1)
    elseif sl == "Fourier"
        enc = fourier_encode
        iscomplex=true
        range = (0,1)
    elseif sl == "Sahand"
        enc = sahand_encode
        iscomplex=true
        range = (0,1)
    elseif sl == "Legendre"
        enc = legendre_encode
        iscomplex = false
        range = (-1,1)
    else
        enc = identity
        iscomplex = false
        range = (0,1)
    end
    return Basis(sl, enc, iscomplex,range)
end



function Base.show(io::IO, E::Basis)
    print(io,E.name)
end

# Splitting up into a time dependent histogram













# container for options with default values

function default_iter()
    @error("No loss_gradient function defined in options")
end
@with_kw struct Options
    nsweeps::Int
    chi_max::Int
    cutoff::Float64
    update_iters::Int
    verbosity::Int
    dtype::DataType
    lg_iter::Function
    bbopt::BBOpt
    track_cost::Bool
    eta::Float64
    rescale::Vector{Bool}
    d::Int
    encoding::Basis
end

function Options(; nsweeps=5, chi_max=25, cutoff=1E-10, update_iters=10, verbosity=1, dtype::DataType=ComplexF64, lg_iter=default_iter, bbopt=BBOpt("CustomGD"),
    track_cost::Bool=(verbosity >=1), eta=0.01, rescale = [false, true], d=2, encoding=Basis("Stoudenmire"))
    Options(nsweeps, chi_max, cutoff, update_iters, verbosity, dtype, lg_iter, bbopt, track_cost, eta, rescale, d, encoding)
end

# type conversions
# These are reasonable to implement because Basis() and BBOpt() are just wrapper types with some validation built in
convert(::Type{Basis}, s::String) = Basis(s)
convert(::Type{BBOpt}, s::String) = BBOpt(s)