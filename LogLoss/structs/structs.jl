using Parameters
using ITensors
using MLJ

import Base.convert
# type aliases
const PCache = Matrix{ITensor}
const PCacheCol = AbstractVector{ITensor} # for view mapping shenanigans
const Maybe{T} = Union{T,Nothing} 

# value types
struct TrainSeparate{Bool} end # value type to determine whether training is together or separate
struct EncodeSeparate{Bool} end # value type for dispatching on whether to encode classes separately
struct DataIsRescaled{Bool} end # Value type to tell fitMPS the data has already been rescaled. Do not use unless you know what you're doing!

# data structures
struct PState
    """Create a custom structure to store product state objects, 
    along with their associated label and type (i.e, train, test or valid)"""
    pstate::MPS
    label::Int # TODO make this a symbol for genericness
    label_index::UInt
end

const TimeseriesIterable = Vector{PState}
struct EncodedTimeseriesSet
    timeseries::TimeseriesIterable
    class_distribution::Vector{Integer}
end
function EncodedTimeseriesSet(class_dtype::DataType=Int64) # empty version
    tsi = TimeseriesIterable(undef, 0)
    class_dist = Vector{class_dtype}(undef, 0) # pays to assume the worst and match types...
    return EncodedTimeseriesSet(tsi, class_dist)
end
Base.isempty(e::EncodedTimeseriesSet) = isempty(e.timeseries) && isempty(class_dist)

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



include("basis_structs.jl")
include("options.jl")


# type conversions
# These are reasonable to implement because Basis() and BBOpt() are just wrapper types with some validation built in
convert(::Type{Basis}, s::String) = Basis(s)
convert(::Type{BBOpt}, s::String) = BBOpt(s)

