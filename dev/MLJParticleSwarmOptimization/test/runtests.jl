using Distributed
addprocs(2)

@everywhere begin
    using MLJParticleSwarmOptimization
    using Random
    using Test
    using ComputationalResources
    using Distributions
    using EvoTrees
    using MLJBase
    using MLJTuning
    using StableRNGs
end

const PSO = MLJParticleSwarmOptimization

include("parameters.jl")
include("update.jl")
include("problems.jl")
include("strategies/basic.jl")
include("strategies/adaptive.jl")