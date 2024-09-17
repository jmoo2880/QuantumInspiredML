abstract type AbstractParticleSwarm <: MLJTuning.TuningStrategy end

struct ParticleSwarmState{T, R, P, I}
    ranges::R
    parameters::P
    indices::I
    X::Matrix{T}
    V::Matrix{T}
    pbest_X::Matrix{T}
    gbest_X::Matrix{T}
    pbest::Vector{T}
    gbest::Vector{T}
end