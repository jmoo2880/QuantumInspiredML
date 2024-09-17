function initialize(r, tuning::AbstractParticleSwarm)
    return initialize(tuning.rng, r, tuning.n_particles)
end

function retrieve!(state::ParticleSwarmState, tuning::AbstractParticleSwarm)
    return retrieve!(tuning.rng, state)
end

function pbest!(state::ParticleSwarmState, measurements, tuning::AbstractParticleSwarm)
    return pbest!(state, measurements, tuning.prob_shift)
end