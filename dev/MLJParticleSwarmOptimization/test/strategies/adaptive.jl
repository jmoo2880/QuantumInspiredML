@testset "AdaptiveParticleSwarm Tuning Hyperparameters" begin
    warning = "AdaptiveParticleSwarm requires at least 3 particles. " *
              "Resetting n_particles=3. " *
              "AdaptiveParticleSwarm requires 1.5 ≤ c1 ≤ 2.5, 1.5 ≤ c2 ≤ 2.5, and " *
              "c1 + c2 ≤ 4. Resetting coefficients c1=2.0, c2=2.0. " *
              "AdaptiveParticleSwarm requires 0 ≤ prob_shift < 1. " *
              "Resetting prob_shift=0.25. "
    ps = @test_logs (:warn, warning) AdaptiveParticleSwarm(
        n_particles=2, c1=3, c2=3, prob_shift=2
    )
    @test ps.n_particles == 3
    @test ps.c1 == 2.0
    @test ps.c2 == 2.0
    @test ps.prob_shift == 0.25
end

@testset "Evolutionary Algorithm" begin
    # Evolutionary factor
    X = [1 1 1; 2 2 2; 10 10 10]
    # If the global best particle is the closest to all particles, then f is 0
    @test PSO._evolutionary_factor(X, 2) == 0
    # If the global best particle is the furthest from all particles, then f is 1
    @test PSO._evolutionary_factor(X, 3) == 1

    # Evolutionary phase
    # Initially pick most likely phase
    @test PSO._evolutionary_phase(0.55, nothing) == 1 # marginal probs [0.75, 0.25, 0, 0]
    @test PSO._evolutionary_phase(0.25, nothing) == 2 # marginal probs [0, 0.5, 0.25, 0]
    @test PSO._evolutionary_phase(0.225, nothing) == 3 # marginal probs [0, 0.25, 0.375, 0]
    @test PSO._evolutionary_phase(0.775, nothing) == 4 # marginal probs [0.25, 0, 0, 0.375]
    # Move to the next phase if possible
    @test PSO._evolutionary_phase(0.55, 1) == 2
    @test PSO._evolutionary_phase(0.25, 2) == 3
    @test PSO._evolutionary_phase(0.775, 4) == 1
    # Stay in the current phase if possible and moving to the next is impossible
    @test PSO._evolutionary_phase(0.55, 2) == 2
    @test PSO._evolutionary_phase(0.25, 3) == 3
    @test PSO._evolutionary_phase(0.775, 1) == 1
    # Pick the most likely phase otherwise
    @test PSO._evolutionary_phase(0.55, 3) == 1
    @test PSO._evolutionary_phase(0.25, 4) == 2
    @test PSO._evolutionary_phase(0.225, 4) == 3
    @test PSO._evolutionary_phase(0.775, 2) == 4

    # Cognitive and social coefficients clamping
    @test PSO._clamp_coefficients(1.0, 1.0) == (1.5, 1.5) # lower bound is 1.5
    @test PSO._clamp_coefficients(3.0, 1.0) == (2.5, 1.5) # upper bound is 2.5
    @test PSO._clamp_coefficients(2.25, 2.25) == (2.0, 2.0) # sum cannot be larger than 4

    # Coefficient adaptive control
    rng = StableRNG(1234)
    # Exploration state
    f, phase = 0.65, 1
    w, c1, c2 = PSO._adapt_parameters(rng, 1.75, 1.75, f, phase)
    @test w == 1 / (1 + 1.5*exp(-2.6*f))
    @test c1 ≥ 1.8 # increase cognitive
    @test c2 ≤ 1.7 # decrease social
    # Exploitation state
    f, phase = 0.35, 2
    w, c1, c2 = PSO._adapt_parameters(rng, 1.75, 1.75, f, phase)
    @test w == 1 / (1 + 1.5*exp(-2.6*f))
    @test 1.775 ≤ c1 ≤ 1.8 # slightly increase cognitive
    @test 1.7 ≤ c2 ≤ 1.725 # slightly decrease social
    # Convergence state
    f, phase = 0.0, 3
    w, c1, c2 = PSO._adapt_parameters(rng, 1.75, 1.75, f, phase)
    @test w == 1 / (1 + 1.5*exp(-2.6*f))
    @test 1.775 ≤ c1 ≤ 1.8 # slightly increase cognitive
    @test 1.775 ≤ c2 ≤ 1.8 # slightly increase social
    # Jumping out state
    f, phase = 1.0, 4
    w, c1, c2 = PSO._adapt_parameters(rng, 1.75, 1.75, f, phase)
    @test w == 1 / (1 + 1.5*exp(-2.6*f))
    @test c1 ≤ 1.7 # decrease cognitive
    @test c2 ≥ 1.8 # increase social
end

const losses = []
const modes = [CPU1(), CPUProcesses(), CPUThreads()]

for acceleration in modes
    @testset "EvoTree Tuning with AdaptiveParticleSwarm and $(typeof(acceleration))" begin
        rng = StableRNG(123)
        features = rand(rng, 10_000) .* 5 .- 2
        X = MLJBase.table(reshape(features, (size(features)[1], 1)))
        y = sin.(features) .* 0.5 .+ 0.5
        y = EvoTrees.logit(y) + randn(rng, size(y))
        y = EvoTrees.sigmoid(y)

        tree = EvoTreeRegressor(rng=rng)
        r1 = range(tree, :max_depth; values=[3:7;])
        r2 = range(tree, :eta; lower=-2, upper=0, scale=exp10)

        baseline_self_tuning_tree = TunedModel(
            model=tree,
            tuning=RandomSearch(rng=StableRNG(1234)),
            resampling=CV(nfolds=5, rng=StableRNG(8888)),
            range=[r1, r2],
            measure=mae,
            n=15,
            acceleration=acceleration
        )
        baseline_mach = machine(baseline_self_tuning_tree, X, y)
        fit!(baseline_mach, verbosity=0)
        baseline_rep = report(baseline_mach)
        baseline_best_loss = baseline_rep.best_history_entry.measurement[1]

        self_tuning_tree = TunedModel(
            model=tree,
            tuning=AdaptiveParticleSwarm(rng=StableRNG(1234)),
            resampling=CV(nfolds=5, rng=StableRNG(8888)),
            range=[r1, r2],
            measure=mae,
            n=15,
            acceleration=acceleration
        )
        mach = machine(self_tuning_tree, X, y)
        fit!(mach, verbosity=0)
        rep = report(mach)
        best_loss = rep.best_history_entry.measurement[1]
        push!(losses, best_loss)

        # There is no reason to expect PSO to be better than
        # RandomSearch, but they should give similar results, say within 10%:

        @test abs(best_loss/baseline_best_loss - 1) < 0.1
    end
end

println("Adaptive PSO losses (see Issue #14):")
(; modes=modes, losses=losses) |> MLJBase.pretty
