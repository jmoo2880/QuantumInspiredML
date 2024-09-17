function optimize(f, ranges, ps::ParticleSwarm, iter)
    fields = getproperty.(ranges, :field)
    state = PSO.initialize(ranges, ps)
    for _ in 1:iter
        PSO.retrieve!(state, ps)
        measurements = [
            f(zip(fields, params)) for params in zip(state.parameters...)
        ]
        PSO.pbest!(state, measurements, ps)
        PSO.gbest!(state)
        PSO.move!(ps.rng, state, ps.w, ps.c1, ps.c2)
    end
    min, particle = findmin(state.pbest)
    return min, state.pbest_X[particle, :]
end

function optimize(f, ranges, ps::AdaptiveParticleSwarm, iter)
    fields = getproperty.(ranges, :field)
    state = PSO.initialize(ranges, ps)
    phase = nothing
    c1, c2 = ps.c1, ps.c2
    for _ in 1:iter
        PSO.retrieve!(state, ps)
        measurements = [
            f(zip(fields, params)) for params in zip(state.parameters...)
        ]
        PSO.pbest!(state, measurements, ps)
        PSO.gbest!(state)
        factor = PSO._evolutionary_factor(state.X, argmin(state.pbest))
        phase = PSO._evolutionary_phase(factor, phase)
        w, c1, c2 = PSO._adapt_parameters(ps.rng, c1, c2, factor, phase)
        PSO.move!(ps.rng, state, w, c1, c2)
    end
    min, particle = findmin(state.pbest)
    return min, state.pbest_X[particle, :]
end

for PS in (ParticleSwarm, AdaptiveParticleSwarm)
    @testset "Optimize Ackley with $PS" begin
        function ackley(x; a=20, b=0.2, c=2π)
            d = length(x)
            return -a * exp(-b * sqrt(sum(x.^2) / d)) -
                exp(sum(cos.(c .* x)) / d) + a + ℯ
        end

        @testset "Numeric Ackley" begin
            r1 = range(Float64, :r1; lower=-2.0, upper=2.0)
            r2 = range(Float64, :r2; lower=-2.0, upper=2.0)
            ps = PS(n_particles=10, rng=StableRNG(1234))
            min, params = optimize([r1, r2], ps, 1_500_000) do pairs
                x = [param for (field, param) in pairs]
                ackley(x)
            end
            @test all(isapprox.(params, 0; atol=3)) # analytical solution
        end

        @testset "Integer Ackley" begin
            r1 = range(Int, :r1; lower=-20, upper=20)
            r2 = range(Int, :r2; lower=-20, upper=20)
            ps = PS(n_particles=10, rng=StableRNG(1234))
            min, params = optimize([r1, r2], ps, 250) do pairs
                x = [param for (field, param) in pairs]
                ackley(x)
            end
            @test all(round.(params) .== 0) # analytical solution
        end

        @testset "Nominal Ackley" begin
            vals = shuffle(StableRNG(1234), -2:2)
            r1 = range(Int, :r1; values=vals)
            r2 = range(Int, :r2; values=vals)
            ps = PS(n_particles=10, rng=StableRNG(1234))
            min, params = optimize([r1, r2], ps, 100) do pairs
                x = [param for (field, param) in pairs]
                ackley(x)
            end
            true_params = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0.] # analytical solution
            @test all(isapprox.(params, true_params; atol=1e-3))
        end

        @testset "Mixed Ackley" begin
            vals = shuffle(StableRNG(1234), -2:2)
            r1 = range(Float64, :r1; lower=-2.0, upper=2.0)
            r2 = range(Int, :r2; lower=-20, upper=20)
            r3 = range(Int, :r3; values=vals)
            ps = PS(n_particles=20, prob_shift=0.15, rng=StableRNG(1234))
            min, params = optimize([r1, r2, r3], ps, 20000) do pairs
                x = [param for (field, param) in pairs]
                ackley(x)
            end
            # Compare with analytical solution
            @test isapprox(params[1], 0; atol=1e-3)
            @test round(params[2]) == 0
            @test argmax(params[3:7]) == 1
        end
    end
end