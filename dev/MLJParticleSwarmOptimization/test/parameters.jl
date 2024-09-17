@testset "Parameters" begin
    # Swarm size
    n = 3
    # Test ranges
    r1 = range(String, :r1; values=["a", "b", "c"])
    r2 = range(Int, :r2; lower=1, upper=3, scale=exp10)
    r3 = range(Float64, :r3; lower=1, upper=Inf, origin=4, unit=1)
    r4 = range(Float32, :r4; lower=-Inf, upper=Inf, origin=0, unit=1)
    rs = (r1, r2, r3, r4)
    # Test distribution types
    Ds = (Uniform, Gamma, Normal)
    # Manually fitted distributions for test ranges
    d1 = Dirichlet(ones(3))
    d2 = Uniform(1, 3)
    d3 = truncated(Gamma(16, 0.25), 1, Inf)
    d4 = Normal(0, 1)
    ds = (d1, d2, d3, d4)
    # Test range representation lengths
    lengths = (3, 1, 1, 1)
    # Test range's corresponding indices in internal representation state.X
    indices = (1:3, 4, 5, 6)
    # Initial hyperparameter representation
    rng = StableRNG(1234)
    X1 = rand(rng, d1, n)'
    X2 = rand(rng, d2, n)
    X3 = rand(rng, d3, n)
    X4 = rand(rng, d4, n)
    Xs = (X1, X2, X3, X4)

    @testset "Initializer" begin
        for (r, D) in zip(rs[2:end], Ds)
            @test PSO._initializer(MLJTuning.boundedness(r)) === D
        end
    end

    @testset "Initialize with distribution types" begin
        rng = StableRNG(1234)
        PSO._initialize(rng, r1, n)
        for (r, D, l, X) in zip(rs[2:end], Ds, lengths[2:end], Xs[2:end])
            r̂, l̂, X̂ = PSO._initialize(rng, r, D, n)[[1,3,4]]
            @test r̂ === r
            @test l̂ == l
            @test X̂ ≈ X
        end
    end

    @testset "Initialize with distributions" begin
        rng = StableRNG(1234)
        for (r, d, l, X) in zip(rs, ds, lengths, Xs)
            r̂, l̂, X̂ = PSO._initialize(rng, r, d, n)[[1,3,4]]
            @test r̂ === r
            @test l̂ == l
            @test X̂ ≈ X
        end
    end

    @testset "Range Indices" begin
        @test PSO._to_indices(lengths) == indices
    end

    @testset "Unsupported distributions" begin
        rng = StableRNG(1234)
        @test_throws ArgumentError PSO._initialize(rng, r1, Uniform, n)
        @test_throws ArgumentError PSO._initialize(rng, r1, Dirichlet, n)
        @test_throws ArgumentError PSO._initialize(rng, r1, d2, n)
        @test_throws ArgumentError PSO._initialize(rng, r1, Dirichlet(ones(4)), n)
        @test_throws ArgumentError PSO._initialize(rng, r2, Dirichlet, n)
        @test_throws ArgumentError PSO._initialize(rng, r2, d1, n)
    end

    @testset "Initialize one range" begin
        rng = StableRNG(1234)
        for (r, l, i, X) in zip(rs, lengths, indices, Xs)
            state = PSO.initialize(rng, r, n)
            @test state.ranges == (r,)
            @test state.indices == (l == 1 ? 1 : 1:l,)
            @test state.X ≈ X
        end
    end

    @testset "Initialize multiple ranges" begin
        rng = StableRNG(1234)
        ranges = [r1, (r2, Uniform), (r3, d3), r4]
        state = PSO.initialize(rng, ranges, n)
        @test state.ranges == rs
        @test state.indices == indices
        @test state.X ≈ hcat(Xs...)
    end

    @testset "Retrieve parameters" begin
        rng = StableRNG(1234)
        ranges = [r1, (r2, Uniform), (r3, d3), r4]
        state = PSO.initialize(rng, ranges, n)
        PSO.retrieve!(rng, state)
        params =  state.parameters
        @test params[1] isa AbstractVector{<:AbstractString}
        @test params[2] isa AbstractVector{<:Integer}
        @test params[3] isa AbstractVector{<:AbstractFloat}
        @test params[4] isa AbstractVector{<:AbstractFloat}
        @test length(params[4]) == 3
        @test all(params) do p
            length(p) == n
        end
    end
end
