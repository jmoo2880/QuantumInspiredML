# MLJParticleSwarmOptimization

Particle swarm optimization for hyperparameter tuning in [MLJ](https://github.com/alan-turing-institute/MLJ.jl).

[![Build Status](https://github.com/JuliaAI/MLJParticleSwarmOptimization.jl/workflows/CI/badge.svg?branch=master)](https://github.com/JuliaAI/MLJTuning.jl/actions)
[![codecov](https://codecov.io/gh/JuliaAI/MLJParticleSwarmOptimization.jl/branch/master/graph/badge.svg?token=W71AMGZ4IW)](https://codecov.io/gh/JuliaAI/MLJParticleSwarmOptimization.jl)

[MLJParticleSwarmOptimization](https://github.com/JuliaAI/MLJParticleSwarmOptimization.jl/) offers a suite of different particle swarm algorithms, extending [MLJTuning](https://github.com/JuliaAI/MLJTuning.jl)'s existing collection of tuning strategies. Currently supported variants and planned releases include:
- [x] `ParticleSwarm`: the original algorithm as conceived by Kennedy and Eberhart [1]
- [x] `AdaptiveParticleSwarm`: Zhan et. al.'s variant with adaptive control of swarm coefficients [2]
- [ ] `OMOPSO`: Sierra and Coello's multi-objective particle swarm variant [3]

## Installation

This package is registered, and can be installed via the Julia REPL:

```julia
julia> ]add MLJParticleSwarmOptimization
```

## Discrete Hyperparameter Handling

Most particle swarm algorithms are designed for problems in continuous domains. To extend support for [MLJ](https://github.com/alan-turing-institute/MLJ.jl)'s integer `NumericRange` and `NominalRange`, we encode discrete hyperparameters with an internal continuous representation, as proposed by Strasser et. al. [4]. See the tuning strategies' documentation and reference the paper for more details.

## Examples

```julia
julia> using MLJ, MLJDecisionTreeInterface, MLJParticleSwarmOptimization, Plots, StableRNGs

julia> rng = StableRNG(1234);

julia> X = MLJ.table(rand(rng, 100, 10));

julia> y = 2X.x1 - X.x2 + 0.05*rand(rng, 100);

julia> Tree = @load DecisionTreeRegressor pkg=DecisionTree verbosity=0;

julia> tree = Tree();

julia> forest = EnsembleModel(atom=tree);

julia> r1 = range(forest, :(atom.n_subfeatures), lower=1, upper=9);

julia> r2 = range(forest, :bagging_fraction, lower=0.4, upper=1.0);
```

### `ParticleSwarm`

```julia
julia> self_tuning_forest = TunedModel(
           model=forest,
           tuning=ParticleSwarm(rng=StableRNG(0)),
           resampling=CV(nfolds=6, rng=StableRNG(1)),
           range=[r1, r2],
           measure=rms,
           n=15
       );

julia> mach = machine(self_tuning_forest, X, y);

julia> fit!(mach, verbosity=0);

julia> plot(mach)
```
![basic](https://github.com/JuliaAI/MLJParticleSwarmOptimization.jl/blob/assets/basic.svg)

### `AdaptiveParticleSwarm`

```julia
julia> self_tuning_forest = TunedModel(
           model=forest,
           tuning=AdaptiveParticleSwarm(rng=StableRNG(0)),
           resampling=CV(nfolds=6, rng=StableRNG(1)),
           range=[r1, r2],
           measure=rms,
           n=15
       );

julia> mach = machine(self_tuning_forest, X, y);

julia> fit!(mach, verbosity=0);

julia> plot(mach)
```

![adaptive](https://github.com/JuliaAI/MLJParticleSwarmOptimization.jl/blob/assets/adaptive.svg)

## References
[1] [Kennedy, J., & Eberhart, R. (1995, November). Particle swarm optimization. In Proceedings of ICNN'95-international conference on neural networks (Vol. 4, pp. 1942-1948). IEEE.](https://ieeexplore.ieee.org/abstract/document/488968/)

[2] [Zhan, Z. H., Zhang, J., Li, Y., & Chung, H. S. H. (2009). Adaptive particle swarm optimization. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 39(6), 1362-1381.](https://ieeexplore.ieee.org/abstract/document/4812104/)

[3] [Sierra, M. R., & Coello, C. A. C. (2005, March). Improving PSO-based multi-objective optimization using crowding, mutation and∈-dominance. In International conference on evolutionary multi-criterion optimization (pp. 505-519). Springer, Berlin, Heidelberg.](https://link.springer.com/chapter/10.1007/978-3-540-31880-4_35)

[4] [Strasser, S., Goodman, R., Sheppard, J., & Butcher, S. (2016, July). A new discrete particle swarm optimization algorithm. In Proceedings of the Genetic and Evolutionary Computation Conference 2016 (pp. 53-60).](https://dl.acm.org/doi/abs/10.1145/2908812.2908935)
