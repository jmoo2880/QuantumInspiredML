using Manifolds, Optim, ManifoldsBase
using ManifoldDiff


# from https://gist.github.com/mateuszbaran/0354c0edfb9cdf25e084a2b915816a09

struct ManifoldWrapper{TM<:AbstractManifold} <: Optim.Manifold
    M::TM
end

function Optim.retract!(M::ManifoldWrapper, x)
    ManifoldsBase.embed_project!(M.M, x, x)
    return x
end

function Optim.project_tangent!(M::ManifoldWrapper, g, x)
    ManifoldsBase.embed_project!(M.M, g, x, g)
    return g
end

#sol = optimize(f, g_FD!, x0, ConjugateGradient(; manifold=ManifoldWrapper(M)))