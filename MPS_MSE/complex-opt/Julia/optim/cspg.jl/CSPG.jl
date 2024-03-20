# Implementation of CSPG from Zhang & Zia
# Adapted from the package SPGBOX.jl, by J. M. Martínez (IMECC - UNICAMP)
# which implements the algorithm of:
#
# NONMONOTONE SPECTRAL PROJECTED GRADIENT METHODS ON CONVEX SETS
# ERNESTO G. BIRGIN, JOSÉ MARIO MARTÍNEZ, AND MARCOS RAYDAN
# SIAM J. O. PTIM. Vol. 10, No. 4, pp. 1196-1211

module CSPG

#using TestItems
using ITensors
using LinearAlgebra

export cspg!
export cspg
export CSPGResult

include("./Manifolds.jl")
include("./CSPGResult.jl")
include("./VAux.jl")
include("./auxiliary_functions.jl")
include("./cspg_main.jl")
end
