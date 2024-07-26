include("../hyperopt.jl")
using JLD2

dloc =  "Data/italypower/datasets/ItalyPowerDemandOrig.jld2"
f = jldopen(dloc, "r")
    Xs_train = read(f, "X_train")
    ys_train = read(f, "y_train")
    # X_test = read(f, "X_test")
    # y_test = read(f, "y_test")
close(f)


setprecision(BigFloat, 128)
Rdtype = Float64

verbosity = 0
test_run = false
track_cost = false
#
encoding = legendre(project=false)
encode_classes_separately = false
train_classes_separately = false

eta_init=0.1
eta_range = (0.001,1)
max_sweeps=5
d_init = 2
d_range = (2,10)
chi_max_init = 15
chi_max_range=(10,60)
exit_early=false
gd = EtaOptimChiDNearestNeighbour(;encoding=encoding,     
    eta_init=eta_init, # if you want eta to be a complex number, fix the indexing for complex numbers in hyperUtils.eta_to_index()
    eta_range = eta_range,
    d_init = d_init, 
    d_range = d_range,
    d_step = 1,
    chi_max_init = chi_max_init, 
    chi_max_range = chi_max_range,
    chi_step=1,
    max_sweeps=max_sweeps,
    max_search_steps=20,
    nfolds=10,
    #method="Best_Eta", 
    max_neighbours=5)

results = hyperopt(gd, Xs_train, ys_train; dir="LogLoss/hyperopt/Benchmarks/IPD/",exit_early=exit_early, sigmoid_transform=true)

# #TODO make the below less jank
get_exemplars(results, nfolds, max_weeps, etas, chi_maxs, ds, encodings; num=5, fix_sweep = !exit_early)

