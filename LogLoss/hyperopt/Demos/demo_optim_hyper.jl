include("../hyperopt.jl")
using JLD2

dloc =  "Data/ecg200/datasets/ecg200.jld2"
f = jldopen(dloc, "r")
    Xs_train = read(f, "X_train")
    ys_train = read(f, "y_train")
    # Xs_test = read(f, "X_test")
    # ys_test = read(f, "y_test")
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

eta_init=0.7
eta_range = (0.001,1)
max_sweeps=5
d_init = 4
d_range = (2,10)
chi_max_init = 25
chi_max_range=(10,60)
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
    nfolds=2,
    method="Best_Eta")

results = hyperopt(gd, Xs_train, ys_train; dir="LogLoss/hyperopt/Benchmarks/ECG200/")

# #TODO make the below less jank
# unfolded = mean(results; dims=1)

# getmissingproperty(f, s::Symbol) = ismissing(f) ? -1 : getproperty(f,s)
# val_accs = getmissingproperty.(unfolded, :acc)
# acc, ind = findmax(val_accs)


# f, swi, etai, di, chmi, ei = Tuple(ind)

# swi = findfirst(val_accs[1, :, etai, di, chmi, ei] .== acc) # make extra extra sure findmax wasnt confused by the sweep format

# println("Best acc $(acc) occured at:\nsweep=$(swi)\neta=$(etas[etai])\nd=$(ds[di])\nchi_max=$(chi_maxs[chmi])\nWith the $(encoding.name) Encoding")


