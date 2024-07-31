include("../hyperopt.jl")

using JLD2

dloc =  "Data/ecg5000/datasets/ecg5000.jld2"
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
exit_early=false
#
encoding = legendre(project=false)
encode_classes_separately = false
train_classes_separately = false


etas = [0.004, 0.007, 0.01, 0.04, 0.07, 0.1, 0.4, 0.7]
max_sweeps=5
ds = [2:6; Int.(ceil.(8:2:15))]
chi_maxs= [15:5:35; 40:10:70]
nfolds=5

gd = GridSearch3D(;encodings = [encoding], etas=etas, max_sweeps=max_sweeps, ds=ds, chi_maxs=chi_maxs,nfolds=nfolds)

results = hyperopt(gd, Xs_train, ys_train; distribute=false, dir="LogLoss/hyperopt/Benchmarks/ECG5000/", sigmoid_transform=true, exit_early=false)


get_exemplar(results, nfolds, max_sweeps, etas, chi_maxs, ds, encodings; num=5, fix_sweep=true)


