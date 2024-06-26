include("hyperopt.jl")


(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits_txt("LogLoss/datasets/ECG_train.txt", 
"LogLoss/datasets/ECG_val.txt", "LogLoss/datasets/ECG_test.txt")


setprecision(BigFloat, 128)
Rdtype = Float64

verbosity = 0
test_run = false
track_cost = false
#
encoding = Basis("Stoudenmire")
encode_classes_separately = false
train_classes_separately = false


etas =[0.5]
max_sweeps=5
ds = [2]
chi_maxs=[15]
nfolds=1

results = hyperopt(encoding, X_train, y_train, X_val, y_val; etas=etas, max_sweeps=max_sweeps, ds=ds, chi_maxs=chi_maxs, nfolds=nfolds)

unfolded = mean(skipmissing(results), dims=1)
val_accs = getproperty.(unfolded, :maxacc)
val, ind = findmax(val_accs)

swi, etai, di, chmi, ei = Tuple(ind)

println("Best acc $(val[1]) occured at:\nsweep=$(val[2])\neta=$(etas[etai])\nd=$(ds[di])\nchi_max=$(chi_max[chmi])\nWith the $(encoding.name) Encoding")


