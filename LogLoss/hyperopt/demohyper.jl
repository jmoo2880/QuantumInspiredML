include("hyperopt.jl")


(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits_txt("LogLoss/datasets/ECG_train.txt", 
"LogLoss/datasets/ECG_val.txt", "LogLoss/datasets/ECG_test.txt")


setprecision(BigFloat, 128)
Rdtype = Float64

verbosity = 0
test_run = false
track_cost = false
#
encoding = Basis("Legendre")
encode_classes_separately = false
train_classes_separately = false


etas = [0.05,0.1,0.5,1]
max_sweeps=5
ds = 3:6
chi_maxs=15:5:25
nfolds=4

results = hyperopt(encoding, X_train, y_train, X_val, y_val; etas=etas, max_sweeps=max_sweeps, ds=ds, chi_maxs=chi_maxs, nfolds=nfolds, distribute=false)


#TODO make the below less jank
unfolded = mean(results; dims=1)

getmissingproperty(f, s::Symbol) = ismissing(f) ? (-1,-1) : getproperty(f,s)
val_accs = getmissingproperty.(unfolded, :maxacc)
val, ind = findmax(val_accs)

f, swi, etai, di, chmi, ei = Tuple(ind)

println("Best acc $(val[1]) occured at:\nsweep=$(val[2])\neta=$(etas[etai])\nd=$(ds[di])\nchi_max=$(chi_maxs[chmi])\nWith the $(encoding.name) Encoding")


