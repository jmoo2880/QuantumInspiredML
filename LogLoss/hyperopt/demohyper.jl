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