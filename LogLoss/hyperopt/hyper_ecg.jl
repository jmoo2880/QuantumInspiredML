include("hyperopt.jl")

(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits_txt("LogLoss/datasets/ECG_train.txt", 
"LogLoss/datasets/ECG_val.txt", "LogLoss/datasets/ECG_test.txt")

Xs = [X_train ; X_val] 
ys = [y_train; y_val] 

setprecision(BigFloat, 128)
Rdtype = Float64

verbosity = 0
test_run = false
track_cost = false
#
encoding = legendre(project=true)
encode_classes_separately = false
train_classes_separately = false


etas = [ 0.01, 0.04, 0.07, 0.1, 0.3, 0.5]
max_sweeps=10
ds = [2:6; Int.(ceil.(8:1.5:15))]
chi_maxs=[20:37; 40:10:70]

results = hyperopt(encoding, Xs, ys; etas=etas, max_sweeps=max_sweeps, ds=ds, chi_maxs=chi_maxs, distribute=false, train_ratio=0.8, sigmoid_transform=false)


#TODO make the below less jank
unfolded = mean(results; dims=1)

getmissingproperty(f, s::Symbol) = ismissing(f) ? -1 : getproperty(f,s)
val_accs = getmissingproperty.(unfolded, :acc)
acc, ind = findmax(val_accs)


f, swi, etai, di, chmi, ei = Tuple(ind)

swi = findfirst(val_accs[1, :, etai, di, chmi, ei] .== acc) # make extra extra sure findmax wasnt confused by the sweep format

println("Best acc $(acc) occured at:\nsweep=$(swi)\neta=$(etas[etai])\nd=$(ds[di])\nchi_max=$(chi_maxs[chmi])\nWith the $(encoding.name) Encoding")


