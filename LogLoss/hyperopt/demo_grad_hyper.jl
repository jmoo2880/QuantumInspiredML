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

eta_init = 0.5
eta_range = (0.001,1)
max_sweeps=10
d_init = 3
d_range = (2,25)
chi_max_init = 35
chi_max_range=(10,70)

results = hyperopt(HGradientDescent(), encoding, Xs, ys; 
    eta_init=eta_init, # if you want eta to be a complex number, fix the indexing for complex numbers in hyperUtils.eta_to_index()
    eta_range = eta_range,
    eta_step_perc=10., # steps in eta as a percentage
    min_eta_eta=0.0005, # minimum step in eta
    eta_eta_init=1.,
    d_init = d_init, 
    d_range = d_range,
    d_step = 1,
    chi_max_init = chi_max_init, 
    chi_max_range = chi_max_range,
    chi_step=1,
    max_sweeps=max_sweeps,
    train_ratio=0.8,
    max_grad_steps=200)


# #TODO make the below less jank
# unfolded = mean(results; dims=1)

# getmissingproperty(f, s::Symbol) = ismissing(f) ? -1 : getproperty(f,s)
# val_accs = getmissingproperty.(unfolded, :acc)
# acc, ind = findmax(val_accs)


# f, swi, etai, di, chmi, ei = Tuple(ind)

# swi = findfirst(val_accs[1, :, etai, di, chmi, ei] .== acc) # make extra extra sure findmax wasnt confused by the sweep format

# println("Best acc $(acc) occured at:\nsweep=$(swi)\neta=$(etas[etai])\nd=$(ds[di])\nchi_max=$(chi_maxs[chmi])\nWith the $(encoding.name) Encoding")


