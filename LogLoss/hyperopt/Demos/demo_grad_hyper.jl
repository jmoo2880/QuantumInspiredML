include("../hyperopt.jl")
using JLD2

dloc =  "Interpolation/paper/ecg200/datasets/ecg200.jld2"
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
encoding = legendre(project=true)
encode_classes_separately = false
train_classes_separately = false

eta_init = 0.01
eta_range = (0.001,1)
max_sweeps=10
d_init = 2
d_range = (2,10)
chi_max_init = 25
chi_max_range=(10,60)

results = hyperopt(HGradientDescent(), encoding, Xs_train, ys_train; 
    eta_init=eta_init, # if you want eta to be a complex number, fix the indexing for complex numbers in hyperUtils.eta_to_index()
    eta_range = eta_range,
    deta_perc=0.1, # steps in eta as a percentage
    min_eta_eta=0.0005, # minimum step in eta
    eta_eta_init=0.001,
    d_init = d_init, 
    d_range = d_range,
    d_step = 1,
    chi_max_init = chi_max_init, 
    chi_max_range = chi_max_range,
    chi_step=1,
    max_sweeps=max_sweeps,
    train_ratio=0.9, 
    max_grad_steps=200)


# #TODO make the below less jank
# unfolded = mean(results; dims=1)

# getmissingproperty(f, s::Symbol) = ismissing(f) ? -1 : getproperty(f,s)
# val_accs = getmissingproperty.(unfolded, :acc)
# acc, ind = findmax(val_accs)


# f, swi, etai, di, chmi, ei = Tuple(ind)

# swi = findfirst(val_accs[1, :, etai, di, chmi, ei] .== acc) # make extra extra sure findmax wasnt confused by the sweep format

# println("Best acc $(acc) occured at:\nsweep=$(swi)\neta=$(etas[etai])\nd=$(ds[di])\nchi_max=$(chi_maxs[chmi])\nWith the $(encoding.name) Encoding")


