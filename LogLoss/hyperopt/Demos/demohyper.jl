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


eta_range = (0.001,10)
max_sweeps=10
ds = 2:5#Int.(ceil.(3:1.5:15))
chi_maxs= 15:5:35#10:5:50

gd = GridSearch(;encodings = [encoding], eta_range=eta_range, max_sweeps=max_sweeps, ds=ds, chi_maxs=chi_maxs,nfolds=10, max_eta_steps=15)

results = hyperopt(gd, Xs_train, ys_train; distribute=false, dir="LogLoss/hyperopt/Benchmarks/ECG200/")



#TODO make the below less jank
unfolded = mean(results; dims=1)

getmissingproperty(f, s::Symbol) = ismissing(f) ? -1 : getproperty(f,s)
val_accs = getmissingproperty.(unfolded, :acc)
acc, ind = findmax(val_accs)


f, swi, etai, di, chmi, ei = Tuple(ind)

swi = findfirst(val_accs[1, :, etai, di, chmi, ei] .== acc) # make extra extra sure findmax wasnt confused by the sweep format

println("Best acc $(acc) occured at:\nsweep=$(swi)\nd=$(ds[di])\nchi_max=$(chi_maxs[chmi])\nWith the $(encoding.name) Encoding")


