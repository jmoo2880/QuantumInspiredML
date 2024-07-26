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
encoding = legendre(project=false, norm=false)
encode_classes_separately = false
train_classes_separately = false


etas = [0.001, 0.004, 0.007, 0.01, 0.04, 0.07, 0.1, 0.3, 0.5]
max_sweeps=5
ds = [2;Int.(ceil.(3:1.5:15))]
chi_maxs=15:5:50

gd = GridSearch3D(;encodings = [encoding], etas=etas, max_sweeps=max_sweeps, ds=ds, chi_maxs=chi_maxs,nfolds=5)

results = hyperopt(gd, Xs_train, ys_train; distribute=false, dir="LogLoss/hyperopt/Benchmarks/IPD/", sigmoid_transform=true, exit_early=false)



#TODO make the below less jank
unfolded = mean(results; dims=1)

getmissingproperty(f, s::Symbol) = ismissing(f) ? -1 : getproperty(f,s)
val_accs = getmissingproperty.(unfolded, :acc)
acc, ind = findmax(val_accs)


f, swi, etai, di, chmi, ei = Tuple(ind)

swi = findfirst(val_accs[1, :, etai, di, chmi, ei] .== acc) # make extra extra sure findmax wasnt confused by the sweep format

println("Best acc $(acc) occured at:\nsweep=$(swi)\nd=$(ds[di])\nchi_max=$(chi_maxs[chmi])\nWith the $(encoding.name) Encoding")




#TODO make the below less jank
unfolded = mean(results; dims=1)

getmissingproperty(f, s::Symbol) = ismissing(f) ? -1 : getproperty(f,s)
val_accs = getmissingproperty.(unfolded, :acc)
acc, ind = findmax(val_accs)


f, swi, etai, di, chmi, ei = Tuple(ind)

swi = findfirst(val_accs[1, :, etai, di, chmi, ei] .== acc) # make extra extra sure findmax wasnt confused by the sweep format

println("Best acc $(acc) occured at:\nsweep=$(swi)\neta=$(etas[etai])\nd=$(ds[di])\nchi_max=$(chi_maxs[chmi])\nWith the $(encoding.name) Encoding")

