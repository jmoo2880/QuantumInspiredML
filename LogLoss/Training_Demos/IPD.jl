include("../RealRealHighDimension.jl")
using MLJBase: train_test_pairs, StratifiedCV
using JLD2

dloc =  "Data/italypower/datasets/ItalyPowerDemandOrig.jld2"
f = jldopen(dloc, "r")
    X_train = read(f, "X_train")
    y_train = read(f, "y_train")
    X_test = read(f, "X_test")
    y_test = read(f, "y_test")
close(f)


setprecision(BigFloat, 128)
Rdtype = Float64

verbosity = 0
test_run = true
track_cost = false
#
encoding = :sahand_legendre_time_dependent #:legendre_no_norm # legendre(project=false)
encode_classes_separately = false
train_classes_separately = false

nsweeps = 3


opts=MPSOptions(; nsweeps=nsweeps, chi_max=14,  update_iters=1, verbosity=verbosity, loss_grad=:KLD,
    bbopt=:TSGO, track_cost=track_cost, eta=0.0719, rescale = (false, true), d=3, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, 
    exit_early=false, sigmoid_transform=true, init_rng=4567, chi_init=4)

print_opts(opts)
nsplits = 30

accs = Vector{Float64}(undef, nsplits)

if test_run
    W, info, train_states, test_states, p = fitMPS(X_train, y_train,  X_test, y_test;  opts=opts, test_run=test_run)
    plot(p)
else
    W, info, train_states, test_states = fitMPS(X_train, y_train,X_test, y_test; opts=opts, test_run=test_run)

    print_opts(opts)
    summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=true);
    sweep_summary(info)
end
