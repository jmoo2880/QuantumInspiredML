include("../RealRealHighDimension.jl")
using MLJBase: train_test_pairs, StratifiedCV
using JLD2
using StatProfilerHTML
using BenchmarkTools

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
test_run = false
track_cost = false
#
encoding = :legendre_no_norm # legendre(project=false)
encode_classes_separately = false
train_classes_separately = false

nsweeps = 3

opts_init =  MPSOptions(; nsweeps=1, chi_max=2,  update_iters=1, verbosity=-10, loss_grad=:KLD,
bbopt=:TSGO, track_cost=track_cost, eta=0.0719, rescale = (false, true), d=1, aux_basis_dim=2, encoding=encoding, 
encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, 
exit_early=false, sigmoid_transform=true, init_rng=4567, chi_init=2, log_level=0)

# compile everything
print("Precompiling...")
fitMPS(X_train, y_train,X_test, y_test; opts=opts_init)
println(" Done")


opts=MPSOptions(; nsweeps=nsweeps, chi_max=50,  update_iters=1, verbosity=verbosity, loss_grad=:KLD,
    bbopt=:TSGO, track_cost=track_cost, eta=0.0719, rescale = (false, true), d=10, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, 
    exit_early=false, sigmoid_transform=true, init_rng=4567, chi_init=4, log_level=0)

print_opts(opts)
nsplits = 30

W, info, train_states, test_states = MPS(), Dict(), EncodedTimeseriesSet(), EncodedTimeseriesSet()
@profilehtml begin
    outs = fitMPS(X_train, y_train, X_test, y_test; opts=opts);
    global W = outs[1];
    global info = outs[2];
    # global train_states = outs[3];
    global test_states  = outs[4];
end;
print()

# print_opts(opts)
# summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=true);
if opts.log_level > 2

    sweep_summary(info)
end

test_loss, test_acc, conf = MSE_loss_acc_conf(W, test_states.timeseries)

@show test_loss, test_acc
@show conf
