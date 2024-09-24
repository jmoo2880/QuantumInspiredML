using JLD2
include("../../LogLoss/RealRealHighDimension.jl")

dloc = "Data/ecg200/datasets/ecg200.jld2"
f = jldopen(dloc, "r")
X_train = read(f, "X_train")
X_test = read(f, "X_test")
y_train = read(f, "y_train")
y_test = read(f, "y_test")
close(f)

Rdtype = ComplexF64
verbosity = 0
test_run = false
track_cost = false
#

encoding = :stoudenmire
encode_classes_separately = false
train_classes_separately = false

#encoding = Basis("Legendre")

opts=MPSOptions(; nsweeps=5, chi_max=35,  update_iters=1, verbosity=verbosity,  loss_grad=:KLD,
    bbopt=:TSGO, track_cost=track_cost, eta=0.0025, rescale = (false, true), d=2, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, exit_early=false, 
    sigmoid_transform=false, init_rng=4567, chi_init=4)
print_opts(opts)

if test_run
    W, info, train_states, test_states, p = fitMPS(X_train, y_train,  X_test, y_test;  opts=opts, test_run=true)
    plot(p)
else
    W, info, train_states, test_states = fitMPS(X_train, y_train, X_test, y_test;  opts=opts, test_run=false)

    print_opts(opts)
    summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=true);
    sweep_summary(info)
end 