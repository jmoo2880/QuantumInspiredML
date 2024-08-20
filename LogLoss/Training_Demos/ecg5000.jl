include("../RealRealHighDimension.jl")
using MLJBase: train_test_pairs, StratifiedCV
using JLD2

dloc =  "Data/ecg5000/datasets/ecg5000.jld2"
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
encoding = :legendre_no_norm #legendre(project=false)
encode_classes_separately = true
train_classes_separately = true

#encoding = Basis("Legendre")



opts=MPSOptions(; nsweeps=10, chi_max=25,  update_iters=1, verbosity=verbosity, loss_grad=:KLD,
    bbopt=:TSGO, track_cost=track_cost, eta=0.1, rescale = (false, true), d=3, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, 
    exit_early=false, sigmoid_transform=true, init_rng=4567, chi_init=4)



print_opts(opts)
nsplits = 30

accs = Vector{Float64}(undef, nsplits+1)

if test_run
    W, info, train_states, test_states, p = fitMPS(X_train, y_train,  X_test, y_test; opts=opts, test_run=true)
    plot(p)
else
    W, info, train_states, test_states = fitMPS(X_train, y_train,X_test, y_test;  opts=opts, test_run=false)

    print_opts(opts)
    summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=true);
    sweep_summary(info)
end

