include("../RealRealHighDimension.jl")
using MLJBase: train_test_pairs, StratifiedCV
using JLD2

dloc =  "Data/ecg200/datasets/ecg200.jld2"
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
encoding = :sahand_legendre
encode_classes_separately = false
train_classes_separately = false

#encoding = Basis("Legendre")
nsweeps=20


opts=MPSOptions(; nsweeps=nsweeps, chi_max=40,  update_iters=1, verbosity=verbosity, loss_grad=loss_grad=:KLD,
    bbopt=:TSGO, track_cost=track_cost, eta=0.01, rescale = (false, true), d=8, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, exit_early=false, sigmoid_transform=true)



print_opts(opts)
nsplits = 30


if test_run
    W, info, train_states, test_states, p = fitMPS(X_train, y_train,  X_test, y_test;  opts=opts, test_run=true)
    plot(p)
else
    W, info, train_states, test_states = fitMPS( X_train, y_train,X_test, y_test;  opts=opts, test_run=false)

    print_opts(opts)
    summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=true);
    sweep_summary(info)
end



test_loss = info["test_KL_div"][1:end-1]
train_loss = info["train_KL_div"][1:end-1]
train_acc = info["train_acc"][1:end-1]
test_acc = info["test_acc"][1:end-1]
sweeps = 0:nsweeps

plot(sweeps, test_loss; label="Test Loss", ylabel="KL Div.", xlabel="sweep", title="ECG200 Loss vs Sweep")
plot!(sweeps, train_loss; label="Train Loss")

plot(sweeps, test_acc; label="Test Acc", ylabel="Accuracy", xlabel="sweep", title="ECG200 Acc vs Sweep")
plot!(sweeps, train_acc; label="Train Acc")