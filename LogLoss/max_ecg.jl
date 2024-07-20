include("RealRealHighDimension.jl")
using MLJBase: train_test_pairs, StratifiedCV
using JLD2

dloc =  "Interpolation/paper/ecg200/datasets/ecg200.jld2"
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
encoding = legendre(project=false)
encode_classes_separately = false
train_classes_separately = false

#encoding = Basis("Legendre")
dtype = encoding.iscomplex ? ComplexF64 : Float64

opts=Options(; nsweeps=5, chi_max=20,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.01, rescale = (false, true), d=3, aux_basis_dim=2, encoding=encoding, 
encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, exit_early=false)



print_opts(opts)
nsplits = 6

accs = Vector{Float64}(undef, nsplits+1)

if test_run
    W, info, train_states, test_states, p = fitMPS(X_train, y_train,  X_test, y_test; random_state=4567, chi_init=4, opts=opts, test_run=true)
    plot(p)
else
    W, info, train_states, test_states = fitMPS(X_train, y_train,X_test, y_test; random_state=4567, chi_init=4, opts=opts, test_run=false)

    print_opts(opts)
    summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=true);
    sweep_summary(info)
end

accs[1] = info["test_acc"][end]

foldrng = MersenneTwister(1)
Xs = [X_train; X_test]
ys = [y_train; y_test]

train_inds = Vector{Vector{Int}}(undef, nsplits)
test_inds = Vector{Vector{Int}}(undef, nsplits)

train_ratio=0.5
nvirt_folds = 2
i = 1
while i <= nsplits - 1
    scv = StratifiedCV(;nfolds=nvirt_folds, rng=foldrng)
    fold_inds_temp = train_test_pairs(scv, eachindex(ys), ys)
    train_inds[i], test_inds[i] = fold_inds_temp[1]
    train_inds[i+1], test_inds[i+1] = fold_inds_temp[2]
    global i += 2
end




for i in 1:nsplits
    tr_inds = train_inds[i]
    te_inds = test_inds[i]
    f_Xs_train = Xs[tr_inds, :]
    f_Xs_test = Xs[te_inds, :]

    f_ys_train = ys[tr_inds]
    f_ys_test = ys[te_inds]
    _, info, _,_ = fitMPS(f_Xs_train, f_ys_train, f_Xs_test, f_ys_test; random_state=4567, chi_init=4, opts=opts, test_run=false)
    accs[i+1] = info["test_acc"][end]
end

