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
test_run = false
track_cost = false
#
encoding = legendre(project=false)
encode_classes_separately = false
train_classes_separately = false

#encoding = Basis("Legendre")
dtype = encoding.iscomplex ? ComplexF64 : Float64

opts=Options(; nsweeps=10, chi_max=14,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=0.0719, rescale = (false, true), d=3, aux_basis_dim=2, encoding=encoding, 
encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, exit_early=false, sigmoid_transform=true)



print_opts(opts)
nsplits = 30

accs = Vector{Float64}(undef, nsplits)

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

train_ratio= length(y_train) / length(y_test)
nvirt_folds = ceil(Int, 1 / (train_ratio))

scv = StratifiedCV(;nfolds=nvirt_folds, rng=foldrng)
fold_inds_temp = train_test_pairs(scv, eachindex(ys), ys)
while length(fold_inds_temp) < nsplits
    global fold_inds_temp = vcat(fold_inds_temp, train_test_pairs(scv, eachindex(ys), ys))
end
i = 1
while i <= nsplits - 1
    train_inds[i+1], test_inds[i+1] = fold_inds_temp[i]
    global i += 1
end




for i in 2:nsplits
    tr_inds = train_inds[i]
    te_inds = test_inds[i]
    f_Xs_train = Xs[tr_inds, :]
    f_Xs_test = Xs[te_inds, :]

    f_ys_train = ys[tr_inds]
    f_ys_test = ys[te_inds]
    _, info, _,_ = fitMPS(f_Xs_train, f_ys_train, f_Xs_test, f_ys_test; random_state=4567, chi_init=4, opts=opts, test_run=false)
    accs[i] = info["test_acc"][end]
end

println("Acc: $(mean(accs))") # Acc: 0.9627758763961071
println("Std: $(std(accs))") # Std: 0.016763937070864134