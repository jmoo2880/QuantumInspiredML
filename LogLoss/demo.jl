include("RealRealHighDimension.jl")


(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits_txt("LogLoss/datasets/ECG_train.txt", 
"LogLoss/datasets/ECG_val.txt", "LogLoss/datasets/ECG_test.txt")

X_train = vcat(X_train, X_val)
y_train = vcat(y_train, y_val)


setprecision(BigFloat, 128)
Rdtype = Float64

verbosity = 0
test_run = false
track_cost = false
#
encoding = Basis("Legendre")
encode_classes_separately = false
train_classes_separately = false

#encoding = Basis("Legendre")
dtype = encoding.iscomplex ? ComplexF64 : Float64

opts=Options(; nsweeps=15, chi_max=15,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
bbopt=BBOpt("CustomGD", "TSGO"), track_cost=track_cost, eta=1e-5, rescale = (false, true), d=7, aux_basis_dim=2, encoding=encoding, 
encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately)



# saveMPS(W, "LogLoss/saved/loglossout.h5")
print_opts(opts)


if test_run
    W, info, train_states, test_states, p = fitMPS(X_train, y_train,  X_test, y_test; random_state=456, chi_init=4, opts=opts, test_run=true)
    plot(p)
else
    W, info, train_states, test_states = fitMPS(X_train, y_train,X_test, y_test; random_state=4756, chi_init=4, opts=opts, test_run=false)

    print_opts(opts)
    summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=true);
    sweep_summary(info)
end