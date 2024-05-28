include("RealRealHighDimension.jl")


(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits_txt("LogLoss/datasets/ECG_train.txt", 
"LogLoss/datasets/ECG_val.txt", "LogLoss/datasets/ECG_test.txt")

X_train = vcat(X_train, X_val)
y_train = vcat(y_train, y_val)


setprecision(BigFloat, 128)
Rdtype = Float64

verbosity = 0
test_run = true
#
encoding = SplitBasis("Hist Split Stoudenmire")

#encoding = Basis("Legendre")
dtype = encoding.iscomplex ? ComplexF64 : Float64

opts=Options(; nsweeps=5, chi_max=15,  update_iters=1, verbosity=verbosity, dtype=dtype, lg_iter=KLD_iter,
bbopt=BBOpt("CustomGD", "TSGO"), track_cost=true, eta=0.025, rescale = (false, true), d=10, aux_basis_dim=2, encoding=encoding)



# saveMPS(W, "LogLoss/saved/loglossout.h5")
print_opts(opts)


if test_run
    W, info, train_states, test_states, p = fitMPS(X_train, y_train, X_val, y_val, X_test, y_test; random_state=456, chi_init=4, opts=opts, test_run=true)
    plot(p)
else
    W, info, train_states, test_states = fitMPS(X_train, y_train, X_val, y_val, X_test, y_test; random_state=456, chi_init=4, opts=opts, test_run=false)

    print_opts(opts)
    summary = get_training_summary(W, train_states, test_states; print_stats=true);
    sweep_summary(info)
end