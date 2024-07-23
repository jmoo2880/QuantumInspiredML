include("../../LogLoss/RealRealHighDimension.jl")
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
encoding = legendre(norm=false)
encode_classes_separately = false
train_classes_separately = false

#encoding = Basis("Legendre")
dtype = encoding.iscomplex ? ComplexF64 : Float64

opts=Options(; nsweeps=20, chi_max=35,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
bbopt=BBOpt("CustomGD"), track_cost=track_cost, eta=0.0025, rescale = (false, true), d=8, aux_basis_dim=2, encoding=encoding, 
encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, exit_early=false)



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

save = false
if save
    scaler = fit_scaler(RobustSigmoidTransform, X_train)
    range = opts.encoding.range

    X_train_scaled = transform_data(scaler, X_train; range=range, minmax_output=opts.minmax)
    X_test_scaled = transform_data(scaler, X_test; range=range, minmax_output=opts.minmax)

    svpath = "Data/ecg200/mps_saves/legendreNN2_d8_chi35.jld2"
    f = jldopen(svpath, "w")
        write(f, "X_train_scaled", X_train_scaled)
        write(f, "y_train", y_train)
        write(f, "X_test_scaled", X_test_scaled)
        write(f, "y_test", y_test);
        write(f, "mps", W)
        write(f, "opts", opts)
    close(f)
end