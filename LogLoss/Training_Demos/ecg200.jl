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
encoding = legendre(project=false)
encode_classes_separately = false
train_classes_separately = false

#encoding = Basis("Legendre")
bbopt = BBOpt("CustomGD", "TSGO")
dtype = encoding.iscomplex ? ComplexF64 : Float64

opts=Options(; nsweeps=5, chi_max=29,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
bbopt=bbopt, track_cost=track_cost, eta=0.1, rescale = (false, true), d=3, aux_basis_dim=2, encoding=encoding, 
encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, exit_early=false, sigmoid_transform=true)



print_opts(opts)
nsplits = 30

accs = Vector{Float64}(undef, nsplits+1)

range = opts.encoding.range
if opts.sigmoid_transform
    # rescale with a sigmoid prior to minmaxing
    scaler = fit_scaler(RobustSigmoidTransform, X_train);
    X_train = permutedims(transform_data(scaler, X_train; range=range, minmax_output=opts.minmax))
    X_test = permutedims(transform_data(scaler, X_test; range=range, minmax_output=opts.minmax))

else
    X_train = permutedims(transform_data(X_train; range=range, minmax_output=opts.minmax))
    X_test = permutedims(transform_data(X_test; range=range, minmax_output=opts.minmax))
end

if test_run
    W, info, train_states, test_states, p = fitMPS(X_train, y_train,  X_test, y_test; random_state=4567, chi_init=4, opts=opts, test_run=true)
    plot(p)
else
    W, info, train_states, test_states = fitMPS(DataIsRescaled{true}(), X_train, y_train,X_test, y_test; random_state=4567, chi_init=4, opts=opts, test_run=false)

    print_opts(opts)
    summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=true);
    sweep_summary(info)
end

