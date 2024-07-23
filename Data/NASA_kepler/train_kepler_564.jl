include("../../LogLoss/RealRealHighDimension.jl")
include("KeplerDataProcessor.jl")
using JLD2

#TODO figure out what josh actually did
# all_kepler = load_dataset("Interpolation/paper/NASA_kepler/datasets/KeplerLightCurves.jld2"); 
# w = 100
# overlap_fraction = 0.0
# discard = [18, 19, 33, 39] ####WHAT#### ?? THe same as 1212
# X_train, X_test, y_train, y_test = make_train_test_split_singleTS(all_kepler, 125, w, discard, overlap_fraction; train_fraction=0.85, return_corrupted_windows=true);

dloc =  "Data/NASA_kepler/c0/sample564.jld2"
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

d = 12
chi_max=34
opts=Options(; nsweeps=10, chi_max=chi_max,  update_iters=1, verbosity=verbosity, dtype=dtype, loss_grad=loss_grad_KLD,
bbopt=BBOpt("CustomGD"), track_cost=track_cost, eta=0.0025, rescale = (false, true), d=d, aux_basis_dim=2, encoding=encoding, 
encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, exit_early=false)



# saveMPS(W, "LogLoss/saved/loglossout.h5")
print_opts(opts)




if test_run
    W, info, train_states, test_states, p = fitMPS(X_train, y_train,  X_test, y_test; random_state=456, chi_init=4, opts=opts, test_run=true)
    plot(p)
else
    W, info, train_states, test_states = fitMPS(X_train, y_train,X_test, y_test; random_state=456, chi_init=4, opts=opts, test_run=false)

    print_opts(opts)
    summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=true);
    sweep_summary(info)
end

save = true
if save
    scaler = fit_scaler(RobustSigmoidTransform, X_train)
    range = opts.encoding.range

    X_train_scaled = transform_data(scaler, X_train; range=range, minmax_output=opts.minmax)
    X_test_scaled = transform_data(scaler, X_test; range=range, minmax_output=opts.minmax)
    svpath = "Data/NASA_kepler/mps_saves/legendreNN_s564_d$(d)_chi$(chi_max).jld2"
    f = jldopen(svpath, "w")
        write(f, "X_train_scaled", X_train_scaled)
        write(f, "y_train", y_train)
        write(f, "X_test_scaled", X_test_scaled)
        write(f, "y_test", y_test);
        write(f, "mps", W)
        write(f, "opts", opts)
    close(f)
end