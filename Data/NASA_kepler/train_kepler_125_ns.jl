include("../../LogLoss/RealRealHighDimension.jl")
include("KeplerDataProcessor.jl")
using JLD2


all_kepler = load_dataset("Data/NASA_kepler/datasets/KeplerLightCurves.jld2"); 
w = 100
overlap_fraction = 0.0
discard = [18, 19, 33, 39] ####WHAT#### ?? THe same as 1212
X_train, X_test, y_train, y_test = make_train_test_split_singleTS(all_kepler, 125, w, discard, overlap_fraction; train_fraction=0.85, return_corrupted_windows=true);

# dloc =  "Data/NASA_kepler/c6/H_s125.jld2"
# f = jldopen(dloc, "w")
#     write(f, "X_train", X_train)
#     write(f, "y_train", y_train)
#     write(f, "X_test", X_test)
#     write(f, "y_test", y_test)
# close(f)



setprecision(BigFloat, 128)
Rdtype = Float64

verbosity = 0
test_run = false
track_cost = false
#
encoding = :legendre_no_norm #legendre(norm=false)
encode_classes_separately = false
train_classes_separately = false

#encoding = Basis("Legendre")

opts=MPSOptions(; nsweeps=10, chi_max=35,  update_iters=1, verbosity=verbosity,  loss_grad=:KLD,
    bbopt=:TSGO, track_cost=track_cost, eta=0.0025, rescale = (false, true), d=12, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, 
    exit_early=false, sigmoid_transform=false, init_rng=4567, chi_init=4)



# saveMPS(W, "LogLoss/saved/loglossout.h5")
print_opts(opts)




if test_run
    W, info, train_states, test_states, p = fitMPS(X_train, y_train,  X_test, y_test; opts=opts, test_run=true)
    plot(p)
else
    W, info, train_states, test_states = fitMPS(X_train, y_train,X_test, y_test; opts=opts, test_run=false)

    print_opts(opts)
    summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=true);
    sweep_summary(info)
end

save = true
if save
    range = opts.encoding.range

    X_train_scaled = transform_data(X_train; range=range, minmax_output=opts.minmax)
    X_test_scaled = transform_data(X_test; range=range, minmax_output=opts.minmax)
    svpath = "Data/NASA_kepler/mps_saves/legendreNN2_ns_d12_chi35.jld2"
    f = jldopen(svpath, "w")
        write(f, "X_train_scaled", X_train_scaled)
        write(f, "y_train", y_train)
        write(f, "X_test_scaled", X_test_scaled)
        write(f, "y_test", y_test);
        write(f, "mps", W)
        write(f, "opts", opts)
    close(f)
end