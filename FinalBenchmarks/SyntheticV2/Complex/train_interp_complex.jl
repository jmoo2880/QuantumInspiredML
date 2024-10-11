include("../../../LogLoss/RealRealHighDimension.jl");
using JLD2

dloc =  "Data/syntheticV2/complex/datasets/eta_0.1_m_disc_range_2_phi_cont_range_0_6.283185307179586_tau_20.0.jld2"
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
encoding = :legendre_no_norm
encode_classes_separately = false
train_classes_separately = false

d = 20
chi_max=100

opts=MPSOptions(; nsweeps=3, chi_max=chi_max,  update_iters=1, verbosity=verbosity, loss_grad=:KLD,
    bbopt=:TSGO, track_cost=track_cost, eta=0.1, rescale = (false, true), d=d, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, 
    exit_early=false, sigmoid_transform=false, init_rng=4567, chi_init=4)

if test_run
    W, info, train_states, test_states, p = fitMPS(X_train, y_train,  X_test, y_test;  opts=opts, test_run=true)
    plot(p)
else
    W, info, train_states, test_states = fitMPS(X_train, y_train,X_test, y_test; chi_init=4, opts=opts, test_run=false)

    print_opts(opts)
    #summary = get_training_summary(W, train_states.timeseries, test_states.timeseries; print_stats=true);
    #sweep_summary(info)
end

range = model_encoding(opts.encoding).range

X_train_scaled = transform_data(X_train; range=range, minmax_output=opts.minmax)
X_test_scaled = transform_data(X_test; range=range, minmax_output=opts.minmax)
svpath = "/Users/joshua/Desktop/QuantumInspiredML/Data/syntheticV2/complex/mps_saves/d1_legendre_ns_d$(d)_chi$(chi_max).jld2"
f = jldopen(svpath, "w")
    write(f, "X_train_scaled", X_train_scaled)
    write(f, "y_train", y_train)
    write(f, "X_test_scaled", X_test_scaled)
    write(f, "y_test", y_test);
    write(f, "mps", W)
    write(f, "opts", opts)
close(f)
