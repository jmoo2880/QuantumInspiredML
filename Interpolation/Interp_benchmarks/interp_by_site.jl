

include("../../LogLoss/RealRealHighDimension.jl")
include("../ForecastingMainNew.jl");
using JLD2
using DataFrames
using StatProfilerHTML
using BenchmarkTools

svpath_ns = "Data/ecg200/mps_saves/legendre_no_norm10_ns_d20_chi100.jld2" #legendre_no_norm_ns_d16_chi60.jld2" #legendreNN2_ns_d8_chi35.jld2"
svpath_gd = "Data/ecg200/mps_saves/legendre_no_normGD_10_ns_d20_chi100.jld2"
dloc =  "Data/ecg200/datasets/ecg200.jld2"



f = jldopen(dloc, "r")
    X_train = read(f, "X_train")
    y_train = read(f, "y_train")
    X_test = read(f, "X_test")
    y_test = read(f, "y_test")
close(f)



###################################3



f = jldopen(svpath_gd, "r")
    mps_gd = read(f, "mps")
    opts_gd = read(f, "opts")
close(f)

opts_gd, _... = safe_options(opts_gd, nothing, nothing)


Xs = [X_train; X_test]
ys = [y_train; y_test]
train_ratio = length(y_train)/length(ys)
num_resamps = 29
splits = [
    if i == 0
        (collect(1:length(y_train)), collect(length(y_train)+1:length(ys)))   
    else
        MLJ.partition(1:length(ys), train_ratio, rng=MersenneTwister(i), stratify=ys) 
    end 
    for i in 0:num_resamps]




mode_range=(-1,1)
xvals=collect(range(mode_range...; step=1E-4))
mode_index=Index(opts_ns.d)
xvals_enc= [get_state(x, opts_ns) for x in xvals]
xvals_enc_it=[ITensor(s, mode_index) for s in xvals_enc];



max_jump=1

inds_tr, inds_te = splits[1]
X_train2, X_test2 = Xs[inds_tr, :], Xs[inds_te, :]
y_train2, y_test2 = ys[inds_tr], ys[inds_te]

nsites = size(X_train2,2)
nendpoints = 0:floor(Int, nsites/2)-1

n1s = sum(y_test2)
n0s = length(y_test2) - n1s

samples = [1:n0s; 1:n1s]
classes = [zeros(Int,n0s); ones(Int,n1s)]
fc_gd = load_forecasting_info_variables(mps_gd, X_train2, y_train2, X_test2, y_test2, opts_gd);

stats = [Vector(undef, length(samples)) for _ in nendpoints]
Threads.@threads for ii in eachindex(nendpoints)
    n = nendpoints[ii]
    interp_sites = 1+n:nsites-n |> collect
    Threads.@threads for i in eachindex(samples)
        class = classes[i]
        instance_idx = samples[i]
        stat1, p1 = any_interpolate_single_timeseries(fc_gd, class, instance_idx, interp_sites, :nearestNeighbour; invert_transform=true, NN_baseline=false, X_train=X_train2, y_train=y_train2, n_baselines=1, plot_fits=false, mode_range=mode_range, xvals=xvals, mode_index=mode_index, xvals_enc=xvals_enc, xvals_enc_it=xvals_enc_it, max_jump=max_jump);
        # push!(ps1, p1...)
        stats[ii][i] = stat1
    end

end
svpath = "Interpolation/Interp_benchmarks/MeanMode_GD_d20chi100_fitpersite.jld2"
f = jldopen(svpath, "w")
    write(f, "fc", fc_gd)
    write(f, "stats", stats)
    write(f, "nendpoints", nendpoints)
close(f)

