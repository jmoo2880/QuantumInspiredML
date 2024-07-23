include("../../LogLoss/RealRealHighDimension.jl")
include("../ForecastingMainNew.jl");
using JLD2
using DataFrames
using StatsBase

function midpoint_benchmark(ys_test, fc, interp_sites, Xs_train, ys_train)
    n1s = sum(ys_test)
    n0s = length(ys_test) - n1s


    samples = [shuffle(1:n0s); shuffle(1:n1s)]
    classes = [zeros(Int,n0s); ones(Int,n1s)]

    ps = []
    stats = []
    MSE_stats = []
    for (i,s) in enumerate(samples)
        class = classes[i]
        instance_idx = s
        stat, MSE_stat, p1 = any_interpolate_single_time_series(fc, class, instance_idx, interp_sites, :directMode; MSE_baseline=true, X_train_scaled=Xs_train, y_train=ys_train);
        push!(ps, p1)
        push!(stats, stat)
        push!(MSE_stats, MSE_stat)
    end

    return stats, MSE_stats, ps
end

function make_interp_sites(n_mid::Integer, Xs::Matrix{Float64})
    npoints = size(Xs, 2)
    start_data = ceil(Int,(npoints+1) / 2 - n_mid/2)

    return [1:(start_data-1); (start_data + n_mid):npoints]
end



svpath = "Data/ecg200/mps_saves/legendreNN2_d8_chi35.jld2"

f = jldopen(svpath, "r")
    X_train_scaled = read(f, "X_train_scaled")
    y_train = read(f, "y_train")
    X_test_scaled = read(f, "X_test_scaled")
    y_test = read(f, "y_test");
    mps = read(f, "mps")
    opts = read(f, "opts")
close(f)

setprecision(BigFloat, 128)
Rdtype = Float64

verbosity = 0
test_run = false
track_cost = false

encoding = legendre(norm=false)
encode_classes_separately = false
train_classes_separately = encode_classes_separately

#encoding = Basis("Legendre")
dtype = encoding.iscomplex ? ComplexF64 : Float64
opts=Options(; nsweeps=20, chi_max=35, update_iters=1, verbosity=-1, dtype=dtype, loss_grad=loss_grad_KLD,
    bbopt=BBOpt("CustomGD"), track_cost=track_cost, eta=0.0025, rescale = (false, true), d=8, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, exit_early=false)

fc = load_forecasting_info_variables(mps, X_train_scaled, y_train, X_test_scaled, y_test, opts);


n_midpoints = [1:2:50; 55:5:95]

stats = Vector{Any}(undef, length(n_midpoints))
MSE_stats = Vector{Any}(undef, length(n_midpoints))
ps = Vector{Any}(undef, length(n_midpoints))

for i in eachindex(n_midpoints)
    n_mid = n_midpoints[i]
    interp_sites = make_interp_sites(n_mid, X_train_scaled)
    stats[i], MSE_stats[i], ps[i] = midpoint_benchmark(y_test, fc, interp_sites, X_train_scaled, y_train)
    
end

idstr = "ecg_"*prod(map(x-> string(x) * "_", n_midpoints))[1:end-1] *".jld2"

path = "Interpolation/Interp_benchmarks/midpoint_benchmarks/" * idstr

jldopen(path, "w") do f
    f["n_midpoints"] = n_midpoints
    f["stats"] = stats
    f["MSE_stats"] = MSE_stats
    f["ps"] = ps
end


