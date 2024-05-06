using ITensors
using Plots
using JLD2
using Base.Threads
using StatsBase
using HDF5
include("/Users/joshua/Documents/QuantumInspiredML/LogLossAlternative/sampling.jl");

function loadMPS(path::String; id::String="W")
    """Loads an MPS from a .h5 file. Returns and ITensor MPS."""
    file = path[end-2:end] != ".h5" ? path * ".h5" : path
    f = h5open("$file","r")
    mps = read(f, "$id", MPS)
    close(f)
    return mps
end

function sliceMPS(W::MPS, class_label::Int)
    """General function to slice the MPS and return the state corresponding to a specific class label."""
    ψ = deepcopy(W)
    decision_idx = findindex(ψ[end], "f(x)")
    decision_state = onehot(decision_idx => (class_label + 1))
    ψ[end] *= decision_state
    normalize!(ψ) 

    return ψ
end;

mps_loaded = loadMPS("/Users/joshua/Documents/QuantumInspiredML/LogLossAlternative/generative_experiment/chinatown/ctown_chi15_mps.h5");
state0 = sliceMPS(mps_loaded, 0)
state1 = sliceMPS(mps_loaded, 1)
@load "/Users/joshua/Documents/QuantumInspiredML/LogLossAlternative/generative_experiment/chinatown/ctown_test_scaled.jld2"
c0_test_idxs = findall(x -> x.== 0, y_test);
c1_test_idxs = findall(x -> x.== 1, y_test);
c0_test_samples = X_test_scaled[c0_test_idxs, :];
c1_test_samples = X_test_scaled[c1_test_idxs, :];

function plot_forecasting_example(class_idx::Int, sample_idx::Int, num_shots::Int; num_tpts_forecast=5)
    """Plot a single forecasting example"""
    info = Dict(0 => Dict("mps_state" => state0, "test_samples" => c0_test_samples),
                1 => Dict("mps_state" => state1, "test_samples" => c1_test_samples))
    time_series_length = size(c0_test_samples, 2) # infer total length
    start_site = time_series_length - num_tpts_forecast
    all_shots_forecast = Matrix{Float64}(undef, num_shots, time_series_length)
    mps_state = info[class_idx]
    # thread the individual trajectories
    @threads for i in 1:num_shots
        all_shots_forecast[i, :] = forecast_mps_sites(mps_state["mps_state"], mps_state["test_samples"][sample_idx,1:start_site], start_site+1)
    end
    mean_ts = mean(all_shots_forecast, dims=1)[1,:]
    std_ts = std(all_shots_forecast, dims=1)[1,:]
    p = plot(collect(1:start_site), mps_state["test_samples"][sample_idx, 1:start_site], lw=2, label="Conditioning data")
    plot!(collect((start_site+1):time_series_length), mean_ts[(start_site+1):end], ribbon=std_ts[(start_site+1):end], label="MPS forecast", ls=:dot, lw=2, alpha=0.5)
    plot!(collect((start_site+1):time_series_length), mps_state["test_samples"][sample_idx, (start_site+1):end], lw=2, label="Ground truth", alpha=0.5)
    xlabel!("Time")
    ylabel!("x")
    title!("Sample $sample_idx, Class $class_idx, $num_tpts_forecast Site Forecast, $num_shots Shots")
    println("Sample $sample_idx sMAPE: $(compute_mape(mean_ts[(start_site+1):end], mps_state["test_samples"][sample_idx,(start_site+1):end], symmetric=true))")
    display(p)
end


function compute_smape_all_c1()
    # compute sMAPE for all test samples in a class using a fixed number of shots
    # and fixed forecasting horizon
    smapes_all = []
    num_shots = 500
    time_series_length = size(c1_test_samples, 2)
    for idx in eachindex(1:size(c1_test_samples,1))
        all_shots_forecast = Matrix{Float64}(undef, num_shots, time_series_length)
        @threads for j in 1:num_shots
            all_shots_forecast[j, :] = forecast_mps_sites(state1, c1_test_samples[idx,1:12], 13)
        end
        mean_ts = mean(all_shots_forecast, dims=1)[1,:]
        smape = compute_mape(mean_ts[13:end], c1_test_samples[idx,13:end]; symmetric=true)
        println("Sample: $idx - sMAPE: $smape")
        push!(smapes_all, smape)
    end
    return smapes_all
end

function compute_smape_all_c0()
    # compute sMAPE for all test samples in a class using a fixed number of shots
    # and fixed forecasting horizon
    smapes_all = []
    num_shots = 500
    time_series_length = size(c0_test_samples, 2)
    for idx in eachindex(1:size(c0_test_samples,1))
        all_shots_forecast = Matrix{Float64}(undef, num_shots, time_series_length)
        @threads for j in 1:num_shots
            all_shots_forecast[j, :] = forecast_mps_sites(state1, c0_test_samples[idx,1:12], 13)
        end
        mean_ts = mean(all_shots_forecast, dims=1)[1,:]
        smape = compute_mape(mean_ts[13:end], c0_test_samples[idx,13:end]; symmetric=true)
        println("Sample: $idx - sMAPE: $smape")
        push!(smapes_all, smape)
    end
    return smapes_all
end

function plot_interp_acausal(class_idx::Int, sample_idx::Int, num_shots::Int, interp_idxs::Vector{Int})
    info = Dict(0 => Dict("mps_state" => state0, "test_samples" => c0_test_samples),
                1 => Dict("mps_state" => state1, "test_samples" => c1_test_samples))
    time_series_length = size(c0_test_samples, 2) # infer total length
    all_shots_interp = Matrix{Float64}(undef, num_shots, time_series_length)

    mps_state = info[class_idx]
    # thread the individual trajectories
    @threads for i in 1:num_shots
        all_shots_interp[i, :] = interpolate_acausal(mps_state["mps_state"], mps_state["test_samples"][sample_idx,:], interp_idxs);
    end
    mean_ts = mean(all_shots_interp, dims=1)[1,:]
    std_ts = std(all_shots_interp, dims=1)[1,:]
    p = plot(mean_ts, ribbon=std_ts, label="MPS Interpolated", lw=2, ls=:dot)
    plot!(mps_state["test_samples"][sample_idx,:], label="Ground truth", lw=2)
    xlabel!("time")
    ylabel!("x")
    title!("Class $class_idx, Sample $sample_idx, $num_shots-Shot MPS Interpolation")
    display(p)
end