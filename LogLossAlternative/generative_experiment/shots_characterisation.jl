using ITensors
using Plots
using JLD2
using Base.Threads
using StatsBase
using HDF5
include("../sampling.jl");

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

mps_loaded = loadMPS("/Users/joshua/Documents/QuantumInspiredML/LogLossAlternative/generative_experiment/saved_data/1000_samples/chi35_mps.h5");
state0 = sliceMPS(mps_loaded, 0)
state1 = sliceMPS(mps_loaded, 1)
@load "/Users/joshua/Documents/QuantumInspiredML/LogLossAlternative/generative_experiment/saved_data/1000_samples/chi35_test.jld2"
c0_test_idxs = findall(x -> x.== 0, y_test);
c1_test_idxs = findall(x -> x.== 1, y_test);
c0_test_samples = X_test_scaled[c0_test_idxs, :];
c1_test_samples = X_test_scaled[c1_test_idxs, :];

function test_shots_repeated()
    num_shots = [100, 500, 1000]
    num_trials = 3
    smape_vals = Matrix{Float64}(undef, length(num_shots), num_trials)

    for (i, ns) in enumerate(num_shots)
        for t in 1:num_trials
            all_shots_forecast = Matrix{Float64}(undef, ns, 100)
            
            for j in 1:ns
                all_shots_forecast[j, :] = forecast_mps_sites(state1, c1_test_samples[1,1:50], 51)
            end
            
            mean_ts = mean(all_shots_forecast, dims=1)[1,:]
            smape = compute_mape(mean_ts[51:end], c1_test_samples[1,51:end])
            
            println("Num shots: $ns - Trial: $t - sMAPE: $smape")
            smape_vals[i, t] = smape
        end
    end
    
    return smape_vals
end

function test_shots_class_subset(subset_size=10)
    num_shots = [100, 500, 1000]
    smape_vals = Matrix{Float64}(undef, subset_size, 3) # each row is a sample, each column is num shots
    # get random subset
    random_idxs = StatsBase.sample(collect(1:size(c1_test_samples, 1)), subset_size; replace=false)
    for (si, ns) in enumerate(num_shots)
        for (idx, s_idx) in enumerate(random_idxs)
            all_shots_forecast = Matrix{Float64}(undef, ns, 100)
            for j in 1:ns
                all_shots_forecast[j, :] = forecast_mps_sites(state1, c1_test_samples[s_idx,1:50], 51)
            end
            mean_ts = mean(all_shots_forecast, dims=1)[1,:]
            smape = compute_mape(mean_ts[51:end], c1_test_samples[s_idx,51:end])
            println("Num shots: $ns - Sample: $s_idx - sMAPE: $smape")
            smape_vals[idx, si] = smape
        end
    end

    return smape_vals

end

function plot_examples_c0(sample_idx)
    all_shots_forecast = Matrix{Float64}(undef, 500, 100)
    for i in 1:500
        all_shots_forecast[i, :] = forecast_mps_sites(state0, c0_test_samples[sample_idx,1:50], 51)
    end
    mean_ts = mean(all_shots_forecast, dims=1)[1,:]
    std_ts = std(all_shots_forecast, dims=1)[1,:]
    plot(collect(1:50), c0_test_samples[sample_idx, 1:50], lw=2, label="Conditioning data")
    plot!(collect(51:100), mean_ts[51:end], ribbon=std_ts[51:end], label="MPS forecast", ls=:dot, lw=2, alpha=0.5)
    plot!(collect(51:100), c0_test_samples[sample_idx, 51:end], lw=2, label="Ground truth", alpha=0.5)
    xlabel!("Time")
    ylabel!("x")
    title!("Sample $sample_idx, Class 0, 50 Site Forecast, 500 Shots")
end
