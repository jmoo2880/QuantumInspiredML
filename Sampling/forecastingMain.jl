include("forecastMetrics.jl");
include("samplingUtils.jl");
using JLD2, HDF5
using StatsPlots, StatsBase

mutable struct forecastable
    mps::MPS
    class::Int
    test_samples::Matrix{Float64}
end

function find_label_index(mps::MPS; lstr="f(x)")
    """Find the label index on the mps
    and extract the number of classes"""
    l_mps = lastindex(ITensors.data(mps))
    posvec = [l_mps, 1:(l_mps-1)...]

    for pos in posvec
        label_idx = findindex(mps[pos], lstr)
        if !isnothing(label_idx)
            num_classes = ITensors.dim(label_idx)
            return label_idx, num_classes, pos
        end
    end
    @warn "Could not find label index on mps."
    return nothing, nothing, nothing
end

function loadMPS(path::String; id::String="W")
    """Loads an MPS from a .h5 file. Returns and ITensor MPS."""
    file = path[end-2:end] != ".h5" ? path * ".h5" : path
    f = h5open("$file","r")
    mps = read(f, "$id", MPS)
    close(f)
    return mps
end

function sliceMPS(label_mps::MPS, class_label::Int)
    """General function to slice the MPS and return the 
    state corresponding to a specific class label."""
    mps = deepcopy(label_mps)
    label_idx, num_classes, pos = find_label_index(mps)
    decision_state = onehot(label_idx => (class_label + 1))
    mps[pos] *= decision_state
    normalize!(mps) 
    return mps
end

function unpack_class_states_and_samples(mps_location::String, 
    scaled_test_samples_loc::String; test_data_name="X_test_scaled",
    test_labels_name="y_test")
    """Function to unpack original labelled mps into individual
    states and the corresponding class (test) samples."""
    mps = loadMPS(mps_location)
    loaded_data = JLD2.load(scaled_test_samples_loc)
    X_test = loaded_data[test_data_name]
    y_test = loaded_data[test_labels_name]
    label_idx, num_classes, _ = find_label_index(mps)
    fcastables = Vector{forecastable}(undef, num_classes)
    for class in 0:(num_classes-1)
        #println(class)
        class_mps = sliceMPS(mps, class)
        idxs = findall(x -> x .== class, y_test)
        test_samples = X_test[idxs, :]
        fcast = forecastable(class_mps, class, test_samples)
        fcastables[(class+1)] = fcast
    end

    return fcastables

end

function forecast_single_time_series(fcastable::Vector{forecastable},
    which_class::Int, which_sample::Int, num_shots::Int, horizon::Int, 
    plot_forecast::Bool=true, get_metrics::Bool=true, plot_dist_error::Bool=false,
    print_metric_table::Bool=true)
    """
    fcastable: Vector of forecatables
    which_class: Class idx
    wich_sample: Sample idx in class
    num_shots: Number of independent trajectories
    horizon: number of sites/time pts. to forecast
    """
    fcast = fcastable[(which_class+1)]
    @assert fcast.class == which_class "Forecastable class does not match queried class."
    mps = fcast.mps
    # get the full time series
    target_time_series_full = fcast.test_samples[which_sample, :]
    # get ranges
    conditioning_sites = 1:(length(target_time_series_full) - horizon)
    forecast_sites = (conditioning_sites[end] + 1):length(mps)
    trajectories = Matrix{Float64}(undef, num_shots, length(target_time_series_full))
    @threads for i in 1:num_shots
        trajectories[i, :] = forecast_mps_sites(mps, target_time_series_full[conditioning_sites], first(forecast_sites))
    end
    # mean traj also includes known sites
    mean_trajectory = mean(trajectories, dims=1)[1,:]
    std_trajectory = std(trajectories, dims=1)[1,:]

    # compute forecast error metrics
    if get_metrics
        metric_outputs = compute_all_forecast_metrics(mean_trajectory[forecast_sites], target_time_series_full[forecast_sites], print_metric_table);
    end

    if plot_forecast
        p = plot(collect(conditioning_sites), target_time_series_full[conditioning_sites], 
            lw=2, label="Conditioning data", xlabel="time", ylabel="x")
        plot!(collect(forecast_sites), mean_trajectory[forecast_sites], 
            ribbon=std_trajectory[forecast_sites], label="MPS forecast", ls=:dot, lw=2, alpha=0.5)
        plot!(collect(forecast_sites), target_time_series_full[forecast_sites], lw=2, label="Ground truth", alpha=0.5)
        title!("Sample $which_sample, Class $which_class, $horizon Site Forecast, $num_shots Shots")
        if plot_dist_error
            xvals, deltas = get_dist_mean_difference(100, 500)
            colormap = cgrad(:diverging_bwr_20_95_c54_n256)
            norm_deltas = (deltas .- minimum(deltas)) ./ (maximum(deltas) - minimum(deltas))
            for i in 1:length(xvals)
                # Calculate the color based on the normalized delta value
                color = colormap[norm_deltas[i]]  
                # Add a horizontal band at the corresponding y-value (xvals[i])
                plot!([101, 103], [xvals[i], xvals[i]], color=color, label="", linewidth=2, alpha=0.9)
            end
        end
        display(p)
    end

    return metric_outputs

end

function forecast_class(fcastable::Vector{forecastable}, 
    which_class::Int, horizon::Int; which_metric = :SMAPE, 
    subset_size::Int=0, num_shots=1000)
    """Compute the forecasting metric of interest
    (sMAPE) on an entire class of test samples for a 
    given horizon size. Optionally specify only a subset 
    (e.g., random subset)"""
    # isolate the class of interest
    fcast = fcastable[(which_class+1)]
    test_samples = fcast.test_samples
    if subset_size > 0
        # generate random subset
        samples = StatsBase.sample(1:size(test_samples, 1), subset_size; replace=false)
    else
        samples = 1:size(test_samples, 1)
    end
    scores = Vector{Float64}(undef, length(samples))
    for (index, sample_index) in enumerate(samples)
        metrics = forecast_single_time_series(fcastable, which_class, sample_index, num_shots, horizon, 
            false, true, false, false)
        scores[index] = metrics[which_metric]
        println("[$index] Sample $sample_index: $(metrics[which_metric])")
    end

    return scores

end

function interpolate_single_time_series(fcastable::Vector{forecastable},
    which_class::Int, which_sample::Int, num_shots::Int, which_sites::Vector{Int})
    """
    fcastable: Vector of forecatables
    which_class: Class idx
    wich_sample: Sample idx in class
    num_shots: Number of independent trajectories
    which_sites: Which particular sites to interpolate
    """
    fcast = fcastable[(which_class+1)]
    @assert fcast.class == which_class "Forecastable class does not match queried class."
    mps = fcast.mps
    target_time_series_full = fcast.test_samples[which_sample, :]
    trajectories = Matrix{Float64}(undef, num_shots, length(target_time_series_full))
    for i in 1:num_shots
        println(i)
        trajectories[i, :] = interpolate_mps_sites(mps, target_time_series_full,
        which_sites)
    end
    # mean traj also includes known sites
    mean_trajectory = mean(trajectories, dims=1)[1,:]
    std_trajectory = std(trajectories, dims=1)[1,:]
    p = plot(mean_trajectory, ribbon=std_trajectory, xlabel="time", ylabel="x", 
        label="MPS Interpolated", ls=:dot, lw=2, alpha=0.8)
    plot!(target_time_series_full, label="Ground Truth", c=:orange, lw=2, alpha=0.7)
    title!("Sample $which_sample, Class $which_class, $(length(which_sites))-site Interpolation")
    display(p)
end



