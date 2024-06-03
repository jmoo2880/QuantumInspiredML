include("./forecastMetrics.jl");
include("./samplingUtils.jl");
using JLD2
using StatsPlots, StatsBase, Plots.PlotMeasures
using ProgressMeter

mutable struct forecastable
    mps::MPS
    class::Int
    test_samples::Matrix{Float64}
    opts::Options
end

function find_label_index(mps::MPS; label_name::String="f(x)")
    """Find the label index on the mps. If the label does not exist,
    assume mps is already spliced i.e., corresponds to a single class."""
    l_mps = lastindex(ITensors.data(mps))
    posvec = [l_mps, 1:(l_mps-1)...]
    # loop through each site and check for label
    for pos in posvec
        label_idx = findindex(mps[pos], label_name)
        if !isnothing(label_idx)
            num_classes = ITensors.dim(label_idx)
            return label_idx, num_classes, pos
        end
    end

    @warn "Could not find label index on mps. Assuming single class mps."

    return nothing, 1, nothing 
end

function get_enc_args_from_opts(opts::Options, X_train_scaled::Matrix, 
    y::Vector{Int})
    """Re-encode the scaled training data using the time dependent
    encoding to get the encoding args."""
    enc_args = []
    if opts.encoding.istimedependent
        enc_args = opts.encoding.init(X_train_scaled, y; opts=opts)
    end

    return enc_args

end

function slice_mps(label_mps::MPS, class_label::Int)
    """Slice an MPS along the specified class label index
    to return a single class state."""
    mps = deepcopy(label_mps)
    label_idx, num_classes, pos = find_label_index(mps)
    if !isnothing(label_idx)
        decision_state = onehot(label_idx => (class_label + 1))
        mps[pos] *= decision_state
        normalize(mps)
    else
        @warn "MPS cannot be sliced, returning original MPS."
    end

    return mps

end






# function loadMPS(path::String; id="mps")
#     """Loads an MPS from a .jld2 file. Returns an ITensor MPS."""
#     file = path[end-4:end] != ".jld2" ? path * ".jld2" : path
#     f = jldopen("$file", "r")
#     mps = read(f, "$id")
#     close(f)
#     return mps
# end

# function sliceMPS(label_mps::MPS, class_label::Int)
#     """General function to slice the MPS and return the 
#     state corresponding to a specific class label."""
#     mps = deepcopy(label_mps)
#     label_idx, num_classes, pos = find_label_index(mps)
#     decision_state = onehot(label_idx => (class_label + 1))
#     mps[pos] *= decision_state
#     normalize!(mps) 
#     return mps
# end

# function unpack_single_class_mps_and_samples(mps_location::String, 
#     scaled_test_samples_loc::String; test_data_name="X_test_scaled",
#     test_labels_name="y_test")
#     # made a seperate function to handle an MPS without a label index
#     mps = loadMPS(mps_location)

#     label_idx, num_classes, _ = find_label_index(mps)
#     if !isnothing(label_idx)
#         error("MPS has label index. There may be more than 1 class.")
#     end

#     loaded_data = JLD2.load(scaled_test_samples_loc)
#     X_test = loaded_data[test_data_name]
#     y_test = loaded_data[test_labels_name]
#     # keep the vector
#     fcastable = Vector{forecastable}(undef, 1)
#     fcast = forecastable(mps, 1, X_test)
#     fcastable[1] = fcast

#     return fcastable
# end

# function unpack_class_states_and_samples(mps_location::String, 
#     scaled_test_samples_loc::String; mps_id::String="mps", 
#     test_data_name="X_test_scaled", test_labels_name="y_test")
#     """Function to unpack original labelled mps into individual
#     states and the corresponding class (test) samples."""
#     mps = loadMPS(mps_location; id=mps_id)
#     loaded_data = JLD2.load(scaled_test_samples_loc)
#     X_test = loaded_data[test_data_name]
#     if all(X_test .>= 0) & all(X_test .<= 1)
#         println("Data in range [0, 1]")
#     elseif all(X_test .>= -1) & all(X_test .<= 1)
#         println("Data in range [-1, 1]")
#     else
#         error("Data has not been rescaled to either [0, 1] or [-1, 1].")
#     end
#     y_test = loaded_data[test_labels_name]

#     # print additional diagonstics
#     println("Dataset has $(size(X_test, 1)) samples.")

#     label_idx, num_classes, _ = find_label_index(mps)
#     fcastables = Vector{forecastable}(undef, num_classes);
#     # if only a single class, skip the slicing
#     for class in 0:(num_classes-1)
#         #println(class)
#         class_mps = sliceMPS(mps, class);
#         println("Class $class mps has local dimension: $(maxdim(class_mps[1])) and $(length(class_mps)) sites.")
#         idxs = findall(x -> x .== class, y_test);
#         test_samples = X_test[idxs, :];
#         fcast = forecastable(class_mps, class, test_samples);
#         fcastables[(class+1)] = fcast;
#     end

#     return fcastables

# end

# function forecast_single_time_series(fcastable::Vector{forecastable},
#     which_class::Int, which_sample::Int, num_shots::Int, horizon::Int, basis::Basis,
#     plot_forecast::Bool=true, get_metrics::Bool=true, plot_dist_error::Bool=false,
#     print_metric_table::Bool=true; use_threaded::Bool=true)
#     """
#     fcastable: Vector of forecatables
#     which_class: Class idx
#     wich_sample: Sample idx in class
#     num_shots: Number of independent trajectories
#     horizon: number of sites/time pts. to forecast
#     uses_threaded: whether (true) or not (false) to use multithreading
#     """
#     fcast = fcastable[(which_class+1)]
#     #@assert fcast.class == which_class "Forecastable class does not match queried class."
#     mps = fcast.mps
#     # extract the local dimension
#     d_mps = maxdim(mps[1])
#     # get the full time series
#     target_time_series_full = fcast.test_samples[which_sample, :]
#     # get ranges
#     conditioning_sites = 1:(length(target_time_series_full) - horizon)
#     forecast_sites = (conditioning_sites[end] + 1):length(mps)
#     trajectories = Matrix{Float64}(undef, num_shots, length(target_time_series_full))
#     if use_threaded
#         p = Progress(num_shots, desc="Trajectories computed...")
#         @threads for i in 1:num_shots
#             trajectories[i, :] = forecast_mps_sites(mps, target_time_series_full[conditioning_sites], first(forecast_sites), basis, d_mps)
#             next!(p)
#         end
#         finish!(p)
#     else
#         for i in 1:num_shots
#             trajectories[i, :] = forecast_mps_sites(mps, target_time_series_full[conditioning_sites], first(forecast_sites), basis, d_mps)
#         end
#     end
#     # mean traj also includes known sites
#     mean_trajectory = mean(trajectories, dims=1)[1,:]
#     std_trajectory = std(trajectories, dims=1)[1,:]

#     # compute forecast error metrics
#     if get_metrics
#         metric_outputs = compute_all_forecast_metrics(mean_trajectory[forecast_sites], target_time_series_full[forecast_sites], print_metric_table);
#     end

#     if plot_forecast
#         p = plot(collect(conditioning_sites), target_time_series_full[conditioning_sites], 
#             lw=2, label="Conditioning data", xlabel="time", ylabel="x")
#         plot!(collect(forecast_sites), mean_trajectory[forecast_sites], 
#             ribbon=std_trajectory[forecast_sites], label="MPS forecast", ls=:dot, lw=2, alpha=0.5)
#         plot!(collect(forecast_sites), target_time_series_full[forecast_sites], lw=2, label="Ground truth", alpha=0.5)
#         title!("Sample $which_sample, Class $which_class, $horizon Site Forecast,\nd = $d_mps MPS, $(basis.name) encoding, $num_shots Shots, Mean/Std")
#         if plot_dist_error
#             xvals, deltas = get_dist_mean_difference(100, basis, d_mps, 500)
#             colormap = cgrad(:diverging_bwr_20_95_c54_n256)
#             norm_deltas = (deltas .- minimum(deltas)) ./ (maximum(deltas) - minimum(deltas))
#             for i in 1:length(xvals)
#                 # Calculate the color based on the normalized delta value
#                 color = colormap[norm_deltas[i]]  
#                 # Add a horizontal band at the corresponding y-value (xvals[i])
#                 plot!([(length(mean_trajectory)+1), (length(mean_trajectory)+2)], [xvals[i], xvals[i]], color=color, label="", linewidth=2, alpha=0.9)
#             end
#         end
#         display(p)
#     end

#     return metric_outputs

# end

# function forecast_single_time_series_mode(fcastable::Vector{forecastable},
#     which_class::Int, which_sample::Int, num_shots::Int, horizon::Int, basis::Basis,
#     plot_forecast::Bool=true, get_metrics::Bool=true,
#     print_metric_table::Bool=true)
#     """Forecast using the histogram mode instead of the sample mean.
#     Bootstrap to get the mean/variance of modes."""
#     fcast = fcastable[(which_class+1)]
#     mps = fcast.mps
#     d_mps = maxdim(mps[1])
#     # get the full time series
#     target_time_series_full = fcast.test_samples[which_sample, :]
#     # get ranges
#     conditioning_sites = 1:(length(target_time_series_full) - horizon)
#     forecast_sites = (conditioning_sites[end] + 1):length(mps)
#     trajectories = Matrix{Float64}(undef, num_shots, length(target_time_series_full))
    
#     p = Progress(num_shots, desc="Trajectories computed...")
#     @threads for i in 1:num_shots
#         trajectories[i, :] = forecast_mps_sites(mps, target_time_series_full[conditioning_sites], first(forecast_sites), basis, d_mps)
#         next!(p)
#     end
#     finish!(p)
    
#     # each site/time pt. has its own distribution, get the mode
#     mode_trajectory = Matrix{Float64}(undef, 3, length(target_time_series_full))

#     for tp in 1:length(target_time_series_full)
#         # only take mode of the actual forecasting sites
#         if tp ∉ conditioning_sites
#             # boostrap the mode and confidence interval
#             mode_est, mode_std_err, ci95 = bootstrap_mode_estimator(get_kde_mode, trajectories[:, tp], 1000)
           
#             mode_trajectory[1, tp] = mode_est
#             mode_trajectory[2, tp] = mode_std_err
#             mode_trajectory[3, tp] = ci95
#         end
#     end

#     # compute forecast error metrics
#     if get_metrics
#         metric_outputs = compute_all_forecast_metrics(mode_trajectory[1, forecast_sites], 
#             target_time_series_full[forecast_sites], print_metric_table);
#     end

#     if plot_forecast
#         p = plot(collect(conditioning_sites), target_time_series_full[conditioning_sites], 
#             lw=2, label="Conditioning data", xlabel="time", ylabel="x")
#         plot!(collect(forecast_sites), mode_trajectory[1, forecast_sites], ribbon=mode_trajectory[3, forecast_sites],
#             label="MPS forecast", ls=:dot, lw=2, alpha=0.5)
#         plot!(collect(forecast_sites), target_time_series_full[forecast_sites], lw=2, label="Ground truth", alpha=0.5)
#         title!("Sample $which_sample, Class $which_class, $horizon Site Forecast,\nd = $d_mps MPS, $(basis.name) encoding, $num_shots Shots, Mode/CI")
#         display(p)
#     end

#     return metric_outputs

# end

# function forecast_single_time_series_mode_analytic(fcastable::Vector{forecastable},
#     which_class::Int, which_sample::Int, horizon::Int, basis::Basis, 
#     plot_forecast::Bool=true, get_metrics::Bool=true, print_metric_table::Bool=true)

#     fcast = fcastable[(which_class+1)]
#     mps = fcast.mps
#     d_mps = maxdim(mps[1])
#     chi_mps = maxlinkdim(mps)
#     target_time_series_full = fcast.test_samples[which_sample, :]
#     conditioning_sites = 1:(length(target_time_series_full) - horizon)
#     forecast_sites = (conditioning_sites[end] + 1):length(mps)

#     mode_ts = forecast_mps_sites_analytic_mode(mps, target_time_series_full[conditioning_sites], first(forecast_sites), basis, d_mps)

#     if get_metrics
#         metric_outputs = compute_all_forecast_metrics(mode_ts[forecast_sites], 
#             target_time_series_full[forecast_sites], print_metric_table);
#     end

#     if plot_forecast
#         p = plot(collect(conditioning_sites), target_time_series_full[conditioning_sites], 
#             lw=2, label="Conditioning data", xlabel="time", ylabel="x")
#         plot!(collect(forecast_sites), mode_ts[forecast_sites],
#             label="MPS forecast", ls=:dot, lw=2, alpha=0.5)
#         plot!(collect(forecast_sites), target_time_series_full[forecast_sites], lw=2, label="Ground truth", alpha=0.5)
#         title!("Sample $which_sample, Class $which_class, $horizon Site Forecast,\nd = $d_mps, χ = $chi_mps, $(basis.name) encoding, Mode")
#         display(p)
#     end

#     return metric_outputs

# end

# function forecast_single_time_series_mean_analytic(fcastable::Vector{forecastable},
#     which_class::Int, which_sample::Int, horizon::Int, basis::Basis,
#     plot_forecast::Bool=true, get_metrics::Bool=true, print_metric_table::Bool=true)

#     fcast = fcastable[(which_class+1)]
#     mps = fcast.mps
#     d_mps = maxdim(mps[1])
#     chi_mps = maxlinkdim(mps)
#     target_time_series_full = fcast.test_samples[which_sample, :]
#     conditioning_sites = 1:(length(target_time_series_full) - horizon)
#     forecast_sites = (conditioning_sites[end] + 1):length(mps)

#     mean_ts, std_ts = forecast_mps_sites_analytic_mean(mps, target_time_series_full[conditioning_sites], first(forecast_sites), basis, d_mps)

#     if get_metrics
#         metric_outputs = compute_all_forecast_metrics(mean_ts[forecast_sites], 
#             target_time_series_full[forecast_sites], print_metric_table);
#     end

#     if plot_forecast
#         p = plot(collect(conditioning_sites), target_time_series_full[conditioning_sites], 
#             lw=2, label="Conditioning data", xlabel="time", ylabel="x")
#         plot!(collect(forecast_sites), mean_ts[forecast_sites], ribbon=std_ts[forecast_sites],
#             label="MPS forecast", ls=:dot, lw=2, alpha=0.5)
#         plot!(collect(forecast_sites), target_time_series_full[forecast_sites], lw=2, label="Ground truth", alpha=0.5)
#         title!("Sample $which_sample, Class $which_class, $horizon Site Forecast,\nd = $d_mps, χ = $chi_mps, $(basis.name) encoding, Mean")
#         display(p)
#     end


#     return metric_outputs

# end

# function forecast_single_time_series_mean_mode(fcastable::Vector{forecastable},
#     which_class::Int, which_sample::Int, num_shots::Int, horizon::Int, basis::Basis,
#     plot_forecast::Bool=true, print_metric_table::Bool=true; mode_resamples=1000)

#     fcast = fcastable[(which_class+1)]
#     mps = fcast.mps
#     d_mps = maxdim(mps[1])
#     target_time_series_full = fcast.test_samples[which_sample, :]
#     conditioning_sites = 1:(length(target_time_series_full) - horizon)
#     forecast_sites = (conditioning_sites[end] + 1):length(mps)

#     trajectories = Matrix{Float64}(undef, num_shots, length(target_time_series_full))

#     p = Progress(num_shots, desc="Trajectories computed...")
#     @threads for i in 1:num_shots
#         trajectories[i, :] = forecast_mps_sites(mps, target_time_series_full[conditioning_sites], first(forecast_sites), basis, d_mps)
#         next!(p)
#     end
#     finish!(p)

#     # mean traj also includes known sites
#     mean_trajectory = mean(trajectories, dims=1)[1,:]
#     std_trajectory = std(trajectories, dims=1)[1,:]

#     # boostrap mode estimate and 95CI
#     mode_trajectory = Matrix{Float64}(undef, 3, length(target_time_series_full))

#     for tp in 1:length(target_time_series_full)
#         if tp ∉ conditioning_sites
#             mode_est, mode_std_err, ci95 = bootstrap_mode_estimator(get_kde_mode, trajectories[:, tp], mode_resamples)
#             mode_trajectory[:, tp] = [mode_est, mode_std_err, ci95]
#         end
#     end

#     # compute forecast error metrics for both estimates
#     metrics_mean = compute_all_forecast_metrics(mean_trajectory[forecast_sites], 
#         target_time_series_full[forecast_sites], print_metric_table)

#     metrics_mode = compute_all_forecast_metrics(mode_trajectory[1, forecast_sites], 
#         target_time_series_full[forecast_sites], print_metric_table)

#     results = Dict(:mean => metrics_mean, :mode => metrics_mode)

#     if plot_forecast
#         p1 = plot(collect(conditioning_sites), target_time_series_full[conditioning_sites], bottom_margin=5mm,
#             left_margin=5mm, lw=2, label="Conditioning data", xlabel="time", ylabel="x")
#         p1 = plot!(collect(forecast_sites), mean_trajectory[forecast_sites], 
#             ribbon=std_trajectory[forecast_sites], label="MPS forecast", ls=:dot, lw=2, alpha=0.5)
#         p1 = plot!(collect(forecast_sites), target_time_series_full[forecast_sites], lw=2, label="Ground truth", alpha=0.5)
#         p1 = title!("Sample $which_sample, Class $which_class,\nd = $d_mps MPS, $(basis.name) encoding, $num_shots Shots, Mean/Std")

#         p2 = plot(collect(conditioning_sites), target_time_series_full[conditioning_sites], 
#             lw=2, label="Conditioning data", xlabel="time", ylabel="x")
#         p2 = plot!(collect(forecast_sites), mode_trajectory[1, forecast_sites],
#             label="MPS forecast", ls=:dot, lw=2, alpha=0.5)
#         p2 = plot!(collect(forecast_sites), target_time_series_full[forecast_sites], lw=2, label="Ground truth", alpha=0.5)
#         p2 = title!("Sample $which_sample, Class $which_class\nd = $d_mps MPS, $(basis.name) encoding, $num_shots Shots, Mode/Std")
#         p = plot(p1, p2, size=(1200, 500))
#         display(p)
#     end

#     return results

# end


# function forecast_class_mean_mode(fcastable::Vector{forecastable},
#     which_class::Int, horizon::Int, basis::Basis; which_metric = :MSE, 
#     subset_size::Int = 0, num_shots=2000)

#     fcast = fcastable[(which_class+1)]
#     test_samples = fcast.test_samples
#     if subset_size > 0
#         # generate random subset
#         samples = StatsBase.sample(1:size(test_samples, 1), subset_size; replace=false)
#     else
#         samples = 1:size(test_samples, 1)
#     end
#     scores = []
#     for (index, sample_index) in enumerate(samples)
#         results = forecast_single_time_series_mean_mode(fcastable, which_class, sample_index, 
#             num_shots, horizon, basis, false, false)
#         push!(scores, (results[:mean], results[:mode]))

#         println("[$index] Sample $sample_index Mean MSE: $(results[:mean][which_metric])")
#         println("[$index] Sample $sample_index Mode MSE: $(results[:mode][which_metric])")
#     end

#     return scores

# end

# function forecast_class_mode_analytic(fcastable::Vector{forecastable},
#     which_class::Int, horizon::Int, basis::Basis; which_metric = :MSE, 
#     subset_size::Int = 0)

#     fcast = fcastable[(which_class+1)]
#     test_samples = fcast.test_samples

#     if subset_size > 0
#         # generate random subset
#         samples = StatsBase.sample(1:size(test_samples, 1), subset_size; replace=false)
#     else
#         samples = 1:size(test_samples, 1)
#     end
#     scores = []

#     for (index, sample_index) in enumerate(samples)
#         results_mode = forecast_single_time_series_mode_analytic(fcastable, which_class, sample_index, 
#             horizon, basis, false, true, false)
#         push!(scores, results_mode)
#         println("[$index] Sample $sample_index Mode MSE: $(results_mode[which_metric])")
#     end

#     return scores

# end

# function forecast_class_mean_mode_analytic(fcastable::Vector{forecastable},
#     which_class::Int, horizon::Int, basis::Basis; which_metric = :MSE, 
#     subset_size::Int = 0)

#     fcast = fcastable[(which_class+1)]
#     test_samples = fcast.test_samples

#     if subset_size > 0
#         # generate random subset
#         samples = StatsBase.sample(1:size(test_samples, 1), subset_size; replace=false)
#     else
#         samples = 1:size(test_samples, 1)
#     end
#     scores = []

#     for (index, sample_index) in enumerate(samples)
#         results_mean = forecast_single_time_series_mean_analytic(fcastable, which_class, sample_index, horizon, 
#             basis, false, true, false)
#         results_mode = forecast_single_time_series_mode_analytic(fcastable, which_class, sample_index, 
#             horizon, basis, false, true, false)
#         push!(scores, (results_mean, results_mode))

#         println("[$index] Sample $sample_index Mean MSE: $(results_mean[which_metric])")
#         println("[$index] Sample $sample_index Mode MSE: $(results_mode[which_metric])")
#     end

#     return scores

# end


# function forecast_class(fcastable::Vector{forecastable}, 
#     which_class::Int, horizon::Int, basis::Basis; which_metric = :MSE, 
#     subset_size::Int=0, num_shots=1000)
#     """Compute the forecasting metric of interest
#     on an entire class of test samples for a 
#     given horizon size. Optionally specify only a subset 
#     (e.g., random subset)"""
#     # isolate the class of interest
#     fcast = fcastable[(which_class+1)]
#     test_samples = fcast.test_samples
#     if subset_size > 0
#         # generate random subset
#         samples = StatsBase.sample(1:size(test_samples, 1), subset_size; replace=false)
#     else
#         samples = 1:size(test_samples, 1)
#     end
#     scores = Vector{Float64}(undef, length(samples))
#     for (index, sample_index) in enumerate(samples)
#         metrics = forecast_single_time_series_mode(fcastable, which_class, sample_index, num_shots, horizon, basis, false, true, false)
#         scores[index] = metrics[which_metric]
#         println("[$index] Sample $sample_index: $(metrics[which_metric])")
#     end

#     return scores

# end

# function interpolate_single_time_series_mode(fcastable::Vector{forecastable}, 
#     which_class::Int, which_sample::Int, which_sites::Vector{Int},
#     basis::Basis)

#     fcast = fcastable[(which_class+1)]
#     mps = fcast.mps
#     d_mps = maxdim(mps[1])
#     chi_mps = maxlinkdim(mps)
#     target_time_series_full = fcast.test_samples[which_sample, :]

#     mode_ts = interpolate_mps_sites_mode(mps, basis, target_time_series_full, which_sites)
#     p = plot(mode_ts, xlabel="time", ylabel="x", 
#         label="MPS Interpolated", ls=:dot, lw=2, alpha=0.8)
#     plot!(target_time_series_full, label="Ground Truth", c=:orange, lw=2, alpha=0.7)
#     title!("Sample $which_sample, Class $which_class, $(length(which_sites))-site Interpolation\nd=$d_mps, χ=$chi_mps, $(basis.name) encoding, Mode")
#     display(p)

# end



# function interpolate_single_time_series(fcastable::Vector{forecastable},
#     which_class::Int, which_sample::Int, num_shots::Int, which_sites::Vector{Int},
#     basis::Basis;
#     use_multi_threaded::Bool=true)
#     """
#     fcastable: Vector of forecatables
#     which_class: Class idx
#     wich_sample: Sample idx in class
#     num_shots: Number of independent trajectories
#     which_sites: Which particular sites to interpolate
#     uses_threaded: whether (true) or not (false) to use multithreading
#     """
#     fcast = fcastable[(which_class+1)]
#     #@assert fcast.class == which_class "Forecastable class does not match queried class."
#     mps = fcast.mps
#     target_time_series_full = fcast.test_samples[which_sample, :]
#     trajectories = Matrix{Float64}(undef, num_shots, length(target_time_series_full))
#     # use conditional multi-threeading
#     if use_multi_threaded
#         p = Progress(num_shots, desc="Interpolation Progress...")
#         @threads for i in 1:num_shots
#             #println(i)
#             trajectories[i, :] = interpolate_mps_sites(mps, basis, target_time_series_full, which_sites)
#             next!(p)
#         end
#         finish!(p)
#     else
#         for i in 1:num_shots
#         #println(i)
#         trajectories[i, :] = interpolate_mps_sites(mps, basis, target_time_series_full, which_sites)
#         end
#     end
#     # mean traj also includes known sites
#     mean_trajectory = mean(trajectories, dims=1)[1,:]
#     std_trajectory = std(trajectories, dims=1)[1,:]
#     p = plot(mean_trajectory, ribbon=std_trajectory, xlabel="time", ylabel="x", 
#         label="MPS Interpolated", ls=:dot, lw=2, alpha=0.8)
#     plot!(target_time_series_full, label="Ground Truth", c=:orange, lw=2, alpha=0.7)
#     title!("Sample $which_sample, Class $which_class, $(length(which_sites))-site Interpolation, \n $num_shots shots")
#     display(p)
# end
