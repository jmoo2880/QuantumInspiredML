include("./forecastMetrics.jl");
include("./samplingUtilsNew.jl");
include("./interpolationUtils.jl");
include("/Users/joshua/QuantumMay/QuantumInspiredML/LogLoss/RealRealHighDimension.jl")

using JLD2
using StatsPlots, StatsBase, Plots.PlotMeasures
using ProgressMeter

mutable struct forecastable
    mps::MPS
    class::Int
    test_samples::Matrix{Float64}
    opts::Options
    enc_args::Vector{Vector{Any}}
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

# probably redundant if enc args are provided externally from training
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

function unpack_forecasting_info(data_loc::String; mps_id::String="mps",
    train_data_name::String="X_train_scaled", test_data_name::String="X_test_scaled",
    opts_name::String="opts")
    """mps_opts_loc contains both trained mps and opts.
    scaled data contains X_train_scaled y_train X_test_scaled y_test."""
    f = jldopen(data_loc, "r")
    mps = read(f, "$mps_id")
    X_train_scaled = read(f, "$train_data_name")
    y_train = read(f, "y_train")
    X_test_scaled = read(f, "$test_data_name")
    y_test = read(f, "y_test")
    opts = read(f, "$opts_name")
    close(f)
    println("Dataset has $(size(X_test_scaled, 1)) samples.")
    label_idx, num_classes, _ = find_label_index(mps)
    fcastables = Vector{forecastable}(undef, num_classes);
    enc_args = get_enc_args_from_opts(opts, X_train_scaled, y_train)
    for class in 0:(num_classes-1)
        class_mps = slice_mps(mps, class)
        println("Class $class mps has local dimension: $(maxdim(class_mps[1])) and $(length(class_mps)) sites.")
        idxs = findall(x -> x .== class, y_test);
        test_samples = X_test_scaled[idxs, :];
        fcast = forecastable(class_mps, class, test_samples, opts, enc_args);
        fcastables[(class+1)] = fcast;
    end

    return fcastables
    
end

function forward_interpolate_single_time_series_sampling(fcastable::Vector{forecastable},
    which_class::Int, which_sample::Int, horizon::Int; num_shots::Int=2000,
    plot_forecast::Bool=true, get_metrics::Bool=true, print_metric_table::Bool=true)
    """Forecast single time series. Produces one trajectory."""

    fcast = fcastable[(which_class+1)]
    mps = fcast.mps
    d_mps = maxdim(mps[1])
    enc_name = fcast.opts.encoding.name
    target_time_series_full = fcast.test_samples[which_sample, :]
    # get ranges
    conditioning_sites = 1:(length(target_time_series_full) - horizon)
    forecast_sites = (conditioning_sites[end] + 1):length(mps)
    trajectories = Matrix{Float64}(undef, num_shots, length(target_time_series_full))
    if fcast.opts.encoding.istimedependent
        @threads for i in 1:num_shots
            trajectories[i, :] = forward_interpolate_trajectory(mps, target_time_series_full[conditioning_sites], 
                first(forecast_sites), fcast.opts, fcast.enc_args)
        end
    else
        @threads for i in 1:num_shots
            trajectories[i, :] = forward_interpolate_trajectory(mps, target_time_series_full[conditioning_sites], 
                first(forecast_sites), fcast.opts)
        end
    end
    # extract summary statistics 
    mean_trajectory = mean(trajectories, dims=1)[1,:]
    std_trajectory = std(trajectories, dims=1)[1,:]

    # compute forecast error metrics
    if get_metrics
        metric_outputs = compute_all_forecast_metrics(mean_trajectory[forecast_sites], 
            target_time_series_full[forecast_sites], print_metric_table)
    end

    # plot forecast
    if plot_forecast
        p = plot(collect(conditioning_sites), target_time_series_full[conditioning_sites],
            lw = 2, label="Conditioning data", xlabel="time", ylabel="x", legend=:outertopright, 
            size=(1000, 500), bottom_margin=5mm, left_margin=5mm)
        plot!(collect(forecast_sites), mean_trajectory[forecast_sites], 
            ribbon=std_trajectory[forecast_sites], label="MPS forecast", ls=:dot, lw=2, alpha=0.5)
        plot!(collect(forecast_sites), target_time_series_full[forecast_sites], lw=2, label="Ground truth", alpha=0.5)
        title!("Sample $which_sample, Class $which_class, $horizon Site Forecast,\nd = $d_mps MPS, $enc_name encoding,\n $num_shots-Shot Mean")
        display(p)
    end

    return metric_outputs

end

function forward_interpolate_single_time_series_directMean(fcastable::Vector{forecastable}, 
    which_class::Int, which_sample::Int, horizon::Int; plot_forecast::Bool=true, 
    get_metrics::Bool=true, print_metric_table::Bool=true)
    """Forward interpolate (forecast) using the direct mean."""

    fcast = fcastable[(which_class+1)]
    mps = fcast.mps
    chi_mps = maxlinkdim(mps)
    d_mps = maxdim(mps[1])
    enc_name = fcast.opts.encoding.name
    target_time_series_full = fcast.test_samples[which_sample, :]
    conditioning_sites = 1:(length(target_time_series_full) - horizon)
    forecast_sites = (conditioning_sites[end] + 1):length(mps)
    # handle both time dependent and time independent encodings
    if fcast.opts.encoding.istimedependent
        mean_ts, std_ts = forward_interpolate_directMean(mps, target_time_series_full[conditioning_sites], 
            first(forecast_sites), fcast.opts, fcast.enc_args)
    else
        mean_ts, std_ts = forward_interpolate_directMean(mps, target_time_series_full[conditioning_sites], 
            first(forecast_sites), fcast.opts)
    end

    if get_metrics
        metric_outputs = compute_all_forecast_metrics(mean_ts[forecast_sites], target_time_series_full[forecast_sites], print_metric_table)
    end

    if plot_forecast
        p = plot(collect(conditioning_sites), target_time_series_full[conditioning_sites], 
            lw=2, label="Conditioning data", xlabel="time", ylabel="x", legend=:outertopright, 
            size=(1000, 500), bottom_margin=5mm, left_margin=5mm)
        plot!(collect(forecast_sites), mean_ts[forecast_sites], ribbon=std_ts[forecast_sites],
            label="MPS forecast", ls=:dot, lw=2, alpha=0.5)
        plot!(collect(forecast_sites), target_time_series_full[forecast_sites], lw=2, label="Ground truth", alpha=0.5)
        title!("Sample $which_sample, Class $which_class, $horizon Site Forecast,\nd = $d_mps, χ = $chi_mps,\n$enc_name encoding, Expectation")
        display(p)
    end

    return metric_outputs
end

function forward_interpolate_single_time_series_directMode(fcastable::Vector{forecastable}, 
    which_class::Int, which_sample::Int, horizon::Int; plot_forecast::Bool=true, 
    get_metrics::Bool=true, print_metric_table::Bool=true)
    """Forward interpolate (forecast) using the direct mode"""

    fcast = fcastable[(which_class+1)]
    mps = fcast.mps
    chi_mps = maxlinkdim(mps)
    d_mps = maxdim(mps[1])
    enc_name = fcast.opts.encoding.name
    target_time_series_full = fcast.test_samples[which_sample, :]
    conditioning_sites = 1:(length(target_time_series_full) - horizon)
    forecast_sites = (conditioning_sites[end] + 1):length(mps)
    # handle both time dependent and time independent encodings
    if fcast.opts.encoding.istimedependent
        mode_ts = forward_interpolate_directMode(mps, target_time_series_full[conditioning_sites], 
            first(forecast_sites), fcast.opts, fcast.enc_args)
    else
        mode_ts = forward_interpolate_directMode(mps, target_time_series_full[conditioning_sites], 
            first(forecast_sites), fcast.opts)
    end

    if get_metrics
        metric_outputs = compute_all_forecast_metrics(mode_ts[forecast_sites], 
            target_time_series_full[forecast_sites], print_metric_table);
    end

    if plot_forecast
        p = plot(collect(conditioning_sites), target_time_series_full[conditioning_sites], 
            lw=2, label="Conditioning data", xlabel="time", ylabel="x", legend=:outertopright, 
            size=(1000, 500), bottom_margin=5mm, left_margin=5mm)
        plot!(collect(forecast_sites), mode_ts[forecast_sites],
            label="MPS forecast", ls=:dot, lw=2, alpha=0.5, c=:magenta)
        plot!(collect(forecast_sites), target_time_series_full[forecast_sites], lw=2, label="Ground truth", alpha=0.5)
        title!("Sample $which_sample, Class $which_class, $horizon Site Forecast,\nd = $d_mps, χ = $chi_mps,\n$enc_name encoding, Mode")
        display(p)
    end

    return metric_outputs

end

function forward_interpolate_single_time_series(fcastable::Vector{forecastable}, 
    which_class::Int, which_sample::Int, horizon::Int, method::Symbol=:directMean; 
    plot_forecast::Bool=true, get_metrics::Bool=true, print_metric_table::Bool=true)

    if method == :directMean 
        metric_outputs = forward_interpolate_single_time_series_directMean(fcastable, which_class, which_sample,
            horizon; plot_forecast=plot_forecast, get_metrics=get_metrics, print_metric_table=print_metric_table)
    elseif method == :directMode
        metric_outputs = forward_interpolate_single_time_series_directMode(fcastable, which_class, which_sample,
        horizon; plot_forecast=plot_forecast, get_metrics=get_metrics, print_metric_table=print_metric_table)
    elseif method == :inverseTform
        metric_outputs = forward_interpolate_single_time_series_sampling(fcastable, which_class, which_sample, horizon;
            num_shots=2000, plot_forecast=plot_forecast, get_metrics=get_metrics, print_metric_table=print_metric_table)
    else
        error("Invalid method. Choose either :directMean (Mean/Std), :directMode, or :inverseTform (inv. transform sampling).")
    end

    return metric_outputs
end

function any_interpolate_single_time_series_sampling(fcastable::Vector{forecastable},
    which_class::Int, which_sample::Int, which_sites::Vector{Int}; num_shots::Int=1000)
    # TO DO -> ADD IN PERFORMANCE METRICS FOR INTERPOLATION 
    fcast = fcastable[(which_class+1)]
    mps = fcast.mps
    chi_mps = maxlinkdim(mps)
    d_mps = maxdim(mps[1])
    enc_name = fcast.opts.encoding.name
    target_time_series_full = fcast.test_samples[which_sample, :]
    trajectories = Matrix{Float64}(undef, num_shots, length(target_time_series_full))
    if fcast.opts.encoding.istimedependent
        @threads for i in 1:num_shots
            trajectories[i, :] = any_interpolate_trajectory(mps, fcast.opts, fcast.enc_args, target_time_series_full, which_sites)
        end
    else
        # time independent encoding
        @threads for i in 1:num_shots
            trajectories[i, :] = any_interpolate_trajectory(mps, fcast.opts, target_time_series_full, which_sites)
        end
    end
    # get summary statistics
    mean_trajectory = mean(trajectories, dims=1)[1,:]
    std_trajectory = std(trajectories, dims=1)[1,:]
    p = plot(mean_trajectory, ribbon=std_trajectory, xlabel="time", ylabel="x", 
        label="MPS Interpolated", ls=:dot, lw=2, alpha=0.8, legend=:outertopright,
        size=(1000, 500), bottom_margin=5mm, left_margin=5mm)
    plot!(target_time_series_full, label="Ground Truth", c=:orange, lw=2, alpha=0.7)
    title!("Sample $which_sample, Class $which_class, $(length(which_sites))-site Interpolation, 
        d = $d_mps, χ = $chi_mps, $enc_name encoding, 
        $num_shots-shot mean")
    display(p)
end

function any_interpolate_single_time_series_directMode(fcastable::Vector{forecastable},
    which_class::Int, which_sample::Int, which_sites::Vector{Int})

    fcast = fcastable[(which_class+1)]
    mps = fcast.mps
    chi_mps = maxlinkdim(mps)
    d_mps = maxdim(mps[1])
    enc_name = fcast.opts.encoding.name
    target_time_series_full = fcast.test_samples[which_sample, :]

    if fcast.opts.encoding.istimedependent
        mode_ts = any_interpolate_directMode(mps, fcast.opts, fcast.enc_args, target_time_series_full, which_sites)
    else
        mode_ts = any_interpolate_directMode(mps, fcast.opts, target_time_series_full, which_sites)
    end
    p = plot(mode_ts, xlabel="time", ylabel="x", 
        label="MPS Interpolated", ls=:dot, lw=2, alpha=0.8, legend=:outertopright,
        size=(1000, 500), bottom_margin=5mm, left_margin=5mm)
    plot!(target_time_series_full, label="Ground Truth", c=:orange, lw=2, alpha=0.7)
    title!("Sample $which_sample, Class $which_class, $(length(which_sites))-site Interpolation, 
        d = $d_mps, χ = $chi_mps, $enc_name encoding, 
        Mode")
    display(p)
end

function any_interpolate_single_time_series_directMean(fcastable::Vector{forecastable},
    which_class::Int, which_sample::Int, which_sites::Vector{Int})

    fcast = fcastable[(which_class+1)]
    mps = fcast.mps
    chi_mps = maxlinkdim(mps)
    d_mps = maxdim(mps[1])
    enc_name = fcast.opts.encoding.name
    target_time_series_full = fcast.test_samples[which_sample, :]

    if fcast.opts.encoding.istimedependent
        mean_ts, std_ts = any_interpolate_directMean(mps, fcast.opts, fcast.enc_args, target_time_series_full, which_sites)
    else
        mean_ts, std_ts = any_interpolate_directMean(mps, fcast.opts, target_time_series_full, which_sites)
    end
    p = plot(mean_ts, ribbon=std_ts, xlabel="time", ylabel="x", 
        label="MPS Interpolated", ls=:dot, lw=2, alpha=0.8, legend=:outertopright,
        size=(1000, 500), bottom_margin=5mm, left_margin=5mm)
    plot!(target_time_series_full, label="Ground Truth", c=:orange, lw=2, alpha=0.7)
    title!("Sample $which_sample, Class $which_class, $(length(which_sites))-site Interpolation, 
        d = $d_mps, χ = $chi_mps, $enc_name encoding, 
        Expectation")
    display(p)
end

function any_interpolate_single_time_series(fcastable::Vector{forecastable},
    which_class::Int, which_sample::Int, which_sites::Vector{Int}, method::Symbol=:directMean)

    if method == :directMean
        any_interpolate_single_time_series_directMean(fcastable, which_class, which_sample, which_sites)
    elseif method == :directMode
        any_interpolate_single_time_series_directMode(fcastable, which_class, which_sample, which_sites)
    elseif method == :inverseTform
        any_interpolate_single_time_series_sampling(fcastable, which_class, which_sample, which_sites; num_shots=1000)
    else
        error("Invalid method. Choose either :directMean (Expect/Var), :directMode, or :inverseTform (inv. transform sampling).")
    end
end
