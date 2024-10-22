include("../LogLoss/RealRealHighDimension.jl");
include("./forecastMetrics.jl");
include("./samplingUtilsNew.jl");
include("./interpolationUtils.jl");

using JLD2
using StatsPlots, StatsBase, Plots.PlotMeasures

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
function get_enc_args_from_opts(
        opts::Options, 
        X_train::Matrix, 
        y::Vector{Int}
    )
    """Rescale and then Re-encode the scaled training data using the time dependent
    encoding to get the encoding args."""

    # transform the data
    # perform the sigmoid scaling
    if opts.sigmoid_transform
        sig_trans = Normalization.fit(RobustSigmoid, X_train)
        X_train_scaled = normalize(permutedims(X_train), sig_trans)
    else
        X_train_scaled = permutedims(X_train)
    end

    if opts.minmax
        minmax = Normalization.fit(MinMax, X_train_scaled)
        normalize!(X_train_scaled, minmax)
    end



    # map to the domain of the encoding
    a,b = opts.encoding.range
    @. X_train_scaled = (b-a) *X_train_scaled + a
    

    if isnothing(opts.encoding.init)
        enc_args = []
    else
        println("Re-encoding the training data to get the encoding arguments...")
        enc_args = opts.encoding.init(X_train_scaled, y; opts=opts)
    end

    return enc_args

end

function slice_mps(label_mps::MPS, class_label::Int)
    """Slice an MPS along the specified class label index
    to return a single class state."""
    mps = deepcopy(label_mps)
    label_idx, num_classes, pos = find_label_index(mps);
    if !isnothing(label_idx)
        decision_state = onehot(label_idx => (class_label + 1));
        mps[pos] *= decision_state;
        normalize(mps);
    else
        @warn "MPS cannot be sliced, returning original MPS."
    end

    return mps

end

function load_forecasting_info_variables(
        mps::MPS, 
        X_train::Matrix{Float64}, 
        y_train::Vector{Int}, 
        X_test::Matrix{Float64}, 
        y_test::Vector{Int},
        opts::AbstractMPSOptions; 
        verbosity::Integer=1
    )
    """No saved JLD File, just pass in variables that would have been loaded 
    from the jld2 file. Need to pass in reconstructed opts struct until the 
    issue is resolved."""

    if opts isa MPSOptions
        _, _, opts = Options(opts)
    end
    

    # extract info
    verbosity > 0 && println("+"^60 * "\n"* " "^25 * "Summary:\n")
    verbosity > 0 && println(" - Dataset has $(size(X_train, 1)) training samples and $(size(X_test, 1)) testing samples.")
    label_idx, num_classes, _ = find_label_index(mps)
    verbosity > 0 && println(" - $num_classes class(es) was detected. Slicing MPS into individual states...")
    fcastables = Vector{forecastable}(undef, num_classes);
    if opts.encoding.istimedependent
        verbosity > 0 && println(" - Time dependent encoding - $(opts.encoding.name) - detected, obtaining encoding args...")
        verbosity > 0 && println(" - d = $(opts.d), chi_max = $(opts.chi_max), aux_basis_dim = $(opts.aux_basis_dim)")
    else
        verbosity > 0 && println(" - Time independent encoding - $(opts.encoding.name) - detected.")
        verbosity > 0 && println(" - d = $(opts.d), chi_max = $(opts.chi_max)")
    end
    enc_args = get_enc_args_from_opts(opts, X_train, y_train)
    for class in 0:(num_classes-1)
        class_mps = slice_mps(mps, class);
        idxs = findall(x -> x .== class, y_test);
        test_samples = X_test[idxs, :];
        fcast = forecastable(class_mps, class, test_samples, opts, enc_args);
        fcastables[(class+1)] = fcast;
    end
    verbosity > 0 && println("\n Created $num_classes forecastable struct(s) containing class-wise mps and test samples.")

    return fcastables

end

function load_forecasting_info(
        data_loc::String;
         mps_id::String="mps",
        train_data_name::String="X_train", 
        test_data_name::String="X_test",
        opts_name::String="opts"
    )

    # yes, there are a lot of checks...
    f = jldopen(data_loc, "r")
    @assert length(f) >= 6 "Expected at least 6 data objects, only found $(length(f))."
    mps = read(f, "$mps_id")
    #@assert typeof(mps) == ITensors.MPS "Expected mps to be of type MPS."
    X_train = read(f, "$train_data_name");
    @assert typeof(X_train) == Matrix{Float64} "Expected training data to be a matrix."
    y_train = read(f, "y_train");
    @assert typeof(y_train) == Vector{Int64} "Expected training labels to be a vector."
    X_test = read(f, "$test_data_name");
    @assert typeof(X_test) == Matrix{Float64} "Expected testing data to be a matrix."
    y_test = read(f, "y_test");
    @assert typeof(y_test) == Vector{Int64} "Expected testing labels to be a vector."
    opts = read(f, "$opts_name");
    @assert typeof(opts) == Options "Expected opts to be of type Options"
    @assert size(X_train, 2) == size(X_test, 2) "Mismatch between training and testing data number of samples."
    # add checks for data range.

    close(f)

    # extract info
    println("+"^60 * "\n"* " "^25 * "Summary:\n")
    println(" - Dataset has $(size(X_train, 1)) training samples and $(size(X_test, 1)) testing samples.")
    label_idx, num_classes, _ = find_label_index(mps)
    println(" - $num_classes class(es) was detected. Slicing MPS into individual states...")
    fcastables = Vector{forecastable}(undef, num_classes);
    if opts.encoding.istimedependent
        println(" - Time dependent encoding - $(opts.encoding.name) - detected, obtaining encoding args...")
        println(" - d = $(opts.d), chi_max = $(opts.chi_max), aux_basis_dim = $(opts.aux_basis_dim)")
    else
        println(" - Time independent encoding - $(opts.encoding.name) - detected.")
        println(" - d = $(opts.d), chi_max = $(opts.chi_max)")
    end
    enc_args = get_enc_args_from_opts(opts, X_train, y_train)
    for class in 0:(num_classes-1)
        class_mps = slice_mps(mps, class);
        idxs = findall(x -> x .== class, y_test);
        test_samples = X_test[idxs, :];
        fcast = forecastable(class_mps, class, test_samples, opts, enc_args);
        fcastables[(class+1)] = fcast;
    end
    println("\n Created $num_classes forecastable struct(s) containing class-wise mps and test samples.")

    return fcastables
    
end

function forward_interpolate_single_timeseries_sampling(
        fcastable::Vector{forecastable},
        which_class::Int, 
        which_sample::Int, 
        horizon::Int; 
        num_shots::Int=2000,
        plot_forecast::Bool=true,
        get_metrics::Bool=true, 
        print_metric_table::Bool=true,
    )
    """Forecast single time series. Produces one trajectory."""

    fcast = fcastable[(which_class+1)]
    mps = fcast.mps
    d_mps = maxdim(mps[1])
    enc_name = fcast.opts.encoding.name
    target_timeseries_full = fcast.test_samples[which_sample, :]
    # get ranges
    conditioning_sites = 1:(length(target_timeseries_full) - horizon)
    forecast_sites = (conditioning_sites[end] + 1):length(mps)
    trajectories = Matrix{Float64}(undef, num_shots, length(target_timeseries_full))
    if fcast.opts.encoding.istimedependent
        @threads for i in 1:num_shots
            trajectories[i, :] = forward_interpolate_trajectory_time_dependent(mps, target_timeseries_full[conditioning_sites], first(forecast_sites), fcast.opts, fcast.enc_args)
        end
    else
        @threads for i in 1:num_shots
            trajectories[i, :] = forward_interpolate_trajectory(mps, target_timeseries_full[conditioning_sites], first(forecast_sites), fcast.opts, fcast.enc_args)
        end
    end
    # extract summary statistics 
    mean_trajectory = mean(trajectories, dims=1)[1,:]
    std_trajectory = std(trajectories, dims=1)[1,:]

    # compute forecast error metrics
    if get_metrics
        metric_outputs = compute_all_forecast_metrics(mean_trajectory[forecast_sites], 
            target_timeseries_full[forecast_sites], print_metric_table)
    end

    # plot forecast
    if plot_forecast
        p = plot(collect(conditioning_sites), target_timeseries_full[conditioning_sites],
            lw = 2, label="Conditioning data", xlabel="time", ylabel="x", legend=:outertopright, 
            size=(1000, 500), bottom_margin=5mm, left_margin=5mm, top_margin=5mm)
        plot!(collect(forecast_sites), mean_trajectory[forecast_sites], 
            ribbon=std_trajectory[forecast_sites], label="MPS forecast", ls=:dot, lw=2, alpha=0.5)
        plot!(collect(forecast_sites), target_timeseries_full[forecast_sites], lw=2, label="Ground truth", alpha=0.5)
        title!("Sample $which_sample, Class $which_class, $horizon Site Forecast,\nd = $d_mps MPS, $enc_name encoding,\n $num_shots-Shot Mean")
        display(p)
    end

    return metric_outputs

end

function forward_interpolate_single_timeseries_directMean(
        fcastable::Vector{forecastable}, 
        which_class::Int, 
        which_sample::Int, 
        horizon::Int; 
        plot_forecast::Bool=true, 
        get_metrics::Bool=true, 
        print_metric_table::Bool=true
    )
    """Forward interpolate (forecast) using the direct mean."""

    fcast = fcastable[(which_class+1)]
    mps = fcast.mps
    chi_mps = maxlinkdim(mps)
    d_mps = maxdim(mps[1])
    enc_name = fcast.opts.encoding.name
    target_timeseries_full = fcast.test_samples[which_sample, :]
    conditioning_sites = 1:(length(target_timeseries_full) - horizon)
    forecast_sites = (conditioning_sites[end] + 1):length(mps)
    # handle both time dependent and time independent encodings
    if fcast.opts.encoding.istimedependent
        mean_ts, std_ts = forward_interpolate_directMean_time_dependent(mps, target_timeseries_full[conditioning_sites], first(forecast_sites), fcast.opts, fcast.enc_args)
        title = "Sample $which_sample, Class $which_class, $horizon Site Forecast,\nd = $d_mps, χ = $chi_mps, aux_dim = $(opts.aux_basis_dim)
            $enc_name encoding, Expectation"
    else
        mean_ts, std_ts = forward_interpolate_directMean(mps, target_timeseries_full[conditioning_sites], first(forecast_sites), fcast.opts, fcast.enc_args)
        title = "Sample $which_sample, Class $which_class, $horizon Site Forecast,\nd = $d_mps, χ = $chi_mps,\n$enc_name encoding, Expectation"
    end

    if get_metrics
        metric_outputs = compute_all_forecast_metrics(mean_ts[forecast_sites], target_timeseries_full[forecast_sites], print_metric_table)
    end

    if plot_forecast
        p = plot(collect(conditioning_sites), target_timeseries_full[conditioning_sites], 
            lw=2, label="Conditioning data", xlabel="time", ylabel="x", legend=:outertopright, 
            size=(1000, 500), bottom_margin=5mm, left_margin=5mm, top_margin=5mm)
        plot!(collect(forecast_sites), mean_ts[forecast_sites], ribbon=std_ts[forecast_sites],
            label="MPS forecast", ls=:dot, lw=2, alpha=0.5)
        plot!(collect(forecast_sites), target_timeseries_full[forecast_sites], lw=2, label="Ground truth", alpha=0.5)
        title!(title)
        display(p)
    end

    return metric_outputs
end

function forward_interpolate_single_timeseries_directMode(
        fcastable::Vector{forecastable}, 
        which_class::Int, 
        which_sample::Int,
        horizon::Int; 
        plot_forecast::Bool=true, 
        get_metrics::Bool=true,
        print_metric_table::Bool=true 
    )
    """Forward interpolate (forecast) using the direct mode"""

    fcast = fcastable[(which_class+1)]
    mps = fcast.mps
    chi_mps = maxlinkdim(mps)
    d_mps = maxdim(mps[1])
    enc_name = fcast.opts.encoding.name
    target_timeseries_full = fcast.test_samples[which_sample, :]
    conditioning_sites = 1:(length(target_timeseries_full) - horizon)
    forecast_sites = (conditioning_sites[end] + 1):length(mps)
    # handle both time dependent and time independent encodings
    if fcast.opts.encoding.istimedependent
        mode_ts = forward_interpolate_directMode_time_dependent(mps, target_timeseries_full[conditioning_sites], 
            first(forecast_sites), fcast.opts, fcast.enc_args)
        title = "Sample $which_sample, Class $which_class, $horizon Site Forecast,\nd = $d_mps, χ = $chi_mps, aux_dim=$(opts.aux_basis_dim) 
            $enc_name encoding, Mode"
    else
        mode_ts = forward_interpolate_directMode(mps, target_timeseries_full[conditioning_sites], first(forecast_sites), fcast.opts, fcast.enc_args)
        title = "Sample $which_sample, Class $which_class, $horizon Site Forecast,\nd = $d_mps, χ = $chi_mps,\n$enc_name encoding, Mode"
    end

    if get_metrics
        metric_outputs = compute_all_forecast_metrics(mode_ts[forecast_sites], 
            target_timeseries_full[forecast_sites], print_metric_table);
    end

    if plot_forecast
        p = plot(collect(conditioning_sites), target_timeseries_full[conditioning_sites], 
            lw=2, label="Conditioning data", xlabel="time", ylabel="x", legend=:outertopright, 
            size=(1000, 500), bottom_margin=5mm, left_margin=5mm, top_margin=5mm)
        plot!(collect(forecast_sites), mode_ts[forecast_sites],
            label="MPS forecast", ls=:dot, lw=2, alpha=0.5, c=:magenta)
        plot!(collect(forecast_sites), target_timeseries_full[forecast_sites], lw=2, label="Ground truth", alpha=0.5)
        title!(title)
        display(p)
    end

    return metric_outputs

end

function forward_interpolate_single_timeseries(
        fcastable::Vector{forecastable}, 
        which_class::Int, 
        which_sample::Int, 
        horizon::Int, 
        method::Symbol=:directMean; 
        plot_forecast::Bool=true, 
        get_metrics::Bool=true, 
        print_metric_table::Bool=true
    )

    if method == :directMean 
        metric_outputs = forward_interpolate_single_timeseries_directMean(fcastable, which_class, which_sample,
            horizon; plot_forecast=plot_forecast, get_metrics=get_metrics, print_metric_table=print_metric_table)
    elseif method == :directMode
        metric_outputs = forward_interpolate_single_timeseries_directMode(fcastable, which_class, which_sample,
        horizon; plot_forecast=plot_forecast, get_metrics=get_metrics, print_metric_table=print_metric_table)
    elseif method == :inverseTform
        metric_outputs = forward_interpolate_single_timeseries_sampling(fcastable, which_class, which_sample, horizon;
            num_shots=2000, plot_forecast=plot_forecast, get_metrics=get_metrics, print_metric_table=print_metric_table)
    else
        error("Invalid method. Choose either :directMean (Mean/Std), :directMode, or :inverseTform (inv. transform sampling).")
    end

    return metric_outputs
end



function NN_interpolate(fcastables::AbstractVector{forecastable},
        which_class::Integer, 
        which_sample::Integer, 
        which_sites::AbstractVector{<:Integer}; 
        X_train::AbstractMatrix{<:Real}, 
        y_train::AbstractVector{<:Integer}, 
        n_ts::Integer=1,
        invert_transform::Bool=true
    )

    fcast = fcastables[(which_class+1)]
    mps = fcast.mps


    target_timeseries_full = fcast.test_samples[which_sample, :]


    known_sites = setdiff(collect(1:length(mps)), which_sites)
    target_series = target_timeseries_full[known_sites]

    c_inds = findall(y_train .== which_class)
    Xs_comparison = X_train[c_inds, known_sites]

    mses = Vector{Float64}(undef, length(c_inds))

    for (i, ts) in enumerate(eachrow(Xs_comparison))
        mses[i] = (ts .- target_series).^2 |> mean
    end
    
    min_inds = partialsortperm(mses, 1:n_ts)
    ts = Vector(undef, n_ts)

    for (i,min_ind) in enumerate(min_inds)
        ts_ind = c_inds[min_ind]
        ts[i] = X_train[ts_ind,:]
    end

    # metric_outputs = compute_all_forecast_metrics(ts[1][which_sites], target_timeseries_full[which_sites])

    return ts


end

function any_interpolate_single_timeseries_sampling(
        fcastable::Vector{forecastable},
        which_class::Int, 
        which_sample::Int, 
        which_sites::Vector{Int}; 
        num_shots::Int=1000, 
        get_metrics::Bool=true
    )

    fcast = fcastable[(which_class+1)]
    mps = fcast.mps
    chi_mps = maxlinkdim(mps)
    d_mps = maxdim(mps[1])
    enc_name = fcast.opts.encoding.name
    target_timeseries_full = fcast.test_samples[which_sample, :]
    trajectories = Matrix{Float64}(undef, num_shots, length(target_timeseries_full))
    if fcast.opts.encoding.istimedependent
        @threads for i in 1:num_shots
            trajectories[i, :] = any_interpolate_trajectory_time_dependent(mps, fcast.opts, fcast.enc_args, target_timeseries_full, which_sites)
        end
    else
        # time independent encoding
        @threads for i in 1:num_shots
            trajectories[i, :] = any_interpolate_trajectory(mps, fcast.opts, fcast.enc_args, target_timeseries_full, which_sites)
        end
    end

    # get summary statistics
    mean_trajectory = mean(trajectories, dims=1)[1,:]
    std_trajectory = std(trajectories, dims=1)[1,:]

    if get_metrics
        metric_outputs = compute_all_forecast_metrics(mean_trajectory[which_sites], 
            target_timeseries_full[which_sites])
    end

    p = plot(mean_trajectory, ribbon=std_trajectory, xlabel="time", ylabel="x", 
        label="MPS Interpolated", ls=:dot, lw=2, alpha=0.8, legend=:outertopright,
        size=(800, 500), bottom_margin=5mm, left_margin=5mm, top_margin=5mm)
    plot!(target_timeseries_full, label="Ground Truth", c=:orange, lw=2, alpha=0.7)
    title!("Sample $which_sample, Class $which_class, $(length(which_sites))-site Interpolation, 
        d = $d_mps, χ = $chi_mps, $enc_name encoding, 
        $num_shots-shot mean")

    return metric_outputs, p

end

function any_interpolate_single_timeseries_directMode(fcastable::Vector{forecastable},
        which_class::Int, 
        which_sample::Int, 
        which_sites::Vector{Int}; 
        get_metrics::Bool=true,
        invert_transform::Bool=true
    )

    fcast = fcastable[(which_class+1)]
    mps = fcast.mps
    chi_mps = maxlinkdim(mps)
    d_mps = maxdim(mps[1])
    enc_name = fcast.opts.encoding.name
    target_timeseries_full = fcast.test_samples[which_sample, :]

    if fcast.opts.encoding.istimedependent
        mode_ts = any_interpolate_directMode_time_dependent(mps, fcast.opts, fcast.enc_args, target_timeseries_full, which_sites)
    else
        mode_ts = any_interpolate_directMode(mps, fcast.opts, fcast.enc_args, target_timeseries_full, which_sites)
    end

    if get_metrics
        metric_outputs = compute_all_forecast_metrics(mode_ts[which_sites], 
            target_timeseries_full[which_sites]);
    end

    p1 = plot(mode_ts, xlabel="time", ylabel="x", 
        label="MPS Interpolated", ls=:dot, lw=2, alpha=0.8, legend=:bottomleft,
        size=(1000, 500), bottom_margin=5mm, left_margin=5mm, top_margin=5mm)
    p1 = plot!(target_timeseries_full, label="Ground Truth", c=:orange, lw=2, alpha=0.7)

    return metric_outputs, p1

end

function any_interpolate_single_timeseries_directMean(fcastable::Vector{forecastable},
        which_class::Int, 
        which_sample::Int, 
        which_sites::Vector{Int}; 
        get_metrics::Bool=true,
        invert_transform::Bool=true
    )

    fcast = fcastable[(which_class+1)]
    mps = fcast.mps
    chi_mps = maxlinkdim(mps)
    d_mps = maxdim(mps[1])
    enc_name = fcast.opts.encoding.name
    target_timeseries_full = fcast.test_samples[which_sample, :]

    if fcast.opts.encoding.istimedependent
        mean_ts, std_ts = any_interpolate_directMean_time_dependent(mps, fcast.opts, fcast.enc_args, target_timeseries_full, which_sites)
    else
        mean_ts, std_ts = any_interpolate_directMean(mps, fcast.opts, fcast.enc_args, target_timeseries_full, which_sites)
    end

    if get_metrics
        metric_outputs = compute_all_forecast_metrics(mean_ts[which_sites], 
            target_timeseries_full[which_sites])
    end

    p1 = plot(mean_ts, ribbon=std_ts, xlabel="time", ylabel="x", 
        label="MPS Interpolated", ls=:dot, lw=2, alpha=0.8, legend=:outertopright,
        size=(1000, 500), bottom_margin=5mm, left_margin=5mm, top_margin=5mm)
    p1 = plot!(target_timeseries_full, label="Ground Truth", c=:orange, lw=2, alpha=0.7)
    p1 = title!("Sample $which_sample, Class $which_class, $(length(which_sites))-site Interpolation, 
        d = $d_mps, χ = $chi_mps, $enc_name encoding, 
        Expectation")

    return metric_outputs, p1

end
"""
Interpolate using the median of the conditional pdf.\n
Uses the (weighted) median absolute deviation to quantify uncertainty. 
"""
function any_interpolate_median(
        fcastable::Vector{forecastable},
        which_class::Int,
        which_sample::Int,
        which_sites::Vector{Int};
        NN_baseline::Bool=true,
        X_train::AbstractMatrix{<:Real}, 
        y_train::AbstractVector{<:Integer}=Int[], 
        n_baselines::Integer=1,
        invert_transform::Bool=true,
        get_metrics::Bool=true,
        full_metrics::Bool=false,
        plot_fits::Bool=true,
        wmad::Bool=false,
        print_metric_table::Bool=false
    )

    # setup interpolation variables
    fcast = fcastable[(which_class+1)]
    mps = fcast.mps
    chi_mps = maxlinkdim(mps)
    d_mps = maxdim(mps[1])
    enc_name = fcast.opts.encoding.name
    target_ts_raw = fcast.test_samples[which_sample, :]

    # transform the data
    # perform the scaling
    if fcast.opts.sigmoid_transform
        sig_trans = fit(RobustSigmoid, X_train)
        target_ts_sig = normalize(reshape(target_ts_raw, :,1), sig_trans)
        
        te_minmax = fit(MinMax, target_ts_sig)
        target_timeseries_full = normalize(target_ts_sig, te_minmax)
    else
        sig_trans = nothing        
        te_minmax = fit(MinMax, X_train)
        target_timeseries_full = normalize(reshape(target_ts_raw, :,1), te_minmax)    
    end

    # rescale if out of bounds
    lb, ub = extrema(target_timeseries_full)
    if lb < 0
        if abs(lb) > 0.01
            @warn "Test set has a value more than 1% below lower bound after train normalization!"
        end
        target_timeseries_full .-= lb
        ub = maximum(target_timeseries_full)
    end

    if ub > 1
        if abs(ub-1) > 0.01
            @warn "Test set has a value more than 1% above upper bound after train normalization!"
        end
        target_timeseries_full ./= ub
    end

    a,b = fcast.opts.encoding.range
    @. target_timeseries_full = (b-a) *target_timeseries_full + a

    target_timeseries_full = reshape(target_timeseries_full, size(target_timeseries_full,1)) # convert back to a vector

    if fcast.opts.encoding.istimedependent
        error("Time dependent option not yet implemented!")
    else
        enc_args = []
        sites = siteinds(mps)

        states = MPS([itensor(fcast.opts.encoding.encode(t, fcast.opts.d, enc_args...), sites[i]) for (i,t) in enumerate(target_timeseries_full)])
        ts, wms = any_interpolate_directMedian(mps, fcast.opts, fcast.enc_args, target_timeseries_full, states, which_sites; wmad=wmad)
        wms .+= ts # add uncertainty onto time series 
    end

    if invert_transform
        ts = reshape((ts .- a) ./ (b-a),:,1)
        wms = reshape((wms .- a) ./ (b-a),:,1)

        denormalize!(ts, te_minmax)
        denormalize!(wms, te_minmax)

        if !isnothing(sig_trans)
            denormalize!(ts, sig_trans)
            denormalize!(wms, sig_trans)
        end

        ts = reshape(ts, size(ts, 1))
        wms = reshape(wms, size(ts, 1))
    end
    new_wms = wms .- ts # remove the time-series, leaving the unscaled uncertainty
    

    if plot_fits
        interp_series = fill(NaN, length(target_ts_raw))
        interp_series[which_sites] = ts[which_sites]
        interp_uncertainties = fill(NaN, length(target_ts_raw))
        interp_uncertainties[which_sites] = new_wms[which_sites]

        observed_series = fill(NaN, length(target_ts_raw))
        obs_pts = setdiff(1:length(target_ts_raw), which_sites)
        observed_series[obs_pts] = target_ts_raw[obs_pts]

        ground_truth = fill(NaN, length(target_ts_raw))
        ground_truth[which_sites] = target_ts_raw[which_sites]
        
        p1 = plot(observed_series, xlabel="time", ylabel="x", 
            label="Observed", ls=:dot, lw=2, legend=:outertopright,
            size=(1000, 500), bottom_margin=5mm, left_margin=5mm, top_margin=5mm, c=:black)

        p1 = plot!(ground_truth, label="Ground truth", c=:black, lw=2, alpha=0.3)
        p1 = plot!(interp_series, label="MPS Interpolated", lw=2, alpha=0.8, c=:red, ribbon=interp_uncertainties,
            fillalpha=0.15)
        p1 = title!("Sample $which_sample, Class $which_class, $(length(which_sites))-site Interpolation, 
            d = $d_mps, χ = $chi_mps, $enc_name encoding, \nMedian" * (wmad ? ", +/- WMAD" : "")
        )
        p1 = [p1] # for type stability
    else
        p1 = []
    end

    if get_metrics
        if full_metrics
            metrics = compute_all_forecast_metrics(ts[which_sites], target_ts_raw[which_sites], print_metric_table)
        else
            metrics = Dict(:MAE => mae(ts[which_sites], target_ts_raw[which_sites]))
        end
    else
        metrics = []
    end

    if NN_baseline
        mse_ts = NN_interpolate(fcastable, which_class, which_sample, which_sites; X_train, y_train, n_ts=n_baselines)

        mse_ts_bounded = mse_ts
        if !invert_transform
            # scale mse_ts to between a and b so it can be plotted on the same axis as ts
            for (i,mse_t) in enumerate(mse_ts)
                mse_norm = fit(MinMax, mse_t)
                mse_ts_bounded[i] = (b-a)*normalize(mse_t, mse_norm) + a
            end
        end

        if plot_fits
            if length(ts) == 1
                p1 = plot!(interp_series_nn, label="Nearest Train Data", c=:orange, lw=2, alpha=0.7, ls=:dot)
            else
                for (i,t) in enumerate(mse_ts_bounded)
                    p1 = plot!(t, label="Nearest Train Data $i", c=:orange,lw=2, alpha=0.7, ls=:dot)
                end

            end
            p1 = [p1] # for type stability
        end

        
        if get_metrics
            if full_metrics
                NN_metrics = compute_all_forecast_metrics(mse_ts[1][which_sites], target_ts_raw[which_sites], print_metric_table)
                for key in keys(NN_metrics)
                    metrics[Symbol("NN_" * string(key) )] = NN_metrics[key]
                end
            else
                metrics[:NN_MAE] = mae(mse_ts[1][which_sites], target_ts_raw[which_sites])
            end
        end
        return metrics, p1
    end

    return metrics, p1
    
end

function any_interpolate_single_timeseries(
        fcastable::Vector{forecastable},
        which_class::Int, 
        which_sample::Int, 
        which_sites::Vector{Int}, 
        method::Symbol=:directMean;
        NN_baseline::Bool=true, 
        X_train::AbstractMatrix{<:Real}, 
        y_train::AbstractVector{<:Integer}=Int[], 
        n_baselines::Integer=1,
        invert_transform::Bool=true, # whether to undo the sigmoid transform 
        get_metrics::Bool=true, # whether to compute goodness of fit metrics
        full_metrics::Bool=false, # whether to compute every metric or just MAE
        plot_fits=true,
        print_metric_table::Bool=false,
        dx::Float64 = 1E-4,
        mode_range=opts.encoding.range,
        xvals::AbstractVector{Float64}=collect(range(mode_range...; step=dx)),
        mode_index=Index(opts.d),
        xvals_enc:: AbstractVector{<:AbstractVector{<:Number}}= [get_state(x, opts, fcastable[1].enc_args) for x in xvals],
        xvals_enc_it::AbstractVector{ITensor}=[ITensor(s, mode_index) for s in xvals_enc],
        max_jump::Union{Number, Nothing}=nothing,
        group_testnorm::Bool=false,
        bad_NN::Bool=false
    )

    # setup interpolation variables
    fcast = fcastable[(which_class+1)]
    X_test = vcat([fc.test_samples for fc in fcastable]...)

    mps = fcast.mps
    chi_mps = maxlinkdim(mps)
    d_mps = maxdim(mps[1])
    enc_name = fcast.opts.encoding.name
    target_ts_raw = fcast.test_samples[which_sample, :]

    # transform the data
    # perform the scaling
    if fcast.opts.sigmoid_transform
        sig_trans = fit(RobustSigmoid, X_train)
        target_ts_sig = normalize(reshape(target_ts_raw, :,1), sig_trans)
        
        if group_testnorm
            te_minmax = fit(MinMax, normalize(X_test, sig_trans))
        else
            te_minmax = fit(MinMax, normalize(X_train, sig_trans))
        end
        target_timeseries_full = normalize(target_ts_sig, te_minmax)
    else
        sig_trans = nothing    
        if group_testnorm
            te_minmax = fit(MinMax, X_test)
        else    
        te_minmax = fit(MinMax, X_train)
        end
        target_timeseries_full = normalize(reshape(target_ts_raw, :,1), te_minmax)    
    end

    target_timeseries = copy(target_timeseries_full)
    target_timeseries[which_sites] .= 0.5 # make it impossible for the unknown region to be used, even accidentally


    # rescale if out of bounds
    lb, ub = extrema(target_timeseries)
    lb_shift = 0
    ub_scale = 1
    if lb < 0
        if abs(lb) > 0.01
            @warn "Test set has a value more than 1% below lower bound after train normalization!"
        end
        target_timeseries_full .-= lb
        lb_shift = lb
        ub = maximum(target_timeseries)
    end

    if ub > 1
        if abs(ub-1) > 0.01
            @warn "Test set has a value more than 1% above upper bound after train normalization!"
        end
        ub_scale = ub
        target_timeseries ./= ub
    end

    a,b = fcast.opts.encoding.range
    @. target_timeseries = (b-a) *target_timeseries + a

    target_timeseries = reshape(target_timeseries, size(target_timeseries,1)) # convert back to a vector

    pred_err = nothing
    if method == :directMean        
        if fcast.opts.encoding.istimedependent
            ts, pred_err = any_interpolate_directMean_time_dependent(mps, fcast.opts, fcast.enc_args, target_timeseries, which_sites)
        else
            ts, pred_err = any_interpolate_directMean(mps, fcast.opts, fcast.enc_args, target_timeseries, which_sites)
        end
    elseif method == :directMedian
        if fcast.opts.encoding.istimedependent
            error("Time dependent option not yet implemented!")
        else
            sites = siteinds(mps)

            states = MPS([itensor(fcast.opts.encoding.encode(t, fcast.opts.d, fcast.enc_args...), sites[i]) for (i,t) in enumerate(target_timeseries)])
            ts, pred_err = any_interpolate_directMedian(mps, fcast.opts, fcast.enc_args, target_timeseries, states, which_sites; dx=dx, mode_range=mode_range, xvals=xvals, xvals_enc=xvals_enc, xvals_enc_it=xvals_enc_it, mode_index=mode_index)
        end
    elseif method == :directMode
        if fcast.opts.encoding.istimedependent
            # xvals_enc = [get_state(x, opts) for x in x_vals]

            ts = any_interpolate_directMode_time_dependent(mps, fcast.opts, fcast.enc_args, target_timeseries, which_sites)
        else
            sites = siteinds(mps)
            
            states = MPS([itensor(fcast.opts.encoding.encode(t, fcast.opts.d, fcast.enc_args...), sites[i]) for (i,t) in enumerate(target_timeseries)])
            ts = any_interpolate_directMode(mps, fcast.opts, fcast.enc_args, target_timeseries, states, which_sites; dx=dx, mode_range=mode_range, xvals=xvals, xvals_enc=xvals_enc, xvals_enc_it=xvals_enc_it, mode_index=mode_index, max_jump=max_jump)
        end
    elseif method == :MeanMode
        if fcast.opts.encoding.istimedependent
            # xvals_enc = [get_state(x, opts) for x in x_vals]
            error("Time dep not implemented for MeanMode")
        else
            sites = siteinds(mps)
            
            states = MPS([itensor(fcast.opts.encoding.encode(t, fcast.opts.d, fcast.enc_args...), sites[i]) for (i,t) in enumerate(target_timeseries)])
            ts = any_interpolate_MeanMode(fcast.mps, fcast.opts, target_timeseries, states, which_sites; dx=dx, mode_range=mode_range, xvals=xvals, xvals_enc=xvals_enc, xvals_enc_it=xvals_enc_it, mode_index=mode_index, max_jump=max_jump)
        end
    elseif method ==:nearestNeighbour
        ts = NN_interpolate(fcastable, which_class, which_sample, which_sites; X_train, y_train, n_ts=1)[1]

        if !invert_transform
            # scale mse_ts to between a and b so it can be plotted on the same axis as ts
            if fcast.opts.sigmoid_transform
                ts_bounded = normalize(reshape(ts,:,1), sig_trans)
                ts_bounded = (b-a).*normalize(ts_bounded, te_minmax) .+ a
            else
                ts_bounded = (b-a).*normalize(reshape(ts,:,1), te_minmax) .+ a
            end
            ts = reshape(ts_bounded, size(ts_bounded, 1))
        end

    else
        error("Invalid method. Choose :directMean (Expect/Var), :directMode, :directMedian, or nearestNeighbour.")
    end


    if invert_transform && !(method == :nearestNeighbour)
        ts = reshape((ts .- a) ./ (b-a),:,1)

        denormalize!(ts, te_minmax)

        if !isnothing(sig_trans)
            denormalize!(ts, sig_trans)
        end


        # undo any extra scaling done to force the ts into [0,1]
        ts .*= ub_scale
        ts .+= lb_shift 

        ts = reshape(ts, size(ts, 1))
    end

    if plot_fits
        p1 = plot(ts, ribbon=pred_err, xlabel="time", ylabel="x", 
            label="MPS Interpolated", ls=:dot, lw=2, alpha=0.8, legend=:outertopright,
            size=(1000, 500), bottom_margin=5mm, left_margin=5mm, top_margin=5mm
        )
        target = invert_transform ? target_ts_raw : target_timeseries_full

        p1 = plot!(target, label="Ground Truth", c=:orange, lw=2, alpha=0.7)
        p1 = title!("Sample $which_sample, Class $which_class, $(length(which_sites))-site Interpolation, 
            d = $d_mps, χ = $chi_mps, $enc_name encoding"
        )
        p1 = [p1] # for type stability
    else
        p1 = []
    end


    if get_metrics
        target = invert_transform ? target_ts_raw : target_timeseries_full
        if full_metrics
            metrics = compute_all_forecast_metrics(ts[which_sites], target[which_sites], print_metric_table)
        else
            metrics = Dict(:MAE => mae(ts[which_sites], target[which_sites]))
        end
    else
        metrics = []
    end

    if NN_baseline
        mse_ts::Vector{Any} = NN_interpolate(fcastable, which_class, which_sample, which_sites; X_train, y_train, n_ts=n_baselines)

        mse_ts_bounded = mse_ts
        if !invert_transform
            # scale mse_ts to between a and b so it can be plotted on the same axis as ts
            for (i,mse_t) in enumerate(mse_ts)
                if fcast.opts.sigmoid_transform
                    mse_ts_bounded[i] = normalize(reshape(mse_t,:,1), sig_trans)
                    mse_ts_bounded[i] = (b-a).*normalize(mse_ts_bounded[i], te_minmax) .+ a
                else
                    mse_ts_bounded[i] = (b-a).*normalize(reshape(mse_t,:,1), te_minmax) .+ a
                end
                mse_ts_bounded[i] = reshape(mse_ts_bounded[i], size(mse_ts_bounded[i], 1))
            end
        end

        if plot_fits 
            if length(ts) == 1
                p1 = plot!(mse_ts_bounded[1], label="Nearest Train Data", c=:red, lw=2, alpha=0.7, ls=:dot)
            else
                for (i,t) in enumerate(mse_ts_bounded)
                    p1 = plot!(t, label="Nearest Train Data $i", c=:red,lw=2, alpha=0.7, ls=:dot)
                end

            end
            p1 = [p1] # for type stability
        end

        
        if get_metrics
            target = invert_transform ? target_ts_raw : target_timeseries_full
            if full_metrics
                NN_metrics = compute_all_forecast_metrics(mse_ts_bounded[1][which_sites], target[which_sites], print_metric_table)
                for key in keys(NN_metrics)
                    metrics[Symbol("NN_" * string(key) )] = NN_metrics[key]
                end
            else
                metrics[:NN_MAE] = mae(mse_ts_bounded[1][which_sites], target[which_sites])
            end
        end
        return metrics, p1
    end

    return metrics, p1
end

function forecast_all(
        fcastable::Vector{forecastable}, 
        method::Symbol, 
        horizon::Int=50;
        metric = :MAE, 
        verbose=false
    )
    """Assess forecasting performance for all classes"""
    num_classes = length(fcastable)
    println("There are $num_classes classes. Evaluating all...")
    all_scores = []
    for (class_idx, fc) in enumerate(fcastable)
        test_samples = fc.test_samples
        class_scores = Vector{Any}(undef, size(test_samples, 1))
        @threads for i in 1:size(test_samples, 1)
            sample_results = forward_interpolate_single_timeseries(fcastable, (class_idx-1), i, horizon, method; 
                plot_forecast=false, get_metrics=true, print_metric_table=false)
            if verbose
                println("[$i] Sample $metric: $(sample_results[metric])")
            end
            class_scores[i] = sample_results[metric]
        end
        push!(all_scores, class_scores)
    end

    return all_scores

end