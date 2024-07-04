using JLD2
using CSV
using DataFrames
using Plots, Plots.PlotMeasures
using StatsBase
using Statistics

mutable struct instance
    class::Int
    name::Symbol
    ts::Vector{Float64}
end

function create_instance(ts::Vector{Float64}, class::Int)
    @assert class in 0:6 "Invalid class."
    label_mapping = Dict(
        0 => :ContactEB,
        1 => :DetachedEB,
        2 => :DeltaScVar,
        3 => :GammaDorVar,
        4 => :NonVar,
        5 => :RotVar,
        6 => :RRLyrVar)
    return instance(class, label_mapping[class], ts)

end

function load_dataset(file::String)

    f = jldopen(file, "r")
    dataset = read(f, "dataset")
    labels = Int.(dataset[:, end]) .-1 # start at index 0 for class 0
    num_instances = size(dataset, 1)
    all_instances = Vector{instance}(undef, num_instances)
    for i in 1:num_instances
        all_instances[i] = create_instance(dataset[i, 1:end-1], labels[i])
    end
    return all_instances
end


function plot_instance(all_instances::Vector{instance}, idx::Int,
        timepts::Union{Nothing, UnitRange}=nothing)
    sample = all_instances[idx]
    if isnothing(timepts)
        idxs = 1:length(sample.ts)
    else
        idxs = timepts
    end
    p = plot(sample.ts[idxs], xlabel = "time (samples)", ylabel = "x", title = "$(sample.name), $idxs", label="", c=palette(:tab10)[(sample.class + 1)], lw=2, 
        ls=:dot)
    display(p)
end

function plot_examples(all_instances::Vector{instance}, which_class::Int, plot_pts::Union{Nothing, UnitRange}=nothing)
    """Plot 20 random examples from a class of interest"""
    if isnothing(plot_pts)
        plot_pts_idxs = 1:length(all_instances[1].ts)
    else
        plot_pts_idxs = plot_pts
    end
    class_labels = [all_instances[i].class for i in 1:length(all_instances)]
    idxs = findall(x -> x .== which_class, class_labels)
    plot_idxs = sample(idxs, 20; replace=false)
    ps = []
    for pidx in plot_idxs
        p = plot(all_instances[pidx].ts[plot_pts_idxs], xlabel="time (samples)", ylabel="x", 
            title="$(all_instances[pidx].name), Sample $pidx", c=palette(:tab10)[(all_instances[pidx].class + 1)], label="")
        push!(ps, p)
    end
    p_final = plot(ps..., layout=(5, 4), size=(1500, 1000), right_margin=5mm, left_margin=5mm, bottom_margin=5mm)
    display(p_final)

end

function make_overlapping_windows(ts::Vector{Float64}, window_size::Int, stride::Int)
    n = length(ts)
    windows = [ts[i:i+window_size-1] for i in 1:stride:n-window_size+1]
    return windows
end

function check_for_artifacts(ts::Vector{Float64}; threshold=10, min_consecutive=2)

    diff_series = diff(ts)
    
    consecutive_count = 0
    max_consecutive = 0
    total_artifact_points = 0

    for diff_value in diff_series
        if diff_value == 0
            consecutive_count += 1
            if consecutive_count >= min_consecutive
                total_artifact_points += 1
            end
        else
            max_consecutive = max(max_consecutive, consecutive_count)
            consecutive_count = 0
        end
    end
    max_consecutive = max(max_consecutive, consecutive_count)

    has_artifacts = max_consecutive > threshold
    
    return has_artifacts
end

function window_single_instance(all_instances::Vector{instance}, which_idx::Int,
    w::Int, overlap_fraction::Float64=0.1; plot_windows=true)
    """Window a single instance and inspect the windows"""
    x = all_instances[which_idx].ts
    stride = w - Int(floor(overlap_fraction * w))
    windows_all = make_overlapping_windows(x, w, stride)
    
    if plot_windows
        ps = []
        for i in 1:length(windows_all)
            if check_for_artifacts(windows_all[i])
                titlefontcolor= :red
            else
                titlefontcolor = :black
            end
            p = plot(windows_all[i, :], xlabel="time (samples)", ylabel="x", label="", title="Window id: $i", 
            titlefontcolor=titlefontcolor, c=palette(:tab10)[(all_instances[which_idx].class +1)])
            push!(ps, p)
        end

        p_final = plot(ps..., size=(2000, 2000))
        display(p_final)
    end

    return windows_all

end

function make_train_test_split_class(all_instances::Vector{instance}, class::Int,
    w::Int, discard_idxs::Vector{Int}; train_ratio::Float64=0.3)

    class_labels = [all_instances[i].class for i in 1:length(all_instances)]
    class_idxs = findall(x -> x .== class, class_labels)

    X_train_all = Vector{Any}()
    X_test_all = Vector{Any}()

    for cidx in class_idxs
        all_windows = window_single_instance(all_instances, cidx, w, 0.0)
        data_mat = permutedims(hcat(all_windows...))
        clean_idxs = setdiff(1:size(data_mat, 1), discard_idxs)
        dirty_idxs = discard_idxs

        clean_data_mat = data_mat[clean_idxs, :]
        dirty_data_mat = data_mat[dirty_idxs, :]

        total_clean = size(clean_data_mat, 1)
        num_train = Int(floor(train_ratio * total_clean))
        train_idxs = sample(1:total_clean, num_train; replace=false)
        candidate_test_idxs = setdiff(1:size(clean_data_mat, 1), train_idxs)
        test_idxs = sample(1:length(candidate_test_idxs), 1)

        X_train = clean_data_mat[train_idxs, :]
        # append the corrupted windows onto the test set
        X_test = vcat(clean_data_mat[test_idxs, :], dirty_data_mat)
        push!(X_train_all, X_train)
        push!(X_test_all, X_test)

    end


    X_train = vcat(X_train_all...)
    y_train = Int.(zeros(size(X_train, 1)))
    X_test = vcat(X_test_all...)
    y_test = Int.(zeros(size(X_test, 1)))

    return X_train, y_train, X_test, y_test

end

function make_train_test_split_singleTS(all_instances::Vector{instance},
    which_idx::Int, w::Int, discard_idxs::Vector{Int}, overlap_fraction::Float64=0.0; 
    train_fraction::Float64=0.8, return_corrupted_windows::Bool=true)
    """Make a train/test split out of a single time series instance.
    First, window the data, then discard the windows with artefacts,
    then split the remaining windows into train/test."""
    all_windows = window_single_instance(all_instances, which_idx, w, overlap_fraction; plot_windows=false)
    data_mat = permutedims(hcat(all_windows...))
    
    clean_idxs = setdiff(1:size(data_mat, 1), discard_idxs)
    println("There are $(length(clean_idxs)) clean windows.")
    dirty_idxs = discard_idxs

    clean_data_mat = data_mat[clean_idxs, :]
    dirty_data_mat = data_mat[dirty_idxs, :]

    # assign train and test
    num_train = Int(floor(size(clean_data_mat, 1) * train_fraction))
    # sample train idxs
    train_idxs = sample(1:size(clean_data_mat, 1), num_train; replace=false)
    test_idxs = setdiff(1:size(clean_data_mat, 1), train_idxs)
    X_train = clean_data_mat[train_idxs, :]
    X_test = clean_data_mat[test_idxs, :]

    if return_corrupted_windows
        # append corrupted windows to the end of the test set
        X_test = vcat(X_test, dirty_data_mat)
    end

    y_train = Int.(zeros(size(X_train, 1)))
    y_test = Int.(zeros(size(X_test, 1)))

    return X_train, X_test, y_train, y_test

end


