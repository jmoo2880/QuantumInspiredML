include("../Interpolation/ForecastingMainNew.jl")
import MLJTuning 
import ProgressMeter



function revise_history!(Xs_train::Matrix, ys_train::Vector, interp_sites::Vector, max_ts_per_class::Integer, history)
    n_maes = length(history) * length(history[1].evaluation.fitted_params_per_fold)
    p = ProgressMeter.Progress(n_maes,
        dt = 0,
        desc = "Evaluating Interp error of $n_maes models:",
        barglyphs = ProgressMeter.BarGlyphs("[=> ]"),
        barlen = 25,
        color = :green
    )    
    ProgressMeter.update!(p,0)


    means = Vector{Float64}(undef, length(history))

    if !(history[1].evaluation isa PerformanceEvaluation)
        error("compact_history must be set to false to use the interpolation_hyperopt hack!")
    end

    for (hi, h) in enumerate(history)
        e = h.evaluation
        nfolds = length(e.fitted_params_per_fold)
        f_per_fold = zeros(Float64, nfolds)
        for fold in 1:nfolds
            # retrieve model and train/test data
            dec, mopts, W = e.fitted_params_per_fold[fold][1]
            # @show hi, fold, W[1][1], W[1][2]
            # @show mopts.eta, mopts.chi_max
            opts, _... = safe_options(mopts, nothing, nothing) # make sure options is abstract

            train_idxs, val_idxs = e.train_test_rows[fold]
            Xs_train_fold, ys_train_fold = Xs_train[train_idxs, :], ys_train[train_idxs]
            Xs_val_fold, ys_val_fold = Xs_train[val_idxs, :], ys_train[val_idxs]


            # precompute encoded data for computational speedup
            mode_range=opts.encoding.range
            xvals=collect(range(mode_range...; step=1E-4))
            mode_index=Index(opts.d)
            xvals_enc= [get_state(x, opts) for x in xvals]
            xvals_enc_it=[ITensor(s, mode_index) for s in xvals_enc];

            fc = load_forecasting_info_variables(W, Xs_train_fold, ys_train_fold, Xs_val_fold, ys_val_fold, opts; verbosity=-1);


            n_c1s = sum(ys_val_fold)
            n_c0s = length(ys_val_fold) - n_c1s

            n0max =  min(max_ts_per_class, n_c0s)
            n1max =  min(max_ts_per_class, n_c1s)

            classes = [zeros(Int,n0max); ones(Int,n1max)]
            sample_idxs = [shuffle(MersenneTwister(fold), 1:n_c0s)[1:n0max]; shuffle(MersenneTwister(fold), 1:n_c1s)[1:n1max]]
            #sample_idxs = [1:n0max; 1:n1max]

            n_ts = length(classes)
            # @show hi, fold, train_idxs[1:5], sample_idxs[1:5]
            @sync @simd for j in 1:n_ts
                Threads.@spawn begin 
                    class = classes[j]
                    sample = sample_idxs[j]

                    stats, _ = any_interpolate_single_timeseries(fc, class, sample, interp_sites, :directMode; NN_baseline=false, X_train=Xs_train_fold, y_train=ys_train_fold,  plot_fits=false, mode_range=mode_range, xvals=xvals, mode_index=mode_index, xvals_enc=xvals_enc, xvals_enc_it=xvals_enc_it);
                    f_per_fold[fold] += stats[:MAE]
                    # @show stats
                    # @show f_per_fold[fold]
                end
            end
            f_per_fold[fold] /= n_ts
            p.counter += 1
            ProgressMeter.updateProgress!(p)
        end
        e.per_fold[1] .= f_per_fold

        h.per_fold[1] .= f_per_fold
        for u in 2:length(e.per_fold)
            e.per_fold[u] .= zeros(nfolds)
            h.per_fold[u] .= zeros(nfolds)
        end
        
        mf = mean(f_per_fold)
        e.measurement[1] = mf
        h.measurement[1] = mf

        means[hi] = mf
    end
    return means
end


# struct InterpolationMeasureHack <: MLJTuning.SelectionHeuristic
#     # f::Function # goodness of fit measure, for example MAE
#     Xs_train::Matrix # timeseries are rows!
#     ys_train::Vector{<:Integer}
#     interp_sites::Vector{<:Integer}
#     max_ts_per_class::Integer

#     function InterpolationMeasureHack(Xs_train::Matrix, ys_train::Vector, interp_sites::Vector, max_ts_per_class::Integer)
#         if maximum(interp_sites) > size(Xs_train, 2) || minimum(interp_sites) < 1
#             error("Trying to interpolate a site not in the timeseries!")
#         end
#         return new(Xs_train, ys_train, interp_sites, max_ts_per_class)
#     end
# end

# InterpolationMeasureHack(Xs_train::Matrix, ys_train::Vector, interp_sites::Vector) = InterpolationMeasureHack(Xs_train, ys_train, interp_sites, 0)


# function MLJTuning.losses(heuristic::InterpolationMeasureHack, history)
#     n_maes = length(history) * length(history[1].evaluation.fitted_params_per_fold)
#     # p = ProgressMeter.Progress(n_maes,
#     #     dt = 0,
#     #     desc = "Evaluating Interp error of $n_maes models:",
#     #     barglyphs = ProgressMeter.BarGlyphs("[=> ]"),
#     #     barlen = 25,
#     #     color = :green
#     # )    
#     # ProgressMeter.update!(p,0)


#     means = Vector{Float64}(undef, length(history))

#     if !(history[1].evaluation isa PerformanceEvaluation)
#         error("compact_history must be set to false to use the interpolation_hyperopt hack!")
#     end

#     for (hi, h) in enumerate(history)
#         e = h.evaluation
#         nfolds = length(e.fitted_params_per_fold)
#         f_per_fold = Vector{Float64}(undef, nfolds)
#         for fold in 1:nfolds
#             # retrieve model and train/test data
#             dec, mopts, W = e.fitted_params_per_fold[fold][1]
#             opts, _... = safe_options(mopts, nothing, nothing) # make sure options is abstract

#             train_idxs, val_idxs = e.train_test_rows[fold]
#             Xs_train, ys_train = heuristic.Xs_train[train_idxs, :], heuristic.ys_train[train_idxs]
#             Xs_val, ys_val = heuristic.Xs_train[val_idxs, :], heuristic.ys_train[val_idxs]


#             # precompute encoded data for computational speedup
#             mode_range=opts.encoding.range
#             xvals=collect(range(mode_range...; step=1E-4))
#             mode_index=Index(opts.d)
#             xvals_enc= [get_state(x, opts) for x in xvals]
#             xvals_enc_it=[ITensor(s, mode_index) for s in xvals_enc];

#             fc = load_forecasting_info_variables(W, Xs_train, ys_train, Xs_val, ys_val, opts; verbosity=-1);


#             n_c1s = sum(ys_val)
#             n_c0s = length(ys_val) - n_c1s

#             n0max = heuristic.max_ts_per_class <= 0 ? n_c0s : min(heuristic.max_ts_per_class, n_c0s)
#             n1max = heuristic.max_ts_per_class <= 0 ? n_c1s : min(heuristic.max_ts_per_class, n_c1s)

#             classes = [zeros(Int,n0max); ones(Int,n1max)]
#             sample_idxs = [shuffle(MersenneTwister(train_idxs[1]), 1:n_c0s)[1:n0max]; shuffle(MersenneTwister(train_idxs[1]), 1:n_c1s)[1:n1max]]
#             n_ts = length(classes)
#             for j in 1:n_ts
#                 class = classes[j]
#                 sample = sample_idxs[j]

#                 stats, _ = any_interpolate_single_timeseries(fc, class, sample, heuristic.interp_sites, :directMode; NN_baseline=false, X_train=Xs_train, y_train=ys_train,  plot_fits=false, mode_range=mode_range, xvals=xvals, mode_index=mode_index, xvals_enc=xvals_enc, xvals_enc_it=xvals_enc_it);
#                 f_per_fold[fold] += stats[:MAE]
#             end
#             f_per_fold[fold] /= n_ts
#             # p.counter += 1
#             # ProgressMeter.updateProgress!(p)
#         end
    
#         e.per_fold[1] .= f_per_fold

#         h.per_fold[1] .= f_per_fold
#         for u in 2:length(e.per_fold)
#             e.per_fold[u] .= zeros(nfolds)
#             h.per_fold[u] .= zeros(nfolds)
#         end
        
#         mf = mean(f_per_fold)
#         @show f_per_fold
#         e.measurement[1] = mf
#         h.measurement[1] = mf

#         means[hi] = mf
#     end
    
#     println("Done")
#     return means
# end

# function MLJTuning.best(heuristic::InterpolationMeasureHack, history)
#     measurements = MLJTuning.losses(heuristic, history)
#     best_index = argmin(measurements)
#     return history[best_index]
# end

# 
# any method that supports naiive selection will support this if used with an MPSClassifier
# MLJTuning.supports_heuristic(any, ::InterpolationMeasureHack) = MLJTuning.supports_heuristic(any, MLJTuning.NaiveSelection())
