include("../../../LogLoss/RealRealHighDimension.jl")
include("../../../Interpolation/ForecastingMainNew.jl");
using JLD2
using Plots
using DelimitedFiles

# load the original ECG200 split 
dloc = "Data/ecg200/datasets/ecg200.jld2"
f = jldopen(dloc, "r")
    X_train = read(f, "X_train")
    y_train = read(f, "y_train")
    X_test = read(f, "X_test")
    y_test = read(f, "y_test")
close(f)

# recombine the original train/test splits
Xs = vcat(X_train, X_test)
ys = vcat(y_train, y_test)

# load the resample indices
rs_f = jldopen("FinalBenchmarks/ECG200/Julia/resample_folds_julia_idx.jld2", "r");
rs_fold_idxs = read(rs_f, "rs_folds_julia");
close(rs_f)

# check that the first resample fold corresp. to the original split
f0_idxs_tr = rs_fold_idxs[0]["train"]
f0_idxs_te = rs_fold_idxs[0]["test"]
X_train_check = Xs[f0_idxs_tr, :]
if any(i -> i != 1, X_train .== X_train_check)
    error("X_train 0th fold does not correspond to original split")
end
y_train_check = ys[f0_idxs_tr]
if any(i -> i != 1, y_train .== y_train_check)
    error("y_train 0th fold does not correspond to original split")
end
X_test_check = Xs[f0_idxs_te, :]
if any(i -> i != 1, X_test .== X_test_check)
    error("X_test 0th fold does not correspond to original split")
end
y_test_check = ys[f0_idxs_te]
if any(i -> i != 1, y_test .== y_test_check)
    error("y_test 0th fold does not correspond to original split")
end

# load the window indices
windows_f = jldopen("FinalBenchmarks/ECG200/Julia/windows_julia_idx.jld2", "r");
window_idxs = read(windows_f, "windows_julia")
close(windows_f)

# test a window
# p = plot(X_train_check[1, :]);
# scatter!(window_idxs[55][1], X_train_check[1, window_idxs[55][1]]);
# display(p)

# set up the MPS hyperparameters
setprecision(BigFloat, 128)
Rdtype = Float64

# training related stuff
verbosity = 0
test_run = false
track_cost = false
encoding = :legendre_no_norm
encode_classes_separately = false
train_classes_separately = false

d = 10
chi_max=20
nsweeps = 5
eta = 0.5
opts=MPSOptions(; nsweeps=nsweeps, chi_max=chi_max,  update_iters=1, verbosity=verbosity, loss_grad=:KLD,
    bbopt=:TSGO, track_cost=track_cost, eta=eta, rescale = (false, true), d=d, aux_basis_dim=2, encoding=encoding, 
    encode_classes_separately=encode_classes_separately, train_classes_separately=train_classes_separately, 
    exit_early=false, sigmoid_transform=false, init_rng=4567, chi_init=4)

W, _, _, _ = fitMPS(X_train, y_train, X_test, y_test, chi_init=4, opts=opts, test_run=false)
fc = load_forecasting_info_variables(W, X_train, y_train, X_test, y_test, opts; verbosity=0)


interp_sites = collect(20:80)
fstyle = font("sans-serif", 23)
stats, p1_ns = any_interpolate_median(fc, 1, 1, interp_sites; NN_baseline=true, 
                X_train=X_train, y_train=y_train, n_baselines=1, plot_fits=true, wmad=true)
plot(p1_ns..., xtickfont=fstyle,ytickfont=fstyle,guidefont=fstyle,titlefont=fstyle,bottom_margin=10mm, left_margin=10mm,xlabel="t")

# main loop -> train -> impute -> train ...
function train_and_impute()
    per_fold_mps = []
    per_fold_nn = []
    for fold_idx in 0:(length(rs_fold_idxs)-1)
        println("Evaluating fold $fold_idx/$((length(rs_fold_idxs)-1))...")
        # step 1 -> extract the training and test sets to train on
        fold_train_idxs = rs_fold_idxs[fold_idx]["train"]
        fold_test_idxs = rs_fold_idxs[fold_idx]["test"]
        X_train_fold = Xs[fold_train_idxs, :]
        y_train_fold = ys[fold_train_idxs]
        X_test_fold = Xs[fold_test_idxs, :]
        y_test_fold = ys[fold_test_idxs]
        # step 2 -> train the MPS on the train and test split 
        W, _, _, _ = fitMPS(X_train_fold, y_train_fold, X_test_fold, y_test_fold; chi_init=4, opts=opts, test_run=false)
        # step 3 -> impute missing data
        fc = load_forecasting_info_variables(W, X_train_fold, y_train_fold, X_test_fold, y_test_fold, opts; verbosity=0)
        println("Finished training, beginning evaluation of imputated values...")
        #interp_sites = collect(20:30)
        # compute over entire dataset
        # loop over each class
        samps_per_class = [size(f.test_samples, 1) for f in fc]
        per_instance_window_scores_mps = []
        per_instance_window_scores_nn = []
        for (i, s) in enumerate(samps_per_class)
            println("Evaluating class $i instances...")
            for inst in 1:s
                # loop over percentage missing
                per_pm_scores_mps = []
                per_pm_scores_nn = []
                for pm in 5:10:95
                    # loop over iterations
                    num_wins = length(window_idxs[pm])
                    per_pm_iter_scores_mps = Vector{Float64}(undef, num_wins)
                    per_pm_iter_scores_nn = Vector{Float64}(undef, num_wins)
                    # thread this part if low d and chi? 
                    @threads for it in 1:num_wins
                        interp_sites = window_idxs[pm][it]
                        stats, _ = any_interpolate_median(fc, (i-1), inst, interp_sites; NN_baseline=true, 
                            X_train=X_train_fold, y_train=y_train_fold, n_baselines=1, plot_fits=false, wmad=true)
                        per_pm_iter_scores_mps[it] = stats[:MAE]
                        per_pm_iter_scores_nn[it] = stats[:NN_MAE]
                    end
                    push!(per_pm_scores_mps, per_pm_iter_scores_mps)
                    push!(per_pm_scores_nn, per_pm_iter_scores_nn)
                end
                push!(per_instance_window_scores_mps, per_pm_scores_mps)
                push!(per_instance_window_scores_nn, per_pm_scores_nn)
            end
        end
        push!(per_fold_mps, per_instance_window_scores_mps)
        push!(per_fold_nn, per_instance_window_scores_nn)
    end
    return per_fold_mps, per_fold_nn
end

# per fold, per instance, per percentage missing, per window
#per_fold_mps, per_fold_nn = train_and_impute()

# f = jldopen("FinalBenchmarks/ECG200/Julia/ecg_benchmark_trial.jld2", "r")
# per_fold_mps = read(f, "per_fold_mps")
# per_fold_nn = read(f, "per_fold_nn")

# # mean across all instances for 5 % missingness
# mean_per_fold_5pt_nn = [mean([per_fold_nn[f][inst][1][w] for inst in 1:100 for w in 1:15]) for f in 1:30]
# mean_per_fold_5pt_mps = [mean([per_fold_mps[f][inst][1][w] for inst in 1:100 for w in 1:15]) for f in 1:30]
# mean_per_fold_95pt_nn = [mean([per_fold_nn[f][inst][10][w] for inst in 1:100 for w in 1:5]) for f in 1:30]
# mean_per_fold_95pt_mps = [mean([per_fold_mps[f][inst][10][w] for inst in 1:100 for w in 1:5]) for f in 1:30]

# # mean across all folds for different % missingness
# mean_per_percent_missing_nn = [mean([mean([per_fold_nn[f][inst][pm][w] for inst in 1:100 for w in 1:length(per_fold_nn[f][inst][pm])]) for f in 1:30]) for pm in 1:10]
# mean_per_percent_missing_mps = [mean([mean([per_fold_mps[f][inst][pm][w] for inst in 1:100 for w in 1:length(per_fold_mps[f][inst][pm])]) for f in 1:30]) for pm in 1:10]
# # add standard error
# std_per_percent_mising_nn = [std([mean([per_fold_nn[f][inst][pm][w] for inst in 1:100 for w in 1:length(per_fold_nn[f][inst][pm])]) for f in 1:30])/sqrt(30) for pm in 1:10]
# std_per_percent_mising_mps = [std([mean([per_fold_mps[f][inst][pm][w] for inst in 1:100 for w in 1:length(per_fold_mps[f][inst][pm])]) for f in 1:30])/sqrt(30) for pm in 1:10]
# # 95 CI
# ci95_per_percent_missing_nn = [1.96 * std([mean([per_fold_nn[f][inst][pm][w] for inst in 1:100 for w in 1:length(per_fold_nn[f][inst][pm])]) for f in 1:30])/sqrt(30) for pm in 1:10]
# ci95_per_percent_mising_mps = [1.96 * std([mean([per_fold_mps[f][inst][pm][w] for inst in 1:100 for w in 1:length(per_fold_mps[f][inst][pm])]) for f in 1:30])/sqrt(30) for pm in 1:10]

# groupedbar([mean_per_percent_missing_nn mean_per_percent_missing_mps],
#     xticks=([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
#         ["5", "15", "25", "35", "45", "55", "65", "75", "85", "95"]),
#     yerr=[ci95_per_percent_missing_nn ci95_per_percent_mising_mps],
#     labels=["1NN" "MPS"],
#     title="30-Fold Averaged ECG200",
#     xlabel = "% missing",
#     ylabel="Mean MAE",
#     legend=:topright
# );
# xflip!(true)