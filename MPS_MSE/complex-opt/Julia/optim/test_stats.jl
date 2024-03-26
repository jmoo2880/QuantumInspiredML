using Plots, Plots.PlotMeasures
using Distributions
using MLBase

# function get_predictions(mps0::MPS, mps1::MPS, pss::Vector{PState})
#     # mps0 overlaps with ORIGINAL class 0 and mps1 overlaps with ORIGINAL class 1
#     @assert length(mps0) == length(mps1) "MPS lengths do not match!"

#     preds = Vector{Int64}(undef, length(pss))
#     all_overlaps_mps0 = Vector{Float64}(undef, length(pss))
#     all_overlaps_mps1 = Vector{Float64}(undef, length(pss))
#     for i in eachindex(pss)
#         ps = pss[i].pstate
#         overlap_mps0 = 1
#         overlap_mps1 = 1
#         for j in eachindex(mps0)
#             overlap_mps0 *= mps0[j] * ps[j]
#             overlap_mps1 *= mps1[j] * ps[j]
#         end
#         overlap_mps0 = abs(overlap_mps0[])
#         overlap_mps1 = abs(overlap_mps1[])
#         pred = 0
#         if overlap_mps1 > overlap_mps0
#             pred = 1
#         end
#         all_overlaps_mps0[i] = overlap_mps0
#         all_overlaps_mps1[i] = overlap_mps1
#         preds[i] = pred
#     end

#     # return overlaps as well for inspection
#     return preds, all_overlaps_mps0, all_overlaps_mps1
        
# end

# function get_overlap(mps::MPS, product_state::PState)
#     """Get the overlap of a single product state with a single mps"""
#     res = 1
#     ps = product_state.pstate
#     for i in eachindex(mps)
#         res *= mps[i] * ps[i]
#     end
#     res = abs(res[])

#     return res

# end

# function overlap_confmat(mps0::MPS, mps1::MPS, pstates::Vector{PState}; plot=false)
#     """Something like a confusion matrix but for median overlaps.
#     Here, mps0 is the mps which overlaps with class 0 and mps1 overlaps w/ class 1"""
#     gt_class_0_idxs = [ps.label .== 0 for ps in pstates]
#     gt_class_1_idxs = [ps.label .== 1 for ps in pstates]
#     # gt class 0, overlap with mps0, we will call this a true negative
#     gt_0_mps_0 = [get_overlap(mps0, ps) for ps in pstates[gt_class_0_idxs]]
#     # gt class 0, overlaps with mps1, false positive
#     gt_0_mps_1 = [get_overlap(mps1, ps) for ps in pstates[gt_class_0_idxs]]
#     # gt class 1, overlap with mps0, false negative
#     gt_1_mps_0 = [get_overlap(mps0, ps) for ps in pstates[gt_class_1_idxs]]
#     # gt class 1, overlaps with mps1, true positive
#     gt_1_mps_1 = [get_overlap(mps1, ps) for ps in pstates[gt_class_1_idxs]]

#     # get medians
#     gt_0_mps_0_median = median(gt_0_mps_0)
#     gt_0_mps_1_median = median(gt_0_mps_1)
#     gt_1_mps_0_median = median(gt_1_mps_0)
#     gt_1_mps_1_median = median(gt_1_mps_1)
#     confmat = [gt_0_mps_0_median gt_0_mps_1_median; gt_1_mps_0_median gt_1_mps_1_median]

#     # dictionary of stats
#     #⟨ps|mps⟩
#     stats = Dict(
#         "Min/Max ⟨0|0⟩" => (minimum(gt_0_mps_0), maximum(gt_0_mps_0)),
#         "Min/Max ⟨1|0⟩" => (minimum(gt_1_mps_0), maximum(gt_1_mps_0)),
#         "Min/Max ⟨0|1⟩" => (minimum(gt_0_mps_1), maximum(gt_0_mps_1)),
#         "Min/Max ⟨1|1⟩" => (minimum(gt_1_mps_1), maximum(gt_1_mps_1)),
#         "MPS States Overlap ⟨1|0⟩" => abs(inner(mps0, mps1))
#     )

#     if plot
#         reversed_confmat = reverse(confmat, dims=1)
#         hmap = heatmap(reversed_confmat,
#         color=:Blues,
#         xticks=(1:size(confmat,2), ["Predicted 0", "Predicted 1"]),
#         yticks=(1:size(confmat,1), ["Actual 1", "Actual 0"]),
#         xlabel="Predicted class",
#         ylabel="Actual class",
#         title="Median Overlap Confusion Matrix")

#         for (i, row) in enumerate(eachrow(reversed_confmat))
#             for (j, value) in enumerate(row)
                
#                 annotate!(j, i, text(string(value), :center, 10))
#             end
#         end

#         display(hmap)

#     end

#     return confmat, stats

# end

function plot_conf_mat(confmat::Matrix)
    reversed_confmat = reverse(confmat, dims=1)
    hmap = heatmap(reversed_confmat,
        color=:Blues,
        xticks=(1:size(confmat,2), ["Predicted 0", "Predicted 1"]),
        yticks=(1:size(confmat,1), ["Actual 1", "Actual 0"]),
        xlabel="Predicted class",
        ylabel="Actual class",
        title="Confusion Matrix")
        
    for (i, row) in enumerate(eachrow(reversed_confmat))
        for (j, value) in enumerate(row)
            
            annotate!(j, i, text(string(value), :center, 10))
        end
    end

    display(hmap)
end

# function get_training_summary(mps0::MPS, mps1::MPS, training_pss::Vector{PState}, testing_pss::Vector{PState})
#     # get final traing acc, final training loss
#     preds_training, overlaps_mps0_training, overlaps_mps1_training = get_predictions(mps0, mps1, training_pss)
#     true_training = [x.label for x in training_pss] # get ground truths
#     acc_training = sum(true_training .== preds_training)/length(training_pss)
#     println("Training Accuracy: $acc_training")

#     # get final testing acc
#     preds_testing, overlaps_mps0_testing, overlaps_mps1_testing = get_predictions(mps0, mps1, testing_pss)
#     true_testing =  [x.label for x in testing_pss] # get ground truths

#     # get overlap between class 0 mps and class 1 mps
#     overlap_mps_states = abs(inner(mps0, mps1))
#     println("Overlap between state 0 MPS and State 1 MPS ⟨ψ0|ψ1⟩ = $overlap_mps_states")

#     # TP, TN, FP, FN FOR TEST SET 
#     acc_testing = sum(true_testing .== preds_testing)/length(testing_pss)
#     println("Testing Accuracy: $acc_testing")
#     r = roc(true_testing, preds_testing)
#     prec = precision(r)
#     println("Precision: $prec")
#     rec = recall(r)
#     println("Recall: $rec")
#     f1 = f1score(r)
#     println("F1 Score: $f1")
#     specificity = true_negative(r) / (true_negative(r) + false_positive(r))
#     println("Specificity: $specificity")
#     sensitivity = true_positive(r) / (true_positive(r) + false_negative(r))
#     println("Sensitivity: $sensitivity")
#     # balanced acc is arithmetic mean of sensitivy and specificity
#     acc_balanced_testing = (sensitivity + specificity) / 2
#     confmat = confusmat(2, (true_testing .+ 1), (preds_testing .+ 1)) # need to offset labels becuase function expects labels to start at 1
#     println("Confusion Matrix: $confmat")
#     # NOTE CONFMAT IS R(i, j) == countnz((gt .== i) & (pred .== j)). So rows (i) are groudn truth and columns (j) are preds

#     stats = Dict(
#         :train_acc => acc_training,
#         :test_acc => acc_testing,
#         :test_balanced_acc => acc_balanced_testing,
#         :precision => prec,
#         :recall => rec,
#         :specificity => specificity,
#         :f1_score => f1,
#         :confmat => confmat
#     )

#     return stats

# end
