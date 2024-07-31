using ITensors
using Random
using Distributions
using Folds
using StatsBase
using Plots
using MLBase
using PrettyTables


function contractMPS(W::MPS, PS::PState)
        N_sites = length(W)
        res = 1
        for i=1:N_sites
            res *= W[i] * conj(PS.pstate[i])
        end

        return res 

end



function MSE_loss_acc_iter(W::MPS, PS::PState)
    """For a given sample, compute the Quadratic Cost and whether or not
    the corresponding prediction (using argmax on deicision func. output) is
    correctly classfified"""
    label_index = PS.label_index # ground truth label
    pos, label_idx = find_label(W)
    y = onehot(label_idx => label_index)


    yhat = contractMPS(W, PS)
    
    diff_sq = abs2.(array(yhat - y))
    sum_of_sq_diff = real(sum(diff_sq))

    loss = 0.5 * sum_of_sq_diff

    # now get the predicted label
    correct = 0
    
    if (argmax(abs.(vector(yhat))) - 1) == PS.label
        correct = 1
    end

    return [loss, correct]

end

function MSE_loss_acc(W::MPS, PSs::TimeseriesIterable)
    """Compute the MSE loss and accuracy for an entire dataset"""
    loss, acc = Folds.reduce(+, MSE_loss_acc_iter(W, PS) for PS in PSs)
    loss /= length(PSs)
    acc /= length(PSs)

    return loss, acc 

end

function MSE_loss_acc_conf_iter(W::MPS, PS::PState)
    """For a given sample, compute the Quadratic Cost and whether or not
    the corresponding prediction (using argmax on decision func. output) is
    correctly classfified"""
    label_index = PS.label_index # ground truth label
    pos, label_idx = find_label(W)
    y = onehot(label_idx => label_index)


    yhat = contractMPS(W, PS)
    
    diff_sq = abs2.(array(yhat - y))
    sum_of_sq_diff = real(sum(diff_sq))

    loss = 0.5 * sum_of_sq_diff

    # now get the predicted label
    correct = 0
    pred = argmax(abs.(vector(yhat))) - 1
    if pred == PS.label
        correct = 1
    end

    nc = ITensors.dim(label_idx)
    conf = zeros(Int64,nc,nc)
    conf[PS.label_index, pred+1] = 1

    return [loss, correct, conf]

end

function MSE_loss_acc_conf(W::MPS, PSs::TimeseriesIterable)
    loss, acc, conf = Folds.reduce(+, MSE_loss_acc_conf_iter(W, PS) for PS in PSs)
    loss /= length(PSs)
    acc /= length(PSs)

    return loss, acc, conf

end

function get_predictions(Ws::Vector{MPS}, pss::TimeseriesIterable)
    # mps0 overlaps with ORIGINAL class 0 and mps1 overlaps with ORIGINAL class 1
    @assert all(length(Ws[1]) .== length.(Ws)) "MPS lengths do not match!"

    preds = Vector{Int64}(undef, length(pss))
    all_overlaps = Vector{Vector{Float64}}(undef, length(pss))
    for i in eachindex(pss)
        psc = conj(pss[i].pstate)
        overlaps = [ITensor(1) for _ in Ws]
        for (wi,w) in enumerate(Ws), j in eachindex(Ws[1])
            overlaps[wi] *= w[j] * psc[j]
        end
        overlaps = abs.(first.(overlaps))
        pred = argmax(overlaps) - 1
        preds[i] = pred

        all_overlaps[i] = overlaps
    end

    # return overlaps as well for inspection
    return preds, all_overlaps
        
end


function overlap_confmat(mps0::MPS, mps1::MPS, pstates::TimeseriesIterable; plot=false)
    """(2 CLASSES ONLY) Something like a confusion matrix but for median overlaps.
    Here, mps0 is the mps which overlaps with class 0 and mps1 overlaps w/ class 1"""
    gt_class_0_idxs = [ps.label_index .== 1 for ps in pstates]
    gt_class_1_idxs = [ps.label_index .== 2 for ps in pstates]
    # gt class 0, overlap with mps0, we will call this a true negative
    gt_0_mps_0 = [get_overlap(mps0, ps) for ps in pstates[gt_class_0_idxs]]
    # gt class 0, overlaps with mps1, false positive
    gt_0_mps_1 = [get_overlap(mps1, ps) for ps in pstates[gt_class_0_idxs]]
    # gt class 1, overlap with mps0, false negative
    gt_1_mps_0 = [get_overlap(mps0, ps) for ps in pstates[gt_class_1_idxs]]
    # gt class 1, overlaps with mps1, true positive
    gt_1_mps_1 = [get_overlap(mps1, ps) for ps in pstates[gt_class_1_idxs]]

    # get medians
    gt_0_mps_0_median = median(gt_0_mps_0)
    gt_0_mps_1_median = median(gt_0_mps_1)
    gt_1_mps_0_median = median(gt_1_mps_0)
    gt_1_mps_1_median = median(gt_1_mps_1)
    confmat = [gt_0_mps_0_median gt_0_mps_1_median; gt_1_mps_0_median gt_1_mps_1_median]

    # dictionary of stats
    #⟨ps|mps⟩
    stats = Dict(
        "Min/Max ⟨0|0⟩" => (minimum(gt_0_mps_0), maximum(gt_0_mps_0)),
        "Min/Max ⟨1|0⟩" => (minimum(gt_1_mps_0), maximum(gt_1_mps_0)),
        "Min/Max ⟨0|1⟩" => (minimum(gt_0_mps_1), maximum(gt_0_mps_1)),
        "Min/Max ⟨1|1⟩" => (minimum(gt_1_mps_1), maximum(gt_1_mps_1)),
        "MPS States Overlap ⟨1|0⟩" => abs(inner(mps0, mps1))
    )

    if plot
        reversed_confmat = reverse(confmat, dims=1)
        hmap = heatmap(reversed_confmat,
        color=:Blues,
        xticks=(1:size(confmat,2), ["Predicted 0", "Predicted 1"]),
        yticks=(1:size(confmat,1), ["Actual 1", "Actual 0"]),
        xlabel="Predicted class",
        ylabel="Actual class",
        title="Median Overlap Confusion Matrix")

        for (i, row) in enumerate(eachrow(reversed_confmat))
            for (j, value) in enumerate(row)
                
                annotate!(j, i, text(string(value), :center, 10))
            end
        end

        display(hmap)

    end

    return confmat, stats

end

function plot_conf_mat(confmat::Matrix)
    reversed_confmat = reverse(confmat, dims=1)
    hmap = heatmap(reversed_confmat,
        color=:Blues,
        xticks=(1:size(confmat,2), ["Predicted $n" for n in 0:(size(confmat,2) - 1)]),
        yticks=(1:size(confmat,1), reverse(["Actual n" for n in 0:(size(confmat,1) - 1)]) ),
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





function get_training_summary(mps::MPS, training_pss::TimeseriesIterable, testing_pss::TimeseriesIterable; print_stats=false,io::IO=stdin)
    # get final traing acc, final training loss

    Ws, l_ind = expand_label_index(mps)
    nclasses = length(Ws)

    preds_training, overlaps = get_predictions(Ws, training_pss)
    true_training = [x.label for x in training_pss] # get ground truths
    acc_training = sum(true_training .== preds_training)/length(training_pss)
    

    # get final testing acc
    preds_testing, overlaps = get_predictions(Ws, testing_pss)
    true_testing =  [x.label for x in testing_pss] # get ground truths

    # get overlap between mps classes
    overlapmat = Matrix{Float64}(undef, nclasses, nclasses)
    for i in eachindex(Ws), j in eachindex(Ws)
        overlapmat[i,j] = abs(dot(Ws[i], Ws[j])) # ITensor dot product conjugates the first argument
    end

    
    confmat = confusmat(nclasses, (true_testing .+ 1), (preds_testing .+ 1)) # need to offset labels becuase function expects labels to start at 1


    # NOTE CONFMAT IS R(i, j) == countnz((gt .== i) & (pred .== j)). So rows (i) are groudn truth and columns (j) are preds
    # tables 
    pretty_table(io,overlapmat;
                    title="Overlap Matrix",
                    title_alignment=:c,
                    title_same_width_as_table=true,
                    header = ["|ψ$n⟩" for n in 0:(nclasses-1)],
                    row_labels = ["⟨ψ$n|" for n in 0:(nclasses-1)],
                    alignment=:c,
                    body_hlines=Vector(1:nclasses),
                    highlighters = Highlighter(f      = (data, i, j) -> (i == j),
                    crayon = crayon"bold" ),
                    formatters = ft_printf("%5.3e"))

    pretty_table(io,confmat;
    title="Confusion Matrix",
    title_alignment=:c,
    title_same_width_as_table=true,
    header = ["Pred. |$n⟩" for n in 0:(nclasses-1)],
    row_labels = ["True |$n⟩" for n in 0:(nclasses-1)],
    body_hlines=Vector(1:nclasses),
    highlighters = Highlighter(f = (data, i, j) -> (i == j), crayon = crayon"bold green" ))


    # TP, TN, FP, FN FOR TEST SET 
    acc_testing = sum(true_testing .== preds_testing)/length(testing_pss)
    r = roc(true_testing, preds_testing)
    prec = precision(r)
    rec = recall(r)
    f1 = f1score(r)
    specificity = true_negative(r) / (true_negative(r) + false_positive(r))
    sensitivity = true_positive(r) / (true_positive(r) + false_negative(r))
    # balanced acc is arithmetic mean of sensitivy and specificity
    acc_balanced_testing = (sensitivity + specificity) / 2

    stats = Dict(
        :train_acc => acc_training,
        :test_acc => acc_testing,
        :test_balanced_acc => acc_balanced_testing,
        :precision => prec,
        :recall => rec,
        :specificity => specificity,
        :f1_score => f1,
        :confmat => confmat
    )

    if print_stats
        statsp = Dict(String(key)=>[stats[key]] for key in filter((s)->s!=:confmat,keys(stats)))
        pretty_table(io,statsp)
        # println("Testing Accuracy: $acc_testing")
        # println("Training Accuracy: $acc_training")
        # println("Precision: $prec")
        # println("Recall: $rec")
        # println("F1 Score: $f1")
        # println("Specificity: $specificity")
        # println("Sensitivity: $sensitivity")
        # println("Balanced Accuracy: $acc_balanced_testing")
    end
    
    return stats

end

function sweep_summary(info;io::IO=stdin)
    """Print a pretty summary of what happened in every sweep"""
    nsweeps = length(info["time_taken"]) - 2
    row_labels = ["Accuracy", "Test KL Div.", "Train KL Div.", "Test MSE", "Time taken"]
    header = vcat(["Initial"],["After Sweep $n" for n in 1:(nsweeps)], ["After Norm"], "Mean")

    data = Matrix{Float64}(undef, length(row_labels), nsweeps+3)

    for (i, key) in enumerate(["test_acc", "test_KL_div", "train_KL_div", "test_loss", "time_taken"])
        data[i,:] = vcat(info[key], [mean(info[key][2:end-1])])
    end

    h1 = Highlighter((data, i, j) -> j < length(header) && data[i, j] == maximum(data[i,1:(end-1)]),
                        bold       = true,
                        foreground = :red )

    h2 = Highlighter((data, i, j) -> j < length(header) && data[i, j] == minimum(data[i,1:(end-1)]),
    bold       = true,
    foreground = :blue )

    pretty_table(io,data;
    row_label_column_title = "",
    header = header,
    row_labels = row_labels,
    body_hlines = Vector(1:length(row_labels)),
    highlighters = (h1,h2),
    alignment=:c)
    #formatters = ft_printf("%.3e"))

end

function print_opts(opts::Options;io::IO=stdin)
    optsD = Dict(String(key)=>[getfield(opts, key)] for key in  fieldnames(Options))
    return pretty_table(io, optsD)
end


function KL_div(W::MPS, test_states::TimeseriesIterable)
    """Computes KL divergence of TS on MPS"""
    Ws, l_ind = expand_label_index(W)

    KLdiv = 0

    for x in test_states, l in eachval(l_ind)
        if x.label_index == l
            qlx = abs2(dot(x.pstate,Ws[l]))
            #qlx = l == 0 ? abs2(dot(x.pstate,W0)) : abs2(dot(x.pstate, W1))
            KLdiv +=  -log(qlx) # plx is 1
        end
    end
    return KLdiv / length(test_states)
end

function KL_div_old(W::MPS, test_states::TimeseriesIterable)
    """Computes KL divergence of TS on MPS, only works for a 2 category label index"""
    W0, W1, l_ind = expand_label_index(W)

    KLdiv = 0

    for x in test_states, l in 0:1
        if x.label == l
            qlx = l == 0 ? abs2(dot(x.pstate,W0)) : abs2(dot(x.pstate, W1))
            KLdiv +=  -log(qlx) # plx is 1
        end
    end
    return KLdiv / length(test_states)
end

function test_dot(W::MPS, test_states::TimeseriesIterable)
    Ws, l_ind = expand_label_index(W)

    outcomes = []
    for (i,ps) in enumerate(test_states)
        inns = [inner(ps.pstate, mps) for mps in Ws]
        cons = Vector(contractMPS(W, ps))

        if !all(isapprox.(inns,cons))
            push!(outcomes,i)
        end
    end
    return outcomes
end

