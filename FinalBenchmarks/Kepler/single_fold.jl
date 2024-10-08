include("../../MLJIntegration/MLJ_integration.jl")
using JLD2
using Plots
using Plots.PlotMeasures
using StatsBase
using Tables
using MLJParticleSwarmOptimization
using PrettyTables
using StableRNGs
import StatisticalMeasures.ConfusionMatrices as CM


f = jldopen("Data/NASA_kepler/datasets/KeplerLightCurveOrig.jld2", "r");
X_train_f = read(f, "X_train")
y_train_f = read(f, "y_train");
X_test_f = read(f, "X_test")
y_test_f = read(f, "y_test");
close(f)

function class_distribution(y_train::Vector{Int}, y_test::Vector{Int})
    train_counts = countmap(y_train)
    test_counts = countmap(y_test)
    tr_classes, tr_vals = collect(keys(train_counts)), collect(values(train_counts))
    te_classes, te_vals = collect(keys(test_counts)), collect(values(test_counts))

    # compute distribution stats
    tr_dist = tr_vals ./ sum(tr_vals)
    te_dist = te_vals ./ sum(te_vals)
    # compute chance level acc
    chance_acc = sum(te_dist.^2)
    println("Distribution adjusted chance accuracy: $(round(chance_acc; digits=4))")

    header = (
        ["Class", "Train", "Test"]
    )
    t = pretty_table(hcat(tr_classes, tr_dist, te_dist); header=header)

    p_train = bar(tr_classes, tr_vals, 
        xlabel="Class", ylabel="Count", title="Train Set",
        c=:lightsteelblue, label="")
    p_test = bar(te_classes, te_vals, 
        xlabel="Class", ylabel="Count", title="Test Set",
        c=:red, label="")
    p = plot(p_train, p_test, size=(1200, 300), bottom_margin=5mm, left_margin=5mm)
    display(p)

end

function plot_examples(class::Int, X::Matrix{Float64}, y::Vector{Int};
    nplot=10, seed=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    pal = palette(:tab10)
    c_idxs = findall(x -> x .== class, y)
    p_idxs = sample(c_idxs, nplot; replace=false)
    ps = [plot(X[idx, :], xlabel="t", ylabel="x", label="", c=pal[class+1]) for idx in p_idxs]
    p = plot(ps..., size=(1000, 500), bottom_margin=5mm, left_margin=5mm, title="C$class")
    display(p)
end

function plot_conf_mat(yhat, y, mach; normalise=false)
    model = mach.model
    eta = model.eta
    chi = model.chi_max
    seed = model.init_rng
    d = model.d 
    # infer the data length
    T = size(mach.data[1], 1)
    cm = CM.confmat(yhat, y);
    confmat = Float64.(CM.matrix(cm));
    if normalise
        # divide each row by row sum to get proportions
        confmat ./= sum(confmat, dims=2)[:, 1]
    end
    reversed_confmat = reverse(confmat, dims=1)
    hmap = heatmap(reversed_confmat,
        color=:Blues,
        xticks=(1:size(confmat,2), ["$n" for n in 0:(size(confmat,2) - 1)]),
        yticks=(1:size(confmat,1), reverse(["$n" for n in 0:(size(confmat,1) - 1)]) ),
        xlabel="Predicted Class",
        ylabel="Actual Class",
        title="Confusion Matrix, η=$(round(eta; digits=3)), χ=$chi, \nd=$d, T=$T, seed=$seed")
        
    for (i, row) in enumerate(eachrow(reversed_confmat))
        for (j, value) in enumerate(row)
            
            annotate!(j, i, text(string(round(value; digits=3)), :center, 10))
        end
    end

    display(hmap)
end

function make_binary_classification(X_train_original::Matrix, y_train_original::Vector, 
    X_test_original::Matrix, y_test_original::Vector, class_a::Int, class_b::Int)
    # takes in the original dataset and returns new dataset containing only two classes
    # map class a to 0 and class b to 1
    if (class_a ∉ unique(y_train_f))
        error("Invalid class a")
    elseif (class_b ∉ unique(y_train_f))
        error("Invalid class b")
    end

    # for train samples 
    class_a_tr_idxs = findall(x -> x .== class_a, y_train_original)
    x_train_a = X_train_original[class_a_tr_idxs, :]
    y_train_a = zeros(eltype(y_train_original), length(class_a_tr_idxs))
    class_b_tr_idxs = findall(x -> x .== class_b, y_train_original)
    x_train_b = X_train_original[class_b_tr_idxs, :]
    y_train_b = ones(eltype(y_train_original), length(class_b_tr_idxs))


    X_train = vcat(x_train_a, x_train_b)
    y_train = vcat(y_train_a, y_train_b)

    # now for test samples
    class_a_te_idxs = findall(x -> x .== class_a, y_test_original)
    x_test_a = X_test_original[class_a_te_idxs, :]
    y_test_a = zeros(eltype(y_test_original), length(class_a_te_idxs))
    class_b_te_idxs = findall(x -> x .== class_b, y_test_original)
    x_test_b = X_test_original[class_b_te_idxs, :]
    y_test_b = ones(eltype(y_test_original), length(class_b_te_idxs))

    X_test = vcat(x_test_a, x_test_b)
    y_test = vcat(y_test_a, y_test_b)

    return X_train, y_train, X_test, y_test

end

function plot_incorrectly_labelled(X_test, y_test, y_preds; zero_indexing::Bool=false)
    """Function to plot the time-series that were incorrectly labelled"""
    # get idxs of incorrectly classified instances
    # use zero indexing to match python outputs for direct comparison
    incorrect = y_test .!= y_preds
    incorrect_idxs = findall(x -> x .== 1, incorrect)
    X_test_mat = Tables.matrix(X_test)
    incorrect_ts = X_test_mat[incorrect_idxs, :]
    ps = []
    for i in 1:(size(incorrect_ts, 1))
        color = y_test[incorrect_idxs[i]] == 0 ? :orange : :blue
        pi = plot(incorrect_ts[i, :], 
            xlabel="t", ylabel="x", 
            title="Actual: $(y_test[incorrect_idxs[i]]), Pred: $(y_preds[incorrect_idxs[i]]), idx: $(incorrect_idxs[i])",
            c=color, label="")
        push!(ps, pi)
    end
    p = plot(ps..., size=(2000, 1200), bottom_margin=5mm, left_margin=5mm)
    display(p)
end

w = 1:100

X_train_sub, y_train_sub, X_test_sub, y_test_sub = make_binary_classification(X_train_f, y_train_f, X_test_f, y_test_f, 2, 4)

X_train = MLJ.table(X_train_sub[:, w])
X_test = MLJ.table(X_test_sub[:, w])
y_train = coerce(y_train_sub, OrderedFactor)
y_test = coerce(y_test_sub, OrderedFactor)

Xs = MLJ.table([X_train_f[:, w]; X_test_f[:, w]])
ys = coerce([y_train_f; y_test_f], OrderedFactor)

exit_early=false

nsweeps=3
chi_max=50
eta=1.0
d=4

mps = MPSClassifier(nsweeps=nsweeps, chi_max=chi_max, eta=eta, d=d, encoding=:Fourier, 
    exit_early=exit_early, init_rng=1234)
mach = machine(mps, X_train, y_train)
MLJ.fit!(mach)
yhat = MLJ.predict(mach, X_test)

@show MLJ.accuracy(yhat, y_test)
MLJ.balanced_accuracy(yhat, y_test)
# get the confusion matrix
plot_conf_mat(yhat, y_test, mach; normalise=false)


############### Do some hyperparameter optimisation ###############
base_mps = MPSClassifier(nsweeps=3, chi_max=30, eta=0.1, d=6, encoding=:Legendre_No_Norm, 
    exit_early=false, init_rng = 9645)

r_eta = MLJ.range(base_mps, :eta, values=[0.1, 0.5, 1.0]);
r_d = MLJ.range(base_mps, :d, values=[3, 4, 5])
r_chi = MLJ.range(base_mps, :chi_max, values=[30, 40, 50, 60])
    
swarm = AdaptiveParticleSwarm(rng=MersenneTwister(0)) 
self_tuning_mps = TunedModel(
        model=base_mps,
        resampling=StratifiedCV(nfolds=5, rng=MersenneTwister(1)),
        tuning=swarm,
        range=[r_eta, r_chi],
        measure=MLJ.misclassification_rate,
        n=9,
        acceleration=CPUThreads()
    );
mach = machine(self_tuning_mps, X_train, y_train)
MLJ.fit!(mach)
@show report(mach).best_model
best = report(mach).best_model
mach = machine(best, X_train, y_train)
MLJ.fit!(mach)
yhat = MLJ.predict(mach, X_test)
@show MLJ.accuracy(yhat, y_test)
@show MLJ.balanced_accuracy(yhat, y_test)

# plot conf mat
plot_conf_mat(yhat, y_test, mach; normalise=true)
