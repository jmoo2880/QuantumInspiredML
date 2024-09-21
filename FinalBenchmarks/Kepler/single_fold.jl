using JLD2
using Plots
using Plots.PlotMeasures
using StatsBase
using PrettyTables
include("../../MLJIntegration/MLJ_integration.jl")


f = jldopen("Data/NASA_kepler/datasets/KeplerLightCurveOrig.jld2", "r");
w = 500
X_train = read(f, "X_train")[:, 1:w];
y_train = read(f, "y_train");
X_test = read(f, "X_test")[:, 1:w];
y_test = read(f, "y_test");

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

# function inspect_sample()
# end
X_train = MLJ.table(X_train)
X_test = MLJ.table(X_test)
y_train = coerce(y_train, OrderedFactor)
y_test = coerce(y_test, OrderedFactor)


Rdtype = Float64

verbosity = 0
test_run = false
track_cost = false
exit_early=false

nsweeps=3
chi_max=50
eta=0.1
d=4

mps = MPSClassifier(nsweeps=nsweeps, chi_max=chi_max, eta=eta, d=d, encoding=:Legendre_No_Norm, 
    exit_early=exit_early, init_rng=4567, sigmoid_transform=false)
mach = machine(mps, X_train, y_train)
MLJ.fit!(mach)
yhat = MLJ.predict(mach, X_test)

@show MLJ.accuracy(yhat, y_test)
MLJ.balanced_accuracy(yhat, y_test)


yhat