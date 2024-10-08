include("../../MLJIntegration/MLJ_integration.jl")
using MLJParticleSwarmOptimization
using MLJ
using Tables
using JLD2
using StableRNGs

# load the original UCR ItalyPowerDemand dataset
dataloc = "Data/ecg200/datasets/ecg200.jld2";
f = jldopen(dataloc, "r")
    X_train_f = read(f, "X_train")
    y_train_f = read(f, "y_train")
    X_test_f = read(f, "X_test")
    y_test_f = read(f, "y_test")
close(f)

X_train = MLJ.table(X_train_f)
y_train = coerce(y_train_f, OrderedFactor)
X_test = MLJ.table(X_test_f)
y_test = coerce(y_test_f, OrderedFactor)

# Combined train and test splits for resampling
Xs = MLJ.table([X_train_f; X_test_f])
ys = coerce([y_train_f; y_test_f], OrderedFactor)

# set params
Rdtype = Float64
encoding = :Legendre_No_Norm
verbosity = 0
test_run = false
track_cost = false
encode_classes_separately = false
train_classes_separately = false
exit_early = false

nsweeps = 1
chi_max = 10
eta = 0.1
d = 3

mps = MPSClassifier(nsweeps=nsweeps, chi_max=chi_max, eta=eta, d=d, encoding=encoding, 
    exit_early=exit_early, init_rng=4567);
r_eta = MLJ.range(mps, :eta, lower=0.001, upper=0.1, scale=:log);
r_d = MLJ.range(mps, :d, lower=2, upper=3)
r_chi = MLJ.range(mps, :chi_max, lower=3, upper=4) 

train_ratio = length(y_train)/length(ys)
num_resamps = 29
splits = [
    if i == 0
        (collect(1:length(y_train)), collect(length(y_train)+1:length(ys)))   
    else
        MLJ.partition(1:length(ys), train_ratio, rng=StableRNG(i), stratify=ys) 
    end 
    for i in 0:num_resamps]

# run hyperparameter optimisation on each fold, then evaluate
per_fold_accs = Vector{Float64}(undef, 30);
per_fold_best_model = Vector{Dict}(undef, 30);
for i in eachindex(splits)
    println("RUNNING FOLD $(i)")
    aps = AdaptiveParticleSwarm(rng=StableRNG(42))
    self_tuning_mps = TunedModel(
        model = mps,
        resampling = StratifiedCV(nfolds=5, rng=StableRNG(1)), 
        tuning = aps,
        range = [r_eta, r_chi, r_d],
        measure=MLJ.misclassification_rate,
        n = 1,
        acceleration=CPUThreads()
    )
    train_idxs = splits[i][1]
    X_train_fold = MLJ.table(Tables.matrix(Xs)[train_idxs, :])
    y_train_fold = ys[train_idxs]

    test_idxs = splits[i][2]
    X_test_fold = MLJ.table(Tables.matrix(Xs)[test_idxs, :])
    y_test_fold = ys[test_idxs]

    mach = machine(self_tuning_mps, X_train_fold, y_train_fold)
    MLJ.fit!(mach)

    # get the best performing model 
    best_model = report(mach).best_model
    mach_best = machine(best_model, X_train_fold, y_train_fold)
    MLJ.fit!(mach_best)
    y_preds = MLJ.predict(mach_best, X_test_fold)
    acc = MLJ.accuracy(y_preds, y_test_fold)
    println("FOLD $i ACC: $acc")
    # extract info
    m = mach_best.model
    info = Dict(
        "d" => m.d,
        "chi_max" => m.chi_max,
        "eta" => m.eta
    )
    per_fold_accs[i] = acc 
    per_fold_best_model[i] = info
end

# jldopen("ECG200Bench_cluster.jld2", "w") do f
#     f["per_fold_accs"] = per_fold_accs
#     f["per_fold_best_model"] = per_fold_best_model
# end
