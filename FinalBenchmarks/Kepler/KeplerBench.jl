include("../../MLJIntegration/MLJ_integration.jl");
using MLJParticleSwarmOptimization
using MLJ
using StableRNGs
using JLD2
using Tables 

# load the modified dataset
dataloc = "Data/NASA_kepler/datasets/KeplerLightCurves_C2_C4.jld2";
w = 1:100 # SET TRUNCATION LENGTH
f = jldopen(dataloc, "r")
    X_train_f = read(f, "X_train")[:, w]
    y_train_f = read(f, "y_train")
    X_test_f = read(f, "X_test")[:, w]
    y_test_f = read(f, "y_test")
close(f)

X_train = MLJ.table(X_train_f)
y_train = coerce(y_train_f, OrderedFactor)
X_test = MLJ.table(X_test_f)
y_test = coerce(y_test_f, OrderedFactor)

# Combined train and test splits for resampling
Xs = MLJ.table([X_train_f; X_test_f])
ys = coerce([y_train_f; y_test_f], OrderedFactor)

# set the base mps
mps = MPSClassifier(nsweeps=2, chi_max=50, eta=0.5, d=3, encoding=:Legendre_No_Norm, 
    exit_early=false, init_rng=9645);

# set the hyperparameter search ranges
r_eta = MLJ.range(mps, :eta, lower=0.1, upper=0.5);
r_d = MLJ.range(mps, :d, lower=3, upper=6)
r_chi = MLJ.range(mps, :chi_max, lower=40, upper=80) 
swarm = AdaptiveParticleSwarm(rng=MersenneTwister(0))
self_tuning_mps = TunedModel(
        model=mps,
        resampling=StratifiedCV(nfolds=5, rng=MersenneTwister(0)),
        tuning=swarm,
        range=[r_eta, r_chi, r_d],
        measure=MLJ.misclassification_rate,
        n=20,
        acceleration=CPUThreads()
    );
train_ratio = length(y_train)/length(ys)
num_resamps = 29
splits = [
    if i == 0
        (collect(1:length(y_train)), collect(length(y_train)+1:length(ys)))   
    else
        MLJ.partition(1:length(ys), train_ratio, rng=StableRNG(i), stratify=ys) 
    end 
    for i in 0:num_resamps]

per_fold_accs = Vector{Float64}(undef, length(splits));
per_fold_bal_accs = Vector{Float64}(undef, length(splits));
best_models = []
for i in eachindex(splits)
    train_idxs = splits[i][1]
    X_train_fold = MLJ.table(Tables.matrix(Xs)[train_idxs, :])
    y_train_fold = ys[train_idxs]

    test_idxs = splits[i][2]
    X_test_fold = MLJ.table(Tables.matrix(Xs)[test_idxs, :])
    y_test_fold = ys[test_idxs]
    mach = machine(self_tuning_mps, X_train_fold, y_train_fold)
    MLJ.fit!(mach)
    best = report(mach).best_model
    mach_best = machine(best, X_train_fold, y_train_fold)
    MLJ.fit!(mach_best)
    yhat = MLJ.predict(mach_best, X_test_fold)
    acc = MLJ.accuracy(yhat, y_test_fold)
    bal_acc = MLJ.balanced_accuracy(yhat, y_test_fold)
    per_fold_accs[i] = acc
    per_fold_bal_accs[i] = bal_acc
    push!(best_models, mach_best)
    println("Fold $i, Acc: $acc, Bal. Acc.: $bal_acc")
end
