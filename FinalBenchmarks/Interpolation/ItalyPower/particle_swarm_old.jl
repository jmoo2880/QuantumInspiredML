include("../../../MLJIntegration/MLJ_integration.jl")
include("../../../MLJIntegration/interpolation_hyperopt_hack.jl")
using MLJParticleSwarmOptimization
using Tables
using JLD2
using StableRNGs

# try to make the annoying blue squiggles go away
import MLJ: fit!, predict
import MLJBase: range


# load the original UCR ItalyPowerDemand dataset
dataloc = "Data/italypower/datasets/ItalyPowerDemandOrig.jld2";
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
verbosity = -5
test_run = false
track_cost = false
encode_classes_separately = false
train_classes_separately = false
exit_early = false

nsweeps = 5
chi_max = 10
eta = 0.1
d = 3

##### Hyper training Parameters ####
num_resamps = 1
n_particles = 3
nCVfolds = 2
nCVevals = 3
train_interp=false
interp_sites= 5:15 |> collect
max_samps=1
####################################

mps = MPSClassifier(nsweeps=nsweeps, 
    chi_max=chi_max, 
    eta=eta, 
    d=d, 
    encoding=encoding, 
    exit_early=exit_early, 
    init_rng=4567,
    log_level=0);

r_eta = range(mps, :eta, lower=0.001, upper=10, scale=:log);
r_d = range(mps, :d, lower=2, upper=8)
r_chi = range(mps, :chi_max, lower=15, upper=20)
r_ts = range(mps, :train_classes_separately, values=[true,false])
r_sig = range(mps, :sigmoid_transform, values=[true,false])
r_enc = range(mps, :encoding, values=[:Legendre_No_Norm, :Fourier])

train_ratio = length(y_train)/length(ys)
splits = [
    if i == 1
        (collect(1:length(y_train)), collect(length(y_train)+1:length(ys)))   
    else
        partition(1:length(ys), train_ratio, rng=StableRNG(i), stratify=ys) 
    end 
    for i in 1:num_resamps]

# run hyperparameter optimisation on each fold, then evaluate
per_fold_accs = Vector{Float64}(undef, num_resamps);
per_fold_best_model = Vector{MPSClassifier}(undef, num_resamps);
accs = [Dict{MPSClassifier, Vector{Float64}}() for _ in 1:num_resamps]
tstart = time()
for i in eachindex(splits)
    println("$(round(time() - tstart,digits=2))s: RUNNING FOLD $(i)/$num_resamps")
    aps = AdaptiveParticleSwarm(rng=StableRNG(42), n_particles=n_particles)
    #lhs = LatinHypercube(rng=StableRNG(42))
    
    train_idxs = splits[i][1]
    X_tr_mat = Tables.matrix(Xs)[train_idxs, :]
    X_train_fold = MLJ.table(X_tr_mat)
    y_train_fold = ys[train_idxs]

    test_idxs = splits[i][2]
    X_te_mat = Tables.matrix(Xs)[test_idxs, :]
    X_test_fold = MLJ.table(X_te_mat)
    y_test_fold = ys[test_idxs]

    if train_interp
        heuristic = InterpolationMeasureHack(MMI.matrix(X_train_fold), Int.(int(y_train_fold)) .-1, interp_sites, max_samps)
    else
        heuristic = MLJTuning.NaiveSelection()
    end

    self_tuning_mps = TunedModel(
        model = mps,
        resampling = StratifiedCV(nfolds=nCVfolds, rng=StableRNG(1)),
        tuning = aps,
        # selection_heuristic=heuristic,
        range = [r_eta, r_chi], #[r_eta, r_chi, r_d, r_ts, r_enc],
        measure=MLJ.misclassification_rate,
        n = nCVevals,
        acceleration = CPUThreads(), # acceleration=CPUProcesses()
        compact_history=false
    )

    controlled = IteratedModel(self_tuning_mps,
                                resampling=nothing,
                                control=[IterationControl.louder(Step(n=1); by=100), NumberLimit(n=1)]
                            )
    mach = machine(controlled, X_train_fold, y_train_fold)
    fit!(mach)

    # save results
    rep = report(mach).model_report

    for split in rep.history
        accs[i][split.model] = split.per_fold[1]
    end

    # get the best performing model 
    best_model = report(mach).best_model
    mach_best = machine(best_model, X_train_fold, y_train_fold)
    fit!(mach_best)
    y_preds = predict(mach_best, X_test_fold)
    acc = accuracy(y_preds, y_test_fold)
    println("FOLD $i ACC: $acc")
    # extract info

    per_fold_accs[i] = acc 
    per_fold_best_model[i] = mach_best.model
end

jldopen("ItalyPowerBench_eta001:10_d2:4_chi15:30_LHS.jld2", "w") do f
    f["accs"] = accs
    f["per_fold_accs"] = per_fold_accs
    f["per_fold_best_model"] = per_fold_best_model
end




