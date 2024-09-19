using Distributed
addprocs(31, exeflags="-t 1") # launch 31 processes, each with one thread
w = workers()

# make the relevant imports on each worker
@everywhere begin
    import Pkg
    Pkg.activate(".")
    include("../../MLJIntegration/MLJ_integration.jl")
    using LinearAlgebra
    BLAS.set_num_threads(1)
    using MLJParticleSwarmOptimization
    using MLJ
    using Tables
    using JLD2
    using StableRNGs
    using Base.Threads
    println("Worker $(myid()) has $(nthreads()) thread(s)!")
end

# load the data on the master process
dataloc = "Data/italypower/datasets/ItalyPowerDemandOrig.jld2"
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

# set ranges
@everywhere begin
    # set params
    Rdtype = Float64
    encoding = :Legendre_No_Norm
    verbosity = 0
    test_run = false
    track_cost = false
    encode_classes_separately = false
    train_classes_separately = false
    exit_early = false
    nsweeps = 5
    chi_max = 10
    eta = 0.1
    d = 3
    mps = MPSClassifier(nsweeps=nsweeps, chi_max=chi_max, eta=eta, d=d, encoding=encoding, 
        exit_early=exit_early, init_rng=4567);
    r_eta = MLJ.range(mps, :eta, lower=0.001, upper=10.0, scale=:log);
    r_d = MLJ.range(mps, :d, lower=3, upper=12)
    r_chi = MLJ.range(mps, :chi_max, lower=15, upper=35) 
end

# make the splits
train_ratio = length(y_train)/length(ys)
num_resamps = 29
splits = [
    if i == 0
        (collect(1:length(y_train)), collect(length(y_train)+1:length(ys)))   
    else
        MLJ.partition(1:length(ys), train_ratio, rng=StableRNG(i), stratify=ys) 
    end 
    for i in 0:num_resamps]

@everywhere function optimise_on_fold(train_idxs, test_idxs, Xs, ys)
    aps = AdaptiveParticleSwarm(rng=StableRNG(42))
    self_tuning_mps = TunedModel(
        model = mps,
        resampling = StratifiedCV(nfolds=5, rng=StableRNG(1)),
        tuning = aps,
        range = [r_eta, r_chi, r_d],
        measure=MLJ.misclassification_rate,
        n = 25,
    )
    X_train_fold = MLJ.table(Tables.matrix(Xs)[train_idxs, :])
    y_train_fold = ys[train_idxs]
    X_test_fold = MLJ.table(Tables.matrix(Xs)[test_idxs, :])
    y_test_fold = ys[test_idxs]
    mach = machine(self_tuning_mps, X_train_fold, y_train_fold)
    MLJ.fit!(mach)
    best_model = report(mach).best_model
    mach_best = machine(best_model, X_train_fold, y_train_fold)
    MLJ.fit!(mach_best)
    
    y_preds = MLJ.predict(mach_best, X_test_fold)
    acc = MLJ.accuracy(y_preds, y_test_fold)
    return (best_model=best_model, accuracy=acc, mach=mach_best)
end

# # Distribute the work across workers
results = @distributed (vcat) for split in splits
    optimise_on_fold(split..., Xs, ys)
end

best_models = [i.best_model for i in results]
accs = [i.accuracy for i in results]
best_machs = [i.mach for i in results]

jldopen("IPD_distrib_res.jld2", "w") do f
    f["best_models"] = best_models
    f["accs"] = accs
    f["best_machs"] = best_machs
end

# clean up - remove workers
rmprocs(w)
