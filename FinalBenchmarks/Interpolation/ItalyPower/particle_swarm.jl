using Distributed

if nprocs() < 30
    addprocs(31, exeflags="-t 1") # 1 thread per physical core  
end
w = workers()

# import relevant packages on all processes
@everywhere begin
    include("../../../MLJIntegration/MLJ_integration.jl")
    using LinearAlgebra
    BLAS.set_num_threads(1)
    using MLJParticleSwarmOptimization
    using MLJ
    using Tables
    using JLD2
    using StableRNGs
    using Base.Threads
    # try to make the annoying blue squiggles go away
    import MLJ: fit!, predict
    import MLJBase: range
    println("Worker $(myid()) has $(nthreads()) thread(s)!")
end

import ProgressMeter




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

####################################
##### Hyper training Parameters ####
num_resamps = 2# 30 # number of train/test splits
n_particles = 4
nCVfolds = 2# number of train/val CV folds within the training set
nCVevals = 1 # number of steps of particle swarm to do
train_interp=true
interp_sites= 5:15 |> collect # which sites to interpolate
max_samps=1#0 # the maximum number of test timeseries to interpolate/compute the MAE of
####################################
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



@everywhere function optimise_on_fold(train_idxs, test_idxs, Xs, ys)
    aps = AdaptiveParticleSwarm(rng=StableRNG(42), n_particles=n_particles)
    #lhs = LatinHypercube(rng=StableRNG(42))
    
    X_tr_mat = Tables.matrix(Xs)[train_idxs, :]
    X_train_fold = MLJ.table(X_tr_mat)
    y_train_fold = ys[train_idxs]
    local accs_per_model = Vector{Float64}(undef, n_particles)
    local models = Vector{MPSClassifier}(undef, n_particles)


    local self_tuning_mps = TunedModel(
        model = mps,
        resampling = StratifiedCV(nfolds=nCVfolds, rng=StableRNG(1)),
        tuning = aps,
        # selection_heuristic=heuristic,
        range = [r_d, r_eta, r_chi, r_sig],
        measure=MLJ.misclassification_rate,
        n = n_particles,
        # acceleration = CPUThreads(), # acceleration=CPUProcesses()
        compact_history=false,
        train_best=false
    )
    mach = machine(self_tuning_mps, X_train_fold, y_train_fold)

    fit!(mach)


    for j in 2:nCVevals
        if train_interp
            revise_history!(MMI.matrix(X_train_fold), Int.(int(y_train_fold)) .-1, interp_sites, max_samps, mach.report[:fit][:history][end-n_particles+1:end])
        end
        self_tuning_mps.n += n_particles
        fit!(mach)
    end

    # calculate measurements for last run
    if train_interp
        revise_history!(MMI.matrix(X_train_fold), Int.(int(y_train_fold)) .-1, interp_sites, max_samps, mach.report[:fit][:history][end-n_particles+1:end])
    end

    # save results
    rep = report(mach)

    for (i, split) in enumerate(rep.history)
        models[i] = split.model
        accs_per_model[i] = split.per_fold[1]

    end

    # get the best performing model 
    best_model = report(mach).best_model
    mach_best = machine(best_model, X_train_fold, y_train_fold)
    fit!(mach_best)


    ## Test
    X_te_mat = Tables.matrix(Xs)[test_idxs, :]
    X_test_fold = MLJ.table(X_te_mat)
    y_test_fold = ys[test_idxs]


    y_preds = predict(mach_best, X_test_fold)
    acc = accuracy(y_preds, y_test_fold)
    println("FOLD $i ACC: $acc")
    # extract info

    per_fold_best_model[i] = mach_best.model
    (best_model=best_model, accuracy=acc, mach=mach_best, models=models, accs_per_model=accs_per_model)

end

# progress bar
entries = @sync begin
    channel = RemoteChannel(()->Channel{Bool}(min(1000, num_resamps+1)), 1)
    p = ProgressMeter.Progress(length(splits),
        dt = 0,
        desc = "Folds",
        barglyphs = ProgressMeter.BarGlyphs("[=> ]"),
        barlen = length(splits),
        color = :orange
        )    

    # printing the progress bar
    begin
        update!(p,0)
        @async while take!(channel)
            p.counter +=1
            ProgressMeter.updateProgress!(p)
        end
    end
end


results = @distributed (vcat) for split in splits
    optimise_on_fold(split..., Xs, ys)
    put!(channel, true)
end
put!(channel, false)

best_models = [i.best_model for i in results]
accs = [i.accuracy for i in results]
best_machs = [i.mach for i in results]
accs_per_model = [i.accs_per_model for i in results]
models_all = [i.models for i in results]


jldopen("FinalBenchmarks/Interpolation/ItalyPower/ItalyPowerBench_Interp.jld2", "w") do f
    f["accs"] = accs
    f["per_fold_accs"] = per_fold_accs
    f["per_fold_best_model"] = per_fold_best_model
    f["accs_per_model"] = accs_per_model
    f["models_all"] = models_all
end




