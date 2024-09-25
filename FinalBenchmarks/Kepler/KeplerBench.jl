include("../../MLJIntegration/MLJ_integration.jl");
using MLJParticleSwarmOptimization
using MLJ
using StableRNGs
using JLD2

# load the modified dataset
dataloc = "Data/NASA_kepler/datasets/KeplerLightCurves_C2_C4.jld2";
w = 1:200 # SET TRUNCATION LENGTH
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
mps = MPSClassifier(nsweeps=5, chi_max=10, eta=0.1, d=3, encoding=:Legendre_No_Norm, 
    exit_early=false, init_rng=4567);

# set the hyperparameter search ranges
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

per_fold_accs = Vector{Float64}(undef, 30);
