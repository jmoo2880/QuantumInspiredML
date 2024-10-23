# script version of the imputation benchmarks notebook for running on the cluster 
import json 
import numpy as np
import matplotlib.pyplot as plt
from pypots.optim import Adam
from pypots.imputation import CSDI, BRITS
from pypots.utils.random import set_random_seed
from pypots.utils.metrics import calc_mae
import pickle
import sys
sys.path.append("Interpolation/Imputation_Algs")
from cdrec.python.recovery import centroid_recovery as CDrec
set_random_seed(1234)
# check that GPU acceleration is enabled
import torch
torch.cuda.device_count()
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"CUDA ENABLED: {torch.cuda.is_available()}")


def evaluate_folds_csdi(Xs, ys, fold_idxs, window_idxs, model):
    fold_scores = dict()
    for fold in range(0, 1):
        print(f"Evaluating fold {fold}/{len(fold_idxs)-1}...")
        # make the splits
        X_train_fold = Xs[fold_idxs[fold]["train"]]
        y_train_fold = ys[fold_idxs[fold]["train"]]
        X_test_fold = Xs[fold_idxs[fold]["test"]]
        y_test_fold = ys[fold_idxs[fold]["test"]]
        # check class distributions
        counts_tr = np.unique(y_train_fold, return_counts=True)[1]
        print(f"Training class distribution: {counts_tr/np.sum(counts_tr)}")
        counts_te = np.unique(y_test_fold, return_counts=True)[1]
        print(f"Testing class distribution: {counts_te/np.sum(counts_te)}")
        print(f"Training CSDI on fold {fold}...")
        model.fit(train_set={'X':X_train_fold})
        print("Finished training!")
        # loop over % missing
        percent_missing_score = dict()
        for pm in window_idxs:
            print(f"Imputing {pm}% missing data over {len(window_idxs[pm])} windows...")
            per_window_scores = dict()
            for (idx, widx) in enumerate(window_idxs[pm]):
                X_test_corrupted = X_test_fold.copy()
                X_test_corrupted[:, widx] = np.nan
                mask = np.isnan(X_test_corrupted) # mask ensures only misisng values are imputed
                csdi_imputed = model.impute(test_set={'X': X_test_corrupted}).squeeze(axis=1)
                errs = [calc_mae(csdi_imputed[i], X_test_fold[i], mask[i]) for i in range(0, X_test_fold.shape[0])] # get individual errors for uncertainty quantification
                per_window_scores[idx] = errs
            percent_missing_score[pm] = per_window_scores
        fold_scores[fold] = percent_missing_score
    return fold_scores


def evaluate_folds_brits(Xs, ys, fold_idxs, window_idxs, model):
    fold_scores = dict()
    for fold in range(0, len(fold_idxs)):
        print(f"Evaluating fold {fold}/{len(fold_idxs)-1}...")
        # make the splits
        X_train_fold = Xs[fold_idxs[fold]["train"]]
        y_train_fold = ys[fold_idxs[fold]["train"]]
        X_test_fold = Xs[fold_idxs[fold]["test"]]
        y_test_fold = ys[fold_idxs[fold]["test"]]
        # check class distributions
        counts_tr = np.unique(y_train_fold, return_counts=True)[1]
        print(f"Training class distribution: {counts_tr/np.sum(counts_tr)}")
        counts_te = np.unique(y_test_fold, return_counts=True)[1]
        print(f"Testing class distribution: {counts_te/np.sum(counts_te)}")
        print(f"Training BRITS on fold {fold}...")
        model.fit(train_set={'X':X_train_fold})
        print("Finished training!")
        # loop over % missing
        percent_missing_score = dict()
        for pm in window_idxs:
            print(f"Imputing {pm}% missing data over {len(window_idxs[pm])} windows...")
            per_window_scores = dict()
            for (idx, widx) in enumerate(window_idxs[pm]):
                X_test_corrupted = X_test_fold.copy()
                X_test_corrupted[:, widx] = np.nan
                mask = np.isnan(X_test_corrupted) # mask ensures only misisng values are imputed
                brits_imputed = model.impute(test_set={'X': X_test_corrupted})
                errs = [calc_mae(brits_imputed[i], X_test_fold[i], mask[i]) for i in range(0, X_test_fold.shape[0])] # get individual errors for uncertainty quantification
                per_window_scores[idx] = errs
            percent_missing_score[pm] = per_window_scores
        fold_scores[fold] = percent_missing_score
    return fold_scores

def evlaute_folds_cdrec(Xs, ys, fold_idxs, window_idxs):
    fold_scores = dict()
    for fold in range(0, len(fold_idxs)):
        print(f"Evaluating fold {fold}/{len(fold_idxs)-1}...")
        # make the splits
        X_train_fold = Xs[fold_idxs[fold]["train"]]
        y_train_fold = ys[fold_idxs[fold]["train"]]
        X_test_fold = Xs[fold_idxs[fold]["test"]]
        y_test_fold = ys[fold_idxs[fold]["test"]]
        # check class distributions
        counts_tr = np.unique(y_train_fold, return_counts=True)[1]
        print(f"Training class distribution: {counts_tr/np.sum(counts_tr)}")
        counts_te = np.unique(y_test_fold, return_counts=True)[1]
        print(f"Testing class distribution: {counts_te/np.sum(counts_te)}")
        print(f"Computing CDrec on fold {fold}...")
        percent_missing_score = dict()
        for pm in window_idxs:
            print(f"Imputing {pm}% missing data over {len(window_idxs[pm])} windows...")
            per_window_scores = dict()
            for (idx, widx) in enumerate(window_idxs[pm]):
                X_test_corrupted = X_test_fold.copy()
                X_test_corrupted[:, widx] = np.nan
                mask = np.isnan(X_test_corrupted) # mask ensures only misisng values are imputed
                Xdata = np.concatenate([X_train_fold.squeeze(), X_test_corrupted.squeeze()])
                cdrec_imputed_raw = CDrec(matrix=Xdata) # using default paramss
                cdrec_imputed = cdrec_imputed_raw[X_train_original.shape[0]:][:].reshape([-1, X_train_original.shape[1], 1]) # only the test data from the concatenated matrix            
                errs = [calc_mae(cdrec_imputed[i], X_test_fold[i], mask[i]) for i in range(0, X_test_fold.shape[0])] # get individual errors for uncertainty quantification
                per_window_scores[idx] = errs
            percent_missing_score[pm] = per_window_scores
        fold_scores[fold] = percent_missing_score
    return fold_scores


# load the original IPD Split 
train_f = np.loadtxt("Data/ecg200/datasets/ECG200_TRAIN.txt")
test_f = np.loadtxt("Data/ecg200/datasets/ECG200_TEST.txt")
X_train = train_f[:, 1:]
y_train = train_f[:, 0]
X_test = test_f[:, 1:]
y_test = test_f[:, 0]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# reshape data for imputation models
X_train_original = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_original =  X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
y_train_original = y_train
y_test_original = y_test
print(X_train_original.shape)
print(X_test_original.shape)

# Combine the train and test splits for resampling
Xs = np.vstack([X_train_original, X_test_original])
print(Xs.shape)
ys = np.concatenate([y_train_original, y_test_original])
print(ys.shape)

# load resample fold indices
with open("FinalBenchmarks/ECG200/Python/resample_folds_python_idx.json", "r") as f:
    resample_fold_idxs_f = json.load(f)
resample_fold_idxs = {int(k): v for k, v in resample_fold_idxs_f.items()}
print(resample_fold_idxs.keys())

X_train_f1 = Xs[resample_fold_idxs[0]["train"]]
X_test_f1 = Xs[resample_fold_idxs[0]["test"]]
y_train_f1 = ys[resample_fold_idxs[0]["train"]]
y_test_f1 = ys[resample_fold_idxs[0]["test"]]

print(X_train_f1.shape)
print(X_test_f1.shape)
print(y_train_f1.shape)
print(y_test_f1.shape)

print(np.all(np.equal(X_train_f1, X_train_original)))
print(np.all(np.equal(y_train_f1, y_train_original)))
print(np.all(np.equal(X_test_f1, X_test_original)))
print(np.all(np.equal(y_test_f1, y_test_original)))

# load imputation window indices
with open("FinalBenchmarks/ECG200/Python/windows_python_idx.json", "r") as f:
    window_idxs_f = json.load(f)
window_idxs = {int(float(k)*100): v for k, v in window_idxs_f.items()}
print(window_idxs.keys())


# run imputation
csdi = CSDI(
    n_steps=len(X_test_original[0]),
    n_features=1, # univariate time series, so num features is equal to one
    n_layers=6,
    n_heads=2,
    n_channels=128,
    d_time_embedding=64,
    d_feature_embedding=32,
    d_diffusion_embedding=128,
    target_strategy="random",
    n_diffusion_steps=50,
    batch_size=32,
    epochs=100,
    patience=None,
    optimizer=Adam(lr=1e-3),
    num_workers=0,
    device=None,
    model_saving_strategy=None
)

fold_scores_csdi = evaluate_folds_csdi(Xs, ys, resample_fold_idxs, window_idxs,csdi)
with open("ECG200_csdi_results.pkl", "wb") as f:
    pickle.dump(fold_scores_csdi, f)

brits = BRITS(
    n_steps=len(X_test_original[0]),
    n_features=1,
    rnn_hidden_size=128,
    batch_size=32,
    epochs=100,
    optimizer=Adam(lr=1e-3),
    num_workers=0,
    device=None, # infer the best device to use
    model_saving_strategy=None
)

fold_scores_brits = evaluate_folds_brits(Xs, ys, resample_fold_idxs, window_idxs, brits)
with open("ECG200_brits_results.pkl", "wb") as f:
    pickle.dump(fold_scores_brits, f)

fold_scores_cdrec = evlaute_folds_cdrec(Xs, ys, resample_fold_idxs, window_idxs)
with open("ECG200_cdrec_results.pkl", "wb") as f:
    pickle.dump(fold_scores_cdrec, f)
