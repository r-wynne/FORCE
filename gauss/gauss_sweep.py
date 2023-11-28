# python library imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from itertools import product
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import RocCurveDisplay, roc_curve, roc_auc_score, auc

import tqdm

from keras import backend as K

from joblib import Parallel, delayed

# machine learning imports
import keras
import tensorflow as tf
from tensorflow.keras import backend
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from tensorflow.keras import regularizers
import sklearn.model_selection
import sklearn

# results = pd.DataFrame(
#     columns=['features', 'nn_signal', 'nn_no_signal', 'cond_exp', 'delta'])

results_nn = pd.DataFrame()
results_auc = pd.DataFrame()

# define the neural network to be used


def get_model(layer_sizes, ninputs=1, noutputs=1):
    model = Sequential()

    # add the layers with their corresponding sizes
    for i, size in enumerate(layer_sizes):
        if i == 0:
            model.add(Dense(size, input_dim=ninputs, activation='relu',
                            # kernel_regularizer=regularizers.L2(1e-5),
                            # bias_regularizer=regularizers.L2(1e-4)
                            ))
        else:
            model.add(Dense(size, activation='relu'))

    # add the output layer and compile
    model.add(Dense(noutputs))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.legacy.Adam(
        learning_rate=1e-2), weighted_metrics=[])
    return model


# define any desired callbacks to be used during training
earlystopping = keras.callbacks.EarlyStopping(monitor='loss', patience=100)

fs = [0, 0.0005, 0.005, 0.05, 0.5]

for f_s in tqdm.tqdm(fs):

    signal_mean, signal_variance = 1, 1
    background_mean, background_variance = 0, 1

    background_size = 1000000
    signal_size = 1000000

    data_1 = np.random.normal(signal_mean, signal_variance, (signal_size, 2))
    data_2 = np.random.normal(
        background_mean, background_variance, (background_size, 2))

    O_s, O_b = data_1[:, 0].reshape(-1, 1), data_2[:, 0].reshape(-1, 1)
    p_s, p_b = data_1[:, 1], data_2[:, 1]

    def likelihood_ratio_function(x):
        return (background_variance / signal_variance) * np.exp((signal_mean-background_mean) * x + 1/2*(background_mean**2 - signal_mean**2))

    features_no_sig = O_b
    targets_no_sig = p_b
    features_sig = np.concatenate((O_b, O_s))
    targets_sig = np.concatenate((p_b, p_s))
    labels = np.concatenate((np.zeros(len(O_b)), np.ones(len(O_s))))

    perm = np.random.permutation(len(labels))

    features_sig, targets_sig, labels = features_sig[perm], targets_sig[perm], labels[perm]

    cond_exp = np.array([background_mean + ((signal_mean - background_mean) * f_s * likelihood_ratio_function(
        feature)) / (1 - f_s + f_s * likelihood_ratio_function(feature)) for feature in features_sig]).flatten()

    cond_exp_plot = np.array([background_mean + ((signal_mean - background_mean) * f_s * likelihood_ratio_function(
        feature)) / (1 - f_s + f_s * likelihood_ratio_function(feature)) for feature in np.linspace(-5, 5, 1000).reshape(-1, 1)]).flatten()

    weights_mask = np.where(labels == 1, f_s / (1 - f_s), 1)
    sample_weights = pd.DataFrame(weights_mask)

    layers = [10, 10, 10, 10]

    def train(X, y):
        model = get_model(layers, ninputs=X.shape[1])
        earlystopping = keras.callbacks.EarlyStopping(
            monitor='loss', patience=100)
        model.fit(X, y, batch_size=10000, epochs=10000, verbose=0,
                  sample_weight=sample_weights, callbacks=[earlystopping], workers=-1)
        return model

    models_sig = Parallel(n_jobs=4)(delayed(train)
                                    (features_sig, targets_sig) for i in range(8))

    preds_sig_auc = []
    preds_plot = []

    tprs = []
    fprs = []
    aucs = []
    tpr_grid = np.linspace(0, 1, 100)
    fpr_grid = np.linspace(0, 1, 100)

    for model in models_sig:

        pred_plot = model.predict(
            np.linspace(-5, 5, 1000).reshape(-1, 1), batch_size=10000, verbose=0)
        preds_plot.append(pred_plot)

        y_pred = model.predict(features_sig, batch_size=10000, verbose=0)

        fpr, tpr, thresholds = roc_curve(labels, y_pred)
        interp_tpr = np.interp(fpr_grid, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        interp_fpr = np.interp(tpr_grid, tpr, fpr)
        interp_fpr[0] = 0.0
        fprs.append(interp_fpr)

        roc_auc = roc_auc_score(labels, y_pred)
        aucs.append(roc_auc)

    mean_pred = np.mean(preds_plot, axis=0)
    std_pred = np.std(preds_plot, axis=0)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_fpr = np.mean(fprs, axis=0)
    mean_fpr[-1] = 1.0

    std_tpr = np.std(tprs, axis=0)
    std_fpr = np.std(fprs, axis=0)

    # mean_auc = auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    result_nn = pd.DataFrame(index=range(len(mean_pred)))

    result_nn = pd.DataFrame(data=np.array([np.linspace(-5, 5, 1000), mean_pred.reshape(-1), std_pred.reshape(-1), cond_exp_plot.reshape(-1), len(mean_pred) * [f_s]]).T,
                             columns=['features', 'mean_nn_signal', 'std_nn_signal', 'cond_exp', 'fs'])

    result_auc = pd.DataFrame(data=np.array([len(mean_fpr) * [f_s], mean_fpr, mean_tpr, len(mean_fpr) * [mean_auc], len(mean_fpr) * [std_auc], std_fpr, std_tpr]).T,
                              columns=['fs', 'mean_fpr', 'mean_tpr', 'mean_auc', 'std_auc', 'std_fpr', 'std_tpr'])

    results_nn = pd.concat([results_nn, result_nn])
    results_auc = pd.concat([results_auc, result_auc])

    del models_sig
    K.clear_session()


X = features_sig
y = labels

cv = KFold(n_splits=4)
classifier = MLPClassifier(batch_size=10000)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for fold, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X[test],
        y[test],
        name=f"ROC fold {fold}",
        alpha=0.3,
        lw=1,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)


std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)


result = pd.DataFrame(data=np.array([mean_fpr, mean_tpr, len(mean_fpr) * [mean_auc], len(mean_fpr) * [std_auc], std_tpr, std_tpr]).T,
                      columns=['mean_fpr', 'mean_tpr', 'mean_auc', 'std_auc', 'std_fpr', 'std_tpr'])

result['fs'] = 'Supervised'

results_auc = pd.concat([results_auc, result])


# model_no_sig = get_model(layers, ninputs=features_sig.shape[1])
# model_no_sig.fit(features_sig, targets_sig, batch_size=10000, epochs=10000,
#                  verbose=1, sample_weight=sample_weights, callbacks=[earlystopping])

# results_fs['features'] = features_sig
# results_fs['nn_signal'] = preds_sig
# results_fs['labels'] = y_true
# # results_fs['nn_no_signal'] = preds_no_sig
# results_fs['cond_exp'] = cond_exp_plot
# results_fs['f_s'] = len(preds_sig) * [f_s / (1 + f_s)]

results_nn.to_csv('gauss_nn_convergence')
results_auc.to_csv('gauss_auc_sweep')
