from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import numpy as np
from numpy.random import seed
import matplotlib
import pandas as pd

from tqdm import tqdm
from tensorflow.keras.layers import Layer

import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legfit, legval
import modplot

from sklearn.metrics import RocCurveDisplay, roc_curve, roc_auc_score

from joblib import Parallel, delayed

# machine learning imports
import keras
import tensorflow as tf
try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
from tensorflow.keras import backend
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU
# from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.neural_network import MLPClassifier
from tensorflow.keras import regularizers
from keras import backend as K
tf.config.run_functions_eagerly(False)


#########################################################################
########################### SIGNIFICANCE  ###############################
#########################################################################


# calculate signficance given features, labels, and model
def Z(preds, labels, bins):

    # get num background events
    background = preds[labels == 0]

    # bin to get distributions
    plt.clf()
    pred_val, bins, _ = plt.hist(preds, bins=bins)
    back_val, bins, _ = plt.hist(background, bins=bins)

    # calculate lr stat
    lr_stat = lr_statistic(pred_val, back_val)

    # return significance (Asimov mean value)
    return np.sqrt(lr_stat)


def Z_2d(preds, masses, labels, bins_x, bins_y):

    plt.clf()
    preds_and_mass, _, _ = np.histogram2d(preds, masses, bins=[bins_x, bins_y])
    preds_and_mass_background, _, _ = np.histogram2d(
        preds[labels == 0], masses[labels == 0], bins=[bins_x, bins_y])

    significance = np.sqrt(lr_statistic(
        preds_and_mass.flatten(), preds_and_mass_background.flatten()))

    return significance

# calculate likelihood ratio statistic given


def lr_statistic(pred, background):

    bin_val = []

    for i in range(len(pred)):
        if pred[i] != 0 and background[i] > 5 and (pred[i] > background[i]):
            bin_val.append(background[i]-pred[i]+pred[i]
                           * np.log(pred[i]/background[i]))
        else:
            bin_val.append(0)

    return 2 * np.sqrt(np.sum(np.array(bin_val)**2))

# calculate likelihood ratio statistic given


def lr_statistic(pred, background):
    bin_val = []
    for i in range(len(pred)):
        if pred[i] != 0 and background[i] > 5 and (pred[i] > background[i]):
            bin_val.append(background[i]-pred[i]+pred[i]
                           * np.log(pred[i]/background[i]))
        else:
            bin_val.append(0)

    return 2 * np.sqrt(np.sum(np.array(bin_val)**2))

#########################################################################
############################# GET MODEL  ################################
#########################################################################
# define the neural network to be used
def get_model(layers=[10], ninputs=13, noutputs=1, activation='relu'):
    model = Sequential()
    noutputs = 1
    # add the layers with their corresponding sizes
    for i, size in enumerate(layers):
        if i == 0:
            model.add(Dense(size, input_dim=ninputs, 
                            activation=activation,
                            # kernel_regularizer=regularizers.L2(1e-6),
                            # bias_regularizer=regularizers.L2(1e-6)
                            )

                      )
            # model.add(Dropout(rate=0.1))
        else:
            model.add(Dense(size, 
                            activation=activation,
                            # kernel_regularizer=regularizers.L2(1e-6),
                            # bias_regularizer=regularizers.L2(1e-6)
                            )
                      )
            # model.add(Dropout(rate=0.1))

    # add the ou
    # tput layer and compile
    model.add(Dense(noutputs))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-2), 
                #   jit_compile=True, 
                  weighted_metrics=[])

    return model

#########################################################################
################ SIGNAL SWEEP WITH ROC UNCERTAINTIES ####################
#########################################################################
def signal_sweep(X, y, labels, fs, layers):

    y_true = np.concatenate([labels, labels])

    models_full = []

    tprs = []
    log_tprs = []
    aucs = []
    fpr_grid = np.linspace(0, 1, 1000)
    fpr_log_grid = np.logspace(-4, 0, 1000)

    weights_mask = np.where(labels == 1, fs, 1)
    sample_weights = pd.DataFrame(np.concatenate([weights_mask, weights_mask]))

    def train():
        model = get_model(layers, ninputs=X.shape[1])
        # earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)
        model.fit(X, y, batch_size=1000, epochs=100, verbose=0,
                  sample_weight=sample_weights, workers=-1)
        return model

    for i in tqdm(range(1)):

        models = Parallel(n_jobs=10)(delayed(train)() for i in range(10))

        y_pred = np.mean([model.predict(X, batch_size=10000, verbose=0)
                          for model in models], axis=0)

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        interp_tpr = np.interp(fpr_grid, fpr, tpr)
        interp_log_tpr = np.interp(fpr_log_grid, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_log_tpr[0] = 0.0
        tprs.append(interp_tpr)
        log_tprs.append(interp_log_tpr)

        roc_auc = roc_auc_score(y_true, y_pred)
        aucs.append(roc_auc)

        models_full.append(models)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_log_tpr = np.mean(log_tprs, axis=0)
    mean_log_tpr[-1] = 1.0

    std_tpr = np.std(tprs, axis=0)
    std_log_tpr = np.std(log_tprs, axis=0)

    # mean_auc = auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    result = pd.DataFrame(data=np.array([len(mean_tpr) * [fs], mean_tpr, mean_log_tpr, len(mean_tpr) * [mean_auc], len(mean_tpr) * [std_auc], std_tpr, std_log_tpr, fpr_grid, fpr_log_grid]).T,
                          columns=['fs', 'mean_tpr', 'mean_log_tpr', 'mean_auc', 'std_auc', 'std_tpr', 'std_log_tpr', 'fpr_grid', 'fpr_log_grid'])

    del models
    K.clear_session()

    return result, models_full


def evaluate_models(X, y, y_true, layers, sample_weights):

    aucs = []
    mses = []

    def train():
        model = get_model(layers, ninputs=X.shape[1])
        earlystopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=50)
        model.fit(X, y, batch_size=1000, epochs=100000, verbose=0,
                  sample_weight=sample_weights, validation_split=0.1, callbacks=[earlystopping], workers=-1)
        return model

    models = Parallel(n_jobs=5)(delayed(train)() for i in range(5))

    for model in models:

        y_pred = model.predict(X, batch_size=1000)

        roc_auc = roc_auc_score(y_true, y_pred)
        aucs.append(roc_auc)
        mses.append(mean_squared_error(y, y_pred))

    return models, aucs, mses


#########################################################################
################ Plots with Background Reduction Factor #################
#########################################################################

def trim_trailing_zeros(value, max_decimal_places=4):
    formatted = f"{value:.{max_decimal_places}f}"
    return formatted.rstrip('0').rstrip('.')

def plot_auc_bs(results, sweep, title, save_as, gauss=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ['black', 'red', 'orange', 'green', 'blue', 'purple']

    i = 0

    for fs in sweep:

        results_fs = results[results.fs == fs]

        tpr_grid = np.linspace(0, 1, 100)
        fpr_grid = 1 - np.linspace(0, 1, 100)

        mean_fpr = 1 - results_fs.mean_fpr
        mean_tpr = results_fs.mean_tpr
        mean_auc = results_fs.mean_auc.iloc[0]
        std_auc = results_fs.std_auc.iloc[0]
        std_fpr = results_fs.std_fpr
        std_tpr = results_fs.std_tpr

        try:
            fs = float(fs)
            if fs > 0:
                if not gauss:
                    fs = fs/(1+fs)
                ax.plot(
                    mean_tpr,
                    fpr_grid,
                    color=colors[i],
                    label=r"$f_S$ = %s (AUC = %0.2f $\pm$ %0.2f)" % (
                        trim_trailing_zeros(fs), mean_auc, std_auc),
                    lw=2,
                    alpha=0.8,
                )
            else:
                ax.plot(
                    mean_tpr,
                    fpr_grid,
                    color=colors[i],
                    label=r"$f_S$ = %s (AUC = %0.2f $\pm$ %0.2f)" % (
                        trim_trailing_zeros(fs), mean_auc, std_auc),
                    lw=2,
                    alpha=0.8,
                )
        except:
            ax.plot(
                mean_tpr,
                fpr_grid,
                color=colors[i],
                # label=r"$f_s$ = %s (AUC = %0.2f $\pm$ %0.2f)" % (fs, mean_auc, std_auc),
                label="%s (AUC = %0.2f $\pm$ %0.2f)" % (fs, mean_auc, std_auc),
                lw=2,
                alpha=0.8,
            )

        ax.errorbar(
            mean_tpr,
            fpr_grid,
            xerr=std_tpr,
            color=colors[i],
            alpha=0.3,
            errorevery=(2*i, 7)
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
        )

        ax.set_xlabel("Signal Efficiency ($\epsilon_S$)", fontsize=14)
        ax.set_ylabel(
        "Background Rejection Factor ($\epsilon_B^{-1}$)", fontsize=14)
        ax.set_title(title, fontsize=16)

        i += 1

    ax.legend(loc="lower left", frameon=False,  prop={'size': 11.5})
    # plt.show()
    if save_as is not None:
        plt.savefig(save_as + '.pdf', bbox_inches='tight')


#########################################################################
################ Plots with Background Reduction Factor #################
#########################################################################

def plot_auc_brf(results, sweep, title, save_as, errors='y', shuffle=False):

    colors = ['black', 'red', 'orange', 'green', 'blue', 'purple']

    i = 0

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,
                                   gridspec_kw={'width_ratios': [5, 1]})
    fig.subplots_adjust(wspace=0)

    for fs in sweep:

        results_fs = results[results.fs == fs]

        mean_auc = results_fs.mean_auc.values[0]
        std_auc = results_fs.std_auc.values[0]

        tpr = results_fs.mean_log_tpr
        fpr_log_grid = results_fs.fpr_log_grid
        brf = 1 / fpr_log_grid

        std_tpr = results_fs.std_tpr

        try:
            fs = float(fs)
            fs = fs/(1+fs)
            if fs > 0:
                ax1.plot(
                    tpr,
                    brf,
                    color=colors[i],
                    label=r"$f_S$ = %s (AUC = %0.2f $\pm$ %0.2f)" % (
                        trim_trailing_zeros(fs), mean_auc, std_auc),
                    lw=2,
                    alpha=0.8,
                )
            else:
                ax1.plot(
                    tpr,
                    brf,
                    color=colors[i],
                    label=r"$f_S$ = %s (AUC = %0.2f $\pm$ %0.2f)" % (
                        trim_trailing_zeros(fs), mean_auc, std_auc),
                    lw=2,
                    alpha=0.8,
                )

        except:
            ax1.plot(
                tpr,
                brf,
                color=colors[i],
                # label=r"$f_s$ = %s (AUC = %0.2f $\pm$ %0.2f)" % (fs, mean_auc, std_auc),
                label="%s (AUC = %0.2f $\pm$ %0.2f)" % (fs, mean_auc, std_auc),
                lw=2,
                alpha=0.8,
            )

        ax2.plot(std_tpr, brf, color=colors[i])

        i += 1

    ax1.set_xlim([-0.05, 1.1])
    ax1.set_xlabel("Signal Efficiency ($\epsilon_S$)", fontsize=14)
    ax1.set_ylabel(
        "Background Rejection Factor ($\epsilon_B^{-1}$)", fontsize=14)

    if title is not None:
        ax1.set_title(title, fontsize=16)


    ax2.set_xlabel("$\Delta \epsilon_S$", fontsize=14)
    ax1.plot(np.linspace(0, 1, 100000), 1 / np.linspace(0.0001, 1, 100000),
             c='black', linestyle='dashed', label='Random, AUC = 0.5')
    ax1.set_yscale('log')
    if shuffle:
        ax1.legend(loc="best", frameon=False,  prop={'size': 11.5})
    else:
        ax1.legend(loc="best", frameon=False,  prop={'size': 11})
    if save_as is not None:
        plt.savefig(save_as + '.pdf', bbox_inches='tight', dpi=100)
    plt.show()

#########################################################################
######################## NN OUTPUT HISTOGRAM  ###########################
#########################################################################

def nn_output_hist(preds, labels, targets, f, save_as=None):

    bin_low, bin_high = min(preds)[0], max(preds)[0]
    preds = preds.reshape((2, int(len(preds)/2))).T
    bins = np.linspace(bin_low, bin_high, 100)

    # design the figure layout
    fig, axes = modplot.axes(ratio_plot=False, figsize=(6, 4), gridspec_update=None,
                             xlabel=r'Model Output', ylabel=r'Density (Normalized)',
                             xlim=(min(bins), max(bins)), ylim=(0.001, 20),
                             xticks=None, yticks=None, xtick_step=None, ytick_step=None)

    # plot the model outputs for signal and background
    histops = {'bins': bins, 'histtype': 'step', 'density': True}
    for j in [0, 1]:
        plt.hist(preds[labels == 0, j], **histops, color='black',
                 label=r"QCD", ls='--' if j else '-')
        plt.hist(preds[labels == 1, j], **histops, color='red',
                 label=r"$Z'\to XY$", ls='--' if j else '-')

    # plot the average mass and resonance masses
    meanm = np.mean(targets[np.concatenate([labels, labels]) == 0], axis=0)
    plt.plot([meanm, meanm], [0, 10], '--', color='gray')
    # plt.text(meanm-0.01, 0.3, r"$\langle O \rangle$")

    # add information stamp to the figure
    modplot.stamp(0.55, 0.99, ax=axes[0], delta_y=0.0475,
                  line_0=None, 
                #   line_1=r"\textsc{Pythia 8} + \textsc{Delphes}",
                  line_1=r"R=1.0 anti-$k_T$ Jets",
                  line_2=r"$p_T>1.2$ TeV, $|\eta| < 2.5$",
                  line_4=r"QCD w. {}$\%\, Z'\to XY$".format(100*f))
    # plt.yscale('log')
    modplot.legend(ax=axes[0], loc='upper left')

    if save_as is not None:
        plt.savefig(save_as + '.pdf', bbox_inches='tight', dpi=100)
    plt.show()


#########################################################################
######################### JET MASS HISTOGRAM ############################
#########################################################################

def jetmass_hist(efps, preds, labels, f, cuts, top_sig, shuffle, save_as=None):
    masses = np.sqrt(efps[:, :, 2]/2)

    def masses_filt(mask):
        return 1000*np.concatenate([masses[mask, 0], masses[mask, 1]])

    bins = 10**np.linspace(1.6, 3.2, 61)
    midbins = (bins[: -1] + bins[1:])/2

    fig, axes = modplot.axes(ratio_plot=True, figsize=(6, 4), gridspec_update=None,
                             xlabel=r'Jet Mass [GeV]', ylabel=r'Events per Bin', ylabel_ratio='Local \n Sig. ($\sigma$)',
                             xlim=(min(bins), max(bins)), ylim=(1, 10000000), ylim_ratio=(-2, top_sig + 2),
                             xticks=[50, 100, 200, 500, 1000], yticks=None, xtick_step=None, ytick_step=None, ytick_ratio_step=2)

    [axes[1].plot(midbins, 0*midbins+k, '-', lw=0.75, alpha=0.5, color='gray')
     for k in range(-2, top_sig+2, 2)]

    # plot the full distribution
    axes[0].hist(masses_filt(labels >= 0), bins=bins,
                 histtype='step', density=False, color='black', label='All Events')

    # plot the distribution for QCD jets
    axes[0].hist(masses_filt(labels == 0), bins=bins, histtype='stepfilled', density=False,
                 alpha=0.2, color='blue', label=r"QCD")

    # plot the distribution for new physics jets
    axes[0].hist(masses_filt(labels == 1), bins=bins, histtype='stepfilled', density=False,
                 alpha=0.5, color='purple', label=r"$Z'\to XY$")

    # plot the distribution after the anomaly cut
    colors = ['black', 'red', 'orange', 'gold', 'green', 'blue', 'purple']

    for i, cut in enumerate(cuts):

        # tag the event if either jet is tagged as new physics
        mask = (preds[:, 0] > cut) * (preds[:, 1] > cut)
        ydata, _, _ = axes[0].hist(masses_filt(
            mask), bins=bins, histtype='step', density=False, color=colors[i % len(colors)])
        yerr = np.sqrt(ydata)  # poisson counting error

        axes[0].errorbar(midbins, ydata, yerr=yerr, fmt='none',
                         color=colors[i % len(colors)])

        # fit background estimate above the trigger turn on
        mask2 = (midbins > 300) * (ydata > 0)

        # mask out the signal region
        high_siglow, high_sighigh = 450, 550
        high_mask = mask2 * ((midbins < high_siglow) +
                             (midbins > high_sighigh))

        xs = np.linspace(min(midbins[high_mask]),
                         max(midbins[high_mask]), 1001)

        yfits1, yfits2 = [], []
        for deg in range(2, 7):
            leg = legfit(np.log(midbins[high_mask]), np.log(
                ydata[high_mask]), deg, w=ydata[high_mask]/yerr[high_mask])

            # the fit curves, overall (1) and bin-by-bin (2)
            yfits1.append(np.exp(legval(np.log(midbins), leg)))
            yfits2.append(np.exp(legval(np.log(xs), leg)))

        # get the central value (deg=5) and envelope (max/min) bin-by-bin
        yfitvals1 = yfits1[3]
        yfituncs1 = np.max(np.abs(yfits1 - yfitvals1[np.newaxis, :]), axis=0)
        axes[1].plot(midbins[mask2], ((ydata - yfitvals1)/np.sqrt(yerr **
                     2 + yfituncs1**2))[mask2],  color=colors[i % len(colors)])
        # plot the fits and uncertainties
        axes[1].fill_between(midbins[mask2], ((ydata - yfitvals1 - yfituncs1)/np.sqrt(yerr**2 + yfituncs1**2))[mask2],
                             ((ydata - yfitvals1 + yfituncs1)/np.sqrt(yerr ** 2 + yfituncs1**2))[mask2], color=colors[i % len(colors)], alpha=0.5, lw=0)
        # plt.plot(midbins[mask1], ((ydata - yfitvals1)/np.sqrt(yerr**2 + yfituncs1**2))[mask1],  color=colors[i%len(colors)])

        # get the central value (deg=5) and envelope (max/min) as a curve
        yfitvals2 = yfits2[3]
        yfituncs2 = np.max(np.abs(yfits2 - yfitvals2[np.newaxis, :]), axis=0)

        # plot the fits and uncertainties
        axes[0].plot(xs, yfitvals2, ls=':', color=colors[i % len(colors)])
        axes[0].fill_between(xs, yfitvals2-yfituncs2, yfitvals2 +
                             yfituncs2, color=colors[i % len(colors)], alpha=0.5, lw=0)

        # fit background estimate above the trigger turn on
        mask1 = (midbins < 325) * (ydata > 0)

        # mask out the signal region
        low_siglow, low_sighigh = 90, 110
        low_mask = mask1 * ((midbins < low_siglow) + (midbins > low_sighigh))

        xs = np.linspace(min(midbins[low_mask]), max(midbins[low_mask]), 1001)

        yfits1, yfits2 = [], []
        for deg in range(2, 7):
            leg = legfit(np.log(midbins[low_mask]), np.log(
                ydata[low_mask]), deg, w=ydata[low_mask]/yerr[low_mask])

            # the fit curves, overall (1) and bin-by-bin (2)
            yfits1.append(np.exp(legval(np.log(midbins), leg)))
            yfits2.append(np.exp(legval(np.log(xs), leg)))

        # get the central value (deg=5) and envelope (max/min) bin-by-bin
        yfitvals1 = yfits1[3]
        yfituncs1 = np.max(np.abs(yfits1 - yfitvals1[np.newaxis, :]), axis=0)
        axes[1].plot(midbins[mask1], ((ydata - yfitvals1)/np.sqrt(yerr **
                     2 + yfituncs1**2))[mask1],  color=colors[i % len(colors)])
        axes[1].fill_between(midbins[mask1], ((ydata - yfitvals1 - yfituncs1)/np.sqrt(yerr**2 + yfituncs1**2))[mask1],
                             ((ydata - yfitvals1 + yfituncs1)/np.sqrt(yerr ** 2 + yfituncs1**2))[mask1], color=colors[i % len(colors)], alpha=0.5, lw=0)

        # plt.plot(midbins[mask1], ((ydata - yfitvals1)/np.sqrt(yerr**2 + yfituncs1**2))[mask1],  color=colors[i%len(colors)])

        # get the central value (deg=5) and envelope (max/min) as a curve
        yfitvals2 = yfits2[3]
        yfituncs2 = np.max(np.abs(yfits2 - yfitvals2[np.newaxis, :]), axis=0)

        # plot the fits and uncertainties
        axes[0].plot(xs, yfitvals2, ls=':', color=colors[i % len(colors)])
        axes[0].fill_between(xs, yfitvals2-yfituncs2, yfitvals2 +
                             yfituncs2, color=colors[i % len(colors)], alpha=0.5, lw=0)

    rect_low = matplotlib.patches.Rectangle(
        (low_siglow, -2), low_sighigh-low_siglow, top_sig + 4, color='gray', alpha=0.33, lw=0)
    axes[1].add_patch(rect_low)

    rect_high = matplotlib.patches.Rectangle(
        (high_siglow, -2), high_sighigh-high_siglow, top_sig + 4, color='gray', alpha=0.33, lw=0)
    axes[1].add_patch(rect_high)

    if shuffle==False:
        # plot formatting
        modplot.stamp(0.035, 0.94, ax=axes[0], delta_y=0.08,
                    line_0=r"QCD w. {}$\%\, Z'\to XY$".format(
                        100*f) if f > 0 else r"QCD",
                    #   line_1=r"\textsc{Pythia 8} + \textsc{Delphes}",
                    #   line_1=r"$R$=1.0 anti-$k_T$ Jets",
                    line_1=r'FORCE Cuts',
                    line_2=r"$p_T>1.2$ TeV, $|\eta| < 2.5$"
        )
    else:
        # plot formatting
        modplot.stamp(0.035, 0.94, ax=axes[0], delta_y=0.08,
                    line_0=r"QCD w. {}$\%\, Z'\to XY$".format(
                        100*f) if f > 0 else r"QCD",
                    #   line_1=r"\textsc{Pythia 8} + \textsc{Delphes}",
                    #   line_1=r"$R$=1.0 anti-$k_T$ Jets",
                    line_1=r'FORCE Cuts, Shuffled Features',
                    line_2=r"$p_T>1.2$ TeV, $|\eta| < 2.5$"
        )
    modplot.legend(ax=axes[0], loc='upper right', fontsize=12)
    # plt.xscale('log');plt.yscale('log');
    # axes[0].set_xticks([50, 100, 200, 500, 1000])
    # axes[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0].set_yscale('log')
    axes[0].yaxis.set_major_locator(
        matplotlib.ticker.LogLocator(numticks=999))
    axes[0].yaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(numticks=999, subs='auto'))
    if top_sig > 16:
        axes[1].set_yticks(np.arange(-4, top_sig, 4))
        # axes[1].set_yticklabels([r'$-2$',0,2,4,6,8,10,12])
        axes[1].set_yticklabels(np.concatenate(
            [np.array([r'$-4$']), np.arange(0, top_sig, 4)]))
    else:
        axes[1].set_yticks(np.arange(-2, top_sig + 2, 2))
        # axes[1].set_yticklabels([r'$-2$',0,2,4,6,8,10,12])
        axes[1].set_yticklabels(np.concatenate(
            [np.array([r'$-2$']), np.arange(0, top_sig + 2, 2)]))

    for ax in axes:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks([50, 100, 200, 500, 1000])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    if save_as is not None:
        plt.savefig(save_as + '.pdf', bbox_inches='tight', dpi=100)

    plt.show()


#########################################################################
######################### DIJET MASS HISTOGRAM ##########################
#########################################################################
def dijetmass_hist(dijetmasses, labels, preds, siglow, sighigh, f, cuts, top_sig, shuffle, save_as=None):

    colors = ['black', 'red', 'orange', 'gold', 'green', 'blue', 'purple']

    bins = 10**np.linspace(3.25, 3.95, 61)

    midbins = (bins[:-1] + bins[1:])/2

    fig, axes = modplot.axes(ratio_plot=True, figsize=(6, 4), gridspec_update=None,
                             xlabel=r'Dijet Mass [GeV]', ylabel=r'Events per Bin', ylabel_ratio='Local \n Sig. ($\sigma$)',
                             xlim=(min(bins), max(bins)), ylim=(1, 10000000), ylim_ratio=(-2, top_sig + 2),
                             xticks=[2000, 3000, 4000, 6000, 8000], yticks=None, xtick_step=None, ytick_step=None, ytick_ratio_step=2)

    [axes[1].plot(midbins, 0*midbins+k, '-', lw=0.75, alpha=0.5, color='gray')
     for k in range(-2, top_sig+2, 2)]

    # plot the full distribution
    axes[0].hist(dijetmasses, bins=bins,
                 histtype='step', density=False, color='black', label='All Events')

    # plot the distribution for new physics events
    axes[0].hist(dijetmasses[labels == 0], bins=bins, histtype='stepfilled', density=False,
                 alpha=0.2, color='blue', label=r"QCD")

    # # plot the distribution for new physics events
    if f > 0:
        axes[0].hist(dijetmasses[labels == 1], bins=bins, histtype='stepfilled', density=False,
                     alpha=0.5, color='purple', label=r"$Z'\to XY$")

    sigs = []

    sr_sigs_after_cut = []
    sigs_after_cut = []

    # plot the distribution after the anomaly cut
    for i, cut in enumerate(cuts):

        # if i > 4: continue
        if cut == None:
            nn_mask = [True for i in range(len(preds))]
        else:
            nn_mask = (preds[:, 0] > cut) + (preds[:, 1] > cut)

        ydata, _, _ = axes[0].hist(dijetmasses[nn_mask], bins=bins,
                                   histtype='step', density=False, color=colors[i % len(colors)])
        yerr = np.sqrt(ydata)  # poisson counting error

        axes[0].errorbar(midbins, ydata, yerr=yerr, fmt='none',
                         color=colors[i % len(colors)])

        # fit background estimate above the trigger turn on
        mask1 = (midbins > 3000) * (ydata > 0)

        # mask out the signal region
        mask = mask1 * ((midbins < siglow) + (midbins > sighigh))

        outmask = ((midbins > siglow) * (midbins < sighigh))

        xs = np.linspace(min(midbins[mask]), max(midbins[mask]), 1001)

        yfits1, yfits2 = [], []
        for deg in range(2, 7):
            leg = legfit(np.log(midbins[mask]), np.log(
                ydata[mask]), deg, w=ydata[mask]/yerr[mask])

            # the fit curves, overall (1) and bin-by-bin (2)
            yfits1.append(np.exp(legval(np.log(midbins), leg)))  # discrete fit
            yfits2.append(np.exp(legval(np.log(xs), leg)))      # smooth fit

        # get the central value (deg=5) and envelope (max/min) bin-by-bin
        yfitvals1 = yfits1[3]
        yfituncs1 = np.max(np.abs(yfits1 - yfitvals1[np.newaxis, :]), axis=0)

        axes[1].plot(midbins[mask1], ((ydata - yfitvals1)/np.sqrt(yerr **
                     2 + yfituncs1**2))[mask1],  color=colors[i % len(colors)])
        # axes[1].errorbar(midbins[mask1], ((ydata - yfitvals1)/np.sqrt(yerr **
        #                                                               2 + yfituncs1**2))[mask1], yerr=[(yfituncs1/np.sqrt(yerr**2 + yfituncs1**2))[mask1],
        #                                                                                                ((yfituncs1)/np.sqrt(yerr ** 2 + yfituncs1**2))[mask1]], color=colors[i % len(colors)])
        axes[1].fill_between(midbins[mask1], ((ydata - yfitvals1 - yfituncs1)/np.sqrt(yerr**2 + yfituncs1**2))[mask1],
                             ((ydata - yfitvals1 + yfituncs1)/np.sqrt(yerr ** 2 + yfituncs1**2))[mask1], color=colors[i % len(colors)], alpha=0.5, lw=0)

        yfitvals2 = yfits2[3]
        yfituncs2 = np.max(np.abs(yfits2 - yfitvals2[np.newaxis, :]), axis=0)

        sigs_after_cut.append(
            (((ydata - yfitvals1)/np.sqrt(yerr**2 + yfituncs1**2))).tolist())
        sr_sigs_after_cut.append(
            (((ydata - yfitvals1)/np.sqrt(yerr**2 + yfituncs1**2))[outmask]).tolist())

        # plot the fits and uncertainties
        axes[0].plot(xs, yfitvals2, ls=':', color=colors[i % len(colors)])
        axes[0].fill_between(xs, yfitvals2-yfituncs2, yfitvals2 +
                             yfituncs2, color=colors[i % len(colors)], alpha=0.5, lw=0)

    rect = matplotlib.patches.Rectangle(
        (siglow, -2), sighigh-siglow, top_sig + 4, color='gray', alpha=0.33, lw=0)
    axes[1].add_patch(rect)

    if shuffle==False:
        # plot formatting
        modplot.stamp(0.035, 0.94, ax=axes[0], delta_y=0.08,
                    line_0=r"QCD w. {}$\%\, Z'\to XY$".format(
                        100*f) if f > 0 else r"QCD",
                    #   line_1=r"\textsc{Pythia 8} + \textsc{Delphes}",
                    #   line_1=r"$R$=1.0 anti-$k_T$ Jets",
                    line_1=r'FORCE Cuts',
                    line_2=r"$p_T>1.2$ TeV, $|\eta| < 2.5$"
        )
    else:
        # plot formatting
        modplot.stamp(0.035, 0.94, ax=axes[0], delta_y=0.08,
                    line_0=r"QCD w. {}$\%\, Z'\to XY$".format(
                        100*f) if f > 0 else r"QCD",
                    #   line_1=r"\textsc{Pythia 8} + \textsc{Delphes}",
                    #   line_1=r"$R$=1.0 anti-$k_T$ Jets",
                    line_1=r'FORCE Cuts, Shuffled Features',
                    line_2=r"$p_T>1.2$ TeV, $|\eta| < 2.5$"
        )

    modplot.legend(ax=axes[0], loc='upper right', fontsize=12)
    axes[0].set_yscale('log')
    # axes[1].set_yticks([-2,0,2,4,6,8,10,12])
    if top_sig > 16:
        axes[1].set_yticks(np.arange(-4, top_sig, 4))
        # axes[1].set_yticklabels([r'$-2$',0,2,4,6,8,10,12])
        axes[1].set_yticklabels(np.concatenate(
            [np.array([r'$-4$']), np.arange(0, top_sig, 4)]))
    else:
        axes[1].set_yticks(np.arange(-2, top_sig + 2, 2))
        # axes[1].set_yticklabels([r'$-2$',0,2,4,6,8,10,12])
        axes[1].set_yticklabels(np.concatenate(
            [np.array([r'$-2$']), np.arange(0, top_sig + 2, 2)]))
    axes[0].yaxis.set_major_locator(
        matplotlib.ticker.LogLocator(numticks=999))
    axes[0].yaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(numticks=999, subs='auto'))
    for ax in axes:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks([2000, 3000, 4000, 6000, 8000])
        for label in ax.get_xticklabels():
            label.set_fontsize(12)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    if save_as is not None:
        plt.savefig(save_as + '.pdf', bbox_inches='tight', dpi=100)

    plt.show()
