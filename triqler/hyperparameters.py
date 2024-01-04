#!/usr/bin/python

from __future__ import print_function

import pandas as pd
import numpy as np
from scipy.stats import hypsecant, gamma, norm, t, cauchy
from scipy.optimize import curve_fit

from . import parsers


def fitPriors(peptQuantRows, params, printImputedVals=False, plot=False):
    params["proteinQuantCandidates"] = np.arange(
        -5.0, 5.0 + 1e-10, 0.01
    )  # log10 of protein ratio
    qc = params["proteinQuantCandidates"]
    params["proteinDiffCandidates"] = np.linspace(
        2 * qc[0], 2 * qc[-1], len(qc) * 2 - 1
    )

    protQuantRows = parsers.filterAndGroupPeptides(
        peptQuantRows, lambda x: not x.protein[0].startswith(params["decoyPattern"])
    )

    (
        imputedVals,
        imputedDiffs,
        observedXICValues,
        protQuants,
        protDiffs,
        protStdevsInGroup,
        protGroupDiffs,
    ) = (list(), list(), list(), list(), list(), list(), list())
    quantRowsCollection = list()
    if params["missingValuePrior"] == "DIA":
        imputed_peptide_group_means = []  # Added for DIA prior
    for prot, quantRows in protQuantRows:
        quantRows, quantMatrix = parsers.getQuantMatrix(quantRows)

        quantMatrixNormalized = [parsers.geoNormalize(row) for row in quantMatrix]
        quantRowsCollection.append((quantRows, quantMatrix))
        geoAvgQuantRow = getProteinQuant(quantMatrixNormalized, quantRows)

        protQuants.extend([np.log10(x) for x in geoAvgQuantRow if not np.isnan(x)])

        args = parsers.getQuantGroups(geoAvgQuantRow, params["groups"], np.log10)
        for group in args:
            if np.count_nonzero(~np.isnan(group)) > 1:
                protDiffs.extend(group - np.mean(group))
                protStdevsInGroup.append(np.std(group))

        quantMatrixFiltered = np.log10(
            np.array([x for x, y in zip(quantMatrix, quantRows) if y.combinedPEP < 1.0])
        )

        # Fit prior based on means of missing value
        if params["missingValuePrior"] == "DIA":
            for peptide in quantMatrixFiltered:
                for group in params["groups"]:
                    n_imputed_values = (
                        len(group) - (~np.isnan(peptide[group])).sum()
                    )  # number of imputed values.
                    if (
                        n_imputed_values == 0
                    ):  # If the condition has no NaN, dont add it to imputed group mean
                        continue
                    else:
                        for n_imputed in range(
                            n_imputed_values
                        ):  # For every imputed value add it to imputed_peptide_group_mean
                            group_means = peptide[group][
                                ~np.isnan(peptide[group])
                            ].mean()
                            if (
                                group_means != np.nan
                            ):  # If not all samples in condition in mean, append it into imputed_peptide_group_means
                                imputed_peptide_group_means.append(group_means)

        # DIA PRIOR COMMENT
        # We can impute in many different ways. The current version adds one condition imputed mean for every missing replicate
        # e.g. if we have [3 NaN NaN] and impute the NaN to 3, we add two 3 to our imputed_peptide_group_means list.
        #
        # We can also modify the code so that
        # 1) We only add one mean per condition to imputed_peptide_group_means, e.g. [3 NaN NaN] adds only 3 one time to imputed_peptide_group_means
        # 2) We can add a filter so that if more than 50% (or x% or x samples) are missing in our condition we do not add it to our imputed_peptide_group_means

        observedXICValues.extend(quantMatrixFiltered[~np.isnan(quantMatrixFiltered)])

        # counts number of NaNs per run, if there is only 1 non NaN in the column, we cannot use it for estimating the imputedDiffs distribution
        numNonNaNs = np.count_nonzero(~np.isnan(quantMatrixFiltered), axis=0)[
            np.newaxis, :
        ]
        xImps = imputeValues(
            quantMatrixFiltered, geoAvgQuantRow, np.log10(geoAvgQuantRow)
        )
        imputedDiffs.extend(
            (xImps - quantMatrixFiltered)[
                (~np.isnan(quantMatrixFiltered)) & (np.array(numNonNaNs) > 1)
            ]
        )
        # imputedVals.extend(xImps[(np.isnan(quantMatrixFiltered)) & (np.array(numNonNaNs) > 1)])

    # Add parameter to params for selecting fitLogitNormal
    if params["missingValuePrior"] == "DIA":
        imputed_peptide_group_means = np.array(imputed_peptide_group_means)
        imputed_peptide_group_means = imputed_peptide_group_means[
            ~np.isnan(imputed_peptide_group_means)
        ]
        imputedValues = imputed_peptide_group_means
        fitLogitNormalDIA(
            observedXICValues, imputedValues, params, plot
        )  # DIA fitLogitNormal - missing value prior
    else:
        fitLogitNormal(
            observedXICValues, params, plot
        )  # old fitLogitNormal - missing value prior

    fitDist(
        protQuants,
        funcHypsec,
        "log10(protein ratio)",
        ["muProtein", "sigmaProtein"],
        params,
        plot,
    )

    fitDist(
        imputedDiffs,
        funcHypsec,
        "log10(imputed xic / observed xic)",
        ["muFeatureDiff", "sigmaFeatureDiff"],
        params,
        plot,
    )

    fitDist(
        protStdevsInGroup,
        funcGamma,
        "stdev log10(protein diff in group)",
        ["shapeInGroupStdevs", "scaleInGroupStdevs"],
        params,
        plot,
        x=np.arange(-0.1, 1.0, 0.005),
    )

    sigmaCandidates = np.arange(0.001, 3.0, 0.001)
    gammaCandidates = funcGamma(
        sigmaCandidates, params["shapeInGroupStdevs"], params["scaleInGroupStdevs"]
    )
    support = np.where(gammaCandidates > max(gammaCandidates) * 0.01)
    params["sigmaCandidates"] = np.linspace(
        sigmaCandidates[support[0][0]], sigmaCandidates[support[0][-1]], 20
    )

    params["proteinPrior"] = funcLogHypsec(
        params["proteinQuantCandidates"], params["muProtein"], params["sigmaProtein"]
    )
    if "shapeInGroupStdevs" in params:
        params["inGroupDiffPrior"] = funcHypsec(
            params["proteinDiffCandidates"], 0, params["sigmaCandidates"][:, np.newaxis]
        )
    else:  # if we have technical replicates, we could use a delta function for the group scaling parameter to speed things up
        fitDist(
            protDiffs,
            funcHypsec,
            "log10(protein diff in group)",
            ["muInGroupDiffs", "sigmaInGroupDiffs"],
            params,
            plot,
        )
        params["inGroupDiffPrior"] = funcHypsec(
            params["proteinDiffCandidates"],
            params["muInGroupDiffs"],
            params["sigmaInGroupDiffs"],
        )


def fitLogitNormal(observedValues, params, plot):
    m = np.mean(observedValues)
    s = np.std(observedValues)
    minBin, maxBin = m - 4 * s, m + 4 * s
    try:
        vals, bins = np.histogram(
            observedValues, bins=np.arange(minBin, maxBin, 0.1), normed=True
        )
    except TypeError:
        # TypeError will be caused by deprecated normed for np >= 1.24
        vals, bins = np.histogram(
            observedValues, bins=np.arange(minBin, maxBin, 0.1), density=True
        )
    bins = bins[:-1]

    popt, _ = curve_fit(funcLogitNormal, bins, vals, p0=(m, s, m - s, s))

    resetXICHyperparameters = False
    if popt[0] < popt[2] - 5 * popt[3] or popt[0] > popt[2] + 5 * popt[3]:
        print(
            "  Warning: muDetect outside of expected region [",
            popt[2] - 5 * popt[3],
            ",",
            popt[2] + 5 * popt[3],
            "]:",
            popt[0],
            ".",
        )
        resetXICHyperparameters = True

    if popt[1] < 0.1 or popt[1] > 2.0:
        print(
            "  Warning: sigmaDetect outside of expected region [0.1,2.0]:", popt[1], "."
        )
        resetXICHyperparameters = True

    if resetXICHyperparameters:
        print(
            "    Resetting mu/sigmaDetect hyperparameters to default values of muDetect = muXIC - 1.0 and sigmaDetect = 0.3"
        )
        popt[1] = 0.3
        popt[0] = popt[2] - 1.0

    print('  params["muDetect"], params["sigmaDetect"] = %f, %f' % (popt[0], popt[1]))
    print('  params["muXIC"], params["sigmaXIC"] = %f, %f' % (popt[2], popt[3]))
    params["muDetect"], params["sigmaDetect"] = popt[0], popt[1]
    params["muXIC"], params["sigmaXIC"] = popt[2], popt[3]
    if plot:
        poptNormal, _ = curve_fit(funcNorm, bins, vals)

        import matplotlib.pyplot as plt

        plt.figure()
        plt.title(
            "Curve fits for muDetect, sigmaDetect, muXIC and sigmaXIC", fontsize=14
        )
        plt.bar(
            bins,
            vals,
            width=bins[1] - bins[0],
            alpha=0.5,
            label="observed distribution",
        )
        plt.plot(
            bins,
            funcLogitNormal(bins, *popt),
            "g",
            label="logit-normal fit",
            linewidth=2.0,
        )
        plt.plot(
            bins,
            0.5 + 0.5 * np.tanh((np.array(bins) - popt[0]) / popt[1]),
            "m",
            label="logit-part fit",
            linewidth=2.0,
        )
        plt.plot(
            bins,
            funcNorm(bins, popt[2], popt[3]),
            "c",
            label="normal-part fit",
            linewidth=2.0,
        )
        plt.plot(
            bins,
            funcNorm(bins, *poptNormal),
            "r",
            label="normal fit (for comparison)",
            linewidth=2.0,
            alpha=0.5,
        )
        plt.ylabel("relative frequency", fontsize=14)
        plt.xlabel("log10(intensity)", fontsize=14)
        plt.legend()
        plt.tight_layout()


# Added for DIAPrior
def fitLogitNormalDIA(observedValues, imputedValues, params, plot):
    m = np.mean(observedValues)
    s = np.std(observedValues)
    minBin, maxBin = (
        -1,
        4,
    )  # DIA prior binning interval... reasoning is that missing values will be close to zero. Higher intensity missing values should not be regarded. We could also go with -2, 6 for example.
    try:
        vals, bins = np.histogram(
            observedValues, bins=np.arange(minBin, maxBin, 0.1), normed=True
        )
    except TypeError:
        # TypeError will be caused by deprecated normed for np >= 1.24
        vals, bins = np.histogram(
            observedValues, bins=np.arange(minBin, maxBin, 0.1), density=True
        )
    bins = bins[:-1]

    DIA_bins = np.arange(minBin, maxBin, 0.1)
    """
    Need to do the binomial weighting to the curve fit
        # We fit the fraction data we have to pmissings
        # binaomial distribution wiki
        #sigma.plot()
        
        # We fit the fraction data we have to pmissings
        popt, pcov = curve_fit(pmissing, xdata, ydata, sigma=sigma)
    """

    df_binned_imputed_mean_nans = pd.cut(imputedValues, DIA_bins, include_lowest=True)
    df_binned_vals = pd.cut(observedValues, DIA_bins, include_lowest=True)
    df_binned_missing_value_fraction = df_binned_imputed_mean_nans.value_counts() / (
        df_binned_vals.value_counts() + df_binned_imputed_mean_nans.value_counts()
    )
    df_binned_missing_value_fraction = pd.DataFrame(
        df_binned_missing_value_fraction.values,
        index=df_binned_missing_value_fraction.index,
        columns=["fraction"],
    )
    df_binned_missing_value_fraction["n_count"] = (
        df_binned_vals.value_counts() + df_binned_imputed_mean_nans.value_counts()
    )
    df_binned_missing_value_fraction.reset_index(inplace=True)
    df_binned_missing_value_fraction.index = bins  # [:-1]
    df_binned_missing_value_fraction = df_binned_missing_value_fraction[
        df_binned_missing_value_fraction.n_count > 10
    ]  # I actually forgot what was the reasoning for the n_count > 10 heuristic...

    xdata = df_binned_missing_value_fraction.index
    ydata = df_binned_missing_value_fraction.fraction
    sigma = np.sqrt(
        ydata * (1 - ydata) / df_binned_missing_value_fraction.n_count
    )  # Used for the binomial weighing of the curve_fit function. An uncertainty estimator, can we write this down again somewhere. I will 100  % forget the reasoning for this in the future
    # I feel like we need to document and explain this somewhere because I have already forgot the reasoning for this.

    # This is to fit the normal logitNormal
    popt, _ = curve_fit(
        funcLogitNormal, bins, vals, p0=(m, s, m - s, s)
    )  # replace  with code below.

    popt_DIA, pcov_DIA = curve_fit(
        funcOneMinusLogit, xdata, ydata, sigma=sigma
    )  # popt_DIA, funcOneMinusLogit is the function called pmissing in dia_sum/script/clean/plot_fraction_missing_values.py. The sigma is for the binomial weighing to the curve_fit

    resetXICHyperparameters = False

    if popt_DIA[1] < 0.005 or popt_DIA[1] > 10.0:  # Heureistically set
        print(
            "  Warning: sigmaDetect outside of expected region [0.005,10.0]:",
            popt_DIA[1],
            ".",
        )
        resetXICHyperparameters = True

    if popt_DIA[0] < -100 or popt_DIA[0] > 100.0:  # Heureistically set
        print(
            "  Warning: muDetect outside of expected region [-100.0,100.0]:",
            popt_DIA[0],
            ".",
        )
        resetXICHyperparameters = True

    if resetXICHyperparameters:
        print(
            "    Resetting mu/sigmaDetect hyperparameters to default DIA values of muDetect = muXIC - 1.0 and sigmaDetect = 0.3"
        )
        popt_DIA[1] = 0.3
        popt_DIA[0] = popt[2] - 1.0

    print(
        '  params["muDetect"], params["sigmaDetect"] = %f, %f'
        % (popt_DIA[0], popt_DIA[1])
    )
    print('  params["muXIC"], params["sigmaXIC"] = %f, %f' % (popt[2], popt[3]))

    params["muDetect"], params["sigmaDetect"] = (
        popt_DIA[0],
        popt_DIA[1],
    )  # popt[0], popt[1] # estimated here
    params["muXIC"], params["sigmaXIC"] = (
        popt[2],
        popt[3],
    )  # not estimated here. We need to
    if plot:
        poptNormal, _ = curve_fit(funcNorm, bins, vals)

        import matplotlib.pyplot as plt

        plt.figure()
        plt.title("Curve fits for muDetect, sigmaDetect, muXIC and sigmaXIC", fontsize=14)
        plt.bar(
            bins, vals, width=bins[1] - bins[0], alpha=0.5, label="observed distribution"
        )
        plt.plot(
            bins, funcLogitNormal(bins, *popt), "g", label="logit-normal fit", linewidth=2.0
        )
        plt.plot(
            bins,
            0.5 + 0.5 * np.tanh((np.array(bins) - popt[0]) / popt[1]),
            "m",
            label="logit-part fit",
            linewidth=2.0,
        )
        plt.plot(
            bins,
            funcNorm(bins, popt[2], popt[3]),
            "c",
            label="normal-part fit",
            linewidth=2.0,
        )
        plt.plot(
            bins,
            funcNorm(bins, *poptNormal),
            "r",
            label="normal fit (for comparison)",
            linewidth=2.0,
            alpha=0.5,
        )
        plt.ylabel("relative frequency", fontsize=14)
        plt.xlabel("log10(intensity)", fontsize=14)
        plt.legend()
        plt.tight_layout()


def fitDist(ys, func, xlabel, varNames, params, plot, x=np.arange(-2, 2, 0.01)):
    try:
        vals, bins = np.histogram(ys, bins=x, normed=True)
    except TypeError:
        # TypeError will be caused by deprecated normed for np >= 1.24
        vals, bins = np.histogram(ys, bins=x, density=True)
    bins = bins[:-1]
    popt, _ = curve_fit(func, bins, vals)
    outputString = (
        ", ".join(['params["%s"]'] * len(popt)) + " = " + ", ".join(["%f"] * len(popt))
    )
    for varName, val in zip(varNames, popt):
        params[varName] = val

    if func == funcHypsec:
        fitLabel = "hypsec fit"
    elif func == funcNorm:
        fitLabel = "normal fit"
    elif func == funcGamma:
        fitLabel = "gamma fit"
    else:
        fitLabel = "distribution fit"
    print("  " + outputString % tuple(varNames + list(popt)))
    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.title("Curve fit for " + " ".join(varNames), fontsize=14)
        plt.bar(bins, vals, width=bins[1] - bins[0], label="observed distribution")
        plt.plot(bins, func(bins, *popt), "g", label=fitLabel, linewidth=2.0)
        if func == funcHypsec:
            poptNormal, _ = curve_fit(funcNorm, bins, vals)
            plt.plot(
                bins,
                funcNorm(bins, *poptNormal),
                "r",
                label="normal fit (for comparison)",
                linewidth=2.0,
                alpha=0.5,
            )

            if False:
                funcStudentT = lambda x, df, mu, sigma: t.pdf(
                    x, df=df, loc=mu, scale=sigma
                )
                poptStudentT, _ = curve_fit(funcStudentT, bins, vals)
                print(poptStudentT)

                funcCauchy = lambda x, mu, sigma: cauchy.pdf(x, loc=mu, scale=sigma)
                poptCauchy, _ = curve_fit(funcCauchy, bins, vals)
                print(poptCauchy)

                plt.plot(
                    bins,
                    funcStudentT(bins, *poptStudentT),
                    "m",
                    label="student-t fit",
                    linewidth=2.0,
                )
                plt.plot(
                    bins,
                    funcCauchy(bins, *poptCauchy),
                    "c",
                    label="cauchy fit",
                    linewidth=2.0,
                )

                funcLogStudentT = lambda x, df, mu, sigma: t.logpdf(
                    x, df=df, loc=mu, scale=sigma
                )
                funcLogNorm = lambda x, mu, sigma: norm.logpdf(x, loc=mu, scale=sigma)
                funcLogCauchy = lambda x, mu, sigma: cauchy.logpdf(
                    x, loc=mu, scale=sigma
                )

                plt.ylabel("relative frequency", fontsize=14)
                plt.xlabel(xlabel, fontsize=14)
                plt.legend()

                plt.figure()
                plt.plot(
                    bins,
                    funcLogHypsec(bins, *popt),
                    "g",
                    label="hypsec log fit",
                    linewidth=2.0,
                )
                plt.plot(
                    bins,
                    funcLogNorm(bins, *poptNormal),
                    "r",
                    label="normal log fit",
                    linewidth=2.0,
                )
                plt.plot(
                    bins,
                    funcLogStudentT(bins, *poptStudentT),
                    "m",
                    label="student-t log fit",
                    linewidth=2.0,
                )
                plt.plot(
                    bins,
                    funcLogCauchy(bins, *poptCauchy),
                    "c",
                    label="cauchy log fit",
                    linewidth=2.0,
                )
        plt.ylabel("relative frequency", fontsize=14)
        plt.xlabel(xlabel, fontsize=14)
        plt.legend()
        plt.tight_layout()


# this is an optimized version of applying parsers.weightedGeomAvg to each of the columns separately
def getProteinQuant(quantMatrixNormalized, quantRows):
    numSamples = len(quantMatrixNormalized[0])

    geoAvgQuantRow = np.array([0.0] * numSamples)
    weights = np.array([[1.0 - y.combinedPEP for y in quantRows]] * numSamples).T
    weights[np.isnan(np.array(quantMatrixNormalized))] = np.nan

    weightSum = np.nansum(weights, axis=0)
    weightSum[weightSum == 0] = np.nan
    geoAvgQuantRow = np.exp(
        np.nansum(np.multiply(np.log(quantMatrixNormalized), weights), axis=0)
        / weightSum
    )
    geoAvgQuantRow = parsers.geoNormalize(geoAvgQuantRow)
    return geoAvgQuantRow


def imputeValues(quantMatrixLog, proteinRatios, testProteinRatios):
    logIonizationEfficiencies = quantMatrixLog - np.log10(proteinRatios)

    numNonZeros = np.count_nonzero(~np.isnan(logIonizationEfficiencies), axis=1)[
        :, np.newaxis
    ] - ~np.isnan(logIonizationEfficiencies)
    np.nan_to_num(logIonizationEfficiencies, False)
    meanLogIonEff = (
        np.nansum(logIonizationEfficiencies, axis=1)[:, np.newaxis]
        - logIonizationEfficiencies
    ) / numNonZeros

    logImputedVals = meanLogIonEff + testProteinRatios
    return logImputedVals


def funcLogitNormal(x, muLogit, sigmaLogit, muNorm, sigmaNorm):
    return logit(x, muLogit, sigmaLogit) * norm.pdf(x, muNorm, sigmaNorm)


def funcNorm(x, mu, sigma):
    return norm.pdf(x, mu, sigma)


def funcHypsec(x, mu, sigma):
    return hypsecant.pdf(x, mu, sigma)
    # return cauchy.pdf(x, mu, sigma)
    # return norm.pdf(x, mu, sigma)


def funcLogHypsec(x, mu, sigma):
    return hypsecant.logpdf(x, mu, sigma)
    # return cauchy.logpdf(x, mu, sigma)
    # return norm.logpdf(x, mu, sigma)


def funcGamma(x, shape, sigma):
    return gamma.pdf(x, shape, 0.0, sigma)


def logit(x, muLogit, sigmaLogit):
    return 0.5 + 0.5 * np.tanh((np.array(x) - muLogit) / sigmaLogit)


# Added for DIA prior from pgm.py
def funcOneMinusLogit(x, muLogit, sigmaLogit):
    return 1.0 - logit(x, muLogit, sigmaLogit) + np.nextafter(0, 1)
