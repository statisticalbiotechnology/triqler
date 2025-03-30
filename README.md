# Triqler: TRansparent Identification-Quantification-Linked Error Rates

[![PyPI version](https://img.shields.io/pypi/v/triqler.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/triqler/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/triqler.svg?logo=python&logoColor=FFE873)](https://pypi.org/project/triqler/)
[![PyPI downloads](https://img.shields.io/pypi/dm/triqler.svg)](https://pypistats.org/packages/triqler)

Triqler is a probabilistic graphical model that propagates error
information through all steps from MS1 feature to protein level,
employing distributions in favor of point estimates, most notably for
missing value imputation. The model outputs posterior probabilities for
fold changes between treatment groups, highlighting uncertainty rather
than hiding it.

For a detailed explanation of how to install and run Triqler
(stand-alone or in combination with MaxQuant, Quandenser or Dinosaur) as
well as how to interpret the results, please read our [Triqler user
manual](https://www.biorxiv.org/content/10.1101/2020.09.24.311605v1).

Brief instructions for installing and running Triqler as well as
descriptions of the input and output formats can be found below.
Instructions for running the converters to the Triqler input format are
available in our
[wiki](https://github.com/statisticalbiotechnology/triqler/wiki).

Supported search engines:
- DIA-NN
- Dinosaur+Percolator
- MaxQuant
- Quandenser

## Method description / Citation

The, M. & Käll, L. (2019). Integrated identification and quantification
error probabilities for shotgun proteomics. *Molecular & Cellular
Proteomics, 18* (3), 561-570. <https://doi.org/10.1074/mcp.RA118.001018>

Truong, P., The, M., & Käll, L. (2023). Triqler for Protein
Summarization of Data from Data-Independent Acquisition Mass
Spectrometry. *Journal of Proteome Research, 22* (4), 1359-1366.
<https://doi.org/10.1021/acs.jproteome.2c00607>

### Installation via `pip`

    pip install triqler

### Installation from source

    git clone https://github.com/statisticalbiotechnology/triqler.git
    cd triqler
    pip install .

## Usage

    usage: triqler [-h] [--out_file OUT] [--fold_change_eval F]
                 [--decoy_pattern P] [--missing_value_prior D] [--min_samples N]
                 [--num_threads N] [--ttest] [--write_spectrum_quants]
                 [--write_protein_posteriors P_OUT]
                 [--write_group_posteriors G_OUT]
                 [--write_fold_change_posteriors F_OUT]
                 [--csv-field-size-limit CSV_FIELD_SIZE_LIMIT]
                 IN_FILE

    positional arguments:
      IN_FILE               List of PSMs with abundances (not log transformed!)
                            and search engine score. See README for a detailed
                            description of the columns.

    optional arguments:
      -h, --help            show this help message and exit
      --out_file OUT        Path to output file (writing in TSV format). N.B. if
                            more than 2 treatment groups are present, suffixes
                            will be added before the file extension. (default:
                            proteins.tsv)
      --fold_change_eval F  log2 fold change evaluation threshold. (default: 1.0)
      --decoy_pattern P     Prefix for decoy proteins. (default: decoy_)
      --missing_value_prior D
                            Distribution to fit for missing value prior. Use "DIA"
                            for using means of NaNs to fit the censored normal
                            distribution. The "default" option fits the censored
                            normal distribution with all observed XIC values.
                            (default: default)
      --min_samples N       Minimum number of samples a peptide needed to be
                            quantified in. (default: 2)
      --num_threads N       Number of threads, by default this is equal to the
                            number of CPU cores available on the device. (default:
                            6)
      --ttest               Use t-test for evaluating differential expression
                            instead of posterior probabilities. (default: False)
      --write_spectrum_quants
                            Write quantifications for consensus spectra. Only
                            works if consensus spectrum index are given in input.
                            (default: False)
      --write_protein_posteriors P_OUT
                            Write raw data of protein posteriors to the specified
                            file in TSV format. (default: )
      --write_group_posteriors G_OUT
                            Write raw data of treatment group posteriors to the
                            specified file in TSV format. (default: )
      --write_fold_change_posteriors F_OUT
                            Write raw data of fold change posteriors to the
                            specified file in TSV format. (default: )
      --csv-field-size-limit CSV_FIELD_SIZE_LIMIT
                            Set a new maximum CSV field size (default: None)

## Example

A sample file `iPRG2016.tsv` is provided in the `example` folder. You
can run Triqler on this file by running the following command:

    python -m triqler --fold_change_eval 0.8 example/iPRG2016.tsv

A detailed example of the different levels of Triqler output can be
found in [Supplementary Note
2](https://www.nature.com/articles/s41467-020-17037-3#Sec13) of the
Quandenser publication.

## Interface

The simplest input format is a tab-separated file consisting of a header
line followed by one PSM per line in the following format:

    run <tab> condition <tab> charge <tab> searchScore <tab> intensity <tab> peptide     <tab> proteins
    r1  <tab> 1         <tab> 2      <tab> 1.345       <tab> 21359.123 <tab> A.PEPTIDE.A <tab> proteinA <tab> proteinB 
    r2  <tab> 1         <tab> 2      <tab> 1.945       <tab> 24837.398 <tab> A.PEPTIDE.A <tab> proteinA <tab> proteinB 
    r3  <tab> 2         <tab> 2      <tab> 1.684       <tab> 25498.869 <tab> A.PEPTIDE.A <tab> proteinA <tab> proteinB
    ...
    r1  <tab> 1         <tab> 3      <tab> 0.452       <tab> 13642.232 <tab> A.NTPEPTIDE.- <tab> decoy_proteinA

Alternatively, if you have match-between-run probabilities, a slightly
more complicated input format can be used as input:

    run <tab> condition <tab> charge <tab> searchScore <tab> spectrumId <tab> linkPEP <tab> featureClusterId <tab> intensity <tab> peptide     <tab> proteins
    r1  <tab> 1         <tab> 2      <tab> 1.345       <tab> 3          <tab> 0.0     <tab> 1                <tab> 21359.123 <tab> A.PEPTIDE.A <tab> proteinA <tab> proteinB 
    r2  <tab> 1         <tab> 2      <tab> 1.345       <tab> 3          <tab> 0.021   <tab> 1                <tab> 24837.398 <tab> A.PEPTIDE.A <tab> proteinA <tab> proteinB 
    r3  <tab> 2         <tab> 2      <tab> 1.684       <tab> 4          <tab> 0.0     <tab> 1                <tab> 25498.869 <tab> A.PEPTIDE.A <tab> proteinA <tab> proteinB
    ...
    r1  <tab> 1         <tab> 3      <tab> 0.452       <tab> 6568       <tab> 0.15    <tab> 9845             <tab> 13642.232 <tab> A.NTPEPTIDE.- <tab> decoy_proteinA

Some remarks:

-   For Triqler to work, it also needs decoy PSMs, preferably resulting
    from a search engine search with a reversed protein sequence
    database concatenated to the target database.
-   The intensities should **not** be log transformed, Triqler will do
    this transformation for you.
-   An intensity of 0 is considered a missing value and the row will be
    discarded.
-   The search engine scores should be such that higher scores indicate
    a higher confidence in the PSM.
-   We recommend usage of well calibrated search engine scores, e.g. the
    SVM scores from Percolator.
-   Do **not** set \--fold_change_eval to 0 or a very low value (\<0.2).
    The fold change posterior distribution always has a certain width,
    reflecting the uncertainty of our estimation. Even if the fold
    change is 0, this distribution will necessarily spill over into low
    fold change values, without there being any ground for differential
    expression.
-   Multiple proteins can be specified at the end of the line, separated
    by tabs. However, it should be noted that Triqler currently discards
    shared peptides.

The output format is a tab-separated file consisting of a header line
followed by one protein per line in the following format:

    q_value <tab> posterior_error_prob <tab> protein <tab> num_peptides <tab> protein_id_PEP <tab> log2_fold_change <tab> diff_exp_prob_<FC> <tab> <condition1>:<run1> <tab> <condition1>:<run2> <tab> ... <tab> <conditionM>:<runN> <tab> peptides

Some remarks:

-   The *q_value* and *posterior_error_prob* columns represent
    respectively the FDR and PEP for the hypothesis that the protein was
    correctly identified and has a fold change larger than the specified
    \--fold_change_eval.
-   The *protein_id_PEP* and *diff_exp_prob\_\<FC\>* columns are simply
    the separate probabilities that make up the above hypothesis test,
    i.e. for correct identification and for fold change respectively.
-   The reported fold change is log2 transformed and is the expected
    value based on the posterior distribution of the fold change.
-   If more than 2 treatment groups are present, separate files will be
    written out for each pairwise comparison with suffixes added before
    the file extension, e.g. proteins.1vs3.tsv.
-   The reported protein expressions per run are the expected value of
    the protein\'s expression in that run. They represent relative
    values (**not** log transformed) to the protein\'s mean expression
    across all runs, which itself would correspond to the value 1.0. For
    example, a value of 1.5 means that the expression in this sample is
    50% higher than the mean across all runs. A second example comparing
    values across samples: if sample1 has a value of 2.0 and sample2 a
    value of 1.5, it means that the expression in sample1 is 33% higher
    than in sample2 (2.0/1.5=1.33). We don\'t necessarily recommend
    using these values for downstream analysis, as the idea is that the
    actual value of interest is the fold change between treatment groups
    rather than between samples.
