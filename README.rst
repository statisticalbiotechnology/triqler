Triqler: TRansparent Identification-Quantification-Linked Error Rates
=====================================================================

Requirements
------------

Python 2 or 3 installation

Packages needed:

- numpy 1.10+
- scipy 0.17+

Installation via ``pip``
************************

::

  pip install triqler

Installation from source
************************

::

  git clone https://github.com/statisticalbiotechnology/triqler.git
  cd triqler
  pip install .

Usage
-----

::

  usage: python -m triqler [-h] [--out_file OUT] [--fold_change_eval F]
                     [--decoy_pattern P] [--min_samples N] [--num_threads N]
                     [--ttest]
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
    --min_samples N       Minimum number of samples a peptide needed to be
                          quantified in. (default: 1)
    --num_threads N       Number of threads, by default this is equal to the
                          number of CPU cores available on the device. (default:
                          8)
    --ttest               Use t-test for evaluating differential expression
                          instead of posterior probabilities. (default: False)

Example
-------

A sample file ``iPRG2016.tsv`` is provided in the ``example`` folder. You can
run Triqler on this file by running the following command:

::

  python -m triqler --fold_change_eval 0.8 example/iPRG2016.tsv

Interface
---------

The simplest input format is a tab-separated file consisting of a header line 
followed by one PSM per line in the following format:

::

  run <tab> condition <tab> charge <tab> searchScore <tab> intensity <tab> peptide     <tab> proteins
  r1  <tab> 1         <tab> 2      <tab> 1.345       <tab> 21359.123 <tab> A.PEPTIDE.A <tab> proteinA <tab> proteinB 
  r2  <tab> 1         <tab> 2      <tab> 1.945       <tab> 24837.398 <tab> A.PEPTIDE.A <tab> proteinA <tab> proteinB 
  r3  <tab> 2         <tab> 2      <tab> 1.684       <tab> 25498.869 <tab> A.PEPTIDE.A <tab> proteinA <tab> proteinB
  ...
  r1  <tab> 1         <tab> 3      <tab> 0.452       <tab> 13642.232 <tab> A.NTPEPTIDE.- <tab> decoy_proteinA


Alternatively, if you have match-between-run probabilities, a slightly more
complicated input format can be used as input:

::

  run <tab> condition <tab> charge <tab> searchScore <tab> spectrumId <tab> linkPEP <tab> featureClusterId <tab> intensity <tab> peptide     <tab> proteins
  r1  <tab> 1         <tab> 2      <tab> 1.345       <tab> 3          <tab> 0.0     <tab> 1                <tab> 21359.123 <tab> A.PEPTIDE.A <tab> proteinA <tab> proteinB 
  r2  <tab> 1         <tab> 2      <tab> 1.345       <tab> 3          <tab> 0.021   <tab> 1                <tab> 24837.398 <tab> A.PEPTIDE.A <tab> proteinA <tab> proteinB 
  r3  <tab> 2         <tab> 2      <tab> 1.684       <tab> 4          <tab> 0.0     <tab> 1                <tab> 25498.869 <tab> A.PEPTIDE.A <tab> proteinA <tab> proteinB
  ...
  r1  <tab> 1         <tab> 3      <tab> 0.452       <tab> 6568       <tab> 0.15    <tab> 9845             <tab> 13642.232 <tab> A.NTPEPTIDE.- <tab> decoy_proteinA

Some remarks:

- For Triqler to work, it also needs decoy PSMs, preferably resulting from a 
  search engine search with a reversed protein sequence database concatenated
  to the target database.
- The intensities should **not** be log transformed, Triqler will do this 
  transformation for you.
- The search engine scores should be such that higher scores indicate a higher
  confidence in the PSM.
- We recommend usage of well calibrated search engine scores, e.g. the
  SVM scores from Percolator.
- Multiple proteins can be specified at the end of the line, separated by tabs. 
  However, it should be noted that Triqler currently discards shared peptides.

The output format is a tab-separated file consisting of a header line followed
by one protein per line in the following format:

::
  
  q_value <tab> posterior_error_prob <tab> protein <tab> num_peptides <tab> protein_id_PEP <tab> log2_fold_change <tab> diff_exp_prob_<FC> <tab> <condition1>:<run1> <tab> <condition1>:<run2> <tab> ... <tab> <conditionM>:<runN> <tab> peptides

Some remarks:

- The reported protein expressions are the expected value of the protein's
  expression in the run. They are calculated relative to the protein's mean 
  expression and are **not** log transformed.
- The reported fold change is log2 transformed and is the expected value based 
  on the posterior distribution of the fold change.
- If more than 2 treatment groups are present, separate files will be written
  out for each pairwise comparison with suffixes added before the file 
  extension, e.g. proteins.1vs3.tsv.

