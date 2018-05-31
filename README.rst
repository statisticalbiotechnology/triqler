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

  usage: python -m triqler [-h] [--out_file OUT] [--decoy_pattern P] [--min_samples N]
                   [--fold_change_eval FC] [--ttest]
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
                          output.tsv)
    --decoy_pattern P     Prefix for decoy proteins. (default: decoy_)
    --min_samples N       Minimum number of samples a peptide needed to be
                          quantified in. (default: 1)
    --fold_change_eval FC
                          log2 fold change evaluation threshold. (default: 1.0)
    --ttest               Use t-test for evaluating differential expression
                          instead of posterior probabilities. (default: False)


Example
-------

A sample file ``iPRG2016.tsv`` is provided in the ``example`` folder. You can
run Triqler on this file by running the following command:

::

  python -m triqler -fold_change_eval 0.8 example/iPRG2016.tsv

The input format is a tab-separated file consisting of a header line followed 
by one PSM per line in the following format:

::

  run <tab> condition <tab> charge <tab> searchScore <tab> intensity <tab> peptide     <tab> proteins
  r1  <tab> 1         <tab> 2      <tab> 1.345       <tab> 21359.123 <tab> A.PEPTIDE.A <tab> proteinA <tab> proteinB 
  r2  <tab> 1         <tab> 2      <tab> 1.945       <tab> 24837.398 <tab> A.PEPTIDE.A <tab> proteinA <tab> proteinB 
  r3  <tab> 2         <tab> 2      <tab> 1.684       <tab> 25498.869 <tab> A.PEPTIDE.A <tab> proteinA <tab> proteinB

Some remarks:

- The intensities should **not** be log transformed, Triqler will do this 
  transformation for you.
- We recommend usage of well calibrated search engine scores, e.g. the
  SVM scores from Percolator.
- Multiple proteins can be specified at the end of the line, separated by tabs. 
  However, it should be noted that Triqler currently discards shared peptides.

The output format is a tab-separated file consisting of a header line followed
by one protein per line in the following format:

::
  
  



