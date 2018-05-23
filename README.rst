Triqler: TRansparent Identification-Quantification-Linked Error Rates
==========================================================================

Requirements
--------------

Python 2 or 3 installation

Packages needed:

- numpy 1.10+
- scipy 0.17+

Installation via ``pip``
*************************

::

  pip install triqler

Usage
-----

::

  python -m triqler [-h] -i I -e E
  optional arguments:
  -h, --help            show this help message and exit
  -i I                  Peptides abundances in CSV format. (default:
                        None)
  -e E                  \|log2 fold change\| soft evaluation threshold


Example
-----

- List of PSMs with abundances (not log transformed!) and search engine score.

::

  python -m triqler -i peptides.csv -e 0.8


