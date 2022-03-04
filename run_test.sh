#!/bin/bash

export OMP_NUM_THREADS=4
# to replace the reference results execute the following line:
#   rename -f 's/iPRG2016./iPRG2016_ref./g' example/iPRG2016.p*; mv example/iPRG2016.tsv.pqr.tsv example/iPRG2016_ref.tsv.pqr.tsv

python3 -m triqler --fold_change_eval 0.8 --out example/iPRG2016.proteins.tsv example/iPRG2016.tsv
if [[ $? -eq 0 ]]; then
  echo "Test succeeded"
else
  echo "Test failed"
  exit
fi

diff -q example/iPRG2016.proteins.1vs2.tsv example/iPRG2016_ref.proteins.1vs2.tsv
if [[ $? -eq 0 ]]; then
  echo "Test 1vs2 succeeded"
else
  echo "Test 1vs2 failed"
  exit
fi

diff -q example/iPRG2016.proteins.1vs3.tsv example/iPRG2016_ref.proteins.1vs3.tsv
if [[ $? -eq 0 ]]; then
  echo "Test 1vs3 succeeded"
else
  echo "Test 1vs3 failed"
  exit
fi

diff -q example/iPRG2016.proteins.2vs3.tsv example/iPRG2016_ref.proteins.2vs3.tsv
if [[ $? -eq 0 ]]; then
  echo "Test 2vs3 succeeded"
else
  echo "Test 2vs3 failed"
  exit
fi

python3 -m triqler --fold_change_eval 0.8 --out example/iPRG2016.proteins.tsv example/iPRG2016.tsv
if [[ $? -eq 0 ]]; then
  echo "Test succeeded"
else
  echo "Test failed"
  exit
fi

#diff -q example/iPRG2016.proteins.1vs2.tsv example/iPRG2016_ref.proteins.1vs2.tsv
#if [[ $? -eq 0 ]]; then
#  echo "Test 1vs2 succeeded"
#else
#  echo "Test 1vs2 failed"
#  exit
#fi

#diff -q example/iPRG2016.proteins.1vs3.tsv example/iPRG2016_ref.proteins.1vs3.tsv
#if [[ $? -eq 0 ]]; then
#  echo "Test 1vs3 succeeded"
#else
#  echo "Test 1vs3 failed"
#  exit
#fi

#diff -q example/iPRG2016.proteins.2vs3.tsv example/iPRG2016_ref.proteins.2vs3.tsv
#if [[ $? -eq 0 ]]; then
#  echo "Test 2vs3 succeeded"
#else
#  echo "Test 2vs3 failed"
#  exit
#fi
