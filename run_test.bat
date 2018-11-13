@echo off

python -m triqler --fold_change_eval 0.8 --out example/iPRG2016.proteins.tsv example/iPRG2016.tsv
if %errorlevel% == 0 (
  echo Test succeeded
) else (
  echo Test failed
)
