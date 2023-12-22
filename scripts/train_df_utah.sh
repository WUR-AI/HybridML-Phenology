#!/bin/bash

cd ..
for SEED in 18 79 5 805 541 299 344 700 831 731
do
  python3 -m runs.fit_eval --model_cls DiffUtahModel --include_temperature --model_name DiffUtahModel_japan_seed${SEED} --seed ${SEED} --num_epochs 20000 --optimizer adam --lr 1e-3 --scheduler_step_size 2000 --scheduler_decay 0.9 --loss_f nll --locations japan_yedoensis_sargentii
done