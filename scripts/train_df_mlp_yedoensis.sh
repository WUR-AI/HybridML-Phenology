#!/bin/bash

for SEED in 18 79 5 805 541 299 344 700 831 731
do
  python3 -m runs.fit_eval --model_cls NNChillModel --include_temperature --model_name NNChillModel_final_japan_seed${SEED}_yedoensis --seed ${SEED} --num_epochs 20000 --optimizer adam --lr 1e-3 --weight_decay 1e-4 --scheduler_step_size 2000 --scheduler_decay 0.9 --loss_f nll --locations japan_yedoensis --parameter_model_thc known_cultivars --parameter_model_thg known_cultivars --parameter_model_tbg known_cultivars  --parameter_model_slg known_cultivars
done