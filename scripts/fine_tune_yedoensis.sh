#!/bin/bash

for SEED in 18 79 5 805 541 299 344 700 831 731
do
  python3 -m runs.fine_tune --model_cls NNChillModel --include_temperature --model_name NNChillModel_final_japan_seed${SEED}_yedoensis --seed ${SEED} --num_epochs 20000 --optimizer adam --lr 1e-3 --locations south_korea --scheduler_step_size 2000 --scheduler_decay 0.9 --loss_f nll --parameter_model_thc known_cultivars --parameter_model_thg known_cultivars --parameter_model_tbg known_cultivars  --parameter_model_slg known_cultivars --plot_level local
#  python3 -m runs.fine_tune --model_cls NNChillModel --include_temperature --model_name NNChillModel_final_japan_seed${SEED}_cultivar --seed ${SEED} --num_epochs 20000 --optimizer adam --lr 1e-3 --locations south_korea --scheduler_step_size 2000 --scheduler_decay 0.9 --loss_f nll --parameter_model_thc known_cultivars --parameter_model_thg known_cultivars --parameter_model_tbg known_cultivars  --parameter_model_slg known_cultivars --plot_level local

done