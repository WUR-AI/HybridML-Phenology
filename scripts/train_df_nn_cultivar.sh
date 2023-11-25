#!/bin/bash

cd ..
#for SEED in 18 79 5 805 541 299 344 700 831 731
#for SEED in 541 299 344 700 831 731
#for SEED in 344 700 831 731
for SEED in 805
#for SEED in 18 79 5 805 541
#for SEED in 299 344 700 831 731
do
#  python3 -m runs.fit_eval --model_cls NNChillModel --include_temperature --model_name NNChillModel_japan_seed${SEED}_cultivar --seed ${SEED} --num_epochs 20000 --optimizer adam --lr 1e-3 --scheduler_step_size 2000 --scheduler_decay 0.9 --loss_f nll --locations japan_yedoensis_sargentii --parameter_model_thc known_cultivars --parameter_model_thg known_cultivars --parameter_model_tbg known_cultivars  --parameter_model_slg known_cultivars
#  python3 -m runs.fit_eval --model_cls NNChillModel --include_temperature --model_name NNChillModel_japan_seed${SEED}_cultivar_holdout --seed ${SEED} --num_epochs 20000 --optimizer adam --lr 1e-3 --scheduler_step_size 2000 --scheduler_decay 0.9 --loss_f nll --locations japan_yedoensis_sargentii --parameter_model_thc known_cultivars --parameter_model_thg known_cultivars --parameter_model_tbg known_cultivars  --parameter_model_slg known_cultivars --hold_out_locations

  python -m runs.fit_eval --model_cls NNChillModel --include_temperature --model_name NNChillModel_switzerland_seed${SEED}_cultivar --seed ${SEED} --num_epochs 20000 --optimizer adam --lr 1e-3 --scheduler_step_size 2000 --scheduler_decay 0.9 --loss_f nll --locations switzerland --parameter_model_thc known_cultivars --parameter_model_thg known_cultivars --parameter_model_tbg known_cultivars  --parameter_model_slg known_cultivars
  python -m runs.fit_eval --model_cls NNChillModel --include_temperature --model_name NNChillModel_switzerland_seed${SEED}_cultivar_holdout --seed ${SEED} --num_epochs 20000 --optimizer adam --lr 1e-3 --scheduler_step_size 2000 --scheduler_decay 0.9 --loss_f nll --locations switzerland --parameter_model_thc known_cultivars --parameter_model_thg known_cultivars --parameter_model_tbg known_cultivars  --parameter_model_slg known_cultivars --hold_out_locations
  python -m runs.fit_eval --model_cls NNChillModel --include_temperature --model_name NNChillModel_southkorea_seed${SEED}_cultivar --seed ${SEED} --num_epochs 20000 --optimizer adam --lr 1e-3 --scheduler_step_size 2000 --scheduler_decay 0.9 --loss_f nll --locations south_korea --parameter_model_thc known_cultivars --parameter_model_thg known_cultivars --parameter_model_tbg known_cultivars  --parameter_model_slg known_cultivars
  python -m runs.fit_eval --model_cls NNChillModel --include_temperature --model_name NNChillModel_southkorea_seed${SEED}_cultivar_holdout --seed ${SEED} --num_epochs 20000 --optimizer adam --lr 1e-3 --scheduler_step_size 2000 --scheduler_decay 0.9 --loss_f nll --locations south_korea --parameter_model_thc known_cultivars --parameter_model_thg known_cultivars --parameter_model_tbg known_cultivars  --parameter_model_slg known_cultivars --hold_out_locations
done

# TODO -- some of seed 805