#!/bin/bash

for SEED in 18 79 5 805 541 299 344 700 831 731
do
  python3 -m runs.fit_eval --model_cls LSTMModel --model_name LSTMModel_final_japan_seed${SEED}_yedoensis --seed ${SEED} --include_temperature --batch_size 512 --num_epochs 20000 --optimizer adam --lr 1e-3 --weight_decay 1e-4 --scheduler_step_size 2000 --scheduler_decay 0.9 --locations japan_yedoensis
  python3 -m runs.fit_eval --model_cls LSTMModel --model_name LSTMModel_final_japan_seed${SEED}_yedoensis_holdout --seed ${SEED} --include_temperature --batch_size 512 --num_epochs 20000 --optimizer adam --lr 1e-3 --weight_decay 1e-4 --scheduler_step_size 2000 --scheduler_decay 0.9 --locations japan_yedoensis --hold_out_locations
  python3 -m runs.fit_eval --model_cls LSTMModel --model_name LSTMModel_final_switzerland_seed${SEED} --seed ${SEED} --include_temperature --batch_size 512 --num_epochs 20000 --optimizer adam --lr 1e-3 --weight_decay 1e-4 --scheduler_step_size 2000 --scheduler_decay 0.9 --locations switzerland
  python3 -m runs.fit_eval --model_cls LSTMModel --model_name LSTMModel_final_switzerland_seed${SEED}_holdout --seed ${SEED} --include_temperature --batch_size 512 --num_epochs 20000 --optimizer adam --lr 1e-3 --weight_decay 1e-4 --scheduler_step_size 2000 --scheduler_decay 0.9 --locations switzerland --hold_out_locations
  python3 -m runs.fit_eval --model_cls LSTMModel --model_name LSTMModel_final_southkorea_seed${SEED} --seed ${SEED} --include_temperature --batch_size 512 --num_epochs 20000 --optimizer adam --lr 1e-3 --weight_decay 1e-4 --scheduler_step_size 2000 --scheduler_decay 0.9 --locations south_korea
  python3 -m runs.fit_eval --model_cls LSTMModel --model_name LSTMModel_final_southkorea_seed${SEED}_holdout --seed ${SEED} --include_temperature --batch_size 512 --num_epochs 20000 --optimizer adam --lr 1e-3 --weight_decay 1e-4 --scheduler_step_size 2000 --scheduler_decay 0.9 --locations south_korea --hold_out_locations
done
