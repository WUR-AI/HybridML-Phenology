#!/bin/bash

for SEED in 18 79 5 805 541 299 344 700 831 731
do
  python -m runs.fit_eval --model_cls LSTMModelLocal --model_name LSTMModelLocal_final_japan_seed${SEED} --seed ${SEED} --include_temperature --batch_size 512 --num_epochs 20000 --optimizer adam --lr 1e-3 --weight_decay 1e-4 --scheduler_step_size 2000 --scheduler_decay 0.9 --locations japan_yedoensis_sargentii
  python -m runs.fit_eval --model_cls LSTMModelLocal --model_name LSTMModelLocal_final_switzerland_seed${SEED} --seed ${SEED} --include_temperature --batch_size 512 --num_epochs 20000 --optimizer adam --lr 1e-3 --weight_decay 1e-4 --scheduler_step_size 2000 --scheduler_decay 0.9 --locations switzerland
  python -m runs.fit_eval --model_cls LSTMModelLocal --model_name LSTMModelLocal_final_southkorea_seed${SEED} --seed ${SEED} --include_temperature --batch_size 512 --num_epochs 20000 --optimizer adam --lr 1e-3 --weight_decay 1e-4 --scheduler_step_size 2000 --scheduler_decay 0.9 --locations south_korea
done
