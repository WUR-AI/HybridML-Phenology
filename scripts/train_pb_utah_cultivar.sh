#!/bin/bash

for SEED in 18 79 5 805 541 299 344 700 831 731
do
  python -m runs.fit_eval --model_cls UtahChillModel --include_temperature --model_name UtahChillModel_yedoensis_seed${SEED} --seed ${SEED} --locations japan_yedoensis
  python -m runs.fit_eval --model_cls UtahChillModel --include_temperature --model_name UtahChillModel_yedoensis_seed${SEED}_holdout --seed ${SEED} --locations japan_yedoensis --hold_out_locations

#  python -m runs.fit_eval --model_cls UtahChillModel --include_temperature --model_name UtahChillModel_sargentii_seed${SEED} --seed ${SEED} --locations japan_sargentii

  python -m runs.fit_eval --model_cls UtahChillModel --include_temperature --model_name UtahChillModel_south_korea_seed${SEED} --seed ${SEED} --locations south_korea
  python -m runs.fit_eval --model_cls UtahChillModel --include_temperature --model_name UtahChillModel_south_korea_seed${SEED}_holdout --seed ${SEED} --locations south_korea --hold_out_locations

  python -m runs.fit_eval --model_cls UtahChillModel --include_temperature --model_name UtahChillModel_switzerland_seed${SEED} --seed ${SEED} --locations switzerland
  python -m runs.fit_eval --model_cls UtahChillModel --include_temperature --model_name UtahChillModel_switzerland_seed${SEED}_holdout --seed ${SEED} --locations switzerland --hold_out_locations
done
