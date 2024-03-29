#!/bin/bash

for SEED in 18 79 5 805 541 299 344 700 831 731
#for SEED in 18
do
#  python3 -m runs.plot_temperature_response_function --include_temperature --model_cls NNChillModel --model_name NNChillModel_japan_seed${SEED} --seed ${SEED} --locations japan_yedoensis
#  python3 -m runs.plot_temperature_response_function --include_temperature --model_cls NNChillModel --model_name NNChillModel_japan_seed18_torch${SEED} --seed 18 --locations japan_yedoensis_sargentii

  python3 -m runs.plot_temperature_response_function --include_temperature --model_cls NNChillModel --model_name NNChillModel_final_japan_seed${SEED}_cultivar --seed ${SEED} --locations japan_yedoensis_sargentii
  python3 -m runs.plot_temperature_response_function --include_temperature --model_cls NNChillModel --model_name NNChillModel_final_japan_seed${SEED}_cultivar_holdout --seed ${SEED} --locations japan_yedoensis_sargentii

  python3 -m runs.plot_temperature_response_function --include_temperature --model_cls NNChillModel --model_name NNChillModel_final_japan_seed${SEED}_yedoensis --seed ${SEED} --locations japan_yedoensis
  python3 -m runs.plot_temperature_response_function --include_temperature --model_cls NNChillModel --model_name NNChillModel_final_japan_seed${SEED}_yedoensis_holdout --seed ${SEED} --locations japan_yedoensis

  python3 -m runs.plot_temperature_response_function --include_temperature --model_cls NNChillModel --model_name NNChillModel_final_japan_seed${SEED} --seed ${SEED} --locations japan_yedoensis_sargentii
  python3 -m runs.plot_temperature_response_function --include_temperature --model_cls NNChillModel --model_name NNChillModel_final_switzerland_seed${SEED} --seed ${SEED} --locations switzerland
  python3 -m runs.plot_temperature_response_function --include_temperature --model_cls NNChillModel --model_name NNChillModel_final_southkorea_seed${SEED} --seed ${SEED} --locations south_korea

  python3 -m runs.plot_temperature_response_function --include_temperature --model_cls NNChillModel --model_name NNChillModel_final_switzerland_seed${SEED}_cultivar --seed ${SEED} --locations switzerland
  python3 -m runs.plot_temperature_response_function --include_temperature --model_cls NNChillModel --model_name NNChillModel_final_southkorea_seed${SEED}_cultivar --seed ${SEED} --locations south_korea


#  python3 -m runs.plot_temperature_response_function --include_temperature --model_cls NNChillModel --model_name NNChillModel_final_japan_era5_seed${SEED} --seed ${SEED} --locations japan_yedoensis_sargentii --temperature_src era5
done