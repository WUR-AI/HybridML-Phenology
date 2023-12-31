#!/bin/bash

cd ..

#SEED=78

for SEED in 18 79 5 805 541 299 344 700 831 731
do
  #python3 -m runs.plot_temperature_response_function --include_temperature --model_cls NNChillModel --model_name NNChillModel_japan_seed${SEED} --seed ${SEED} --locations japan_yedoensis
  python3 -m runs.plot_temperature_response_function --include_temperature --model_cls NNChillModel --model_name NNChillModel_japan_seed${SEED}_decay --seed ${SEED} --locations japan_yedoensis_sargentii
done