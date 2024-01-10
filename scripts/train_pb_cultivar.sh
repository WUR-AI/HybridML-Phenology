#!/bin/bash

for SEED in 18
do
  python -m runs.fit_eval --model_cls UtahChillModel --include_temperature --model_name UtahChillModel_yedoensis_seed${SEED} --seed ${SEED} --locations japan_yedoensis
  python -m runs.fit_eval --model_cls UtahChillModel --include_temperature --model_name UtahChillModel_sargentii_seed${SEED} --seed ${SEED} --locations japan_sargentii
  python -m runs.fit_eval --model_cls UtahChillModel --include_temperature --model_name UtahChillModel_south_korea_seed${SEED} --seed ${SEED} --locations south_korea
  python -m runs.fit_eval --model_cls UtahChillModel --include_temperature --model_name UtahChillModel_switzerland_seed${SEED} --seed ${SEED} --locations switzerland
done

