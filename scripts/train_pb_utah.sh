#!/bin/bash

for SEED in 18 79 5 805 541 299 344 700 831 731
do
  python3 -m runs.fit_eval --model_cls LocalUtahChillModel --include_temperature --model_name LocalUtahChillModel_seed${SEED} --seed ${SEED}
done
