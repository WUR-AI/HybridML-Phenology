#!/bin/bash

#for SEED in 18 78 5 805 541
cd ..
for SEED in 805 541
do

  python3 -m runs.fit_eval --model_cls LocalChillHoursModel --include_temperature --model_name LocalChillHoursModel_seed${SEED} --seed ${SEED}
done

