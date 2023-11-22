#!/bin/bash

cd ..
for SEED in 18 79 5 805 541
#for SEED in 299 344 700 831 731
do
  python3 -m runs.fine_tune --model_cls NNChillModel --include_temperature --num_epochs 20000 --optimizer adam --lr 1e-3 --locations south_korea --parameter_model known_cultivars --model_name NNChillModel_japan_seed${SEED}_cultivar --plot_level local --seed ${SEED}
done