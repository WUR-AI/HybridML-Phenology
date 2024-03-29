#!/bin/bash

python3 -m runs.plot_unit_progression --model_cls NNChillModel --include_temperature --locations japan_yedoensis_sargentii --model_name NNChillModel_final_japan_seed18 --seed 18

python3 -m runs.plot_unit_progression --model_cls NNChillModel --include_temperature --locations switzerland --model_name NNChillModel_final_switzerland_seed18 --seed 18

python3 -m runs.plot_unit_progression --model_cls NNChillModel --include_temperature --locations south_korea --model_name NNChillModel_final_southkorea_seed18 --seed 18
