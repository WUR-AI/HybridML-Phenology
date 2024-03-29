#!/bin/bash

#
# SCRIPT TO GENERATE ALL RESULTS
# RUNNING EVERYTHING TAKES A LONG TIME (IN THE ORDER OF WEEKS WITH THE HARDWARE WE USED)
#

# FIT PROCESS-BASED MODELS
./scripts/train_pb_hour.sh
./scripts/train_pb_utah.sh
./scripts/train_pb_days.sh

# LEARN CHILL FUNCTION
./scripts/train_df_mlp.sh

# LEARN CHILL FUNCTION WITH CULTIVAR PARAMETERS
./scripts/train_df_mlp_cultivar.sh

# LEARN CHILL FUNCTION WITH CULTIVAR PARAMETERS AND HELD OUT LOCATIONS
./scripts/train_df_mlp_cultivar_holdout.sh

# LEARN A CHILL FUNCTION FOR THE YEDOENSIS CULTIVAR
./scripts/train_df_mlp_yedoensis.sh
# FREEZE CHILL FUNCTION WEIGHTS AND FIT REMAINING PARAMETERS TO DATA IN SOUTH KOREA
./scripts/fine_tune_yedoensis.sh

# FIT PROCESS-BASED MODELS WITH CULTIVAR PARAMETERS
./scripts/train_pb_hour_cultivar.sh
./scripts/train_pb_utah_cultivar.sh
./scripts/train_pb_days_cultivar.sh

# ABLATION
# - TRAIN DIFFERENTIABLE APPROXIMATION TO UTAH MODEL
./scripts/train_df_utah.sh
./scripts/train_df_mlp_era5.sh

# FIGURES


# PLOT UTAH CHILL FUNCTION AND DOUBLE LOGISTIC APPROXIMATION
python3 -m runs.plot_logistic

# PLOT TEMPERATURE RESPONSE
./scripts/plot_learned_temperature_function.sh