#!/bin/bash

# Parameters
FORECASTING_HORIZON=96
PAST_HISTORY=384
FREQ='1min'
TRAIN_LAG=15
TEST_LAG=15
ROLLING_METHOD='sum'
ROLLING_WINDOW=1 # 1 means no rolling window
FIT_SCALER='no'

echo '#######' 'FH'$FORECASTING_HORIZON-'PH'$PAST_HISTORY '#######'
python ./experiments/main.py --forecasting-horizon $FORECASTING_HORIZON --past-history $PAST_HISTORY --train-lag $TRAIN_LAG --test-lag $TEST_LAG --freq $FREQ --rolling-method $ROLLING_METHOD --rolling-window $ROLLING_WINDOW --fit-scaler $FIT_SCALER
