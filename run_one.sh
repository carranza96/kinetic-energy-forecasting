#!/bin/bash

# Parameters
FORECASTING_HORIZON=60
PAST_HISTORY=240

FREQ='1min'
TRAIN_LAG=1
TEST_LAG=1

echo '#######' 'FH'$FORECASTING_HORIZON-'PH'$PAST_HISTORY '#######'
python ./experiments/main.py --forecasting-horizon $FORECASTING_HORIZON --past-history $PAST_HISTORY --train-lag $TRAIN_LAG --test-lag $TEST_LAG --freq $FREQ
