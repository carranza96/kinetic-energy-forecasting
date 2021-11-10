#!/bin/bash

# Parameters
FORECASTING_HORIZON=96
PAST_HISTORY=384
FREQ='1min'
TRAIN_LAG=15
TEST_LAG=15

echo '#######' 'FH'$FORECASTING_HORIZON-'PH'$PAST_HISTORY '#######'
python ./experiments/main.py --forecasting-horizon $FORECASTING_HORIZON --past-history $PAST_HISTORY --train-lag $TRAIN_LAG --test-lag $TEST_LAG --freq $FREQ
