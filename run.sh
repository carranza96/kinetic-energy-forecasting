#!/bin/bash
FREQ='1min'
TRAIN_LAG=1
TEST_LAG=1

# Grid parameters
FORECASTING_HORIZONS=(60 240 720 1440)
PAST_HISTORY_FACTORS=(2 4)

for fh in ${FORECASTING_HORIZONS[*]}; do
    for ph in ${PAST_HISTORY_FACTORS[*]}; do
        echo '#######' 'FH'$fh-'PH'$((fh*ph)) '#######'
        python ./experiments/main.py --forecasting-horizon $fh --past-history $((fh*ph)) --train-lag $TRAIN_LAG --test-lag $TEST_LAG --freq $FREQ
    done
done