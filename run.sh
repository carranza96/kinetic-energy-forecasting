#!/bin/bash
FREQ='1min'
TRAIN_LAG=1
TEST_LAG=1
ROLLING_METHOD='sum'
ROLLING_WINDOW=1 # 1 means no rolling window
DAYS_TEST=90

# Grid parameters
FORECASTING_HORIZONS=(15 30 60 240)
PAST_HISTORY_FACTORS=(2 4 6)
# PAST_HISTORY_FACTORS=(2 4 6 8 10 20 40 60 80 100)
# PAST_HISTORY_FACTORS=(6)

for fh in ${FORECASTING_HORIZONS[*]}; do
    for ph in ${PAST_HISTORY_FACTORS[*]}; do
        echo '#######' 'FH'$fh-'PH'$((fh*ph)) '#######'
        python ./experiments/main.py --days-test $DAYS_TEST --forecasting-horizon $fh --past-history $((fh*ph)) --train-lag $TRAIN_LAG --test-lag $TEST_LAG --freq $FREQ --rolling-window $ROLLING_WINDOW
    done
done