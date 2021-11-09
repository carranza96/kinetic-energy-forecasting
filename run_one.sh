#!/bin/bash

# Parameters
FORECASTING_HORIZON=96
PAST_HISTORY=384
FREQUENCY=15min

echo '#######' 'FH'$FORECASTING_HORIZON-'PH'$PAST_HISTORY-'FREQ'$FREQUENCY '#######'
python ./experiments/main.py --forecasting-horizon $FORECASTING_HORIZON --past-history $PAST_HISTORY --frequency $FREQUENCY
