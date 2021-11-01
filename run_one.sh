#!/bin/bash

# Parameters
FORECASTING_HORIZON=60
PAST_HISTORY=240

echo '#######' 'FH'$FORECASTING_HORIZON-'PH'$PAST_HISTORY '#######'
python ./experiments/main.py --forecasting-horizon $FORECASTING_HORIZON --past-history $PAST_HISTORY
