#!/bin/bash

for num_prev in 0; do
    for times in {1..10}; do
        python cartpole_dqn_past_N_various_architecture.py --num_prev $num_prev;
        sleep 1
    done
done