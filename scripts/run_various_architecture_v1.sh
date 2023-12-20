#!/bin/bash

for num_prev in 19; do
    for times in {1..15}; do
        python cartpole_dqn_past_N_various_architecture_cartv1.py --num_prev $num_prev;
        sleep 1
    done
done