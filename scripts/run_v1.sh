#!/bin/bash

for num_prev in 0; do
    for times in {1..10}; do
        python cartpole_dqn_past_N_cartv1.py --num_prev $num_prev
    done
done