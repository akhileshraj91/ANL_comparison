#! /bin/sh.

python RL_model_hardware.py max-range-config.yaml -- ones-stream-full 33554432 100 optimal
ID = ps
echo ID
python RL_model_hardware.py max-range-config.yaml -- ones-stream-full 33554432 100 minimum
python RL_model_hardware.py max-range-config.yaml -- ones-stream-full 33554432 100 maximum
python plotting_histories_RL.py




