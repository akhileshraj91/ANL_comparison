#!/bin/sh
export LC_ALL=C
set -e -u -f -o pipefail

declare -r PROGRAM=${0##*/}

count=0

while [ $count -le 10 ]  
do
        filename='maximum'
	new_name=$filename'_'$count
	python RL_model_hardware.py max-range-config.yaml -- ones-stream-full 33554432 10000 $new_name
        #python enforce_max_power.py max-range-config.yaml -- ones-stream-full 100 100 $new_name
        filename='optimal'
        new_name=$filename'_'$count
        python RL_model_hardware.py max-range-config.yaml -- ones-stream-full 33554432 10000 $new_name
        #python enforce_max_power.py max-range-config.yaml -- ones-stream-full 100 100 $new_name
        filename='minimum'
        new_name=$filename'_'$count
	python RL_model_hardware.py max-range-config.yaml -- ones-stream-full 33554432 10000 $new_name
        python enforce_max_power.py max-range-config.yaml -- ones-stream-full 100 100 $new_name
 	count=$(($count+1))
done


