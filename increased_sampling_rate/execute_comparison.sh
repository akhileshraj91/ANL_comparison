#/bin/sh
export LC_ALL=C
set -e -u -f -o pipefail

declare -r PROGRAM=${0##*/}

count=0
filename='optimal'
while [ $count -le 0 ]  
do
#	count=$(($count+1))
	new_name=$filename'_'$count
	echo $new_name
	python RL_model_hardware.py max-range-config.yaml -- ones-stream-full 33554432 10000 $new_name
        count=$(($count+1))
done


