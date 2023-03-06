
#!/bin/sh



search_dir=./experiment_results/
for entry in "$search_dir"*
do
  echo $entry
  base_name=$(basename ${entry})
  echo $base_name
  #python plotting_histories_RL.py
done
