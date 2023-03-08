#!/bin/bash

# preliminaries.sh: Preliminary measures script for STREAM benchmark
#
# -----------------------------------------------------------------------------
#
# Runs preliminary experiments: measure a single datapoint for various
# experiment plans.
# In particular, we do not take care of minimizing any bias due to the order in
# which benchmarks are run, or due to the machine, or …
#
# -----------------------------------------------------------------------------
#
# authors: Raphaël Bleuse <raphael.bleuse@inria.fr>
# date: 2021-02-16
# version: 0.4


declare -rA RUNNERS=(
        [controller]="controller --controller-configuration"
        [identification]="identification --experiment-plan"
)
declare -rA RUNNERS_POSTRUN_SNAPSHOT_FILES=(
        [controller]="controller-runner.log"
        [identification]="identification-runner.log"
)
        declare -r runner="${1}"
        declare -r input="${2}"


	if xpctl "${runner_cmd[@]}" "${cfg}" -- \
                        "${BENCHMARK}" --iterationCount="${ITERATION_COUNT}" --problemSize="${PROBLEM_SIZE}"
                then
                        # identify execution as successful
                        touch "${LOGDIR}/SUCCESS"
                        tar --append --file="${archive}" --directory="${LOGDIR}" -- SUCCESS
                else
                        # identify execution as failed
                        touch "${LOGDIR}/FAIL"
                        tar --append --file="${archive}" --directory="${LOGDIR}" -- FAIL
                fi
