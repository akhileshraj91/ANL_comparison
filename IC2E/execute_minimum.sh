#!/bin/bash


export LC_ALL=C  # ensure we are working with known locales
#set -e -u -f -o pipefail # safer shell script

declare -r PROGRAM=${0##*/}


# parameters  -----------------------------------------------------------------

# xpctl subcommand for each supported runner
declare -rA RUNNERS=(
	[controller]="controller --controller-configuration"
	[identification]="identification --experiment-plan"
)

declare -r BENCHMARK='stream_c'
declare -r ITERATION_COUNT='10_000'
declare -r PROBLEM_SIZE='33_554_432'


# configuration  --------------------------------------------------------------

declare -r LOGDIR='./results/minimum'  # all relative paths are relative to $LOGDIR
declare -r DATADIR='./good_dyna'
declare -r OUTPUTDIR='./results/minimum'

declare -r PARAMS_FILE='parameters.yaml'
declare -r TOPOLOGY_FILE='topology.xml'


# files to snapshot before running the experiment
declare -ra PRERUN_SNAPSHOT_FILES=(
	"${PARAMS_FILE}"
	"${TOPOLOGY_FILE}"
)

# pseudo-files from /proc to record
declare -ra SYSTEM_STATE_SNAPSHOT_FILES=(
	/proc/cpuinfo
	/proc/iomem
	/proc/loadavg
	/proc/meminfo
	/proc/modules
	/proc/stat
	/proc/sys/kernel/hostname
	/proc/uptime
	/proc/version
	/proc/vmstat
	/proc/zoneinfo
)

# common (i.e., shared by runners) files to snapshot once the experiment is done
declare -a POSTRUN_SNAPSHOT_FILES=(
	# outputs
	dump_pubMeasurements.csv
	dump_pubProgress.csv
	controller-runner.log
	#nrm.log
	#time-metrics.csv
)

# runner-specific files to snapshot once the experiment is done
declare -rA RUNNERS_POSTRUN_SNAPSHOT_FILES=(
	[controller]="controller-runner.log"
	[identification]="identification-runner.log"
)

# helper functions  -----------------------------------------------------------

function dump_parameters {
	declare -r timestamp="${1}"
	declare -r runner="${2}"
	declare -r cfg="${3}"
	declare -r benchmark="${4}"
	declare -r extra="${*:5}"

	cat <<- EOF > "${OUTPUTDIR}/${PARAMS_FILE}"
		timestamp: ${timestamp}
		runner: ${runner}
		config-file: ${cfg##*/}
		benchmark: ${benchmark}
		extra: ${extra}
	EOF
}


function snapshot_system_state {
	archive="${1}"
	subdir="${2}"

	# create unique namespace to work with
	wd=$(mktemp --directory)
	mkdir "${wd}/${subdir}"

	# snapshot
	for pseudofile in "${SYSTEM_STATE_SNAPSHOT_FILES[@]}"; do
		saveas="$(basename "${pseudofile}")"
		cat "${pseudofile}" > "${wd}/${subdir}/${saveas}"
	done

	# archive
	tar --append --file="${archive}" --directory="${wd}" --transform='s,^,sysstate/,' -- "${subdir}"

	# clean unique namespace
	rm --recursive --force -- "${wd}"
}

echo "$DATADIR"
count=1

for cfg in "$DATADIR"/*
do
	if [[ ${cfg} == *"dynamics"* ]]; then
		while [ $count -le 10 ]
		do
			echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>${cfg}"
			timestamp="$(date --iso-8601=seconds)"
			archive="${OUTPUTDIR}/preliminaries_${BENCHMARK}_${timestamp}.tar"
			dump_parameters "${timestamp}" "${RUNNER}" "${cfg}" "${BENCHMARK}" "--iterationCount=${ITERATION_COUNT} --problemSize=${PROBLEM_SIZE}"
			lstopo --output-format xml --whole-system --force "${OUTPUTDIR}/${TOPOLOGY_FILE}"
			tar --create --file="${archive}" --files-from=/dev/null
			tar --append --file="${archive}" --transform='s,^.*/,,' -- "${cfg}"
			tar --append --file="${archive}" --directory="${OUTPUTDIR}" -- "${PRERUN_SNAPSHOT_FILES[@]}"
			snapshot_system_state "${archive}" 'pre'
			python minimum.py max-range-config.yaml -- ones-stream-full 33554432 10000 ${cfg}
			# retrieve benchmark logs and snapshot post-run state
			tar --append --file="${archive}" --directory="${OUTPUTDIR}" -- "${POSTRUN_SNAPSHOT_FILES[@]}"
			touch "${OUTPUTDIR}/SUCCESS"
			tar --append --file="${archive}" --directory="${OUTPUTDIR}" -- SUCCESS
			snapshot_system_state "${archive}" 'post'
			# compress archive
			xz --compress "${archive}"
			sleep 5
			python  enforce_max_power.py max-range-config.yaml
			echo __________________________________________________________________________________________________
			sleep 10
			count=$(($count+1))
		done
	fi

done
