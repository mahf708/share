#!/bin/bash
#
# Chained PBS Pro submission for piControl ACE2-EAMv3 inference.
# 5,840 coupled steps per segment (1 step = 5 days = 80 years/segment).
#
# Usage: bash submit_picontrol.sh [start_segment] [end_segment]
#   Defaults: start=0, end=59
#   Example: bash submit_picontrol.sh      # submit all 60
#            bash submit_picontrol.sh 5     # resume from segment 5
#            bash submit_picontrol.sh 5 10  # submit segments 5-10

set -euo pipefail

INFERENCE_DIR=${HOME}/inference
SCRATCH_DIR=/gpfs/fs0/globalscratch/ac.ngmahfouz
TEMPLATE=${INFERENCE_DIR}/picontrol.yaml
RUN_DIR=${SCRATCH_DIR}/picontrol_run
ACE_VENV=${HOME}/ace/.venv/bin/activate

IC_DIR=${INFERENCE_DIR}/2025-11-24-E3SMv3-piControl-100yr-coupled-IC

# First segment's initial conditions
FIRST_OCEAN_IC=${IC_DIR}/0401-01_ocean_ic.nc
FIRST_ATMOS_IC=${IC_DIR}/0401-01_atmosphere_ic.nc

# PBS Pro settings — adjust as needed
ACCOUNT=E3SM
QUEUE=gpu
WALLTIME=05:00:00
NGPUS=1

# Note: Swing allows max 10 queued jobs and 4 running per user.
# Submit in batches of 10 if running more segments.
# Use 3rd argument to chain after an existing job ID.
START_SEG=${1:-0}
END_SEG=${2:-9}
CHAIN_AFTER=${3:-""}

mkdir -p "${RUN_DIR}"

PREV_JOBID="${CHAIN_AFTER}"

for seg in $(seq "$START_SEG" "$END_SEG"); do
    seg_label=$(printf "seg_%04d" "$seg")
    seg_dir=${RUN_DIR}/${seg_label}
    yaml_file=${RUN_DIR}/${seg_label}.yaml

    # Skip segments that already completed (have a restart file)
    if [[ -f "${seg_dir}/atmosphere/restart.nc" && -f "${seg_dir}/ocean/restart.nc" ]]; then
        echo "Segment ${seg_label} already complete, skipping."
        continue
    fi

    # Determine initial conditions
    if [[ "$seg" -eq 0 ]]; then
        ocean_ic=${FIRST_OCEAN_IC}
        atmos_ic=${FIRST_ATMOS_IC}
    else
        prev_label=$(printf "seg_%04d" $((seg - 1)))
        ocean_ic=${RUN_DIR}/${prev_label}/ocean/restart.nc
        atmos_ic=${RUN_DIR}/${prev_label}/atmosphere/restart.nc
    fi

    # Generate segment yaml from template
    sed \
        -e "s|EXPERIMENT_DIR_PLACEHOLDER|${seg_dir}|" \
        -e "s|OCEAN_IC_PLACEHOLDER|${ocean_ic}|" \
        -e "s|ATMOS_IC_PLACEHOLDER|${atmos_ic}|" \
        "${TEMPLATE}" > "${yaml_file}"

    # Write a PBS job script for this segment
    job_script=${RUN_DIR}/${seg_label}.pbs
    cat > "${job_script}" <<PBSEOF
#!/bin/bash -l
#PBS -N pictl_${seg_label}
#PBS -A ${ACCOUNT}
#PBS -q ${QUEUE}
#PBS -l select=1:ngpus=${NGPUS}
#PBS -l walltime=${WALLTIME}
#PBS -o ${RUN_DIR}/${seg_label}_pbs.out
#PBS -j oe

cd \${PBS_O_WORKDIR:-${RUN_DIR}}
source ${ACE_VENV}
python -m fme.coupled.inference ${yaml_file}
PBSEOF

    # Build qsub command
    QSUB_CMD="qsub"

    # Chain to previous job
    if [[ -n "$PREV_JOBID" ]]; then
        QSUB_CMD+=" -W depend=afterok:${PREV_JOBID}"
    fi

    # Submit
    JOBID=$(${QSUB_CMD} "${job_script}")

    echo "Submitted ${seg_label}: job ${JOBID} (IC: ${ocean_ic})"
    PREV_JOBID=${JOBID}
done

echo ""
echo "Submitted segments ${START_SEG}-${END_SEG} as a chain."
echo "Monitor with: qstat -u \$USER"
echo "Total simulated time: $((( END_SEG - START_SEG + 1 ) * 80)) years"
