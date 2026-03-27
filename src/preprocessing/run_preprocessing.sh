#!/bin/bash

set -e

FROM_STEP=1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --from-step) FROM_STEP="$2"; shift ;;
    esac
    shift
done

run_step () {
    STEP_NUM=$1
    if [ "$FROM_STEP" -le "$STEP_NUM" ]; then
        return 0
    else
        return 1
    fi
}

if run_step 1; then
echo "=============================="
echo "STEP 1 — preprocess raw tables"
echo "=============================="

python -m src.preprocessing.01_preprocess_raw_data \
    --in data/raw_data \
    --out data/intermediates
fi

if run_step 2; then
echo "=============================="
echo "STEP 2 — build patient states"
echo "=============================="

python -m src.preprocessing.02_build_patient_states \
    data/intermediates/patient_states \
    --data data
fi

if run_step 3; then
echo "=============================="
echo "STEP 3 — build mask and delta"
echo "=============================="

python -m src.preprocessing.03_build_mask_and_delta \
    data/intermediates/patient_states/patient_states.csv \
    data/intermediates/patient_states/patient_states_clean.csv \
    --mask-out data/intermediates/patient_states/mask.csv \
    --delta-out data/intermediates/patient_states/delta.csv
fi

if run_step 4; then
echo "=============================="
echo "STEP 4 — build states and actions"
echo "=============================="

python -m src.preprocessing.04_build_states_and_actions \
    data/intermediates/patient_states/patient_states_clean.csv \
    data/intermediates/patient_states/actions.csv
fi

if run_step 5; then
echo "=============================="
echo "STEP 5 — build sepsis cohort"
echo "=============================="

python -m src.preprocessing.05_build_sepsis_cohort \
    --states data/intermediates/patient_states/patient_states_clean.csv \
    --actions data/intermediates/patient_states/actions.csv \
    --demog data/intermediates/demog.csv \
    --mask data/intermediates/patient_states/mask.csv \
    --delta data/intermediates/patient_states/delta.csv \
    --qstime data/intermediates/patient_states/qstime.csv \
    --output-dir data/final_cohort \
    --keep-multiple-icu-stays
fi

if run_step 6; then
echo "=============================="
echo "STEP 6 — build reward"
echo "=============================="

python -m src.preprocessing.06_build_reward \
    data/final_cohort/patient_states_filtered.csv \
    data/final_cohort/patient_states_with_reward.csv \
    --outcome-file data/final_cohort/sepsis_cohort.csv
fi

if run_step 7; then
echo "=============================="
echo "STEP 7 — build MedDreamer episodes"
echo "=============================="

python -m src.preprocessing.07_build_meddreamer_episodes \
    --states data/final_cohort/patient_states_with_reward.csv \
    --actions data/final_cohort/actions_filtered.csv \
    --mask data/final_cohort/mask_filtered.csv \
    --delta data/final_cohort/delta_filtered.csv \
    --cohort data/final_cohort/sepsis_cohort.csv \
    --output data/meddreamer_dataset
fi

echo "=============================="
echo "PREPROCESSING PIPELINE DONE"
echo "=============================="