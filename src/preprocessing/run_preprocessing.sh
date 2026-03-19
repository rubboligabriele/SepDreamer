#!/bin/bash

set -e  # stop if any command fails

echo "=============================="
echo "STEP 1 — preprocess raw tables"
echo "=============================="

python -m src.preprocessing.01_preprocess_raw_data \
    --in data/raw_data \
    --out data/intermediates


echo "=============================="
echo "STEP 2 — build patient states"
echo "=============================="

python -m src.preprocessing.02_build_patient_states \
    data/intermediates/patient_states \
    --data data


echo "=============================="
echo "STEP 3 — build mask and delta"
echo "=============================="

python -m src.preprocessing.03_build_mask_and_delta \
    data/intermediates/patient_states/patient_states.csv \
    data/intermediates/patient_states/patient_states_clean.csv \
    --mask-out data/intermediates/patient_states/mask.csv \
    --delta-out data/intermediates/patient_states/delta.csv


echo "=============================="
echo "STEP 4 — build states and actions"
echo "=============================="

python -m src.preprocessing.04_build_states_and_actions \
    data/intermediates/patient_states/patient_states_clean.csv \
    data/intermediates/patient_states/actions.csv


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
    --output-dir data/final_cohort


echo "=============================="
echo "STEP 6 — build reward"
echo "=============================="

python -m src.preprocessing.06_build_reward \
    data/final_cohort/patient_states_filtered.csv \
    data/final_cohort/patient_states_with_reward.csv \
    --outcome-file data/final_cohort/sepsis_cohort.csv


echo "=============================="
echo "PREPROCESSING PIPELINE DONE"
echo "=============================="