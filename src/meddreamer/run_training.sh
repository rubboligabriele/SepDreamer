#!/bin/bash
source /scratch/$USER/meddreamer/env/bin/activate
cd ~/MedDreamer-615C

python -u -m src.meddreamer.main \
  --configs behavior \
  --datadir data/meddreamer_dataset \
  --logdir logs \
  --device cuda \
  --ckptdir logs/2026-03-21/17-02-22_default_log_mimic_sepsis \
  --ckptepoch 2000