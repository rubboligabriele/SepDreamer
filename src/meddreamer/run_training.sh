#!/bin/bash
source /scratch/$USER/meddreamer/env/bin/activate
cd ~/MedDreamer-615C

python -m src.meddreamer.main \
  --configs defaults \
  --datadir data/meddreamer_dataset \
  --logdir logs \
  --device cuda \
  --epochs 100 \
  --batch_size 12 \
  --train_batch_length 10 \
  --eval_every 20 \
  --log_every 20 \
  --save_every 50