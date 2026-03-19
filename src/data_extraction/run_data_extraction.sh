#!/bin/bash

set -e  # stop if any command fails

echo "=============================="
echo "DATA EXTRACTION FROM BIGQUERY"
echo "=============================="

python -m src.data_extraction.extract \
    src/data_extraction/client_secret.json \
    mimic-iv-486516 \
    --out data/raw_data \
    --skip-existing

echo "=============================="
echo "DATA EXTRACTION DONE"
echo "=============================="