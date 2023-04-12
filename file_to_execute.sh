#!/bin/sh

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd md_cnn/model_training
tar -xvf ../../input_data/fasta_files.tar.gz -C ../../input_data
python3 run_MDCNN_for_conformal_prediction.py parameter_files/conv_conv_pool_accuracy.txt