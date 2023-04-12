import os

# Se mettre dans le dossier "md_cnn/model_training" avec la commande suivante

#sudo apt install python3.8-venv
os.system("python3 -m venv venv")
os.system("source venv/bin/activate")
os.system("pip install -r requirements.txt")
os.chdir("md_cnn/model_training")
os.system("tar -xf ../../input_data/fasta_files.tar.gz -C ../../input_data")
os.system("python3 run_MDCNN_for_conformal_prediction.py parameter_files/conv_conv_pool_accuracy.txt")