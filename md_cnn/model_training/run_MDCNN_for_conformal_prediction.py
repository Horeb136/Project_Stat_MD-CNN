#!/usr/bin/env python
# coding: utf-8
"""
Runs multitask model with conv-conv-pool architecture:
- training on entire train set
- accuracy evaluation on held-out test set
This is the architecture used for the final MD-CNN model

Authors:
	Michael Chen (original version)
	Anna G. Green
	Chang Ho Yoon
"""

import sys
import glob
import os
import yaml
import sparse

import tensorflow as tf
import keras.backend as K
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tb_cnn_codebase import *

drugs = ['RIFAMPICIN', 'ISONIAZID', 'PYRAZINAMIDE',
             'ETHAMBUTOL', 'STREPTOMYCIN', 'LEVOFLOXACIN',
             'CAPREOMYCIN', 'AMIKACIN', 'MOXIFLOXACIN',
             'OFLOXACIN', 'KANAMYCIN', 'ETHIONAMIDE',
             'CIPROFLOXACIN']
num_drugs = len(drugs)

def run():

    def get_conv_nn():

		#TODO: replace X.shape with passed argument
        model = models.Sequential()
		#TODO: add filter size argument
        filter_size = 12 #see the article page 11
        model.add(layers.Conv2D(
            64, (5, filter_size),
            data_format='channels_last',
            activation='relu',
            input_shape = X.shape[1:]
        ))
        model.add(layers.Lambda(lambda x: K.squeeze(x, 1)))
        model.add(layers.Conv1D(64, 12, activation='relu'))
        model.add(layers.MaxPooling1D(3))
        model.add(layers.Conv1D(32, 3, activation='relu'))
        model.add(layers.Conv1D(32, 3, activation='relu'))
        model.add(layers.MaxPooling1D(3))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu', name='d1'))
        model.add(layers.Dense(256, activation='relu', name='d2'))
        model.add(layers.Dense(13, activation='sigmoid', name='d4'))

        opt = Adam(learning_rate=np.exp(-1.0 * 9))

        model.compile(optimizer=opt,
                      loss=masked_multi_weighted_bce,
                      metrics=[masked_weighted_accuracy])

        return model

    class myCNN:
        """
        Class for handling CNN functionality

        """
        def __init__(self):
            self.model = get_conv_nn()
            self.epochs = N_epochs

        def fit_model(self, X_train, y_train, X_val=None, y_val=None):
            """
            X_train: np.ndarray
                n_strains x 5 (one-hot) x longest locus length x no. of loci
                Genotypes of isolates used for training
            y_train: np.ndarray
                Labels for isolates used for training

            X_val: np.ndarray (optional, default=None)
                Optional genotypes of isolates in validation set

            y_val: np.ndarray (optional, default=None)
                Optional labels for isolates in validation set

            Returns
            -------
            pd.DataFrame:
                training history (accuracy, loss, validation accuracy, and validation loss) per epoch

            """
            if X_val is not None and y_val is not None:
                history = self.model.fit(
                    X_train, y_train,
                    epochs=self.epochs,
                    validation_data=(X_val, y_val),
                    batch_size=128
                )
                print('\nhistory dict:', history.history)
                return pd.DataFrame.from_dict(data=history.history)
            else:
                history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=128)
                print('\nhistory dict:', history.history)
                return pd.DataFrame.from_dict(data=history.history)

        def predict(self, X_val):
            return np.squeeze(self.model.predict(X_val))


    ### Prepare the test data - held out strains
    def compute_y_pred(df_geno_pheno_test):
        """
        Computes predicted phenotypes
        """

        # Get the numeric encoding for current subset of isolates
        df_geno_pheno_test = df_geno_pheno_test.fillna('-1')
        y_all_test, y_all_test_array = rs_encoding_to_numeric(df_geno_pheno_test, drugs)

        # Make sure that we have phenotype for at least one drug
        ind_with_phenotype_test = y_all_test.index[y_all_test.sum(axis=1) != -num_drugs] + int(df_geno_pheno_test.index[0])
        ind_with_phenotype_test_0index = y_all_test.index[y_all_test.sum(axis=1) != -num_drugs]

        # Get x indices for which we have phenotype
        X = X_sparse_test[ind_with_phenotype_test]
        print("the shape of X_test is {}".format(X.shape))

        y_test = y_all_test_array[ind_with_phenotype_test_0index]
        del y_all_test_array
        del y_all_test

        print("the shape of y_test is {}".format(y_test.shape))

        print('Predicting for test data...')
        y_pred = model.predict(X.todense())

        return y_pred, y_test


    _, input_file = sys.argv

    # load kwargs from config file (input_file)
    kwargs = yaml.safe_load(open(input_file, "r"))
    print(kwargs)

    output_path = kwargs["output_path"]
    N_epochs = kwargs["N_epochs"]
    filter_size = kwargs["filter_size"]
    pkl_file_sparse_train = kwargs['pkl_file_sparse_train']
    pkl_file_sparse_test = kwargs['pkl_file_sparse_test']

    # Determine whether pickle already exists
    if os.path.isfile(kwargs["pkl_file"]):
        print("pickle file already exists, proceeding with modeling")
    else:
        print("creating genotype phenotype pickle")
        make_geno_pheno_pkl(**kwargs)

    df_geno_pheno = pd.read_pickle(kwargs["pkl_file"])
    y_all_train, y_array = rs_encoding_to_numeric(df_geno_pheno.query("category=='set1_original_10202'"), drugs)


    if os.path.isfile(pkl_file_sparse_train) and os.path.isfile(pkl_file_sparse_test):
        print("X input already exists, loading X")
        X_sparse_train = sparse.load_npz(pkl_file_sparse_train)
        X_sparse_test = sparse.load_npz(pkl_file_sparse_test)

    else:
        print("creating X pickle")
        X_all = create_X(df_geno_pheno)
        X_sparse = sparse.COO(X_all)
        del X_all
        sparse.save_npz("X_all.pkl", X_sparse, compressed=False)

        print("splitting X into train and test")
        X_sparse = sparse.load_npz("X_all.pkl.npz")
        X_sparse_test, X_sparse_train = split_into_traintest(X_sparse, df_geno_pheno, "set1_original_10202")

    # ### obtain isolates with at least 1 resistance status to length of drugs
    ind_with_phenotype = np.where(y_all_train.sum(axis=1) != -num_drugs)

    X = X_sparse_train[ind_with_phenotype]
    print("the shape of X is {}".format(X.shape))

    y = y_array[ind_with_phenotype]
    print("the shape of y is {}".format(y.shape))

    ### Train the model on the entire training set - no CV splits
    saved_model_path = kwargs['saved_model_path']

    if os.path.isdir(saved_model_path):
        model = models.load_model(saved_model_path, custom_objects={
            'masked_weighted_accuracy': masked_weighted_accuracy,
            "masked_multi_weighted_bce": masked_multi_weighted_bce
        })
    else:
        print("Did not find model", saved_model_path)
        model = myCNN()
        X_train = X.todense()
        print('fitting..')
        alpha_matrix = load_alpha_matrix(kwargs["alpha_file"], y, df_geno_pheno, **kwargs)
        history = model.fit_model(X_train, alpha_matrix)
        history.to_csv(output_path + "history.csv")
        model.save(saved_model_path)
    #
    # ## Get the thresholds for evaluation
    print("Predicting for training data...")
    y_train_pred = model.predict(X.todense())
    y_train = y_array[ind_with_phenotype]

    # Select the prediction threshold for each drug based on TRAINING SET DATA
    _data = []
    for idx, drug in enumerate(drugs):
        # Calculate the threshold from the TRAINING data, not the test data
        print("getting threshold for", drug)
        val__ = get_threshold_val(y_train[:, idx], y_train_pred[:, idx])
        val__["drug"] = drug
        _data.append(val__)
    threshold_data = pd.DataFrame(_data)
    threshold_data.to_csv(kwargs['threshold_file'])

    drug_to_threshold = {x:y for x,y in zip(threshold_data.drug, threshold_data.threshold)}

    ## Compute AUC for training set data
    results = compute_drug_auc_table(y_train, y_train_pred, drug_to_threshold)
    results.to_csv(f"{output_path}_training_set_drug_auc.csv")

    # Compute AUC for test set data
    df_geno_pheno = df_geno_pheno.query("category != 'set1_original_10202'")
    df_geno_pheno = df_geno_pheno.reset_index(drop=True)

    y_pred, y_test = compute_y_pred(df_geno_pheno)
    results = compute_drug_auc_table(y_test, y_pred, drug_to_threshold)
    results.to_csv(f"{output_path}_test_set_drug_auc.csv")
    #y_pred.to_csv("y_pred.csv")
    #y_test.to_csv("y_test.csv")
    np.savetxt("y_pred.csv", y_pred, delimiter=',')
    np.savetxt("y_test.csv", y_test, delimiter=',')
run()
