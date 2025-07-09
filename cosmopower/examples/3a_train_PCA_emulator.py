#!/usr/bin/env/python

# Author: A. Spurio Mancini, H. T. Jense

import os, glob
import numpy as np
import cosmopower as cp
import tensorflow as tf
from cosmopower import cosmopower_PCA, cosmopower_PCAplusNN
from cosmopower.parser import YAMLParser
from cosmopower.dataset import Dataset

"""
In the previous file, we created a training dataset for a linear P(k,z) emulator.
Here, we will train the emulator over this dataset.
"""
parser = YAMLParser("example.yaml")

network_settings = parser.settings("Pk/lin")

print(network_settings)

"""
We assume that we simply want to train a dense neural network.
There is also the `3_train_emulator.py` file which does the same as this
file but trains an NN type emulator.
"""
input_files = glob.glob(os.path.join(parser.path, "spectra", "Pk_lin.*hdf5"))
output_file = os.path.join(parser.path, "networks", parser.network_filename("Pk/lin"))

print("Training the linear P(k) emulator.")
print(f"Input files: {input_files}")
print(f"Output file: {output_file}")

settings = parser.get_traits("Pk/lin")
parameters = parser.network_input_parameters("Pk/lin")
modes = parser.modes("Pk/lin")
datasets = [Dataset(parser, "Pk/lin", os.path.basename(f)) for f in input_files]

"""
We need to set up the PCA manually.
You can do this through the cosmopower pipeline via a traits block like:
  n_traits:
    n_hidden: [512, 512, 512]
  p_traits:
    n_pcas: 100
    n_batches: 2
where `n_pcas` is the number of PCA components retained, and `n_batches` is the
number of batches over which the PCA is trained.
"""
n_pcas = 25
n_batches = 2

pca = cosmopower_PCA(parameters=parameters, modes=modes, n_pcas=n_pcas,
                     n_batches=n_batches, verbose=True)

network = cosmopower_PCAplusNN(cp_pca=pca, verbose=True, trainable=True,
                               **settings.get("n_traits", {}))

with tf.device("/device:CPU:0"):
    network.train(training_data=datasets,
                  filename_saved_model=output_file,
                  **parser.network_training_parameters("Pk/lin"))
