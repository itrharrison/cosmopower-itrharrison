#!/usr/bin/env/python

# Author: A. Spurio Mancini, H. T. Jense

import cosmopower as cp
from cosmopower.parser import YAMLParser

"""
Compare the example.yaml file. It contains a basic overview of a linear P(k,z)
emulator we want to train from 4,000 example spectra in LCDM parameters. In
addition, we generate 500 validation spectra.

The `YAMLParser` object can load this file for us.
"""
parser = YAMLParser("example.yaml")

parser.setup_path(force_clean=True)

print(parser.all_parameters)

samples, validation_samples = parser.get_parameter_samples(return_validation = True)

print(samples)

parser.save_samples_to_file(samples, validation_samples)
