#!/usr/bin/env/python

# Author: A. Spurio Mancini, H. T. Jense

import cosmopower as cp
from cosmopower.parser import YAMLParser

"""
Compare the example.yaml file. It contains a basic overview of a linear P(k,z)
emulator we want to train from 400,000 example spectra in LCDM parameters.

The `YAMLParser` object can load this file for us.
"""
parser = YAMLParser("example.yaml")

parser.setup_path(force_clean=True)

print(parser.all_parameters)

lhc = parser.get_parameter_samples()

print(lhc)

parser.save_samples_to_file(lhc)
