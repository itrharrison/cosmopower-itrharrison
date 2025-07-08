#!/usr/bin/env/python

# Author: A. Spurio Mancini, H. T. Jense

import numpy as np
import cosmopower as cp
from cosmopower.parser import YAMLParser
from cosmopower.spectra import init_boltzmann_code, get_boltzmann_spectra
from cosmopower.dataset import Dataset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

"""
In the previous file, we generated an LHC and saved it to the file
`example/spectra/parameters.hdf5`. Now we'll generate the spectra associated
with this dataset.
"""
parser = YAMLParser("example.yaml")

"""
The function will load the samples from a file (if you've generated one,
otherwise it can also just create a new one from scratch).
"""
samples = parser.get_parameter_samples()
print(samples.keys())

"""
Cosmopower has a series of built-in interpreters for boltzmann codes. We can
directly load the code from the parser.
"""
boltzmann_code = init_boltzmann_code(parser)
extra_args = parser.boltzmann_extra_args

"""
First we'll simply generate the first sample and plot it directly.
"""
params = {k: samples[k][0] for k in parser.boltzmann_inputs}

print("Plot a spectrum with these parameters:")
for k in params:
    print(f"{k:10s} = {params[k]:.3e}")

"""
Note that this sample does not include the redshift z, we use that separately.
"""
get_boltzmann_spectra(parser, boltzmann_code, params, ["Pk/lin"], extra_args)

"""
And now we plot k vs P(k)
"""
k = parser.modes("Pk/lin")
Pk = boltzmann_code["Pk/lin"]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize = (8, 6))

ax.plot(k, Pk)
ax.semilogx()
ax.semilogy()

ax.set_xlabel(r"k")
ax.set_ylabel(r"P(k)")
ax.set_title(f"Linear matter power spectrum at $z = {samples['z'][0]:.3f}$")

plt.show()

"""
Now we save this to a dataset
"""
dataset = Dataset(parser, "Pk/lin", "Pk_lin.hdf5")
network_params = np.array([
    samples[k][0] for k in parser.network_input_parameters("Pk/lin")
])

with dataset:
    dataset.write_data(0, network_params, Pk)

"""
Now we just do this many more times!
"""
dataset.open()
for n in tqdm(range(1, parser.nsamples)):
    params = {k: samples[k][n] for k in parser.boltzmann_inputs}
    if get_boltzmann_spectra(parser, boltzmann_code, params, ["Pk/lin"], extra_args):
        # This function returns False if the sample is invalid, so it's a
        # simple check to discard invalid datapoints.
        network_params = np.array([
            samples[k][n] for k in parser.network_input_parameters("Pk/lin")
        ])
        dataset.write_data(n, network_params, Pk)
dataset.close()


