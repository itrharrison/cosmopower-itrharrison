#!/usr/bin/env/python

# Author: H. T. Jense

"""
Cosmopower comes with a built-in wrapper for cobaya. This allows for very fast
parameter inference from cosmological data.

In this example, we will run a quick cosmological chain using cobaya to infer
cosmological parameters from the Planck 2018 data. Make sure you have cobaya
installed in your python environment.

We provide the `planck_cmb.yaml` yaml file to prescribe some simple parameter
ranges for a CMB emulator that computes the TT, TE, and EE CMB power spectra
that are measured by Planck.

Using the tutorials provided before, you can train your own emulator to compute
these quantities. This can take a while, so alternatively you can take a look
at the pre-generated emulators provided at
<https://github.com/cosmopower-organization>, which can be used with the
framework provided here.
"""

from cobaya import run
from cobaya.yaml import yaml_load
from cobaya.install import install
from cobaya.model import get_model
from cobaya.output import get_output
from cobaya.sampler import get_sampler

model = r"""
likelihood:
  planck_2018_lowl.EE:
  planck_2018_lowl.TT:
  planck_2018_highl_plik.TTTEEE_lite_native:

theory:
  cosmopower:
    package_file: planck_cmb.yaml
    stop_at_error: true

params:
  ombh2:
    prior:
      min: 0.020
      max: 0.024
    ref: 0.022
  omch2:
    prior:
      min: 0.10
      max: 0.14
    ref: 0.12
  cosmomc_theta:
    prior:
      min: 103.7e-4
      max: 104.4e-4
    ref: 104.1e-4
  logA:
    prior:
      min: 2.9
      max: 3.2
    ref: 3.05
  ns:
    prior:
      min: 0.92
      max: 1.01
    ref: 0.96
  tau:
    prior:
      min: 0.01
      max: 0.12
    ref: 0.05
  H0:
    derived: true
  sigma8:
    derived: true
"""

yaml = yaml_load(model)

install(yaml)

# Create the cobaya model.
model = get_model(model)

# The best-fit values from the Planck paper.
plik_best_fit = {
	"ombh2" : 22.383e-3,
	"omch2" : 12.011e-2,
	"cosmomc_theta" : 104.0909e-4,
	"logA" : 3.0448,
	"ns" : 0.96605,
	"tau" : 0.0543,
	"A_planck": 1.0,
}

# Compute the log-posterior at best-fit.
logpost = model.logposterior(plik_best_fit, as_dict = True)
print(logpost)

# Write to some output file
output = get_output(prefix="chains/example_planck", resume=False, force=True)

# Create an MCMC sampler for cobaya.
sampler = get_sampler({"mcmc": None}, model=model, output=output)

# Run the sampler!
run(sampler)
