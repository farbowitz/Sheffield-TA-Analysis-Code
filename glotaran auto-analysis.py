# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 21:29:18 2022

@author: Daniel
"""

import pyglotaran_extras
import glotaran.io

import os

import pandas as pd
import xarray as xr


#recommended in documentation
from glotaran.io import load_model
from glotaran.io import load_parameters
from glotaran.io import save_dataset
from glotaran.io.prepare_dataset import prepare_time_trace_dataset
from glotaran.optimization.optimize import optimize
from glotaran.project.scheme import Scheme

from glotaran.testing.simulated_data.sequential_spectral_decay import DATASET as dataset

print(dataset)

location = 'C:/Users/Daniel/Desktop/Solar Cell Technology MSc/Thesis Project - Water-based nanoparticle ink OPVs/TA Spectroscope data/PBDB-T;ITIC/19_03_2021 - UV-Vis Emmas Solutions/'
test_file = 'itic_532_0.25mW_magicangle.csv'

test_path = location+test_file

df = pd.read_csv(test_path)

#essential cleaning
##remove qnd recast metadata
###try/except

#good enough for the moment
n = 13
df_metadata = df.tail(n)
print(df_metadata)

df = df.head(len(df)-n)


df = df.set_index('0.000000000')
df = df.rename_axis('spectral')
df = df.rename_axis('time', axis=1)



#cast all as numeric
df.index = pd.to_numeric(df.index)
#coerce isn't right - double check 0.0000000.1 issue
df.columns = pd.to_numeric(df.columns, errors='coerce')
for col in df.columns:
    df[col] = pd.to_numeric(df[col])
print(df)

#Convert to single multiindexed dataframe from numpy array to use pandas's to_xarray() function
a = df.to_numpy()
#double-check that reshape preserves correct order of data
a = a.reshape(df.size,1)

idx = pd.MultiIndex.from_product((df.index,df.columns), names=('spectral','time'))
df2 = pd.DataFrame(a, index=idx, columns=['data'])

print(df2.to_xarray())

'''
#recast df to xarray
array = xr.DataArray(df)
print(array)
'''