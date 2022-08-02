import pandas as pd 
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import logging
from pbdbt_itic_analysis_as_py import Dataset, Dataset_with_quantities
import sys

l = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
l.addHandler(stream_handler)
logging.basicConfig(format='%(levelname)s: %(asctime)s %(message)s', level=logging.INFO)


def find_nearest_index(array, number, direction=None): 
    idx = None
    #addressing lists, pandas series, and similar
    try:
      array = np.asarray(array)
    except:
      l.info('Not compatible with np.array')

    #need to address nan values? multi-dim arrays?
    if direction is None:
        _delta = np.abs(array-number)
        _delta_positive = _delta
    elif direction == 'backward':
        _delta = number - array
        _delta_positive = _delta[_delta > 0]
    elif direction == 'forward':
        _delta = array - number
        _delta_positive = _delta[_delta >= 0]
    idx = np.where(_delta == _delta_positive.min())
    #check that this is singular value?
    return idx

def find_nearest_value(array, number, direction=None):

  index = find_nearest_index(array, number, direction=direction)
  #check vs data types
  array = np.asarray(array)

  return array[index]





class Multifile_Analysis():
    def __init__(self, *args) -> None:
        '''
        Args should be Datafiles which each take file paths according to the convention. Use the import_data function 
        '''
        try:
            self.args = args
            self.xarrays = [obj.xarray for obj in args]
        except TypeError:
            l.warning('At least one argument not valid Datafile type')

    def plot_multiple_files(self, axis, normalize_data=False, normalize_slice=True, *vars):
        '''
        axis: 'time' or 'spectral' depending on which type of graph you would like, corresponding
        *vars should be desired wavelengths for the kinetics or times for the spectra, respectively
        '''
        if axis != ('time' or 'spectral'):
            l.warning("Invalid axis passed. Only 'time' or 'spectral' allowed.")
            return None
        elif axis == 'time':
            alt_axis = 'spectral'
        elif axis == 'spectral':
            alt_axis = 'time'


        #set multiple cmaps? 

        for Datafile in self.args:
            
            
            if normalize_data:
                xar = Datafile.normalize_data()
            else:
                xar = Datafile.xarray

            x = xar[axis]

            #sort vars to give good color map representation
            for var in sorted(vars):
                if alt_axis == 'spectral':
                    y = xar.sel(spectral=var, method='nearest').to_array().T
                elif alt_axis == 'time':
                    y = xar.sel(spectral=var, method='nearest').to_array().T
                if normalize_slice:
                    if np.abs(y.min()) > np.abs(y.max()):
                        y = y/np.abs(y.min())
                    else:
                        y = y/np.abs(y.max())
                #str(np.around(Datafile.fluence,1))+' μJ/cm²/pulse'

                plt.plot(x,y, label=str(var)+' nm')
                if axis == 'time':
                    plt.xscale('symlog')

        
        plt.legend()
        plt.show()

    def plot_comparatives(self):
        '''
        If you want to compare several multi-datafile graphs using plt.subplots
        '''
        pass



test_filepath1 = 'C:/Users/Daniel/Desktop/Programming/PBDBT-ITIC data/PBDB-T;ITIC/longtime TA/HDF5 data/07-10_redo.hdf5'
test_filepath2 = 'C:/Users/Daniel/Desktop/Programming/PBDBT-ITIC data/PBDB-T;ITIC/longtime TA/HDF5 data/07-50.hdf5'
test_filepath3 = 'C:/Users/Daniel/Desktop/Programming/PBDBT-ITIC data/PBDB-T;ITIC/longtime TA/HDF5 data/07-100.hdf5'
test_filepath4 = 'C:/Users/Daniel/Desktop/Programming/PBDBT-ITIC data/PBDB-T;ITIC/longtime TA/HDF5 data/07-500.hdf5'

obj1 = Dataset_with_quantities(test_filepath1)
obj2 = Dataset_with_quantities(test_filepath2)
obj3 = Dataset_with_quantities(test_filepath3)
obj4 = Dataset_with_quantities(test_filepath4)

Multifile_Analysis(obj1, obj2, obj3, obj4).plot_multiple_files('time', True, True, 650)
