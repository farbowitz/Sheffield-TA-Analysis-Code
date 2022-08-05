import pandas as pd 
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import logging
from pbdbt_itic_analysis_as_py import Dataset, Dataset_with_quantities
import sys
from matplotlib import cm
import types

l = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
l.addHandler(stream_handler)
logging.basicConfig(format='%(levelname)s: %(asctime)s %(message)s', level=logging.INFO)

#cmap takes values from 0 to 1
#figure out best place to assign
cmap = cm.get_cmap('viridis')


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

    def assign_labels(self, obj, axis, purpose, var):

        #for time data, assign either pump beam power or fluence, if possible        
        if axis == 'time':
            if type(obj) == Dataset_with_quantities:
                label = '%s' % float('%.2g' % obj.fluence)+' μJ/cm²/pulse'
            elif type(obj) == Dataset:
                label = obj.metadata['pump power']
            else:
                l.warning("Multifile_analysis class can only take arguments that are of the classes 'Dataset' or 'Dataset_with_quantitites', imported from pdbdt_itic_analysis_as_py.py")
                return None, None
            title = '{} in {} Kinetics at {} nm'.format(obj.material, obj.solvent, var)
            x_axis_label = '{} ({})'.format(axis, obj.time_units)
            

        elif axis == 'spectral':

            pass

        else:
            l.warning("To label data, axis can only take on 'time' or 'spectral'.")
            return None, None
            
            




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
                #define the other axis
                if alt_axis == 'spectral':
                    y = xar.sel(spectral=var, method='nearest').to_array().T
                elif alt_axis == 'time':
                    y = xar.sel(spectral=var, method='nearest').to_array().T

                #handle normalization
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

    def time_to_color(self, cmap, xar, val, *vars):
        #color should represent time, preferably on a log scale
        #should be subject to max and min values, map logarithmically to [0,1]
        time_max = np.max(xar['time'])
        time_min = np.min(xar['time'])
        #lower time scale if factor of 10 bigger than highest var?
        var_max = sorted(vars)[-1]
        var_min = sorted(vars)[1]
        if var_max > time_max or var_min < time_min:
            l.warning('Variable range {}-{} outside of time range: {}-{}'.format(var_min, var_max, time_min, time_max))
        #should probably work from absolute scale of logs, place on linear interval from 1-max value
        c_val = np.log(1+val-time_min)/np.log(1+time_max-time_min)
        return cmap(c_val)


    def _plot_component(self, Datafile, axis, axes, index_x, index_y, cmap=cm.get_cmap('viridis'), normalize_data = False, normalize_slice = False, *vars):
        if axis != ('time' or 'spectral'):
            l.warning("Invalid axis passed. Only 'time' or 'spectral' allowed.")
            return None
        elif axis == 'time':
            alt_axis = 'spectral'
        elif axis == 'spectral':
            alt_axis = 'time'
                   
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

                #assign color to each plot, might need conditional
                if alt_axis == 'time':
                    color = self.time_to_color(cmap, xar, var, vars)   
                else:
                    var_min = sorted(vars)[1]
                    var_max = sorted(vars)[-1]
                    color = cmap((var - var_min)/(var_max-var_min))

                axes[index_x, index_y].plot(x,y, label=str(var)+' nm', color=color)
                #how to title each graph?
                axes[index_x, index_y].set_title()
            
                if axis == 'time':
                    axes[index_x, index_y].xscale('symlog')

    def plot_comparatives(self, axis='spectral', *vars):
        '''
        If you want to compare several multi-datafile graphs using plt.subplots. Assumes you will use all the Datafiles.
        '''

        def squarest_dims(number):
            #find nearest square larger than number
            root = np.ceil(np.sqrt(number))
            #can it fit into n by (n-1) dimensions?
            if number < ((root-1)*root):
                return (root-1), root
            else:
                return root, root
    
        def plot_within_subplots(Datafile, axes, index_x, index_y, *vars):
            self._plot_component(Datafile, axis, axes, index_x, index_y, False, False, vars)
         
            #axes[index_x, index_y].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            #axes[index_x,index_y].title(np.average([x1,x2]))

        nrows, ncols = squarest_dims(len(self.args))
        fig, axes = plt.subplots(nrows=int(nrows), ncols=int(ncols))
        idx = 0
        for Datafile in self.args:
            index_x = int(np.floor(idx/ncols))
            index_y = int(idx % ncols)
            plot_within_subplots(Datafile, axes, index_x, index_y, vars)
            idx += 1
        fig.show()
        


#NEW CONCEPT: MAKE MULTIFILE XARRAY WITH ALL PROJECT VARIABLES, EACH TAKING ON A TYPE AND UNIT FOR MORE GENERALIZED PLOTTING

class MultiDataset():
    '''
    Inputs
    File paths to your datasets
    List of variable types (i.e. what you would name a graph axis for that variable)
    List of variable base units (one string if variable is continuous/float-type; list of values associate with corresponding type if variable is discrete/string-type). Put None if not needed.
    **kwargs: 
    - Optional map of file names to assigned variables. One can also perform path mapping in Dataset class.
    See __init__ for example of default case.
    Check Dataset class for other assumptions.
    '''
    def __init__(self, paths, var_types=['ΔA', 'time', 'wavelength', 'fluence', 'material', 'solvent'], var_units=['a.u.', 'ns', 'nm','μJ/cm²/pulse', None, None], **kwargs):
        self.variables = var_types
        self.assign_permanent_attributes(var_types, var_units)
        #create empty xarray with all variables in place
        #additional var list for visual mapping properties?
        pass

    def assign_permanent_attributes(self, var_types, var_units):
        #check lengths agree
        if len(var_types) == len(var_units):
            pass
        else:
            l.warning('Mismatch between var_types and var_units lengths: {} and {}'.format(len(var_types), len(var_units)))
            return None

        for i in range(len(var_types)):
            
            var_type = var_types[i]
            var_unit = var_units[i]
            if self.is_units(var_unit):
                setattr(self, var_type+'_units', var_unit)
        

    def create_individual_dataset_and_attributes(self, path):
        #check if default var_types 'time', 'spectral', and 'ΔA' as well as default var_units are listed.
        obj = Dataset_with_quantities(path)
        #check lengths agree
       
        
        for var_type in self.variables:
            var_type_attr_name = var_type+'_values'
            temp_attr_name = 'current_'+var_type+'_values'

            #check if dataset already has such values, if so, inherit it (e.g. Dataset should already have)
            #either it has values which were already defined in the Dataset class and a unit type, or it has values defined in var_units 
            if hasattr(obj, var_type_attr_name):
                setattr(self, temp_attr_name, getattr(obj,var_type_attr_name))
            elif hasattr(obj, var_type):
                #i.e. should be listed as just var_type if value is singular
                value = getattr(obj, var_type)               
                setattr(self, temp_attr_name, np.asarray(value))
            else:
                l.warning('Cannot find attributes {} or {} in Dataset {}'.format(var_type, var_type_attr_name, path))

        #construct xarray from info
        not_ΔA = [var for var in self.variables if var != 'ΔA']

        xar = xr.Dataset(data_vars={'ΔA': (['wavelength', 'time'],getattr(self, 'current_ΔA_values'))}, coords= {var_name:getattr(self, 'current_{}_values'.format(var_name)) for var_name in not_ΔA})

        return xar
        
    def merge_arrays(self, paths):
        return xr.merge([self.create_individual_dataset_and_attributes(path) for path in paths])
            


            



    def merge_current_values_to_list(self, var_type):
        var_type_attr_name = var_type+'_values'
        #create attribute if it doesn't already exist
        if not hasattr(self, var_type_attr_name):
            setattr(self, var_type_attr_name, getattr(self, 'current_'+var_type_attr_name))
        else:
            #compare both lists and flatten
            setattr(self, var_type_attr_name, getattr(self, var_type_attr_name).extend(getattr(self, 'current_'+var_type_attr_name)))



    

    def add_xarray_dim_along_new_varible(self, var_type):
        for value in self.var_type.values:
            obj = self.create_individual_dataset_and_attributes()
            #if value is in getattr(obj)
            #Append to xarray dimension of variable type?

    def plot_single_2D_graph_without_show(self):
        pass

    def plot_single_2D_graph_with_show(self):
        self.plot_single_2D_graph_without_show()
        #plt.legend()
        plt.show()
        pass

    def plot_multiple_2D_graph_without_show(self):
        for var in vars:
            self.plot_single_2D_graph_without_show()
            #assign legends
        pass

    def add_in_units(self, string):
        pass

    def is_units(self, unit):
        if type(unit) == str:
            return True
        elif unit is None:
            return False
        else:
            l.warning('Incorrect type {} placed into var_units list. Values should only be strings or None.'.format(type(unit)))
            return None









test_filepath1 = 'C:/Users/Daniel/Desktop/Programming/PBDBT-ITIC data/PBDB-T;ITIC/longtime TA/HDF5 data/07-10_redo.hdf5'
test_filepath2 = 'C:/Users/Daniel/Desktop/Programming/PBDBT-ITIC data/PBDB-T;ITIC/longtime TA/HDF5 data/07-50.hdf5'
test_filepath3 = 'C:/Users/Daniel/Desktop/Programming/PBDBT-ITIC data/PBDB-T;ITIC/longtime TA/HDF5 data/07-100.hdf5'
test_filepath4 = 'C:/Users/Daniel/Desktop/Programming/PBDBT-ITIC data/PBDB-T;ITIC/longtime TA/HDF5 data/07-500.hdf5'
'''
obj1 = Dataset_with_quantities(test_filepath1)
obj2 = Dataset_with_quantities(test_filepath2)
obj3 = Dataset_with_quantities(test_filepath3)
obj4 = Dataset_with_quantities(test_filepath4)
'''
paths = [test_filepath1, test_filepath2]
print(MultiDataset(paths).merge_arrays(paths))
