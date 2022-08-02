


##anything time related?
from datetime import datetime
from datetime import timedelta

##dataframes and analysis
import pandas as pd
import numpy as np
import scipy
from scipy import optimize
from scipy.interpolate import interp1d
from scipy.signal import argrelmax
import matplotlib.pyplot as plt


##fitting for kinetics
from lmfit.models import GaussianModel
from matplotlib.collections import LineCollection 

##coloring
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

##import data
import glob
import os
import h5py
import logging
import xarray as xr
import sys

##glotaran analysis?
import pyglotaran_extras
import glotaran.io
from yaml import load


l = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
l.addHandler(stream_handler)


folder = "C:/Users/Daniel/Desktop/Programming/PBDBT-ITIC data/PBDB-T;ITIC/longtime TA/HDF5 data/"
all_files = glob.glob(os.path.join(folder, '*.hdf5'))
Spec_data = {}
Sorb_data = {}
Irr_data = {}
titles = []
os.chdir(folder)

#IMPORTANT USER-DEFINED FUNCTIONS

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




def mkdir(existing_folder, new_folder):
  if existing_folder[-1] != '/':
    existing_folder += '/'
  if new_folder[-1] != '/':
    new_folder += '/'
  directory = existing_folder+new_folder
  if not os.path.exists(directory):
          os.makedirs(directory)
  return directory

def get_sweep(spectrum_name):
  return spectrum_name[0:7], spectrum_name+'.Dtc'


#adapted from HDF5 file converter, outputs various .Dtc (csv-type?) files in separate folders
##Originally by David Bossanyi (I think)
def convert_hdf5_file(key, folder=folder):
        fname = str(os.path.basename(os.path.normpath(key)))
        l.info('starting file <{0}>'.format(fname))

        #removes .hdf5
        filebasename = fname[0:-5]

        #personal preference: create new subfolder /CSV files/ if one doesn't already exist
        main_folder = mkdir(folder, 'CSV files')
        savedir = mkdir(main_folder, filebasename)
        

        with h5py.File(key, "r") as f:
            array = np.array(f['Average']).T
            fpath = os.path.join(savedir, 'average_dTT.Dtc')
            l.info('saving averaged dT/T data to {0}'.format(str(fpath)))
            np.savetxt(fpath, array, delimiter=',')
            fpath = os.path.join(savedir, '_datetimeinfo.txt')
            g = f.get('Average')
            with open(fpath, 'w') as fmd:
                fmd.write('start: '+str(g.attrs['start date'])+' '+str(g.attrs['start time'])+'\n')
                fmd.write('end: '+str(g.attrs['end_date'])+' '+str(g.attrs['end_time'])+'\n')

            #get metadata
            array = np.array(f['Metadata'])  # might want the transpose ???
            g = f.get('Metadata')
            fpath = os.path.join(savedir, 'metadata.txt')
            with open(fpath, 'w') as fmd:
                for key in g.attrs:
                    fmd.write(str(key)+': '+str(g.attrs[key])+'\n')  
            l.info('saving metadata to {0}'.format(str(fpath)))

            #include individual spectra
            newsavedir = mkdir(savedir, 'sweeps')
            wavelength = np.array(f['Average'])[0,1:]
            group = f['Spectra']
            for spectrum_name in group.keys():
                sweep, name = get_sweep(spectrum_name)
                folder = mkdir(newsavedir, sweep)
                spectrum = np.array(group[spectrum_name])
                array = np.vstack((wavelength, spectrum)).T
                fpath = os.path.join(folder, name)
                l.info('saving spectrum to {0}'.format(fpath))
                np.savetxt(fpath, array, delimiter=',')

            #include individual sweeps
            newsavedir = mkdir(savedir, 'sweeps')
            group = f['Sweeps']
            for sweep in group.keys():
                folder = mkdir(newsavedir, sweep)
                array = np.array(group[sweep]).T
                fpath = os.path.join(folder, 'dTT.Dtc')
                l.info('saving sweep dT/T data to {0}'.format(fpath))
                np.savetxt(fpath, array, delimiter=',')
                fpath = os.path.join(folder, 'datetimeinfo.txt')
                attr = group.get(sweep)
                with open(fpath, 'w') as fmd:
                    fmd.write(str(attr.attrs['date'])+' '+str(attr.attrs['time'])+'\n')
        f.close()
        l.info('finished file <{0}>'.format(fname))


class Dataset:
  '''
  key: should refer to filepath for file to be used, init will automatically check for filetype

  default type to convert to is 2D numpy array 
  '''
  def __init__(self, key):
    self.key = key
    l.info('Loading data from {}'.format(key))
    self.data, self.metadata = self.route()
    #use these instead of self.data
    self.times = self.data[0,1:]
    self.spectra = self.data[1:, 0]
    self.values = self.data[1:,1:]

    self.xarray = self.array_to_xarray()
    self.clean_data()




  def route(self):
    key = self.key
    if key.endswith('.hdf5'):
      data, metadata = self.load_data_hdf5()
    elif key.endswith('.csv' | '.Dtc' | '.txt'):
      data, metadata = self.load_data_csv()
    else:
      l.info('File type not valid. Check extension.')
      data, metadata = None, None
    return data, metadata

  def load_data_hdf5(self):
    key = self.key
    if key.endswith('.hdf5'):
      with h5py.File(key, "r") as f:
        data = np.array(f['Average']).T
      
        
        l.info('loading averaged dT/T data')
        
        #get metadata
        #array = np.array(f['Metadata'])  # might want the transpose ???
        g = f.get('Metadata')
        l.info('loading metadata')
        metadata = dict((str(key), str(g.attrs[key])) for key in g.attrs)
        g = f.get('Average')
        metadata['start'] = str(g.attrs['start date'])+' '+str(g.attrs['start time'])
        metadata['end'] = str(g.attrs['end_date'])+' '+str(g.attrs['end_time'])
        return data, metadata 
    else:
      l.info('Not a hdf5 file. Check Dataset class methods.')
      return None, None
  
  def load_data_csv(self):
    key = self.key
    if key.endswith('.csv'|'.txt'|'.Dtc'):
      all_data = np.genfromtxt(key, delimiter=',')

    else:
      l.info('Not a CSV file. Check Dataset class methods.')
      return None, None

  def plot_2d_data_by_index(self):
    values = self.xarray['data']
    plt.imshow(values, cmap='RdBu_r')
    plt.colorbar()
    plt.show() 

  def clean_data(self):
    self.remove_background_spectra()
    #UP NEXT: Adjust for bad t0 measurement
    #self.check_for_nas()

  def remove_background_spectra(self, n_to_avg = 10):
    '''
      

      Parameters
      ----------
      n_to_avg : TYPE, optional
          DESCRIPTION. The default is 10.

      Returns
      -------
      None.

     
    #working from original array of arrays
    first_n_times = self.values[:, :(n_to_avg)]
    avg_list = []
    for array in first_n_times:
      avg_list.append(np.average(array))
    avg_array = np.asarray(avg_list)
    values = self.values.T
    adjusted_values = values - avg_array
    self.values = adjusted_values.T
    '''
    
    xar = self.xarray
    #select subsection, bizarre indexing for xarray datasets, requires full list of indices
    region_to_avg = xar.isel(time=list(range(n_to_avg)), spectral=list(range(len(xar['spectral']))))
    #average spectra along time axis
    avg_spectrum = region_to_avg.mean(dim='time')
    #remove average from all spectra
    xar = xar-avg_spectrum
    #return as self.xarray
    self.xarray = xar
    

  def check_for_nas(self):
    values = self.values
    print(np.isnan(values).any())

  def extract_metadata(self, all_data):
    pass

  def standardize_metadata(self):
    pass

  def normalize_data(self):
    #find max from xarray
    if self.xarray:
      xar = self.xarray
      if np.abs(xar.min()) > np.abs(xar.max()):
        xar = xar/np.abs(xar.min())
      else:
        xar = xar/np.abs(xar.max())
      return xar
    else:
      data = self.values
      max = max([arr.max() for arr in data])
      data = data/max
      self.values = data
      return data 

  '''
  DONE IN _with_quantities version
  def calculate_fluence(self):
    metadata = self.xarray.attrs
    #check that metadata exists first
    try:
      power = metadata.get('pump power')
      pump_wavelength = metadata.get('pump wavelength')
    except:
      l.info('Cannot find attributes "pump power" and "pump wavelength"')
    #clean strings
    pwr = float(power.replace('uW', ''))
    p_wl = int(pump_wavelength.replace('nm', ''))
    freq = 10000

    #Energy per pulse
    hc = 1.98644586*(10**(-10))
    
    OD = 0.4
    beam_area = 3.1415926536*(0.01**2)
    photon_E = hc/p_wl
    photons_absorbed = 1 - (10**(-OD))
    #Excitation Density
    #return photons_absorbed*P*0.8/(f*photon_E*beam_area)
    #E per pulse per area
    return photons_absorbed*pwr/(freq*beam_area*photon_E)
  '''

  def calculate_dispersion(self):
    for wl in self.xarray.spectral:
      #find max of spectrum
      #check 5-10% drop from peak (both directions) occurs within the same order of magnitude
      #new ys -- proposed time adjustments per spectrum
      #check as dispersion equation
      #update data based on dispersion model
      pass
    pass


  #turns array of arrays data into xarray Dataset structure (for pyglotaran)
  def array_to_xarray(self):
    #add in metadata?
    metadata = self.metadata
    time = self.times
    spectral = self.spectra
    data_ = self.values.T
    xarr = xr.Dataset(data_vars={'data':(['time', 'spectral'],data_)}, coords= {'time':time, 'spectral':spectral}, attrs=metadata)
    #xarr = xr.DataArray(data_, dims=('time', 'spectral'), coords={'time':time, 'spectral':spectral}, attrs=metadata)
    #Removal of bad datapoint for my dataset
    xarr = xarr.drop(find_nearest_value(xarr['time'], 2733), dim='time')
    return xarr
  
  def array_to_pandas(self):
    pass

  '''
  Original idea to print 2d heatmap with axis labels in
  def plot_2D_heatmap(self, data, name, colormaps=cm.get_cmap('RdBu_r', 100)):
    """
    Helper function to plot data with associated colormap.
    """
    n = len(colormaps)
    fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 5),
                            constrained_layout=True, squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(data.spectral,data.time, data.data, cmap=cmap, rasterized=True)
        fig.colorbar(psm, ax=ax)
        ax.title.set_text(name)
    plt.xlim(0.9,3.0)
    yvals = np.asarray([-1, 0, 1, 5, 10, 50, 100, 500, 1000])
    ylogs = np.around(np.log10(yvals + 6), 2)
    plt.ylim(ylogs[0], ylogs[-1])
    plt.yticks(ylogs, labels=yvals)
    plt.xlabel('Energy(eV)')
    plt.ylabel('Time (ps)')
    plt.show()
    '''

'''
beam_reference_list = [-1444.799957, -1438.349957, -1431.899958, -1425.449958, -1418.999958, -1412.549958, -1406.099958, -1399.649959, -1393.199959, -1386.749959, -1380.299959, -1373.849959, -1367.39996, -1360.94996, -1354.49996, -1348.04996, -1341.59996, -1335.149961, -1328.699961, -1322.249961, -1315.799961, -1309.349961, -1302.899961, -1296.449962, -1289.999962, -1283.549962, -1277.099962, -1270.649962, -1264.199963, -1257.749963, -1251.299963, -1244.849963, -1238.399963, -1231.949964, -1225.499964, -1219.049964, -1212.599964, -1206.149964, -1199.699965, -1193.249965, -1186.799965, -1180.349965, -1173.899965, -1167.449965, -1160.999966, -1154.549966, -1148.099966, -1141.649966, -1135.199966, -1128.749967, -1122.299967, -1115.849967, -1109.399967, -1102.949967, -1096.499968, -1090.049968, -1083.599968, -1077.149968, -1070.699968, -1064.249969, -1057.799969, -1051.349969, -1044.899969, -1038.449969, -1031.999969, -1025.54997, -1019.09997, -1012.64997, -1006.19997, -999.74997, -993.299971, -986.849971, -980.399971, -973.949971, -967.499971, -961.049972, -954.599972, -948.149972, -941.699972, -935.249972, -928.799973, -922.349973, -915.899973, -909.449973, -902.999973, -896.549973, -890.099974, -883.649974, -877.199974, -870.749974, -864.299974, -857.849975, -851.399975, -844.949975, -838.499975, -832.049975, -825.599976, -819.149976, -812.699976, -806.249976, -799.799976, -793.349977, -786.899977, -780.449977, -773.999977, -767.549977, -761.099977, -754.649978, -748.199978, -741.749978, -735.299978, -728.849978, -722.399979, -715.949979, -709.499979, -703.049979, -696.599979, -690.14998, -683.69998, -677.24998, -670.79998, -664.34998, -657.899981, -651.449981, -644.999981, -638.549981, -632.099981, -625.649981, -619.199982, -612.749982, -606.299982, -599.849982, -593.399982, -586.949983, -580.499983, -574.049983, -567.599983, -561.149983, -554.699984, -548.249984, -541.799984, -535.349984, -528.899984, -522.449985, -515.999985, -509.549985, -503.099985, -496.649985, -490.199986, -483.749986, -477.299986, -470.849986, -464.399986, -457.949986, -451.499987, -445.049987, -438.599987, -432.149987, -425.699987, -419.249988, -412.799988, -406.349988, -399.899988, -393.449988, -386.999989, -380.549989, -374.099989, -367.649989, -361.199989, -354.74999, -348.29999, -341.84999, -335.39999, -328.94999, -322.49999, -316.049991, -309.599991, -303.149991, -296.699991, -290.249991, -283.799992, -277.349992, -270.899992, -264.449992, -257.999992, -251.549993, -245.099993, -238.649993, -232.199993, -225.749993, -219.299994, -212.849994, -206.399994, -199.949994, -193.499994, -187.049994, -180.599995, -174.149995, -167.699995, -161.249995, -154.799995, -148.349996, -141.899996, -135.449996, -128.999996, -122.549996, -116.099997, -109.649997, -103.199997, -96.749997, -90.299997, -83.849998, -77.399998, -70.949998, -64.499998, -58.049998, -51.599998, -45.149999, -38.699999, -32.249999, -25.799999, -19.349999, -12.9, -6.45, 0.0, 6.45, 12.9, 19.349999, 25.799999, 32.249999, 38.699999, 45.149999, 51.599998, 58.049998, 64.499998, 70.949998, 77.399998, 83.849998, 90.299997, 96.749997, 103.199997, 109.649997, 116.099997, 122.549996, 128.999996, 135.449996, 141.899996, 148.349996, 154.799995, 161.249995, 167.699995, 174.149995, 180.599995, 187.049994, 193.499994, 199.949994, 206.399994, 212.849994, 219.299994, 225.749993, 232.199993, 238.649993, 245.099993, 251.549993, 257.999992, 264.449992, 270.899992, 277.349992, 283.799992, 290.249991, 296.699991, 303.149991, 309.599991, 316.049991, 322.49999, 328.94999, 335.39999, 341.84999, 348.29999, 354.74999, 361.199989, 367.649989, 374.099989, 380.549989, 386.999989, 393.449988, 399.899988, 406.349988, 412.799988, 419.249988, 425.699987, 432.149987, 438.599987, 445.049987, 451.499987, 457.949986, 464.399986, 470.849986, 477.299986, 483.749986, 490.199986, 496.649985, 503.099985, 509.549985, 515.999985, 522.449985, 528.899984, 535.349984, 541.799984, 548.249984, 554.699984, 561.149983, 567.599983, 574.049983, 580.499983, 586.949983, 593.399982, 599.849982, 606.299982, 612.749982, 619.199982, 625.649981, 632.099981, 638.549981, 644.999981, 651.449981, 657.899981, 664.34998, 670.79998, 677.24998, 683.69998, 690.14998, 696.599979, 703.049979, 709.499979, 715.949979, 722.399979, 728.849978, 735.299978, 741.749978, 748.199978, 754.649978, 761.099977, 767.549977, 773.999977, 780.449977, 786.899977, 793.349977, 799.799976, 806.249976, 812.699976, 819.149976, 825.599976, 832.049975, 838.499975, 844.949975, 851.399975, 857.849975, 864.299974, 870.749974, 877.199974, 883.649974, 890.099974, 896.549973, 902.999973, 909.449973, 915.899973, 922.349973, 928.799973, 935.249972, 941.699972, 948.149972, 954.599972, 961.049972, 967.499971, 973.949971, 980.399971, 986.849971, 993.299971, 999.74997, 1006.19997, 1012.64997, 1019.09997, 1025.54997, 1031.999969, 1038.449969, 1044.899969, 1051.349969, 1057.799969, 1064.249969, 1070.699968, 1077.149968, 1083.599968, 1090.049968, 1096.499968, 1102.949967, 1109.399967, 1115.849967, 1122.299967, 1128.749967, 1135.199966, 1141.649966, 1148.099966, 1154.549966, 1160.999966, 1167.449965, 1173.899965, 1180.349965, 1186.799965, 1193.249965, 1199.699965, 1206.149964, 1212.599964, 1219.049964, 1225.499964, 1231.949964, 1238.399963, 1244.849963, 1251.299963, 1257.749963, 1264.199963, 1270.649962, 1277.099962, 1283.549962, 1289.999962, 1296.449962, 1302.899961, 1309.349961, 1315.799961, 1322.249961, 1328.699961, 1335.149961, 1341.59996, 1348.04996, 1354.49996, 1360.94996, 1367.39996, 1373.849959, 1380.299959, 1386.749959, 1393.199959, 1399.649959, 1406.099958, 1412.549958, 1418.999958, 1425.449958, 1431.899958, 1438.349957]
beam_stack = [beam_reference_list, [0]*len(beam_reference_list), beam_reference_list, [0]*len(beam_reference_list)]
beam_reference_df = pd.DataFrame(np.asarray(beam_stack).T, columns = ['Pos X [µm]', 'X Value [%]','Pos Y [µm]', 'Y Value [%]'])
'''

class Dataset_with_quantities(Dataset):
  def __init__(self, key) -> None:
     super().__init__(key)
     #self.import_beam_data('path')
     #self.import_absorption_data('path2')
     self.assign_relevant_quantities()

  #BEAM STUFF
  ##IMPORT

  def import_beam_data(self):
    #assume probe beam gets centered at center of pump beam
    #how to line up metadatas?
    #current arrangement, beam info assigned to day folders
    ##check all files in master folder to see if they contain the same name
    top_folder = 'C:/Users/Daniel/Desktop/Programming/PBDBT-ITIC data/PBDB-T;ITIC/longtime TA/Dan F/'
    for root, dirs, files in os.walk(top_folder):
      for file in files:
        if file == os.path.basename(self.key):
          dir_up = root.split(sep='\\')[0]+'/'
          #confirm dates match
          
          beam_folder = dir_up + 'beam_info/'
          
          if os.path.exists(beam_folder):
            #look in folder for files that end in .txt and have 'pump'/'probe' and 'xy' in them
            for beam_file in os.listdir(beam_folder):
              if beam_file.endswith('.txt') and 'xy' in beam_file:
                beam_path = beam_folder+beam_file
                if 'probe' in beam_file:
                  probe_df = pd.read_csv(beam_path, sep='\t', header = 7, encoding='unicode_escape')
                  #new_probe_df = pd.merge(beam_reference_df, probe_df, how='outer', on=['Pos X [µm]', 'Pos Y [µm]'])
                  #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                   #print(new_probe_df)
                if 'pump' in beam_file:
                  pump_df = pd.read_csv(beam_path, sep='\t', header = 7, encoding='unicode_escape')
    return probe_df, pump_df


  ##CENTERING

  #defined for beam output data put into pandas dataframe
  def x_center_from_DataFrame(self, df, var='X'):
    x_values = np.asarray(df[var+' Value [%]'].fillna(0))
    x_index = np.asarray(df['Pos {} [µm]'.format(var)].fillna(0))
    x_peaks, pvals = scipy.signal.find_peaks(np.asarray(x_values), height=50)
    #main peak where val is highest
    x_peak = x_peaks[np.where(pvals['peak_heights'] == pvals['peak_heights'].max())]
    x_center = x_index[x_peak]
    
    

    '''
    #plt.plot(x_index, x_values)
    #plt.axvline(x_center, 0, 80)
    #plt.plot(x_index, out.best_fit)
    #plt.show()

    '''
    return x_center



  def xy_center_from_DataFrame(self, df):
    x_center = self.x_center_from_DataFrame(df, 'X')
    y_center = self.x_center_from_DataFrame(df, 'Y')
    return x_center, y_center

  def center_axis(self, df, value, var='X'):
    df['Pos {} [µm]'.format(var)] = df['Pos {} [µm]'.format(var)] - value
    return df 

  def center_axes(self, df, val_x, val_y):
    df = self.center_axis(df, val_x, 'X')
    df = self.center_axis(df, val_y, 'Y')
    return df

  def center_pump_probe(self):
    probe_df, pump_df = self.import_beam_data()
    #center of gaussian beam is max in X and Y direction
    probe_x, probe_y = self.xy_center_from_DataFrame(probe_df)
    pump_x, pump_y = self.xy_center_from_DataFrame(pump_df)
    probe_df = self.center_axes(probe_df, probe_x, probe_y)
    pump_df = self.center_axes(pump_df, pump_x, pump_y)
    return probe_df, pump_df

  ##RADIUS (SECONDARY METHOD TO COMPUTE FLUENCE)
  def x_radius_from_dataframe(self, df, var='X'):
    x_values = np.asarray(df[var+' Value [%]'].fillna(0))
    x_index = np.asarray(df['Pos {} [µm]'.format(var)].fillna(0))

    #MODEL WITH LMFIT - works, but overly complicated and occasionally inaccurate. Probably better to go by peaks.
    model = GaussianModel()
    pars = model.guess(x_values, x=x_index)
    out = model.fit(x_values, pars, x=x_index)
    for name, param in out.params.items():
      if name =='sigma':
        x_sigma = param.value

    #go out 3sigma (99% of beam should be sufficient), change from micrometers to centimeters (default unit for fluence)
    x_radius = 3*x_sigma/10000
    return x_radius
  def xy_radius_from_dataframe(self, df):
    x_radius = self.x_radius_from_dataframe(df, 'X')
    y_radius = self.x_radius_from_dataframe(df, 'Y')
    return x_radius, y_radius
  def area_from_radii(self, df):
    x_radius, y_radius = self.xy_radius_from_dataframe(df)
    #using ellispe area formula
    return np.pi*x_radius*y_radius
  def probe_area_from_radii(self):
    probe_df, pump_df = self.import_beam_data()
    return self.area_from_radii(probe_df)
  
  def probe_proportion_of_pump(self):
    probe_df, pump_df = self.import_beam_data()
    #can basically just assume they're centered and only use sigma/radius data
    probe_x_radius, probe_y_radius = self.xy_radius_from_dataframe(probe_df)
    pump_x_radius, pump_y_radius = self.xy_radius_from_dataframe(pump_df)
    #what proportion of pump beam is probe beam occupying?
    print('Probe X: {}  Probe Y: {}   Pump X: {}  Pump Y: {}'.format(probe_x_radius, probe_y_radius, pump_x_radius, pump_y_radius))
    prop_x = (1-2*scipy.stats.norm.cdf(-probe_x_radius, scale=pump_x_radius))
    prop_y = (1-2*scipy.stats.norm.cdf(-probe_y_radius, scale=pump_y_radius))
    print('Proportion in X: {}  Proportion in Y: {}'.format(prop_x, prop_y))
    #is this right?
    return np.average([prop_x, prop_y])
  
 




  ##LASER PLOTTING
  
    
  def plot_laser_axis(self, df, var='X'):
    x_values = np.asarray(df[var+' Value [%]'].fillna(0))
    x_index = np.asarray(df['Pos {} [µm]'.format(var)].fillna(0))
    plt.plot(x_index, x_values)
    plt.show()

  def plot_laser_two_axes(self, df):
    self.plot_laser_axis(df, 'X')
    self.plot_laser_axis(df, 'Y')

  def pos_mesh_from_df(self, df):
    x_index = np.asarray(df['Pos {} [µm]'.format('X')].fillna(0))
    y_index = np.asarray(df['Pos {} [µm]'.format('Y')].fillna(0))
    x_mesh, y_mesh = np.meshgrid(x_index, y_index)
    return x_mesh, y_mesh

  def plot_laser_2D_wo_show(self, df):
    x_mesh, y_mesh = self.pos_mesh_from_df(df)
    x_values = np.asarray(df['X Value [%]'].fillna(0))
    y_values = np.asarray(df['Y Value [%]'].fillna(0))
    x_val_mesh, y_val_mesh = np.meshgrid(x_values, y_values)
    

    
    z_mesh = x_val_mesh*y_val_mesh
    print(z_mesh)

    plt.figure().add_subplot(111).contourf(x_mesh, y_mesh, z_mesh, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)
    return z_mesh

  def normalize_z_mesh(self, z_mesh):
    full_s = sum(sum(z_mesh))
    return z_mesh/full_s

  def state_strength_est(self):
    probe_df, pump_df = self.center_pump_probe()
    probe_mesh = self.normalize_z_mesh(self.plot_laser_2D_wo_show(probe_df))
    pump_mesh = self.normalize_z_mesh(self.plot_laser_2D_wo_show(pump_df))
    int_mesh = probe_mesh*pump_mesh
    x_mesh, y_mesh = self.pos_mesh_from_df(probe_df)
    plt.figure().add_subplot(111).contourf(x_mesh, y_mesh, int_mesh, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)
  

  
  def plot_lasers_2D(self):
    probe_df, pump_df = self.center_pump_probe()
    self.plot_laser_2D_wo_show(probe_df)
    self.plot_laser_2D_wo_show(pump_df)
    plt.show()

  def fluence_from_beams(self):
    probe_df, pump_df = self.center_pump_probe()





  def plot_possible_probe_areas(self):
    #guess at probe beam radius in cm
    probe_beam_radius = 1
    #probe area in cm^2
    self.probe_area = np.pi*(probe_beam_radius**2)
    
    pass

  #assign naming convention for steady-state spectra
  def import_absorption_data(self, path):
    pass
  def assign_relevant_quantities(self):
    try:
        power = self.metadata.get('pump power')
        pump_wavelength = self.metadata.get('pump wavelength')
    except:
        l.info('Cannot find attributes "pump power" and "pump wavelength"')
    #clean strings
    if ('uW' in power) and ('nm' in pump_wavelength): 
      self.pump_power = float(power.replace('uW', ''))
      self.pump_wavelength = int(pump_wavelength.replace('nm', ''))
    else:
      l.info('Power and wavelength not standardized to units uW and nm.')
    
    #frequency in Hz
    self.laser_frequency = 10000
    #planck constant in ev*nm
    hc = 1239.81
    uJ_to_eV = 6.2415*(10**12)

    self.thickness = 100 * (10**(-9))
    #not sure how to get this
    self.molar_absorbance_epsilon = None
    #avogadros = 6.02 * (10**23)
    self.beam_proportion = self.probe_proportion_of_pump()
    self.probe_area = self.probe_area_from_radii()


    #CALCULATIONS
    ##convert energy from uJ to eV
    self.energy_per_pulse = uJ_to_eV*self.pump_power*self.beam_proportion/self.laser_frequency
    self.photons_per_pulse = self.energy_per_pulse*self.pump_wavelength/hc
    


    ##more correctly, molecules not in ground state per pulse
    #self.excitons_per_pulse = (10**(-self.absorbance_at_wavelength))*self.photons_per_pulse
    #self.exciton_density_per_area = self.excitons_per_pulse/self.probe_area
    #self.exciton_density_per_volume = self.exciton_density_per_area/self.thickness
    #exciton density should also be -deltaA_GSB*avogadro's_number/(molar_absorbance_epsilon*thickness), as a check

    #fluence in uJ/(cm^2)/pulse
    self.fluence = (self.beam_proportion*self.pump_power/self.probe_area)/self.laser_frequency




    pass


      
from glotaran.utils.ipython import display_file
        
from glotaran.io import load_model
from glotaran.io import load_parameters
from glotaran.io import save_dataset
from glotaran.io.prepare_dataset import prepare_time_trace_dataset
from glotaran.optimization.optimize import optimize
from glotaran.project.scheme import Scheme

###NEXT THING: IMPORT STEADY-STATE ABSORPTION DATA AND USE FOR CALCULATIONS


test_filepath = 'C:/Users/Daniel/Desktop/Programming/PBDBT-ITIC data/PBDB-T;ITIC/longtime TA/HDF5 data/07-50.hdf5'
qs = Dataset(test_filepath)
#print('Pump power: {} uW  Probe area: {} cm^2  Fluence: {} uJ/cm^2/pulse'.format(qs.pump_power, qs.probe_area, qs.fluence))



#probe_df, pump_df = qs.import_beam_data()
#let's try working with pyglotaran again
dataset = qs.xarray

#print(dataset)



#False for np.ndarray.any(np.isnan(qs.values)), qs.times, qs.spectra
'''
dataset = prepare_time_trace_dataset(dataset)
print(dataset)
dataset = dataset.fillna(0)
plot_data = dataset.data_singular_values.sel(singular_value_index=range(0, 10))
plot_data.plot(yscale="log", marker="o", linewidth=0, aspect=2, size=5)

#this is where you write the model to model.yaml and parameter guesses to parameters.yaml

model = load_model('model.yaml')
print(model.validate())
parameters = load_parameters("parameters.yaml")
print(model.validate(parameters=parameters))
scheme = Scheme(model, parameters, {"dataset1": dataset})
result = optimize(scheme)
print(result)

'''

for filename in all_files:
  pass
  '''
  #Simplify name
  new_name = filename.replace(".hdf5", "").replace(folder,"")
  #import data
  #keys are ['Average', 'Metadata', 'Spectra', 'Sweeps']
  obj= Dataset(filename)
  print(obj.xarray.attrs)
  '''
  '''
  #GLOTARAN
  model = load_model('model.yaml')
  parameters = load_parameters('parameters.yaml')
  scheme = Scheme(model, parameters, {"dataset1": dataset})
  result = optimize(scheme)
  print(result)
  '''
  '''
  dataset = prepare_time_trace_dataset(dataset)
  plot_data = dataset.data_singular_values.sel(singular_value_index=range(0, 10))
  plot_data.plot(yscale="log", marker="o", linewidth=0, aspect=2, size=5)
  '''
  '''
  with h5py.File(filename, "r") as f:
    g = f.get('Average')
    date = g.attrs['start date']
    print(type(date))

  '''


        
        
  '''
  #Some cleaning
  df.drop(df.tail(14).index, inplace=True)
  df = df.rename({'0.00000E+0.1':'0.000000000'}, axis=1)
  cols = df.columns.values
  cols[0] = 'Wavelength(nm)'
  df.columns = cols
  df = df.set_index('Wavelength(nm)')
  #remove rows (wavelengths) with at least 3 na values
  df = df.dropna(axis=0, thresh=3)

  #change strings to numbers
  df = df.astype(float)
  df['Energy(eV)'] = 1239.842/df.index.astype(float)
  df = df.set_index('Energy(eV)')
  df.columns = np.around(pd.to_numeric(df.columns), 2)

  titles.append(new_name)
  Spec_data[new_name] = df

Spec_data

'''