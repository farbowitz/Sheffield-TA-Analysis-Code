#add in function to remove data times < 0?
def remove_times_before(df, time):
    new_df = df.drop(df.columns[df.columns<time], axis=1)
    return new_df


#Modeling
#Kinetic algorithm
from scipy.optimize import curve_fit
from lmfit import Model, Parameter, Parameters, report_fit
from lmfit.models import ExponentialModel

def single_exp(x, a, t):
    return a * np.exp(-(x)/t)
'''
def double_exp(x, a1, a2, t1, t2):
  return a1*np.exp(-1*(x)/t1) + (a2)*np.exp(-1*(x)/t2)

def tri_exp(x, a1, t1, a2, t2, a3, t3):
  return a1*np.exp(-1*(x)/t1) + (a2)*np.exp(-1*(x)/t2) + (a3)*np.exp(-1*(x)/t3)
'''
def multi_exp(order):
    model = Model(single_exp, prefix='e1_')
    for i in range(1, len(order)):
     model += Model(single_exp, prefix = 'e%d_' % (i+1))
    params = model.make_params()
    #guess values for params?
    return model, params

def power_law(x,a1,t1):
  return t1/(x**a1)










class Kinetic:
    def __init__(self, dataset, wavelength, cmap = cmap):
        self.cmap = cmap
        self.wl = wavelength
        self.data = self.get_kinetics(dataset, wavelength)
        pass
        
    def get_kinetics(self, df, wl):
        k_df = pd.DataFrame()
        df = remove_times_before(df, -1)
        try:
            k_df = df.loc[wl]
        except:
            k_df = df.loc[wl-find_nearest(df.index,wl, None)]
        return k_df


    def graph_without_show(self, cmap=cmap, model=None):
        k_df = self.data
        #coloring it similarly to spectra
        colormap = cm.get_cmap(cmap)
        #overdoing it?
        #color_list = [colormap(time/len(k_df)) for time in k_df.index.tolist()]
        color_list = [colormap(i/len(k_df)) for i in range(len(k_df))]
        plt.scatter(k_df.index, k_df, marker='x', color=color_list)
        plt.gca().set_xscale('log')
        plt.set_cmap(cmap)
        '''
        #trying to put together continuous color line
        segs = [[wl]+[float(k_df.loc[wl])] for wl in k_df.index]
        line_segs = LineCollection(segs)
        print(line_segs)
        '''
        
    def graph(self):
        self.graph_without_show()
        plt.show()

    def normalize(self):
        self.data = self.data/self.data.max()

    def model_kinetics(self, order = 3):

        #find time of peak
        tmax = x[pd.DataFrame(y).idxmax()]
        x_fit = x[x>=tmax]
        y_fit = y[x>=tmax]
        model, params = multi_exp(order)
        #parameter guesses by order of magnitude
        for i in range(len(order)):
          params['e%d_'%(i+1)+'a'] = (10**(-i))*x_fit[0]  #or use y.max()?
          params['e%d_'%(i+1)+'t'] = (10**i)*0.1


        result = model.fit(y_fit, params, x=x_fit)

        print(result.fit_report(min_correl=0.5))
        result.plot_fit()
        plt.show()



        '''df = pd.DataFrame(data={'a1':a1, 't1':t1}, index=[0])
        print(df)

        plt.plot(x_fit, result.best_fit, '--', color=colormap(i/10), label='Best Fit')
        plt.legend(loc='best')
        return result'''

    def k_fit(self, func, name=name):
        #find time of peak
        tmax = self.data.idxmax()
        x = self.data.index
        y = self.data
        x_fit = x[x>=tmax]
        y_fit = y[x>=tmax]
        model = Model(func, independent_vars=['x'])
        params = Parameters()
        params.add('a1', value=0.5, min=-1, max=1)
        params.add('t1', value=5, min=0, max=700000)


        result = model.fit(y_fit, params, x=x_fit)

  

        a1 = np.around(result.values['a1'], 2)

        t1 = np.around(result.values['t1'], 1)



        df = pd.DataFrame(data={'a1':a1, 't1':t1}, index=[0])
        print(df)

        plt.plot(x_fit, result.best_fit, '--', color=colormap(1/10), label='Best Fit')
        plt.legend(loc='best')
        return result
        

#Is there an easier way to call up all the data and still execute different desired functions
for name in data_NIR:
    df = data_NIR[name]
    obj = Kinetic(df, 936)
    obj.normalize()
    obj.graph_without_show()
    obj.k_fit(single_exp)
    plt.show()







WLs_of_interest = [450, 590, 610, 660, 700, 750, 930, 1000, 1200,1250, 1300]
WL_UV = sorted(i for i in WLs_of_interest if i <= 800)
WL_NIR = sorted(i for i in WLs_of_interest if i > 800)

def find_nearest_index(array, number, direction=None): 
    idx = -1
    if direction is None:
        ser = np.abs(array-number)
        idx = ser.get_loc(ser.min())
    elif direction == 'backward':
        _delta = number - array
        _delta_positive = _delta[_delta > 0]
        if not _delta_positive.empty:
            idx = _delta.get_loc((_delta_positive.min()))
    elif direction == 'forward':
        _delta = array - number
        _delta_positive = _delta[_delta >= 0]
        if not _delta_positive.empty:
            idx = _delta.get_loc(_delta_positive.min())
    return idx
colormap = cm.get_cmap('plasma')
styles = [':', '--', '-.', (0, (3, 5, 1, 5)), (0, (1, 10)), (0, (1, 1)), (0, (3, 10, 1, 10, 1, 10)), (0, (5, 1)), (0, (3, 10, 1, 10)), (0, (5, 5)),(0, (3, 1, 1, 1)),  (0, (1, 1))]
def k_chart(data, name, guess, func):

  dex = find_nearest_index(data.index, guess)
  region = ''
  if guess > 800:
    region = 'NIR'
  elif guess <= 800:
    region = 'UV-Vis'
  #dd = data.iloc[dex-10:dex+10].mean()
  #data2 = dd[dd.index>1]
  #data3 = dd[(dd.index>1) & (dd.index<10)]
  #data4 = dd[(dd.index>400) & (dd.index<4500)]
  #update = pd.DataFrame(data4, columns = ['vals'])
  #slope = linregress(np.asarray(data4.index), np.asarray(np.log10(abs(data4))))[0]
  #intercept = linregress(np.asarray(data4.index), np.asarray(np.log10(abs(data4))))[1]
  #stderr = linregress(np.asarray(data4.index), np.asarray(np.log10(abs(data4))))[4]
  #update['estimate'] = 10**(slope*update.index+intercept)
  #+" -- Slope: "+str(np.around(slope, 6))+ " Std Err: "+str(np.around(stderr,8))

  
  
  i = WLs.index(guess)
  label = str(guess)+' nm'

  peak_to_norm =  data.iloc[dex-10:dex+10].mean().max()
  trough_to_norm = data.iloc[dex-10:dex+10].mean().min()
  norm = max(abs(peak_to_norm), abs(trough_to_norm))
  x = np.asarray(data.columns)

  y = abs(np.asarray(data.iloc[dex-10:dex+10].mean())/norm)
  tmax = x[pd.DataFrame(y).idxmax()]
  x=x-tmax
  y=y[x>7]
  x=x[x>7]
  plt.scatter(x,y, label=label)
  #plt.plot(np.asarray(update.index), np.asarray(update['estimate']), linewidth=0.8, color='black')

  k_fit(x, y, power_law, name)
  
  plt.ylabel('ΔA (normalized)')
  plt.yscale('log')
  plt.xscale('symlog')
  plt.ylim(0.01,1.2)
  plt.xlim(0,6000)
  plt.xlabel('Time (ps)')








def k_chart_bk(data, guess, label):
  f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w')

  # plot the same data on both axes
  ax.plot(x, y)
  ax2.plot(x, y)

  ax.set_xlim(0,7.5)
  ax2.set_xlim(40,42.5)

  # hide the spines between ax and ax2
  ax.spines['right'].set_visible(False)
  ax2.spines['left'].set_visible(False)
  ax.yaxis.tick_left()
  ax.tick_params(labelright='off')
  ax2.yaxis.tick_right()

# This looks pretty good, and was fairly painless, but you can get that
# cut-out diagonal lines look with just a bit more work. The important
# thing to know here is that in axes coordinates, which are always
# between 0-1, spine endpoints are at these locations (0,0), (0,1),
# (1,0), and (1,1).  Thus, we just need to put the diagonals in the
# appropriate corners of each of our axes, and so long as we use the
# right transform and disable clipping.

  d = .015 # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
  kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
  ax.plot((1-d,1+d), (-d,+d), **kwargs)
  ax.plot((1-d,1+d),(1-d,1+d), **kwargs)

  kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
  ax2.plot((-d,+d), (1-d,1+d), **kwargs)
  ax2.plot((-d,+d), (-d,+d), **kwargs)

# What's cool about this is that now if we vary the distance between
# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'
