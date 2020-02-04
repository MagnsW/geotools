import numpy as np
import itertools
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

class WSModel:
  '''This class creates a waterspeed model

  To initialize a model with 25m shot point interval, type my_waterspeedmodel_25m_SPI = waterspeed.WSModel(25.00)

  Optional parameters are (with default values): maxcurrent=2, pop_min=5000, pop_max=15000, wsp_bins=[-np.inf, 2.5, 3.0, 3.5, 5.0, 5.5, 6.0, np.inf]
  maxcurrent, pop_min and pop_max parameters define the plot boundaries. wsp_bins define the water speed intervals.
   
  If you want to make the waterspeed plot only, call the method plotwaterspeed. 
  Example: my_waterspeedmodel_25m_SPI.plotwaterspeed()

  If you want to make a combined plot with water speed and bottom speed, first set the the BSP intervals with the setpoplimits method.
  
  Example: my_waterspeedmodel_25m_SPI.setpoplimit(5000)
  
  5000 here is then the minimum pop-interval in ms and defines the maximum BSP. Max BSP will be printed out when method is run.
  Optional parameter is bsplowlimit, which has a default value of 3.0. If you want to change this, see example below.
  
  Example: my_waterspeedmodel_25m_SPI.setpoplimit(5000, bsplowlimit=2.5)

  To plot the combined waterspeed / bottom speed plot, run the plotcombspeed method.
  Example: my_waterspeedmodel_25m_SPI.plotcombspeed()

  '''
  def __init__(self, sp_int, maxcurrent=2, pop_min=5000, pop_max=15000, wsp_bins=None):
    '''
    To initialize a model with 25m shot point interval, type my_waterspeedmodel_25m_SPI = waterspeed.WSModel(25.00)

    Optional parameters are (with default values): maxcurrent=2, pop_min=5000, pop_max=15000, wsp_bins=[-np.inf, 2.5, 3.0, 3.5, 5.0, 5.5, 6.0, np.inf]
    maxcurrent, pop_min and pop_max parameters define the plot boundaries. wsp_bins define the water speed intervals.

    '''

    self.sp_int = float(sp_int)
    self.maxtailcurrent = -maxcurrent
    self.maxheadcurrent = maxcurrent
    self.currents = self._makecurrents()
    self.pop_min = pop_min
    self.pop_max = pop_max
    self.popintervals, self.pop_inc = self._makepopintervals()
    
    if not wsp_bins:
      self.wsp_bins = [-np.inf, 2.5, 3.0, 3.5, 5.0, 5.5, 6.0, np.inf]
    else:
      wsp_bins.insert(0, -np.inf)
      wsp_bins.append(np.inf)
      self.wsp_bins = wsp_bins
    self.dataframe = self._makedataframe()

  def _makecurrents(self):
    current_inc = (self.maxheadcurrent - self.maxtailcurrent) / 80
    #print(current_inc)
    return np.round(np.arange(self.maxtailcurrent, self.maxheadcurrent + current_inc, current_inc), decimals=3)

  def _makepopintervals(self):
    pop_inc = int((self.pop_max - self.pop_min) / 40)
    return list(range(self.pop_min, self.pop_max + pop_inc, pop_inc)), pop_inc

  def _makedataframe(self):
    df = pd.DataFrame(columns = ['Pop Interval (ms)', 'Current (knots)'])
    combinations = itertools.product(self.popintervals, self.currents)
    for combo in combinations:
      df.loc[df.shape[0],:] = combo
    df['Shot Point Interval (m)'] = self.sp_int
    df['BSP (m/s)'] = df['Shot Point Interval (m)'] / df['Pop Interval (ms)'] * 1000
    df['BSP (knots)'] = df['BSP (m/s)'] * 3600/1852
    df['WSP (knots)'] = df['BSP (knots)'] + df['Current (knots)']
    df['WSP Interval (knots)'] = pd.cut(df['WSP (knots)'], bins=self.wsp_bins)
    return df

  def plotwaterspeed(self, cleanreclength=None, minpopint=None, currentmark=1.0):
    '''
    
    If you want to make the waterspeed plot only, call the method plotwaterspeed. 
    Example: my_waterspeedmodel_25m_SPI.plotwaterspeed()

    Optional parameters is cleanreclength and minpopint which will draw the horizontal lines at corrsponding bottom speeds.
    cleanreclength could for example be set to the required clean record and minpopint could be the clean record length plus the dither window.
    Last optional parameter is currentmark, which will draw a vertical dashed line at a current that you want to highlight in the plot.

    '''
    title = 'SP interval: ' + str(self.sp_int) + 'm'
    dftoplot = self.dataframe
    dftoplot['WSP Interval (knots)'] = dftoplot['WSP Interval (knots)'].astype(str)
    #dftoplot = self.dataframe.sort_values(by=['WSP Interval (knots)'], ascending=False)
    use_fontdict = {'fontsize': 20, 'fontweight' : 20, 'verticalalignment': 'baseline', 'horizontalalignment': 'center'}
    sns.set_style("whitegrid")
    plt.figure(figsize=(18, 10))
    plt.gca().invert_yaxis()
    #plt.ylim((4000,12000))
    plt.title(title, fontdict=use_fontdict)
    if cleanreclength:
      plt.axhline(cleanreclength, color='black')
      plt.text(self.maxheadcurrent, cleanreclength, 'Clean record length: '+str(cleanreclength) +'ms', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))
    if minpopint:
      plt.axhline(minpopint, color='black')
      plt.text(self.maxheadcurrent, minpopint, 'Pop interval: '+str(minpopint) +'ms', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))
    plt.axvline(0, linestyle=":", color='black')
    plt.axvline(-currentmark, linestyle=":", color='black')
    plt.axvline(currentmark, linestyle=":", color='black')
    #sns.scatterplot(x=self.dataframe['Current (knots)'], y=self.dataframe['Pop Interval (ms)'], hue=self.dataframe['WSP Issue'], hue_order=choicelist, s=100, marker='o', palette="coolwarm")
    #sns.scatterplot(x=self.dataframe['Current (knots)'], y=self.dataframe['Pop Interval (ms)'], hue=self.dataframe['WSP Interval (knots)'], s=100, marker='o', palette="coolwarm")
    sns.scatterplot(x=dftoplot['Current (knots)'], y=dftoplot['Pop Interval (ms)'], hue=dftoplot['WSP Interval (knots)'], s=100, marker='o', palette="coolwarm_r")
    plt.text(self.maxtailcurrent/2, self.pop_max + 2 * self.pop_inc, 'Tail currents', fontdict=use_fontdict)
    plt.text(self.maxheadcurrent/2, self.pop_max + 2 * self.pop_inc, 'Head currents', fontdict=use_fontdict)
    #
    #plt.text(4, req_rec_length2, 'Record length with dither: '+str(req_rec_length2) +'ms', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))
    plt.legend(framealpha=1, loc='lower left')
    plt.show()

  def setbsplimits(self, bsp_bins):
    bsp_bins.insert(0, -np.inf)
    bsp_bins.append(np.inf)
    self.bsp_bins = bsp_bins
    self.dataframe['BSP Interval (knots)'] = pd.cut(self.dataframe['BSP (knots)'], bins=bsp_bins)

  def setpoplimit(self, poplimit, bsplowlimit=3.0):
    '''

    If you want to make a combined plot with water speed and bottom speed, first set the the BSP intervals with the setpoplimits method.
  
    Example: my_waterspeedmodel_25m_SPI.setpoplimit(5000)
  
    5000 here is then the minimum pop-interval in ms and defines the maximum BSP. Max BSP will be printed out when method is run.
    Optional parameter is bsplowlimit, which has a default value of 3.0. If you want to change this, see example below.
    
    Example: my_waterspeedmodel_25m_SPI.setpoplimit(5000, bsplowlimit=2.5)

    '''
    maxbottomspeed = np.around(self.sp_int / poplimit * 1000 * 3600/1852, decimals=2)
    print("Max BSP set to: " + str(maxbottomspeed))
    print("Low BSP limit set to: " + str(bsplowlimit))
    self.dataframe['Max BSP (knots)'] = maxbottomspeed
    condlist_bsp = [self.dataframe['BSP (knots)'] > self.dataframe['Max BSP (knots)'], 
                  ((self.dataframe['BSP (knots)'] <= self.dataframe['Max BSP (knots)']) & (self.dataframe['BSP (knots)'] >= bsplowlimit)),
                    self.dataframe['BSP (knots)'] < bsplowlimit]
    choicelist_bsp = ['BSP > ' + str(maxbottomspeed) + ' kn (data loss)',
                      'No issue',
                      'BSP < ' + str(bsplowlimit) + ' kn']
    self.dataframe['BSP Interval (knots)'] = (np.select(condlist_bsp, choicelist_bsp))

  def plotcombspeed(self, cleanreclength=None, minpopint=None, currentmark=1.0):
    '''
    
    To plot the combined waterspeed / bottom speed plot, run the plotcombspeed method.
    Example: my_waterspeedmodel_25m_SPI.plotcombspeed()

    Optional parameters is cleanreclength and minpopint which will draw the horizontal lines at corrsponding bottom speeds.
    cleanreclength could for example be set to the required clean record and minpopint could be the clean record length plus the dither window.
    Last optional parameter is currentmark, which will draw a vertical dashed line at a current that you want to highlight in the plot.

    This function will fail if setpoplimit is not run first to define the BSP limits.

    '''
    title = 'SP interval: ' + str(self.sp_int) + 'm'
    dftoplot = self.dataframe
    dftoplot['WSP Interval (knots)'] = dftoplot['WSP Interval (knots)'].astype(str)
    use_fontdict = {'fontsize': 20, 'fontweight' : 20, 'verticalalignment': 'baseline', 'horizontalalignment': 'center'}
    sns.set_style("whitegrid")
    plt.figure(figsize=(18, 10))
    plt.gca().invert_yaxis()
    #plt.ylim((4000,12000))
    plt.title(title, fontdict=use_fontdict)
    if cleanreclength:
      plt.axhline(cleanreclength, color='black')
      plt.text(self.maxheadcurrent, cleanreclength, 'Clean record length: '+str(cleanreclength) +'ms', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))
    if minpopint:
      plt.axhline(minpopint, color='black')
      plt.text(self.maxheadcurrent, minpopint, 'Pop interval: '+str(minpopint) +'ms', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))
    plt.axvline(0, linestyle=":", color='black')
    plt.axvline(-currentmark, linestyle=":", color='black')
    plt.axvline(currentmark, linestyle=":", color='black')
    dftoplot = dftoplot.rename(columns={'WSP Interval (knots)': 'WSP Interval [kn] (Outer disc)', 'BSP Interval (knots)': 'BSP Interval (Inner dot)'})
    sns.scatterplot(x=dftoplot['Current (knots)'], y=dftoplot['Pop Interval (ms)'], hue=dftoplot['WSP Interval [kn] (Outer disc)'], s=120, marker='o', palette="coolwarm_r")
    try:
      sns.scatterplot(x=dftoplot['Current (knots)'], y=dftoplot['Pop Interval (ms)'], hue=dftoplot['BSP Interval (Inner dot)'], s=30, marker='o', palette="coolwarm_r")
    except:
      print("Pop limit not set. You need to run the method .setpoplimit() on your water speed model first to plot the BSP dots.")
    plt.text(self.maxtailcurrent/2, self.pop_max + 2 * self.pop_inc, 'Tail currents', fontdict=use_fontdict)
    plt.text(self.maxheadcurrent/2, self.pop_max + 2 * self.pop_inc, 'Head currents', fontdict=use_fontdict)
    plt.legend(framealpha=1)
    plt.show()


