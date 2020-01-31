import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
from geotools.input_tools import make_df_from_columndata

class OffsetData:
    '''
    Help function goes here
    '''

    def __init__(self, filenames):
        self.filenames = filenames
        self.configurations = list(filenames.keys())
        self.df = self._readtodataframe()
                
    def _readtodataframe(self):
        df_list = []
        for configuration in self.filenames:
            df = pd.DataFrame()
            print("Reading configuration: ", configuration)
            print("From file: ", self.filenames[configuration])
            df = make_df_from_columndata(self.filenames[configuration])
            df['Configuration'] = configuration
            df_list.append(df) 
        df_combined = pd.concat(df_list)
        return df_combined.reset_index()

    def plot_swarm(self, offsetlim=800, xtickinc=4):
        sns.set_style("whitegrid")
        plt.figure(figsize=(16, 12))
        sns.swarmplot(x='MidPtX', y='Offset', data=self.df[(self.df['Offset'] <= offsetlim)], hue='Configuration')
        plt.ylim(0, offsetlim)
        locs, labels = plt.xticks()
        plt.xticks(rotation=-45)
        plt.xticks(locs[::xtickinc], labels[::xtickinc])

        plt.show()

    def makeoffsetclass(self, minoff=0, maxoff=600, incr=25):
        offset_breaks = range(minoff, maxoff+incr, incr)
        offset_planes = pd.IntervalIndex.from_breaks(offset_breaks)
        #print(offset_planes)
        self.df['Offsetclass'] = pd.cut(self.df['Offset'], offset_planes)

    def hist_offsetclass(self):
        try:
            plt.figure(figsize=(16, 12))
            sns.countplot(x='Offsetclass', data=self.df, hue='Configuration')
            plt.xticks(rotation=-45)
            plt.show()
        except ValueError:
            print("Error: Did you forget to run the makeoffsetclass method?")

    def _offsetclassbinning(self, df_input):
        df_binned = df_input.groupby(['Offsetclass', 'MidPtX']).count().rename(columns={'ShotNo': 'Count'}).filter(['Count']).reset_index()
        df_binned_pivot = df_binned.pivot('Offsetclass', 'MidPtX', 'Count')
        return df_binned_pivot

    def plot_offset_bins(self):
        for configuration in self.configurations:
            df_binned_pivot = self._offsetclassbinning(self.df[self.df['Configuration']==configuration])
            plt.figure(figsize=(30, 12))
            sns.set_style('dark')
            pal = sns.color_palette('Reds')
            sns.heatmap(df_binned_pivot, cmap=pal, vmin=1, vmax=18, linewidths=0.1)
            plt.title(configuration)
            plt.xticks(rotation=-90)
            plt.show()
        
    def compute_offset_components(self):
        print("Computing offset components...")
        self.df['OffsetY'] = self.df.apply(lambda row: row.Offset*np.cos(np.deg2rad(row.AzSrc)), axis=1)
        self.df['OffsetX'] = self.df.apply(lambda row: row.Offset*np.sin(np.deg2rad(row.AzSrc)), axis=1)
        print("Done!")

    def _getmaxlength(self, arr): 
        # intitialize count 
        count = 0 
        # initialize max 
        result = 0 
        for i in range(0, len(arr)): 
            # Reset count when True is found 
            if arr[i]: 
                count = 0
            # If False is found, increment count 
            # and update result if count  
            # becomes more. 
            else: 
                # increase count 
                count+= 1 
                result = max(result, count)              
        return result

    def _count_empty_bins(self, configuration):
        df_binned_pivot = self._offsetclassbinning(self.df[self.df['Configuration']==configuration])
        empty_bins_count = pd.notnull(df_binned_pivot).reset_index().apply(lambda row: self._getmaxlength(row), axis=1)
        return empty_bins_count

    def plot_empty_bins(self):
        plt.figure(figsize=(16, 12))
        sns.set_style('whitegrid')
        for configuration in self.configurations:
            count_for_conf = self._count_empty_bins(configuration)
            plt.plot(count_for_conf, label=configuration)
        plt.legend()
        plt.xlabel('Offset Class (25m increment)')
        plt.ylabel('Number consecutive empty xline bins')
        plt.title('Number of Empty x-line bins afo Offset Class')
        plt.ylim(0, 25)



    