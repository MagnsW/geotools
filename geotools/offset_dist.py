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

    def x_limit(self, lower=None, upper=None):
        if not lower:
            lower = -np.inf
        if not upper:
            upper = np.inf
        self.df = self.df[(self.df['MidPtX'] >= lower) & (self.df['MidPtX'] <= upper)]

    def offset_limit(self, lower=None, upper=None):
        if not lower:
            lower = -np.inf
        if not upper:
            upper = np.inf
        self.df = self.df[(self.df['Offset'] >= lower) & (self.df['Offset'] <= upper)]

    def plot_swarm(self, offsetlim=800, xtickinc=4):
        sns.set_style("whitegrid")
        plt.figure(figsize=(16, 12))
        sns.swarmplot(x='MidPtX', y='Offset', data=self.df[(self.df['Offset'] <= offsetlim)], hue='Configuration')
        plt.ylim(offsetlim, 0)
        locs, labels = plt.xticks()
        plt.xticks(rotation=-45)
        plt.xticks(locs[::xtickinc], labels[::xtickinc])

        plt.show()

    def makeoffsetclass(self, minoff=0, maxoff=600, incr=25):
        offset_breaks = range(minoff, maxoff+incr, incr)
        offset_planes = pd.IntervalIndex.from_breaks(offset_breaks)
        #print(offset_planes)
        self.df['Offsetclass'] = pd.cut(self.df['Offset'], offset_planes)
        self.offset_inc = incr
        self.num_offset_planes = len(offset_planes)

    def hist_offsetclass(self):
        try:
            plt.figure(figsize=(16, 12))
            sns.countplot(x='Offsetclass', data=self.df, hue='Configuration')
            plt.xticks(rotation=-45)
            plt.show()
        except ValueError:
            print("Error: Did you forget to run the makeoffsetclass method?")

    def fill_empty_sublines(self):
        for configuration in self.configurations:
            midpoints_current = self.df[self.df['Configuration']==configuration]['MidPtX'].unique()
            midpoints_current.sort()
            x_min = midpoints_current.min()
            x_max = midpoints_current.max()
            midpoint_diffs = []
            for i in range(0, len(midpoints_current)):
                if i > 0:
                    midpoint_diff = midpoints_current[i] - midpoints_current[i-1]
                    midpoint_diffs.append(midpoint_diff)
            bin_size = min(midpoint_diffs)
            print(f'Bin size seems to be {bin_size} for the configuration {configuration}.')
            midpoints_complete = np.arange(min(midpoints_current), max(midpoints_current)+bin_size, bin_size)
            midpoints_current_set = set(midpoints_current)
            midpoints_complete_set = set(midpoints_complete)
            midpoints_add = midpoints_complete_set.difference(midpoints_current_set)
            dict_to_add = {}
            # for midpoint in midpoints_add:
            #     dict_to_add['MidPtX'] = midpoint
            #     dict_to_add['Configuration'] = configuration
            #     print("Adding midpoint: ", dict_to_add)
            # 
            dict_to_add['MidPtX'] = list(midpoints_add)
            dict_to_add['Configuration'] = [configuration] * len(midpoints_add)
            print(f"Adding the following midpoints to {configuration}: ")
            df_to_add_temp= pd.DataFrame(dict_to_add)
            print(df_to_add_temp)
            self.df = self.df.append(df_to_add_temp, ignore_index=True, sort=True)

    def _make_nan_bins(self, input):
        cols = input.columns
        cols_diff = []
        for i in range(0, len(cols)):
            #print(i, cols[i])
            if i>0:
                col_diff = cols[i] - cols[i-1]
                #print(col_diff)
                cols_diff.append(col_diff)
        bin_size = min(cols_diff)
        new_cols = np.arange(min(cols), max(cols)+bin_size, bin_size)
        cols_set = set(cols)
        cols_new_set = set(new_cols)
        cols_to_add = cols_new_set.difference(cols_set)
        #print("Will add the following nan columns: ", cols_to_add)
        for col in cols_to_add:
            print(col)
            input[col]=np.nan
        return input.sort_index(axis=1), bin_size


    def offsetclassbinning(self):
        self.binned = {}
        self.bin_size = {}
        #self.binned_temp = {}
        for configuration in self.configurations:
            df_binned = pd.DataFrame()
            df_binned = self.df[self.df['Configuration']==configuration].groupby(['Offsetclass', 'MidPtX']).count().rename(columns={'ShotNo': 'Count'}).filter(['Count']).reset_index()
            df_binned_pivot = df_binned.pivot('Offsetclass', 'MidPtX', 'Count')
            #self.binned[configuration] = df_binned_pivot
            self.binned[configuration], self.bin_size[configuration] = self._make_nan_bins(df_binned_pivot)
            print(f'Bin size seems to be {self.bin_size[configuration]} for the configuration {configuration}.')
            #self.binned_temp[configuration] = df_binned

    # def _offsetclassbinning(self, df_input):
    #     df_binned = df_input.groupby(['Offsetclass', 'MidPtX']).count().rename(columns={'ShotNo': 'Count'}).filter(['Count']).reset_index()
    #     df_binned_pivot = df_binned.pivot('Offsetclass', 'MidPtX', 'Count')
    #     return df_binned_pivot


    def plot_offset_bins(self, maxfold=18):
        for configuration in self.configurations:
            #df_binned_pivot = self._offsetclassbinning(self.df[self.df['Configuration']==configuration])
            plt.figure(figsize=(30, 12))
            sns.set_style('dark')
            pal = sns.color_palette('Reds')
            #sns.heatmap(df_binned_pivot, cmap=pal, vmin=1, vmax=18, linewidths=0.1)
            sns.heatmap(self.binned[configuration], cmap=pal, vmin=1, vmax=maxfold, linewidths=0.1)
            plt.title(configuration)
            plt.xticks(rotation=-90)
            plt.show()
        
    def compute_offset_components(self):
        print("Computing offset components...")
        self.df['OffsetY'] = self.df.apply(lambda row: row.Offset*np.cos(np.deg2rad(row.AzSrc)), axis=1)
        self.df['OffsetX'] = self.df.apply(lambda row: row.Offset*np.sin(np.deg2rad(row.AzSrc)), axis=1)
        print("Done!")

    # def _getmaxlength(self, arr): 
    #     # intitialize count 
    #     count = 0 
    #     # initialize max 
    #     result = 0 
    #     for i in range(0, len(arr)): 
    #         # Reset count when True is found 
    #         if arr[i]: 
    #             count = 0
    #         # If False is found, increment count 
    #         # and update result if count  
    #         # becomes more. 
    #         else: 
    #             # increase count 
    #             count+= 1 
    #             result = max(result, count)              
    #     return result

    def _count_empty_bins(self, configuration):
        def getmaxlength(arr):
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
 
        empty_bins_count = pd.notnull(self.binned[configuration]).reset_index().apply(lambda row: getmaxlength(row), axis=1)
        return empty_bins_count

    def plot_empty_bins(self, range=False, y_max=None):
        if not y_max:
            if range:
                y_max = 500
            else:
                y_max = 25
        plt.figure(figsize=(16, 12))
        #sns.set_style('whitegrid')
        
        sns.set_style('dark', {'axes.grid': True, 
            'xtick.bottom': True, 
            'xtick.top': True,
            'ytick.left': True, 
            'ytick.right': True,
            })
        plt.xticks(np.arange(0, self.num_offset_planes,1))
        plt.yticks(np.arange(0, y_max, np.ceil(y_max/25)))
        for configuration in self.configurations:
            if range:
                count_for_conf = self._count_empty_bins(configuration) * self.bin_size[configuration]
            else:
                count_for_conf = self._count_empty_bins(configuration)
            plt.plot(count_for_conf, label=configuration)
        plt.legend()
        plt.xlabel(f'Offset Class ({self.offset_inc}m increment)')
        if range:
            plt.ylabel('X-line range of consecutive empty bins [m]')
            plt.title('Empty x-line Range afo Offset Class')
            plt.ylim(0, y_max)
        else:
            plt.ylabel('Number consecutive empty xline bins')
            plt.title('Number of Consecutive Empty x-line bins afo Offset Class')
            plt.ylim(0, y_max)

    def make_stats(self, attr='AzSrc', minoff=200, maxoff=400):
        stats = {}
        for configuration in self.configurations:
            temp = self.df[(self.df['Offset'] <= maxoff) & (self.df['Offset'] >= minoff)]
            temp = temp[(temp['Configuration'] == configuration)]
            stats_temp = temp.groupby(['MidPtX'])[attr].describe().reset_index()
            print("Configuration: ", configuration)
            print(stats_temp)
            stats[configuration] = stats_temp
        self.stats = stats


        
    