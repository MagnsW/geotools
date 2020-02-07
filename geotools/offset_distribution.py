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

    def __init__(self, filenames, minoffset=0, maxoffset=800, incoffset=25):
        self.filenames = filenames
        self.configurations = list(filenames.keys())
        self.minoffset = minoffset
        self.maxoffset = maxoffset
        self.incoffset = incoffset
        self.df = self._readtodataframe()
        self._compute_offset_components()
        self._fill_empty_sublines() 
        self._makeoffsetclass()
        self._offsetclassbinning()
        print("------INIT FINISHED--------")

    def _readtodataframe(self):
        df_list = []
        for configuration in self.filenames:
            df = pd.DataFrame()
            print("Reading configuration: ", configuration)
            print("From file: ", self.filenames[configuration])
            df = make_df_from_columndata(self.filenames[configuration])
            df['Configuration'] = configuration
            df = df[(df['Offset'] >= self.minoffset) & (df['Offset'] <= self.maxoffset)]            
            df_list.append(df) 
        df_combined = pd.concat(df_list)
        print(f"Offsets limited to [{self.minoffset}, {self.maxoffset}]")
        return df_combined.reset_index()

    def _compute_offset_components(self):
        print("Computing offset components...")
        self.df['OffsetY'] = self.df.apply(lambda row: row.Offset*np.cos(np.deg2rad(row.AzSrc)), axis=1)
        self.df['OffsetX'] = self.df.apply(lambda row: row.Offset*np.sin(np.deg2rad(row.AzSrc)), axis=1)
        print("Done!")

    def _fill_empty_sublines(self):
        self.bin_sizes = {}
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
            self.bin_sizes[configuration] = bin_size
            self.df = self.df.append(df_to_add_temp, ignore_index=True, sort=True)

    def _makeoffsetclass(self):
        offset_breaks = np.arange(self.minoffset, self.maxoffset+self.incoffset, self.incoffset)
        offset_planes = pd.IntervalIndex.from_breaks(offset_breaks)
        print(f"Generating the following offset planes: {offset_planes}")
        self.df['Offsetclass'] = pd.cut(self.df['Offset'], offset_planes)
        self.num_offset_planes = len(offset_planes)

    def _offsetclassbinning(self):
        self.offset_binned = {}
        for configuration in self.configurations:
            df_binned = pd.DataFrame()
            df_binned = self.df[self.df['Configuration']==configuration].groupby(['Offsetclass', 'MidPtX']).count().rename(columns={'ShotNo': 'Count'}).filter(['Count']).reset_index()
            df_binned_pivot = df_binned.pivot('Offsetclass', 'MidPtX', 'Count')
            self.offset_binned[configuration] = df_binned_pivot
            #self.binned[configuration], self.bin_size[configuration] = self._make_nan_bins(df_binned_pivot)
            #print(f'Bin size seems to be {self.bin_size[configuration]} for the configuration {configuration}.')

