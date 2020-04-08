import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
from GStools.input_tools import make_df_from_columndata

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
        print(f"Input data MidPtX range is: {self.df['MidPtX'].min()} to {self.df['MidPtX'].max()}")
        self._compute_offset_components()
        self._fill_empty_sublines() 
        self._makeoffsetclass()
        self._offsetclassbinning()
        # self.attribs = ['AzRec', 'AzSrc', 'DipRec', 'DipSrc', 
        #                 'Offset', 'RefAngIn', 'RefAngOut', 'Ttime']
        self.attribs = ['AzSrc', 'DipSrc', 'Offset', 'RefAngIn', 'Ttime']
        #self.attribs = ['AzSrc']
        self.df_attribs_stats = self._make_stats()
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
        print(f"Offsets limited to [{self.minoffset}, {self.maxoffset}]m")
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
            dict_to_add['MidPtX'] = sorted(list(midpoints_add))
            dict_to_add['Configuration'] = [configuration] * len(midpoints_add)
            print(f"Adding the following midpoints to {configuration}: ")
            df_to_add_temp= pd.DataFrame(dict_to_add)
            if len(df_to_add_temp) > 0:
                print(df_to_add_temp)
            else:
                print("No midpoints added.")
            self.bin_sizes[configuration] = bin_size
            self.df = self.df.append(df_to_add_temp, ignore_index=True, sort=True)

    def _makeoffsetclass(self):
        offset_breaks = np.arange(self.minoffset, self.maxoffset+self.incoffset, self.incoffset)
        offset_planes = pd.IntervalIndex.from_breaks(offset_breaks)
        print(f"Generating {len(offset_planes)} offset planes")
        self.df['Offsetclass'] = pd.cut(self.df['Offset'], offset_planes)
        self.offset_planes = offset_planes
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
            #print(col)
            input[col]=np.nan
        return input.sort_index(axis=1)

    def _make_stats(self):
        stats = pd.DataFrame()
        print("Computing attribute statistics...")
        for attrib in self.attribs:
            print(f"Computing for attribute {attrib}")
            temp = self.df.groupby(['MidPtX', 'Configuration', 'Offsetclass'])[attrib].describe().reset_index()
            temp['Attribute'] = attrib
            stats = stats.append(temp, ignore_index=True)

        print("Done!")

        print(f"Columns in attibute statistics dataset is: {list(stats.columns)}")
        return stats

    def x_limit(self, lower=None, upper=None):
        if not lower:
            lower = -np.inf
        if not upper:
            upper = np.inf
        self.df = self.df[(self.df['MidPtX'] >= lower) & (self.df['MidPtX'] <= upper)]
        print("Regenerating attibute statistics...")
        self.df_attribs_stats = self._make_stats()
        print("Rebinning...")
        self._offsetclassbinning()
        

    def offset_limit(self, lower=None, upper=None):
        if not lower:
            lower = -np.inf
        if not upper:
            upper = np.inf
        self.df = self.df[(self.df['Offset'] >= lower) & (self.df['Offset'] <= upper)]
        print("Regenerating attibute statistics...")
        self.df_attribs_stats = self._make_stats()
        print("Rebinning...")
        self._offsetclassbinning()

    def plot_swarm(self, xtickinc=4):
        sns.set_style("whitegrid")
        plt.figure(figsize=(16, 12))
        sns.swarmplot(x='MidPtX', y='Offset', data=self.df, hue='Configuration')
        plt.ylim(self.maxoffset, 0)
        locs, labels = plt.xticks()
        plt.xticks(rotation=-45)
        plt.xticks(locs[::xtickinc], labels[::xtickinc])
        plt.show()


    def plot_attrib(self, attrib='AzSrc', descriptor='count', minval=None, maxval=None, palette=None):
        for configuration in self.configurations:
            #print(f"len(self.df_attribs_stats): {len(self.df_attribs_stats)}")
            self.temp = self.df_attribs_stats[(self.df_attribs_stats['Configuration'] == configuration) & (self.df_attribs_stats['Attribute'] == attrib)]
            #print(f"len(temp): {len(temp)}")
            #print(f"config: {configuration}, columns:  {temp.columns}")
            self.temp_pivot = self.temp.pivot(index='Offsetclass', columns='MidPtX', values=descriptor).sort_values('Offsetclass')
            self.temp_pivot = self._make_nan_bins(self.temp_pivot)
            #print("Here comes temp_pivot")
            #print(self.temp_pivot)
            plt.figure(figsize=(30, 12))
            sns.set_style('dark')
            #pal = sns.color_palette(palette)
            #sns.heatmap(df_binned_pivot, cmap=pal, vmin=1, vmax=18, linewidths=0.1)
            
            sns.heatmap(self.temp_pivot, cmap=palette, vmin=minval, vmax=maxval, linewidths=0.1)
            #sns.heatmap(self.temp_pivot, linewidths=0.1)
            plt.title(f"Attribute: {attrib}; Descriptor: {descriptor}; Configuration: {configuration}")
            plt.xticks(rotation=-90)
            plt.yticks(rotation=0)
            plt.show()

    def plot_offset_bins(self, maxfold=None, palette=None):
        for configuration in self.configurations:
            #df_binned_pivot = self._offsetclassbinning(self.df[self.df['Configuration']==configuration])
            plt.figure(figsize=(30, 12))
            sns.set_style('dark')
            pal = sns.color_palette(palette)
            #sns.heatmap(df_binned_pivot, cmap=pal, vmin=1, vmax=18, linewidths=0.1)
            sns.heatmap(self.offset_binned[configuration], cmap=pal, vmin=0, vmax=maxfold, linewidths=0.1)
            plt.title(configuration)
            plt.xticks(rotation=-90)
            plt.yticks(rotation=0)
            plt.show()

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
 
        empty_bins_count = pd.notnull(self.offset_binned[configuration]).reset_index().apply(lambda row: getmaxlength(row), axis=1)
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
                count_for_conf = self._count_empty_bins(configuration) * self.bin_sizes[configuration]
            else:
                count_for_conf = self._count_empty_bins(configuration)
            plt.plot(count_for_conf, label=configuration)
        plt.legend()
        plt.xlabel(f'Offset Class ({self.incoffset}m increment)')
        if range:
            plt.ylabel('X-line range of consecutive empty bins [m]')
            plt.title('Empty x-line Range afo Offset Class')
            plt.ylim(0, y_max)
        else:
            plt.ylabel('Number consecutive empty xline bins')
            plt.title('Number of Consecutive Empty x-line bins afo Offset Class')
            plt.ylim(0, y_max)

    def plot_stat_for_offset_class(self, offsetplane, config, attrib, descriptor='50%', swarm=True, y_lim=None):
        df_temp = self.df[self.df['Configuration'] == config]
        bins = df_temp['MidPtX'].unique()
        df_temp = df_temp[(df_temp['Offsetclass'] == self.offset_planes[offsetplane])]
        bins_kept = df_temp['MidPtX'].unique()
        bins_to_add = set(bins).difference(set(bins_kept))
        dict_to_add = {}
        dict_to_add['MidPtX'] = sorted(list(bins_to_add))
        df_to_add = pd.DataFrame(dict_to_add)
        df_temp = df_temp.append(df_to_add, ignore_index=True, sort=True)
        df_temp_graph = self.df_attribs_stats[(self.df_attribs_stats['Attribute'] == attrib) & (self.df_attribs_stats['Offsetclass'] == self.offset_planes[offsetplane])]
        df_temp_graph = df_temp_graph[df_temp_graph['Configuration'] == config]
        bins_graph_kept = df_temp_graph['MidPtX'].unique()
        df_temp_graph = df_temp_graph.append(df_to_add, ignore_index=True, sort=True).sort_values(['MidPtX']).reset_index()
        plt.figure(figsize=(24, 12))
        if y_lim:
            plt.ylim(*y_lim)
        sns.set_style('darkgrid')
        if swarm:
            sns.swarmplot(x='MidPtX', y=attrib, data=df_temp.reset_index(), hue='Configuration', hue_order=self.configurations)
        else:
            sns.boxplot(x='MidPtX', y=attrib, data=df_temp.reset_index(), hue='Configuration', hue_order=self.configurations)
        sns.lineplot(data=df_temp_graph[descriptor], color='grey', style='.')
        locs, labels = plt.xticks()
        plt.xticks(locs[::1], labels[::1], rotation=-90)
        plt.title(f"Subselection: Offset: {self.offset_planes[offsetplane]}")
        plt.show()

    def plot_stat_for_offset_class_comp(self, offsetplane, attrib, descriptor, dist=False, hist=True, kde=False, histbins=(0, 180, 5), linewidth=3):
        df_temp_graph = self.df_attribs_stats[(self.df_attribs_stats['Attribute'] == attrib) & (self.df_attribs_stats['Offsetclass'] == self.offset_planes[offsetplane])]
        df_temp_graph = df_temp_graph.sort_values(['Configuration', 'MidPtX']).reset_index()
        df_diff = pd.DataFrame()
        for config in self.configurations:
            df_diff_temp = df_temp_graph[df_temp_graph['Configuration'] == config][[descriptor]]
            df_diff_temp = df_diff_temp.diff().apply(np.abs)
            df_diff_temp.columns = [descriptor + ' AbsDiff']
            df_diff = df_diff.append(df_diff_temp)
        df_temp_graph = pd.concat([df_temp_graph, df_diff],axis=1)
        plt.figure(figsize=(24, 12))
        sns.set_style('darkgrid')
        if dist:
            for config in self.configurations:
                #sns.distplot(df_temp_graph[df_temp_graph['Configuration'] == config][descriptor + ' AbsDiff'].dropna(), bins = list(np.arange(*histbins)), kde=False, hist_kws={"histtype": "step", "linewidth": linewidth}, label=config)
                sns.distplot(df_temp_graph[df_temp_graph['Configuration'] == config][descriptor + ' AbsDiff'].dropna(), bins = list(np.arange(*histbins)), kde=kde, hist=hist, hist_kws={"histtype": "step", "linewidth": linewidth}, label=config)
        else:
            sns.lineplot(x='MidPtX', y=descriptor + ' AbsDiff', data=df_temp_graph, hue='Configuration', hue_order=self.configurations)
        plt.legend()
        plt.title(f"Subselection: Offset: {self.offset_planes[offsetplane]} for attribute: {attrib}")
        plt.show()


    # def _make_stats(self):
    #     stats = {}
    #     print("Computing attribute statistics...")
    #     for attrib in self.attribs:
    #         stats[attrib] = {}
    #         for configuration in self.configurations:
    #             temp = self.df[(self.df['Offset'] <= self.maxoffset) & (self.df['Offset'] >= self.minoffset)]
    #             temp = temp[(temp['Configuration'] == configuration)]
    #             stats_temp = temp.groupby(['MidPtX'])[attrib].describe().reset_index()
    #             #print("Configuration: ", configuration)
    #             #print(stats_temp)
    #             stats[attrib][configuration] = stats_temp
    #     print("Done!")
    #     return stats

