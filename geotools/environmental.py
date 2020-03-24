import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from geotools.input_tools import make_df_from_columndata

class EnvDataAzm:
    '''
    Help function
    '''
    def __init__(self, filename, azimuth=180, datasetname=''):
        self.filename = filename
        if datasetname:
            self.datasetname=datasetname
        else:
            self.datasetname=filename
        self.df = make_df_from_columndata(filename)
        self.df = self._slice_azm(azimuth)

    def _slice_azm(self, azimuth):
        df_temp = self.df.loc[(self.df['Azimuth'] == azimuth)].reset_index(drop=True)
        print(f'Azimuth slice at {azimuth} degrees created')
        return df_temp.drop(['Azimuth'], axis=1)

    def plot(self):
        value_to_plot = self.df.columns[2]
        #print(f'Creating pivot table for {value_to_plot}...')
        df_pivot = self.df.pivot(index='Depth', columns='Range', values=value_to_plot)
        plt.figure(figsize=(16,8))
        plt.title(self.datasetname + '; ' + value_to_plot)
        sns.heatmap(df_pivot, cmap='jet')
        plt.show()

    def limit(self, maxoffset=np.inf, maxdepth=np.inf):
        self.df = self.df[self.df['Range'] <= maxoffset]
        self.df = self.df[self.df['Depth'] <= maxdepth]

    def mirror_azm(self):
        data_negative = self.df.copy(deep=True)
        data_zof = self.df.loc[self.df['Range'] == self.df['Range'].min()].reset_index(drop=True)
        data_zof['Range'] = 0
        data_negative['Range'] = data_negative['Range'].apply(np.negative)
        data_negative_sort = data_negative.sort_values(by=['Range', 'Depth'])
        combined_data = pd.concat([data_negative_sort, data_zof, self.df]).reset_index(drop=True)
        self.df = combined_data
        self.datasetname += ' - Mirrored'

    def shift_and_limit(self, shift=0, minval=-np.inf, maxval=np.inf):
        self.df['Range'] = self.df['Range'] + shift
        self.df = self.df.loc[(self.df['Range'] >= minval) & (self.df['Range'] <= maxval)].reset_index(drop=True)
        self.datasetname += ' - Shifted'

    def _dB_sum(self, amp1, amp2):
        x = amp1/10
        y = amp2/10
        amp_sum = 10*(np.log10(10**x + 10**y))
        return amp_sum

    def dB_add(self, other):
        value_to_sum = self.df.columns[2]
        #df_temp = self.df.join(other.df, rsuffix='-r', on)
        df_temp = pd.merge(self.df, other.df, how='inner', on=['Range', 'Depth'])
        #print(df_temp)
        self.df[value_to_sum] = df_temp.apply(lambda x: self._dB_sum(x[value_to_sum + '_x'], x[value_to_sum + '_y']), axis=1)
        self.datasetname += ' - Added ' + other.datasetname

    def rename(self, newname):
        self.datasetname = newname


def plot_graph(depth, *argv, x_lim=None, y_lim=None):
    plt.figure(figsize=(16, 8))
    sns.set_style('darkgrid')
    if x_lim:
        plt.xlim(*x_lim)
    if y_lim:
        plt.ylim(*y_lim)
    for dataset in argv:
        print(dataset.datasetname)
        plt.title(f'Depth: {depth} m')
        value_to_plot = dataset.df.columns[2]
        df_temp = dataset.df[dataset.df['Depth'] == depth]
        sns.regplot(x=df_temp['Range'], y=df_temp[value_to_plot], fit_reg=False, label=dataset.datasetname, marker='.')
    plt.legend()
    plt.show()
