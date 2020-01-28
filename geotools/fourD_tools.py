
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from geotools.input_tools import make_df_from_columndata
import warnings
warnings.filterwarnings("ignore")


class RepeatabilityData:
    def __init__(self, filenames):
        self.filenames = filenames
        self.df = self._readtodataframe()

    def _readtodataframe(self):
        df_list = []
        for scenario in self.filenames:
            df = pd.DataFrame()
            print(scenario)
            print(self.filenames[scenario])
            df = make_df_from_columndata(self.filenames[scenario])
            df['Scenario'] = scenario
            df_list.append(df) 
        df_combined = pd.concat(df_list)
        return df_combined

    def offsetsplit(self, channo):
        self.df['Channel Range'] = self.df['BasRecvNo'].apply(
            lambda x: '< '+str(channo) if ((x % 408 < channo) & (x % 408 > 0)) 
            else '> '+str(channo))

    def plot_all_dist(self):
        self.plot_dist(attrib='SouRecDist', title='Delta S + Delta R', plttype='violin')
        self.plot_dist(attrib='OffsDiff', title='Delta Offset', plttype='violin')
        self.plot_dist(attrib='SouDist', title='Delta S', plttype='violin')
        self.plot_dist(attrib='RecDist', title='Delta R', plttype='violin')
        self.plot_dist(attrib='AzimDiff', title='Delta Azimuth', plttype='violin')

    def plot_dist(self, attrib='SouRecDist', title='Delta S + Delta R', plttype='violin', maxval=None):
        sns.set_style("darkgrid")
        sns.set_context('paper')
        plt.figure(figsize=(15,12))
        plt.title(title)
        plt.xticks(rotation=20)
        plt.gca().invert_yaxis()
        if maxval:
            plt.ylim(maxval, -maxval/20)
        if plttype == 'violin':
            if 'Channel Range' in self.df.columns:
                sns.set_palette('Paired')
                sns.violinplot(data=self.df, x='Scenario', y=attrib, hue='Channel Range', split=True, cut=0, scale='count')
            else:
                sns.violinplot(data=self.df, x='Scenario', y=attrib, cut=0, scale='count')
        else:
            if 'Channel Range' in self.df.columns:
                sns.set_palette('Paired')
                sns.boxplot(data=self.df, x='Scenario', y=attrib, hue='Channel Range')
            else:
                ns.boxplot(data=self.df, x='Scenario', y=attrib)
        plt.show()

    
