
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from geotools.input_tools import make_df_from_columndata
import warnings
warnings.filterwarnings("ignore")


class RepeatabilityData:
    '''
    Make a repeatability dataset by: 
    my_repeatability_data = Reapeatability(scenario_files)
    scenario_files must be a python dictionary with scenario as key and filename
    as value.
    Example:

    scenario_files = {
    '10x75m - 2deg on 8x75m': 'Gap_REFf2-2_on_POLf0-s.A1X',
    '12x75m - 2deg on 8x75m': 'Gap_12x75f2-2_on_POLf0-s.A1X',
    '19x37.5m - 2deg on 8x75m': 'Gap_19x37f2-2_on_POLf0-s.A1X',
    '19x37.5m - 4deg on 8x75m': 'Gap_19x37f4-4_on_POLf0-s.A1X',
    '19x37.5m - 150perc Fanning - 2deg on 8x75m': 'Gap_19x37fanf2-2_POLf0-s.A1X',
    '19x37.5m - 150perc Fanning - 4deg on 8x75m': 'Gap_19x37fanf4-4_POLf0-s.A1X',
    } 

    The my_repeatability_data will be a dataframe. This can be inspected by
    using print(my_repeatability_data.df). Check documentation on 
    pandas dataframes.

    Once crated, you can split on offset, using the offsetsplit method. This is
    an optional operation. This does not affect the data, it just adds and 
    populates the column 'Channel Range'.
    Example:

    my_repeatability_data.offsetsplit(100, 408)

    This will split the channels at channel no 100. 408 is the total number of 

    This will return the same dataframe with an extra column called 
    'Channel Range'

    For plotting the distributions, use the method plot_dist
    Example:
    my_repeatability_data.plot_dist()

    With no arguments, the SouRecDist attribute will be plotted. Other 
    attributes can be plotted by selection using the attrib parameter.
    Example:
    my_repeatability_data.plot_dist(attrib='OffsDiff', title='Delta Offset', maxval=75)

    The maxval is the maximum value (y-limit) in the plot.

    If you want to plot all attributes with default parameters, use the 
    plot_all_dist method.
    Example:
    my_repeatability_data.plot_all_dist()

    '''


    def __init__(self, filenames):
        self.filenames = filenames
        self.scenarios = list(filenames.keys())
        self.df = self._readtodataframe()
        self.plot_params = {
            'SouRecDist': ['Delta S + Delta R', 150],
            'OffsDiff': ['Delta Offset', 50],
            'SouDist': ['Delta S', 75],
            'RecDist': ['Delta R', 75],
            'AzimDiff': ['Delta Azimuth', 8],
            }
        self.summaries = self._makesummaries()

    def _readtodataframe(self):
        df_list = []
        for scenario in self.filenames:
            df = pd.DataFrame()
            print("Reading scenario: ", scenario)
            print("From file: ", self.filenames[scenario])
            df = make_df_from_columndata(self.filenames[scenario])
            df['Scenario'] = scenario
            df_list.append(df) 
        df_combined = pd.concat(df_list)
        return df_combined

    def _makesummaries(self):
        keys_dict = {}
        for key in self.plot_params:
            keys_dict[key] = self._make_full_summary(key)
            print("Making summary for: ", key)
        return keys_dict

    
    def _make_summary(self, scenario, attribute):
        df = self.df[self.df.Scenario == scenario]
        summary = df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])[[attribute]].round(decimals=1)
        summary = summary.T
        summary['Scenario'] = scenario
        summary = summary.set_index(['Scenario'])
        return summary

    def _make_full_summary(self, attribute='SouRecDist'):
        summary_list = []
        for scenario in self.scenarios:
            summary = self._make_summary(scenario, attribute)
            summary_list.append(summary)
        full_summary = pd.concat(summary_list).reset_index(level='Scenario')
        return full_summary

    def plot_summary_stylish(self, attrib):
        cm = sns.light_palette("red", as_cmap=True)
        s = self.summaries[attrib].style.set_caption(self.plot_params[attrib][0]).background_gradient(cmap=cm).hide_index().hide_columns(['count'])
        return s

    def plot_count_stylish(self):
        cm = sns.light_palette("green", as_cmap=True)
        s = self.summaries['SouRecDist'][['Scenario','count']].style.set_caption('Number of Traces').background_gradient(cmap=cm).hide_index()
        return s

    def offsetsplit(self, channo, chanperstrm):
        '''
        Split offsets in nears and fars. First argument is where the split is
        done (f.ex. channel 100), second argument is the total number of channels
        in one streamer, (f.ex. 480).
        This function returns the same dataframe, but with an additional column
        called 'Channel Range'
        '''

        self.df['Channel Range'] = self.df['BasRecvNo'].apply(
            lambda x: '< '+str(channo) if ((x % chanperstrm < channo) & (x % chanperstrm > 0)) 
            else '> '+str(channo))

    def plot_dist(self, attrib, plttype='violin', maxval=None):
        title = self.plot_params[attrib][0]
        if not maxval:
            maxval = self.plot_params[attrib][1]
        sns.set_style("darkgrid")
        sns.set_context('poster')
        plt.figure(figsize=(20,16))
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

    
