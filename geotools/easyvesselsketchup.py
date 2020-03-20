import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from ipywidgets import interactive

class VesselSetup:
    '''
    Instantiate you vessel using:
    my_vessel = easyvesselsketchup.VesselSetup(no of sources, no of streamers, streamer separation)
    Example:
    my_vessel = easyvesselsketchup.VesselSetup(4, 10, 75)
    Maximum number of sources is 6
    Then plot interactively using the interactive plot method.
    my_vessel.interactive_plot(gridspacing, source position slider increment, gridshift=True/False)

    '''

    def __init__(self, no_sources, no_streamers, strm_sep):
        if no_sources > 6:
            no_sources = 6
            print("Number of sources above max. Set to 6")
        self.no_sources = int(no_sources)
        self.no_streamers = int(no_streamers)
        self.strm_sep = strm_sep
        self.outerstrmpos = (no_streamers-1)*strm_sep/2
        self.recs = np.arange(-self.outerstrmpos, self.outerstrmpos+strm_sep, strm_sep)
        
    def interactive_plot(self, gridspace, slider_inc, gridshift=True):
        recs = self.recs
        no_sources = self.no_sources
        if gridshift:
            gridspaceshift = gridspace/2.0
        else:
            gridspaceshift = 0
        grid = np.arange(-self.outerstrmpos-gridspaceshift, self.outerstrmpos+self.strm_sep-gridspaceshift, gridspace)

        def f(source_1, source_2, source_3, source_4, source_5, source_6):
            sopos = [source_1, source_2, source_3, source_4, source_5, source_6][:no_sources]
            midpt = []
            for source in sopos:
              for rec in recs:
                midpt.append((source + rec)/2)
            sns.set_style("whitegrid")
            f, ax = plt.subplots(figsize=(24,2)) 
            plt.setp(ax,xticks=grid)
            sns.scatterplot(x=sopos, y=1, marker='*', s=100, label='Sources', color='blue')
            sns.scatterplot(x=recs, y=0.75, marker='v', s=100, label='Receivers', color='orange')
            sns.scatterplot(x=midpt, y=0, label='Midpoints', alpha=0.5, color='green')
            cur_axes = plt.gca()
            cur_axes.axes.get_xaxis().set_ticklabels([])
            cur_axes.axes.get_yaxis().set_ticklabels([])
            plt.legend()
            plt.show()
        interactive_plot = interactive(f, source_1=(-self.outerstrmpos, self.outerstrmpos, slider_inc), 
                                          source_2=(-self.outerstrmpos, self.outerstrmpos, slider_inc),
                                          source_3=(-self.outerstrmpos, self.outerstrmpos, slider_inc),
                                          source_4=(-self.outerstrmpos, self.outerstrmpos, slider_inc),
                                          source_5=(-self.outerstrmpos, self.outerstrmpos, slider_inc),
                                          source_6=(-self.outerstrmpos, self.outerstrmpos, slider_inc))
        return interactive_plot