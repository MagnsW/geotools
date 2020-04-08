import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from GStools.input_tools import make_df_from_columndata

class IModJob:
    def __init__(self):
        self.df_vels = pd.DataFrame()
        self.df_plm_model = pd.DataFrame()
        self.df_plm_mutes = pd.DataFrame()
        self.df_targets = pd.DataFrame()
        self.df_vessel = pd.DataFrame()
        self.offset_classes = {}
        self.max_depth = None
        self.max_vel = None
        self.airvel = 330.0
        self.offsetgroups = {}
        print('Initialization complete!')

    def read_col_data(self, filename):
        self.df_vels = make_df_from_columndata(filename).reset_index().drop(columns=['numRows'])
        self.df_vels.columns = ['Depth', 'Vp']
        self.df_vels = self.df_vels[['Depth', 'Vp']]

    def make_model(self, waterdepth, watervel=None, step=None, intervals=None, max_depth=None, Qp=200.0):
        if not max_depth:
            max_depth = self.df_vels['Depth'].max()
        self.max_depth = max_depth
        if not watervel:
            watervel = self.df_vels['Vp'].min()
        print('Water velocity set to: ' + str(watervel))
        print('Max depth is: ' + str(max_depth))
        if not step:
            step = 500
        if not intervals:
            intervals = list(np.arange(waterdepth, max_depth + step, step))
        print('The depths defined in the model are: ' + str(intervals))
        intfno = list(range(1, len(intervals)+1))
        depths = [0, waterdepth]
        avg_vels = [self.airvel, watervel]
        densities = [0.01, 1.00]
        Qps = [100000, 10000]
        Qss = [100000, 10000]
        for i in range(len(intervals)-1):            
            avg_vel = self.df_vels[(self.df_vels['Depth'] >= intervals[i]) & (self.df_vels['Depth'] <= intervals[i+1])]['Vp'].mean().round(1)
            #print(i, intervals[i], intervals[i+1], avg_vel)
            density = np.around(0.309588 * avg_vel**0.25, 2)
            depths.append(intervals[i+1])
            avg_vels.append(avg_vel)
            densities.append(density)
            Qps.append(np.around(Qp, 1))
            Qss.append(np.around(3*Qp/4, 1))
        Vs = np.around([(vp/(3**0.5) if vp > watervel else 0) for vp in avg_vels], 1)
        self.df_plm_model = pd.DataFrame(list(zip(intfno, depths, avg_vels, Vs, densities, Qps, Qss)), columns=['Intf', 'Depth', 'Vp', 'Vs', 'Density', 'Qp', 'Qs'])
        self.max_vel = self.df_vels[self.df_vels['Depth'] < max_depth]['Vp'].max()

    def plot_vels(self):
        plt.figure(figsize=(12, 16))
        sns.set_style('whitegrid')
        sns.regplot(x=self.df_vels['Vp'], y=self.df_vels['Depth'], fit_reg=False)
        sns.lineplot(x=self.df_plm_model['Vp'], y=self.df_plm_model['Depth'], drawstyle='steps-post')
        #plt.gca().invert_yaxis()
        plt.ylim(self.max_depth, -100)
        plt.xlim(self.airvel, self.max_vel+200)
        plt.show()

    def compute_mute_offset(self, percmute):
        model = self.df_plm_model
        model['Thickness'] = model.diff()[['Depth']]
        model['dOWT'] = model['Thickness'] / model['Vp']
        model['OWT'] = model.cumsum()['dOWT']
        model['TWT'] = model['OWT']*2        
        Smax = 1 + percmute/100
        muteangle = np.arccos(1/Smax)
        muteangledeg = np.round(np.degrees(muteangle))
        print('Mute angle is: ' + str(muteangledeg))
        model['Mute angle - rad'] = muteangle
        model['Mute angle - deg'] = muteangledeg
        model['Constant'] = np.sin(muteangle) / model['Vp']
        intf_list = list(model['Intf'])[1:]
        vp_list = list(model['Vp'])[1:]
        depth_list = list(model['Thickness'])[1:]
        constant = list(model['Constant'])[1:]
        twt_list = list(model['TWT'])[1:]
        mute_onset = [np.nan]
        for i in range(len(intf_list)):
            const = constant[i]
            #print('i is: ' + str(i))
            #print('Target is: ' + str(intf_list[i]))
            #print('TWT is: ' + str(twt_list[i]))
            x = []
            for j, intf in enumerate(intf_list[:i]):        
                #print(i, j, intf, vp_list[j])
                alpha = np.arcsin(const * vp_list[j])
                x.append(depth_list[j] * np.tan(alpha))
                #print('alpha = ' + str(np.degrees(alpha)))
            x.append(depth_list[i] * np.tan(muteangle))
            #print('x = ' + str(x))
            x_sum = sum(x) * 2
            #print('x_sum is: ' + str(x_sum) + ' trace no: ' + str(x_sum/12.5))
            #print('----')
            mute_onset.append(x_sum)
        coltitle = str(percmute) + '% /' + str(muteangledeg)  + ' deg mute offset'
        model[coltitle] = mute_onset
        
        self.df_plm_mutes = model[['Intf', 'Depth', 'Thickness', 'TWT', coltitle]]

    def makeoffsetclass(self, start, end, number=4):
        offsetclasses = list(np.round(np.linspace(start, end, num=number + 1), 0))
        #print(offsetclasses)
        #print(len(offsetclasses))
        offset = {}
        for i in range(len(offsetclasses)-1):
            #print(offsetclasses[i], offsetclasses[i+1])
            offsetuple = (offsetclasses[i], offsetclasses[i+1])
            offset['Group' + str(i+1)] = offsetuple
        print(offset)
        self.offsetgroups = offset







