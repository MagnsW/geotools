import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from GStools.input_tools import make_df_from_columndata
from GStools import velconvert

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
        self.muteperc = None
        self.offsetgroups = {}
        self.targets = {}
        self.vessel = {}
        self.project_name = None
        print('Initialization complete!')

    def read_col_data(self, filename):
        self.df_vels = make_df_from_columndata(filename).reset_index().drop(columns=['numRows'])
        self.df_vels.columns = ['Depth', 'Vp']
        self.df_vels = self.df_vels[['Depth', 'Vp']]

    def velconvert_twt_avg_to_depth_int(self, timeunit='ms'):
        self.df_vels = velconvert.twt_avg_to_depth_int(self.df_vels, timeunit=timeunit)
        self.df_vels = self.df_vels.rename(columns={'Vint': 'Vp'}, errors='raise')

    def velconvert_depth_avg_to_depth_int(self):
        self.df_vels = velconvert.depth_avg_to_depth_int(self.df_vels)
        self.df_vels = self.df_vels.rename(columns={'Vint': 'Vp'}, errors='raise')

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

    def plot_vels(self, rotate=True):
        plt.figure(figsize=(12, 16))
        sns.set_style('whitegrid')
        
        if rotate:
            sns.regplot(x=self.df_vels['Vp'], y=self.df_vels['Depth'], fit_reg=False)
            sns.lineplot(x=self.df_plm_model['Vp'], y=self.df_plm_model['Depth'], drawstyle='steps-post')
            plt.ylim(self.max_depth, -100)
            plt.xlim(self.airvel, self.max_vel+200)
        else:
            sns.regplot(y=self.df_vels['Vp'], x=self.df_vels['Depth'], fit_reg=False)
            sns.lineplot(y=self.df_plm_model['Vp'], x=self.df_plm_model['Depth'], drawstyle='steps-pre')
            plt.xlim(-100, self.max_depth,)
            plt.ylim(self.airvel, self.max_vel+200)
        
        plt.show()

    def compute_mute_offset(self, percmute):
        self.muteperc = percmute
        model = self.df_plm_model.copy(deep=True)
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

    def set_targets(self, perc_populated=50):
        offset_col_name = self.df_plm_mutes.columns[-1]
        print('Offset column name: ' + offset_col_name)
        for key in self.offsetgroups:
            print('Assigning target to Group: ' + key)
            group_start = self.offsetgroups[key][0]
            group_end = self.offsetgroups[key][1]
            print(group_start, group_end)
            req_offset = group_start + (group_end - group_start) * perc_populated/100
            print('Required Offset: ' + str(req_offset))
            df_temp = self.df_plm_mutes[self.df_plm_mutes[offset_col_name] >= req_offset]
            print('Target identified: ' + str(df_temp.head(1)['Intf'].values[0]))
            self.targets[key] = df_temp.head(1)['Intf'].values[0]
        print(self.targets)

    def make_vessel(self, name, no_sources, no_strm, subline_sep, spi, no_chan, first_rec_x, source_name):
        req_rec_length = self.df_plm_mutes[self.df_plm_mutes['Intf'] == list(self.targets.values())[-1]]['TWT'].values[0]
        print('Deepest target TWT(s) is: ' + str(req_rec_length))
        rec_length = int(np.ceil(req_rec_length + 2) * 1000)
        print('Record length set to: ' + str(rec_length))
        self.vessel = {
            "Name": name,
            "VesselSampleInterval": 2,
            "VesselRecordingLength": rec_length, 
            "VesselNumberOfSources" : no_sources, 
            "VesselNumberOfStreamers": no_strm,
            "VesselSubSurfaceLineSep": subline_sep,
            "VesselShotPointDistance": spi,
            "VesselStreamerDepth": 8,
            "VesselNumberOfGroupsPerStreamer": no_chan,
            "VesselDefaultStreamerX": first_rec_x,
            "VesselGroupInterval": 12.5,
            "VesselSingleNotionalSource": source_name}




    def _make_intro_string(self):
        string_intro = f"""<!DOCTYPE PGS_N2_JOB>
<Pages ParentID="workspace" ID="root">
<Page Enabled="yes" ID="root" Expanded="yes">
<Parameter ID="GlobalProject">
<Entity name="Project">
<Key name="name" state="default">{self.project_name}</Key>
</Entity>
</Parameter>
<Parameter ID="GlobalOverwrite" state="default">Yes</Parameter>
<Page Enabled="yes" ID="DataManagerRoot" Expanded="yes">
<Parameter ID="DataMgrDefaultProject">
<Entity name="Project">
<Key name="name" state="default">{self.project_name}</Key>
</Entity>
</Parameter>
<Page Enabled="yes" ID="DataMgrModel" Expanded="yes">
<Parameter ID="PlaneLayerModelSpec">
<Entity name="Project">
<Key name="name" state="default">{self.project_name}</Key>
<Entity name="PlaneLayerModel">
<Key name="name" state="default">*</Key>
</Entity>
</Entity>
</Parameter>
"""
        return string_intro

    def _make_model_string(self):
        def to_xml(df, item='item', field='Parameter'):
            def row_to_xml(row):
                xml = ['<ParameterGroup ID="'+item+'">']
                for i, col_name in enumerate(row.index):
                    xml.append('  <Parameter ID="{0}" state="changed">{1}</Parameter>'.format(col_name, row.iloc[i]))
                xml.append('</ParameterGroup>')
                return '\n'.join(xml)
            res = '\n'.join(df.apply(row_to_xml, axis=1))
            return res
        
        df_model_interfaces = self.df_plm_model[['Intf', 'Depth']]
        df_model_interfaces = df_model_interfaces.rename(columns={'Intf': "ModelRowLabel", 'Depth': "ModelInterfaceMidPointZ"})
        df_model_interfaces["ModelInterfaceMidPointX"] = 0
        df_model_interfaces["ModelInterfaceMidPointY"] = 0
        df_model_interfaces["ModelInterfaceDip"] = 0
        df_model_interfaces["ModelInterfaceAzim"] = 0
        df_model_interfaces = df_model_interfaces[['ModelRowLabel', 'ModelInterfaceMidPointX', 'ModelInterfaceMidPointY', 'ModelInterfaceDip', 'ModelInterfaceAzim']]

        
        xml_model_interfaces = to_xml(df_model_interfaces, item='ModelInterface')

        return xml_model_interfaces

    def generate_job(self, project_name):
        self.project_name = project_name
        string1 = self._make_intro_string()
        string2 = self._make_model_string()
        full_string = string2
        #_make_model()
        #_make_vessel()
        #_make_modeling()
        print(full_string)

















