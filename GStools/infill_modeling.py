import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from GStools.input_tools import make_df_from_columndata, make_df_from_segy
from GStools import velconvert

class IModJob:
    '''The IModJob class creates an Infill Modeling Job object. 
    Step 1: To initialize, use:

    my_imod_job = infill_modeling.IModJob()
    
    Step 2: Create velocity function: 
    Next step is to read in the 1D velocity function. Depending on the format
    of the input, the following commands should work:

    Nucleus column data where columns are TWT/depth and velocity:
    my_imod_job.read_col_data(filename.A1X)

    Nucleus column data where columns are depth and TWT :
    my_imod_job.read_col_data_depth_twt(depth_twt_filename)
    Note that this will create average velocites which will need to be converted
    to interval velocities using the method velconvert_depth_avg_to_depth_int
    described later.

    Segy:
    my_imod_job.read_segy(segy_filename.sgy)

    After being read in, the velocities can be printed by:
    In Jupyter:
    my_imod_job.df_vels 
    or outside Jupyter: 
    print(my_imod_job.df_vels)

    For a graph: 
    my_imod_job.df_vels.plot(x='Depth', y='Vp')

    For velocity conversions, the following methods may be used:
    my_imod_job.velconvert_depth_avg_to_depth_int()
    my_imod_job.velconvert_twt_avg_to_depth_int()

    Also more samples may be added with the method:
    my_imod_job.add_depth_samples()
    This should ensure that there are enough samples between the PLM interfaces
    so that average properties may be computed for each layer. This method can
    also increase the depth of the velocity function by flooding. 
    (maxdepth argument)

    Step 3: Create the PLM:
    my_imod_job.make_model(waterdepth=285, max_depth=4000, watervel=1550, Qp=120)
    If watervelocity is not stated, the velocity from the velocity function:
    my_imod_job.df_vels will be used. 

    For a QC plot of the velocity samples and the PLM velocities:
    my_imod_job.plot_vels()

    To view the PLM:
    my_imod_job.df_plm_model

    Step 4: Define the mute.
    For 100% stretch mute (or 60 degree angle mute):

    my_imod_job.compute_mute_offset(100)

    To list the mute offsets on each interface in the PLM:
    my_imod_job.df_plm_mutes

    Step 5: Define offset classes:
    my_imod_job.makeoffsetclass(100, 4100, 4)

    Step 6: Set targets for each offset class:
    my_imod_job.set_targets(50)
    The 50 in this examples means that each offset group will have 50% unmuted
    traces for the target selected

    Step 7: Define vessel:
    
    my_imod_job.make_vessel(name='TestVessel', no_sources=3, no_strm=12, subline_sep=18.75, spi=12.5, no_chan=640, first_rec_x=250, source_name='3280T__060_2000_080')
    
    Inspect vessel: print(my_imod_job.vessel)

    Step 8: Make the specs:
    
    my_imod_job.make_specs()

    Inspect specs: print(my_imod_job.specs)

    Step 9: Create Nucleus job: 
    my_imod_job.generate_job(project_path, project_name, filename)

    with for example:
    project_path = '/lus/ossi001/GeoSupportNV/' (path to nucleus project)
    project_name = '2020_04_Auto_Infill' (Nucleus project name)
    filename= 'Infill_job.Top Page.Workspace.J1X' (file name of Nucleus job file)

    The job will also print to terminal and may be copy-pasted into an editor 
    if that is more conveinient. If no filename is given, only the terminal 
    output will be produced





    '''


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
        self.project_path = None
        self.plm_name = 'PLM'
        self.reflection_times = {}
        self.specs = {}
        print('Initialization complete!')

    def read_col_data(self, filename):
        self.df_vels = make_df_from_columndata(filename).reset_index().drop(columns=['numRows'])
        self.df_vels.columns = ['Depth', 'Vp']
        self.df_vels = self.df_vels[['Depth', 'Vp']]

    def read_col_data_depth_twt(self, filename, timeunit='ms'):
        if timeunit == 's':
            c = 1
        elif timeunit == 'ms':
            c = 1000
        else:
            print('Wrong time unit given. Defaulting to miliseconds')
            c = 1000
        df_temp = make_df_from_columndata(filename).reset_index().drop(columns=['numRows'])
        df_temp.columns = ['Depth', 'TWT']
        df_temp['Vp'] = 2 * c * df_temp['Depth'] / df_temp['TWT']
        print(df_temp)
        df_temp = df_temp.interpolate(limit_direction='backward')
        print(df_temp)
        self.df_vels = df_temp[['Depth', 'Vp']]
        print('Velocity calculated is average, use method "velconvert_depth_avg_to_depth_int" before model building')

    def read_segy(self, filename):
        self.df_vels = make_df_from_segy(filename)
        self.df_vels.columns = ['Depth', 'Vp']
        self.df_vels = self.df_vels[['Depth', 'Vp']]
        self.df_vels = self.df_vels.interpolate(limit_direction='backward')


    def velconvert_twt_avg_to_depth_int(self, timeunit='ms'):
        self.df_vels = velconvert.twt_avg_to_depth_int(self.df_vels, timeunit=timeunit)
        self.df_vels = self.df_vels.rename(columns={'Vint': 'Vp'}, errors='raise')
        self.df_vels = self.df_vels.interpolate(limit_direction='backward')

    def velconvert_depth_avg_to_depth_int(self):
        self.df_vels = velconvert.depth_avg_to_depth_int(self.df_vels)
        self.df_vels = self.df_vels.rename(columns={'Vint': 'Vp'}, errors='raise')
        self.df_vels = self.df_vels.interpolate(limit_direction='backward')

    def add_depth_samples(self, maxdepth=None, inc=50):
        if maxdepth:
            max_depth = maxdepth
        else:
            max_depth = self.df_vels['Depth'].max()
        depth_resampled = np.arange(0, max_depth, inc)
        df_temp = pd.DataFrame(depth_resampled, columns=['Depth'])
        df_temp['Vp'] = np.nan
        self.df_vels = self.df_vels.append(df_temp)
        self.df_vels = self.df_vels.drop_duplicates(subset=['Depth'], keep='first')
        self.df_vels = self.df_vels.sort_values(by=['Depth'])
        self.df_vels = self.df_vels.interpolate(limit_direction='both')

    def make_model(self, waterdepth, watervel=None, step=None, intervals=None, max_depth=None, Qp=200.0, name=None):
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
        if name:
            self.plm_name = name
        intfno = list(range(1, len(intervals)+1))
        depths = [0, waterdepth]
        avg_vels = [self.airvel, watervel]
        densities = [0.01, 1.00]
        Qps = [100000, 10000]
        Qss = [100000, 10000]
        for i in range(len(intervals)-1):
            avg_vel = np.around(self.df_vels[(self.df_vels['Depth'] >= intervals[i]) & (self.df_vels['Depth'] <= intervals[i+1])]['Vp'].mean())
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
            sns.regplot(x=self.df_vels['Vp'], y=self.df_vels['Depth'], fit_reg=False, label='Input Velocities')
            sns.lineplot(x=self.df_plm_model['Vp'], y=self.df_plm_model['Depth'], drawstyle='steps-post', label='Model Velocities')
            plt.ylim(self.max_depth, -100)
            plt.xlim(self.airvel, self.max_vel+200)
        else:
            sns.regplot(y=self.df_vels['Vp'], x=self.df_vels['Depth'], fit_reg=False, label='Input Velocities')
            sns.lineplot(y=self.df_plm_model['Vp'], x=self.df_plm_model['Depth'], drawstyle='steps-pre', label='Model Velocities')
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
            self.reflection_times[key] = df_temp.head(1)['TWT'].values[0]
        print(self.targets)

    def make_vessel(self, name, no_sources, no_strm, subline_sep, spi, no_chan, first_rec_x, source_name):
        source_depth = int(source_name[7:10]) /10
        strm_depth = source_depth + 1
        print(f'Streamer depth set to: {source_depth}m, streamer depth will be set to {strm_depth}m')
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
            "VesselStreamerDepth": strm_depth,
            "VesselNumberOfGroupsPerStreamer": no_chan,
            "VesselDefaultStreamerX": first_rec_x,
            "VesselGroupInterval": 12.5,
            "VesselSingleNotionalSource": source_name}

    def make_specs(self):
        #df_specs = pd.DataFrame(columns=('Group', 'amplow', 'amphigh', 'timelow', 'timehigh'))
        lasti = len(self.offsetgroups)-1
        self.specs = {}
        for i, group in enumerate(self.offsetgroups):
            if i == 0:
                self.specs[group] = {'amplow': 1.0, 'amphigh': 2.0, 'timelow': 0.5, 'timehigh': 1.0}
            elif i == lasti:
                self.specs[group] = {'amplow': 2.0, 'amphigh': 4.0, 'timelow': 1.0, 'timehigh': 2.0}
            else:
                self.specs[group] = {'amplow': 1.5, 'amphigh': 3.0, 'timelow': 0.75, 'timehigh': 1.5}


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
</Parameter>"""
        return string_intro

    def _make_model_string(self):
        def to_xml(df, item='item', field='Parameter'):
            def row_to_xml(row):
                xml = ['<ParameterGroup ID="'+item+'">']
                for i, col_name in enumerate(row.index):
                    value = row.iloc[i]
                    if i == 0:
                        value = int(value)
                    xml.append('<Parameter ID="{0}" state="changed">{1}</Parameter>'.format(col_name, value))
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
        df_model_interfaces = df_model_interfaces[['ModelRowLabel', 'ModelInterfaceMidPointX', 'ModelInterfaceMidPointY', 'ModelInterfaceMidPointZ', 'ModelInterfaceDip', 'ModelInterfaceAzim']]
        df_model_interfaces = df_model_interfaces[:-1] #Dropping last interface

        xml_model_interfaces = to_xml(df_model_interfaces, item='ModelInterface')

        df_model_layers = self.df_plm_model[['Intf', 'Vp', 'Vs', 'Density', 'Qp', 'Qs']]
        df_model_layers = df_model_layers.rename(columns={'Intf': "ModelRowLabel", 'Vp': "ModelLayerVp", 'Vs': "ModelLayerVs", 'Density': "ModelLayerDensity", 'Qp': "ModelLayerQp", 'Qs': "ModelLayerQs"})
        #print(df_model_interfaces.dtypes)
        #print(df_model_layers.dtypes)
        xml_model_layers = to_xml(df_model_layers, item='ModelLayer')

        xml_intro_string = f"""
<Page Enabled="yes" ID="PlaneLayerModelEdit" Expanded="yes">
<Parameter ID="PlaneLayerModelSpec">
<Entity name="Project">
<Key name="name" state="default">{self.project_name}</Key>
<Entity name="PlaneLayerModel">
<Key name="name" state="changed">{self.plm_name}</Key>
</Entity>
</Entity>
</Parameter>
<Parameter ID="ModelSeparator" state="changed"></Parameter>
<ParameterGroup ID="ModelOrigo">
<Parameter ID="ModelOrigoEast" state="default">0</Parameter>
<Parameter ID="ModelOrigoNorth" state="default">0</Parameter>
</ParameterGroup>
<ParameterGroup ID="ModelSize">
<Parameter ID="ModelSizeX" state="changed">20000</Parameter>
<Parameter ID="ModelSizeY" state="changed">20000</Parameter>
<Parameter ID="ModelSizeZ" state="changed">{self.df_plm_model['Depth'].max()}</Parameter>
</ParameterGroup>
<Parameter ID="ModelRotation" state="default">0</Parameter>
<Parameter ID="ModelSeparator" state="changed"></Parameter>
<Parameter ID="ModelNumberOfLayers" state="changed">{len(self.df_plm_model)-1}</Parameter>
<Parameter ID="ModelNumberOfDiffractors" state="default">0</Parameter>
<Parameter ID="ModelSeparator" state="changed"></Parameter>
<Parameter ID="ModelLayerUpdateOption" state="default">No</Parameter>
<Page Enabled="yes" ID="ModelInterfaces" Expanded="yes">
"""
        xml_mid_string = """
</Page>
<Page Enabled="yes" ID="ModelLayers" Expanded="yes">
"""     
        xml_end_string = """
</Page>
</Page>
</Page>"""

        entire_string = xml_intro_string + xml_model_interfaces + xml_mid_string + xml_model_layers + xml_end_string

        return entire_string

    def _make_vessel_string(self):
        string = f"""
<Page Enabled="yes" ID="DataMgrVessel" Expanded="yes">
<Parameter ID="VesselSpec">
<Entity name="Project">
<Key name="name" state="default">{self.project_name}</Key>
<Entity name="Vessel">
<Key name="name" state="default">*</Key>
</Entity>
</Entity>
</Parameter>
<Page Enabled="yes" ID="VesselEdit" Expanded="yes">
<Parameter ID="VesselSpec">
<Entity name="Project">
<Key name="name" state="default">{self.project_name}</Key>
<Entity name="Vessel">
<Key name="name" state="changed">{self.vessel['Name']}</Key>
</Entity>
</Entity>
</Parameter>
<Parameter ID="VesselSampleInterval" state="default">{self.vessel['Name']}</Parameter>
<Parameter ID="VesselRecordingLength" state="changed">{self.vessel['VesselRecordingLength']}</Parameter>
<Parameter ID="VesselSourceSeparator" state="changed"></Parameter>
<Parameter ID="VesselSourceModus" state="default">Standard source configuration</Parameter>
<Parameter ID="VesselSourceType" state="default">Notional sources</Parameter>
<Parameter ID="VesselNumberOfSources" state="changed">{self.vessel['VesselNumberOfSources']}</Parameter>
<Parameter ID="VesselGeometrySeparator" state="changed"></Parameter>
<Parameter ID="VesselNumberOfStreamers" state="changed">{self.vessel['VesselNumberOfStreamers']}</Parameter>
<Parameter ID="VesselSubSurfaceLineSep" state="changed">{self.vessel['VesselSubSurfaceLineSep']}</Parameter>
<ParameterGroup ID="VesselSailLineSepGroup">
<Parameter ID="VesselSailLineOption" state="default">No</Parameter>
<Parameter ID="VesselSailLineSep" state="default">600</Parameter>
</ParameterGroup>
<Parameter ID="VesselRecalculateConfiguration" state="default">No</Parameter>
<Parameter ID="VesselShotPointDistance" state="changed">{self.vessel['VesselShotPointDistance']}</Parameter>
<Parameter ID="VesselSecondsPerDegree" state="default">30</Parameter>
<Parameter ID="VesselStreamerSeparator" state="changed"></Parameter>
<Parameter ID="VesselStreamerModus" state="default">Standard streamer configuration</Parameter>
<Parameter ID="VesselStreamerType" state="default">Conventional</Parameter>
<Parameter ID="VesselStreamerDepth" state="default">{self.vessel['VesselStreamerDepth']}</Parameter>
<Parameter ID="VesselNumberOfGroupsPerStreamer" state="changed">{self.vessel['VesselNumberOfGroupsPerStreamer']}</Parameter>
<Parameter ID="VesselDefaultStreamerX" state="changed">{self.vessel['VesselDefaultStreamerX']}</Parameter>
<Parameter ID="VesselGroupInterval" state="default">{self.vessel['VesselGroupInterval']}</Parameter>
<Parameter ID="VesselFeatherType" state="default">None</Parameter>
<Parameter ID="VesselCalcStackFold" state="default">120</Parameter>
<Parameter ID="VesselReceiverarraySeparator" state="changed"></Parameter>
<ParameterGroup ID="VesselHydrophoneArrayGroup">
<Parameter ID="VesselHydrophoneArrayOption" state="default">Regular</Parameter>
<Parameter ID="VesselHydrophoneArray">
<Entity name="Project">
<Key name="name" state="default">{self.project_name}</Key>
<Entity name="ReceiverArray">
<Key name="name" state="default">*</Key>
</Entity>
</Entity>
</Parameter>
</ParameterGroup>
<Parameter ID="VesselNumberOfPhones" state="default">16</Parameter>
<Parameter ID="VesselGroupLength" state="default">12.5</Parameter>
<ParameterGroup ID="VesselGeophoneArrayGroup">
<Parameter ID="VesselGeophoneArrayOption" state="default">Regular</Parameter>
<Parameter ID="VesselGeophoneArray">
<Entity name="Project">
<Key name="name" state="default">{self.project_name}</Key>
<Entity name="ReceiverArray">
<Key name="name" state="default">*</Key>
</Entity>
</Entity>
</Parameter>
</ParameterGroup>
<Parameter ID="VesselNumberOfPhonesGeoStr" state="default">16</Parameter>
<Parameter ID="VesselGroupLengthGeoStr" state="default">12.5</Parameter>
<Page Enabled="yes" ID="VesselNotionalsPageS" Expanded="yes">
<Parameter ID="VesselSingleNotionalSource">
<Entity name="Project">
<Key name="name" state="default">{self.project_name}</Key>
<Entity name="Wavelet">
<Key name="name" state="changed">{self.vessel['VesselSingleNotionalSource']}</Key>
<Key name="type" state="changed">Notional Source Signature</Key>
</Entity>
</Entity>
</Parameter>
<Parameter ID="VesselNumberOfGathers" state="default">1</Parameter>
<Parameter ID="VesselNumberOfSubSources" state="default">1</Parameter>
<Parameter ID="VesselNotionalFilter" state="default">GeoStr LChyd_3/7-214/341</Parameter>
<ParameterGroup ID="VesselFilterFrequency">
<Parameter ID="VesselLowFrequency" state="default">3</Parameter>
<Parameter ID="VesselHighFrequency" state="default">214</Parameter>
</ParameterGroup>
<ParameterGroup ID="VesselFilterSlope">
<Parameter ID="VesselLowSlope" state="default">7</Parameter>
<Parameter ID="VesselHighSlope" state="default">341</Parameter>
</ParameterGroup>
<Parameter ID="VesselSourceDepthNot" state="default">6</Parameter>
</Page>
</Page>
</Page>
</Page>"""
        return string

    def _make_end_string(self):
        string = f"""
</Page>
</Pages>"""
        return string

    def _make_modeling_string(self):
        def make_one_block(group):
            trace_start = int(self.reflection_times[group] *1000) - 200
            trace_end = int(self.reflection_times[group] *1000) + 300
            string = f"""
<Page Enabled="yes" ID="PlaneLayerRoot" Expanded="yes">
<Parameter ID="GlobalProject">
<Entity name="Project">
<Key name="name" state="default">{self.project_name}</Key>
</Entity>
</Parameter>
<Page Enabled="yes" ID="Pkpstm_root" Expanded="yes">
<Parameter ID="Kpstm_RunMode" state="default">Generate and Migrate</Parameter>
<Parameter ID="Kpstm_AnalysisOption" state="changed">Infill analysis</Parameter>
<Parameter ID="Kpstm_MemorySeparator" state="changed"></Parameter>
<Parameter ID="Kpstm_MemoryOption" state="changed">Specify</Parameter>
<Parameter ID="Kpstm_MaxMemory" state="changed">4</Parameter>
<Parameter ID="Kpstm_ModelSeparator" state="changed"></Parameter>
<Parameter ID="Kpstm_Model">
<Entity name="Project">
<Key name="name" state="changed">{self.project_name}</Key>
<Entity name="PlaneLayerModel">
<Key name="name" state="changed">{self.plm_name}</Key>
</Entity>
</Entity>
</Parameter>
<ParameterGroup ID="Kpstm_Reflectors">
<Parameter ID="Kpstm_Reflectoroption" state="default">Select one interface</Parameter>
<Parameter ID="Kpstm_Reflectorinterface" state="changed">{self.targets[group]}</Parameter>
</ParameterGroup>
<Parameter ID="Kpstm_DiffPrimOption" state="default">No</Parameter>
<Parameter ID="Kpstm_DiffMultOption" state="default">No</Parameter>
<Parameter ID="Kpstm_VesselSeparator" state="changed"></Parameter>
<Parameter ID="Kpstm_Vessel">
<Entity name="Project">
<Key name="name" state="changed">{self.project_name}</Key>
<Entity name="Vessel">
<Key name="name" state="changed">{self.vessel['Name']}</Key>
</Entity>
</Entity>
</Parameter>
<Parameter ID="Kpstm_SurveySeparator" state="changed"></Parameter>
<Parameter ID="Kpstm_SurveyOption" state="default">Automatically Computed</Parameter>
<Parameter ID="Kpstm_Saillinedirection" state="default">90</Parameter>
<Parameter ID="Kpstm_SaillineCentre" state="default">Sailline Center</Parameter>
<Parameter ID="Kpstm_Saillineoption" state="default">Uniform sailline direction</Parameter>
<Parameter ID="Kpstm_NoBulkLines" state="default">10</Parameter>
<Parameter ID="Kpstm_BinSeparator" state="changed"></Parameter>
<ParameterGroup ID="Kpstm_Bincenter">
<Parameter ID="Kpstm_bincenter_x" state="default">10000</Parameter>
<Parameter ID="Kpstm_bincenter_y" state="default">10000</Parameter>
</ParameterGroup>
<ParameterGroup ID="Kpstm_Binnumbers">
<Parameter ID="Kpstm_binnumbers_in" state="default">1</Parameter>
<Parameter ID="Kpstm_binnumbers_cr" state="default">120</Parameter>
</ParameterGroup>
<ParameterGroup ID="Kpstm_Binsize">
<Parameter ID="Kpstm_binsize_in" state="default">6.25</Parameter>
<Parameter ID="Kpstm_binsize_cr" state="changed">{self.vessel['VesselSubSurfaceLineSep']}</Parameter>
</ParameterGroup>
<Parameter ID="Kpstm_ModellingSeparator" state="changed"></Parameter>
<Parameter ID="Kpstm_GeomSpreading" state="default">Yes</Parameter>
<Parameter ID="Kpstm_Reflection" state="default">Yes</Parameter>
<Parameter ID="Kpstm_Absorption" state="default">Yes</Parameter>
<Parameter ID="Kpstm_SRdirectivity" state="default">Yes</Parameter>
<Parameter ID="Kpstm_ReceiverGhostOption" state="changed">Standard calculations</Parameter>
<Parameter ID="Kpstm_SourceGhostOption" state="default">Yes</Parameter>
<ParameterGroup ID="Kpstm_WhiteNoise">
<Parameter ID="Kpstm_white_noise" state="default">No</Parameter>
<Parameter ID="Kpstm_whitenoiselevel" state="default">10</Parameter>
</ParameterGroup>
<ParameterGroup ID="Kpstm_WeatherNoise">
<Parameter ID="Kpstm_weather_noise" state="default">No</Parameter>
<Parameter ID="Kpstm_weathernoiselevel" state="default">10</Parameter>
</ParameterGroup>
<ParameterGroup ID="Kpstm_BarnacleNoise">
<Parameter ID="Kpstm_Barnacle_noise" state="default">No</Parameter>
<Parameter ID="Kpstm_Barnaclenoiselevel" state="default">10</Parameter>
</ParameterGroup>
<Parameter ID="Kpstm_AngleMuteSeparator" state="changed"></Parameter>
<Parameter ID="Kpstm_AngleMuteOption" state="default">No</Parameter>
<Parameter ID="Kpstm_MigrationSeparator" state="changed"></Parameter>
<Parameter ID="Kpstm_Redatum" state="default">Yes</Parameter>
<ParameterGroup ID="Kpstm_Tracetime">
<Parameter ID="Kpstm_traceout_start" state="changed">{trace_start}</Parameter>
<Parameter ID="Kpstm_traceout_end" state="changed">{trace_end}</Parameter>
</ParameterGroup>
<ParameterGroup ID="Kpstm_Mute">
<Parameter ID="Kpstm_Muteoption" state="changed">Stretch mute</Parameter>
<Parameter ID="Kpstm_Stretchmute" state="changed">{self.muteperc}</Parameter>
</ParameterGroup>
<Parameter ID="Kpstm_NoOfOffsetMutePairs" state="default">16</Parameter>
<ParameterGroup ID="Kpstm_Taper">
<Parameter ID="Kpstm_Taperoption" state="default">Yes</Parameter>
<Parameter ID="Kpstm_Taperfactor" state="default">10</Parameter>
</ParameterGroup>
<Parameter ID="Kpstm_Hitcount" state="default">No</Parameter>
<Parameter ID="Kpstm_Phaseshift" state="default">3D</Parameter>
<ParameterGroup ID="Kpstm_Antialias">
<Parameter ID="Kpstm_antialias_in" state="default">6.25</Parameter>
<Parameter ID="Kpstm_antialias_cr" state="default">{self.vessel['VesselSubSurfaceLineSep']}</Parameter>
</ParameterGroup>
<Parameter ID="Kpstm_NoOfMigrApertPairs" state="default">4</Parameter>
<Parameter ID="Kpstm_InfillSeparator" state="changed"></Parameter>
<Parameter ID="Kpstm_OffsetOption" state="changed">No</Parameter>
<ParameterGroup ID="Kpstm_OffsetMigration">
<Parameter ID="Kpstm_offsetused_start" state="changed">{self.offsetgroups[group][0]}</Parameter>
<Parameter ID="Kpstm_offsetused_end" state="changed">{self.offsetgroups[group][1]}</Parameter>
</ParameterGroup>
<Parameter ID="Kpstm_FresnelOption" state="changed">Yes</Parameter>
<Parameter ID="Kpstm_infillHoleSize" state="default">24</Parameter>
<Parameter ID="Kpstm_infillHoleOption" state="default">Centered</Parameter>
<Parameter ID="Kpstm_infillHoleStart" state="default">49</Parameter>
<Parameter ID="Kpstm_FresnelColumnData">
<Entity name="Project">
<Key name="name" state="default">{self.project_name}</Key>
<Entity name="ColumnData">
<Key name="name" state="changed">{group}_FresnW</Key>
</Entity>
</Entity>
</Parameter>
<Parameter ID="Kpstm_OutputSeparator" state="changed"></Parameter>
<Parameter ID="Kpstm_Output_Mig">
<Entity name="Project">
<Key name="name" state="default">{self.project_name}</Key>
<Entity name="SyntheticData">
<Key name="name" state="changed">{group}</Key>
<Key name="type" state="default">Stacked</Key>
</Entity>
</Entity>
</Parameter>
<Page Enabled="yes" ID="Kpstm_MigrTimeOffsetPairs" Expanded="yes">
<ParameterGroup ID="Kpstm_migrtimeoffset">
<Parameter ID="Kpstm_MigrTime" state="default">266.666656494141</Parameter>
<Parameter ID="Kpstm_MigrOffset" state="default">115.470053678379</Parameter>
</ParameterGroup>
<ParameterGroup ID="Kpstm_migrtimeoffset">
<Parameter ID="Kpstm_MigrTime" state="default">851.462005615234</Parameter>
<Parameter ID="Kpstm_MigrOffset" state="default">404.145187874326</Parameter>
</ParameterGroup>
<ParameterGroup ID="Kpstm_migrtimeoffset">
<Parameter ID="Kpstm_MigrTime" state="default">1369.06033325195</Parameter>
<Parameter ID="Kpstm_MigrOffset" state="default">692.820322070273</Parameter>
</ParameterGroup>
<ParameterGroup ID="Kpstm_migrtimeoffset">
<Parameter ID="Kpstm_MigrTime" state="default">4955.65774536133</Parameter>
<Parameter ID="Kpstm_MigrOffset" state="default">2886.75134195947</Parameter>
</ParameterGroup>
</Page>
</Page>
<Page Enabled="yes" ID="InfillAnalysis_root" Expanded="yes">
<Parameter ID="Infill_InputName">
<Entity name="Project">
<Key name="name" state="default">{self.project_name}</Key>
<Entity name="SyntheticData">
<Key name="name" state="default">*</Key>
<Key name="type" state="default">Stacked</Key>
</Entity>
</Entity>
</Parameter>
<Parameter ID="Infill_DataSeparator" state="changed"></Parameter>
<Parameter ID="Infill_HoleOption" state="default">All holes</Parameter>
<Parameter ID="Infill_DirectionOption" state="default">Crossline</Parameter>
<Parameter ID="Infill_LineNumber" state="default">1</Parameter>
<Parameter ID="Infill_PickingSeparator" state="changed"></Parameter>
<Parameter ID="Infill_TimeshiftOption" state="default">No</Parameter>
<Parameter ID="Infill_TimeShiftMs" state="default">10</Parameter>
<Parameter ID="Infill_EventpickingOption" state="changed">Trough</Parameter>
<Parameter ID="Infill_AmplitudeInterpolation" state="default">Interpolated</Parameter>
<Page Enabled="yes" ID="InfillQC" Expanded="yes">
<Parameter ID="Infill_InputName">
<Entity name="Project">
<Key name="name" state="default">{self.project_name}</Key>
<Entity name="SyntheticData">
<Key name="name" state="changed">{group}</Key>
<Key name="type" state="changed">Stacked</Key>
</Entity>
</Entity>
</Parameter>
<Parameter ID="Infill_DataSeparator" state="changed"></Parameter>
<Parameter ID="Infill_HoleOption" state="default">All holes</Parameter>
<Parameter ID="Infill_DirectionOption" state="default">Crossline</Parameter>
<Parameter ID="Infill_LineNumber" state="default">1</Parameter>
<ParameterGroup ID="Infill_Coverage">
<Parameter ID="Infill_CoverageOpt" state="default">No</Parameter>
<Parameter ID="Infill_CoverageValue" state="default">30</Parameter>
</ParameterGroup>
<Parameter ID="Infill_PickingSeparator" state="changed"></Parameter>
<ParameterGroup ID="Infill_TraceWindowTime">
<Parameter ID="Infill_tracewindow_start" state="default">{trace_start}</Parameter>
<Parameter ID="Infill_tracewindow_end" state="default">{trace_end}</Parameter>
</ParameterGroup>
<Parameter ID="Infill_TimeshiftOption" state="default">No</Parameter>
<Parameter ID="Infill_TimeShiftMs" state="default">10</Parameter>
<ParameterGroup ID="Infill_PickingWindow">
<Parameter ID="Infill_pickwindow_start" state="default">{trace_start}</Parameter>
<Parameter ID="Infill_pickwindow_end" state="default">{trace_end}</Parameter>
</ParameterGroup>
<Parameter ID="Infill_EventpickingOption" state="changed">Trough</Parameter>
<Parameter ID="Infill_AmplitudeInterpolation" state="default">Interpolated</Parameter>
<Parameter ID="Infill_PlotSeparator" state="changed"></Parameter>
<Parameter ID="Infill_AmpRelDisplayOption" state="default">Yes</Parameter>
<Parameter ID="Infill_AmpAbsDisplayOption" state="changed">Yes</Parameter>
<Parameter ID="Infill_TraceDisplayOption" state="changed">Yes</Parameter>
<Parameter ID="Infill_AmplitudeAxisOption" state="default">Automatic</Parameter>
<ParameterGroup ID="Infill_AxisSpecifications">
<Parameter ID="Infill_axisminimum" state="default">-60</Parameter>
<Parameter ID="Infill_axismaximum" state="default">0</Parameter>
</ParameterGroup>
<Parameter ID="Infill_OutputSeparator" state="changed"></Parameter>
<Parameter ID="Infill_OutputPick" state="default">No</Parameter>
<Parameter ID="Infill_TimeAmplFile">
<Entity name="Project">
<Key name="name" state="default">{self.project_name}</Key>
<Entity name="ColumnData">
<Key name="name" state="default">*</Key>
</Entity>
</Entity>
</Parameter>
</Page>
<Page Enabled="yes" ID="InfillSpec" Expanded="yes">
<Parameter ID="Infill_InputName">
<Entity name="Project">
<Key name="name" state="default">{self.project_name}</Key>
<Entity name="SyntheticData">
<Key name="name" state="changed">{group}</Key>
<Key name="type" state="changed">Stacked</Key>
</Entity>
</Entity>
</Parameter>
<Parameter ID="Infill_MergeFresnel" state="default">Yes</Parameter>
<Parameter ID="Infill_InputFresnel">
<Entity name="Project">
<Key name="name" state="default">{self.project_name}</Key>
<Entity name="ColumnData">
<Key name="name" state="changed">{group}_FresnW</Key>
</Entity>
</Entity>
</Parameter>
<Parameter ID="Infill_DataSeparator" state="changed"></Parameter>
<Parameter ID="Infill_HoleOption" state="default">All holes</Parameter>
<Parameter ID="Infill_DirectionOption" state="default">Crossline</Parameter>
<Parameter ID="Infill_LineNumber" state="default">1</Parameter>
<Parameter ID="Infill_PickingSeparator" state="changed"></Parameter>
<ParameterGroup ID="Infill_TraceWindowTime">
<Parameter ID="Infill_tracewindow_start" state="default">{trace_start}</Parameter>
<Parameter ID="Infill_tracewindow_end" state="default">{trace_end}</Parameter>
</ParameterGroup>
<Parameter ID="Infill_TimeshiftOption" state="default">No</Parameter>
<Parameter ID="Infill_TimeShiftMs" state="default">10</Parameter>
<ParameterGroup ID="Infill_PickingWindow">
<Parameter ID="Infill_pickwindow_start" state="default">{trace_start}</Parameter>
<Parameter ID="Infill_pickwindow_end" state="default">{trace_end}</Parameter>
</ParameterGroup>
<Parameter ID="Infill_EventpickingOption" state="changed">Trough</Parameter>
<Parameter ID="Infill_AmplitudeInterpolation" state="default">Interpolated</Parameter>
<Parameter ID="Infill_TrafficLightSeparator" state="changed"></Parameter>
<Parameter ID="Infill_FrequencyThreshold" state="default">3</Parameter>
<Parameter ID="Infill_FrequencySpec" state="default">50</Parameter>
<ParameterGroup ID="Infill_AmplitudeThreshold">
<Parameter ID="Infill_AmplitudeGood" state="changed">{self.specs[group]['amplow']}</Parameter>
<Parameter ID="Infill_AmplitudeBad" state="changed">{self.specs[group]['amphigh']}</Parameter>
</ParameterGroup>
<ParameterGroup ID="Infill_TimeshiftThreshold">
<Parameter ID="Infill_TimeShiftGood" state="changed">{self.specs[group]['timelow']}</Parameter>
<Parameter ID="Infill_TimeShiftBad" state="changed">{self.specs[group]['timehigh']}</Parameter>
</ParameterGroup>
<Parameter ID="Infill_SpecifyContours" state="default">Automatic</Parameter>
<ParameterGroup ID="Infill_SpecifyTimeShift">
<Parameter ID="Infill_ContoursTimeShiftMin" state="default">1</Parameter>
<Parameter ID="Infill_ContoursTimeShiftMax" state="default">2</Parameter>
<Parameter ID="Infill_ContoursTimeShiftInc" state="default">0.5</Parameter>
</ParameterGroup>
<ParameterGroup ID="Infill_SpecifyAmplitude">
<Parameter ID="Infill_ContoursAmplitudeMin" state="default">1</Parameter>
<Parameter ID="Infill_ContoursAmplitudeMax" state="default">2</Parameter>
<Parameter ID="Infill_ContoursAmplitudeInc" state="default">0.5</Parameter>
</ParameterGroup>
<Parameter ID="Infill_AnalysisAreaSeparator" state="changed"></Parameter>
<ParameterGroup ID="Infill_CoverageSpec">
<Parameter ID="Infill_CoverageSpecMin" state="default">0</Parameter>
<Parameter ID="Infill_CoverageSpecMax" state="default">100</Parameter>
<Parameter ID="Infill_CoverageSpecInc" state="default">10</Parameter>
</ParameterGroup>
<ParameterGroup ID="Infill_HoleSizeSpec">
<Parameter ID="Infill_HoleSizeSpecMin" state="default">0</Parameter>
<Parameter ID="Infill_HoleSizeSpecMax" state="default">200</Parameter>
</ParameterGroup>
<Parameter ID="Infill_HoleSizeIncLog" state="changed">{self.vessel['VesselSubSurfaceLineSep']}</Parameter>
<Parameter ID="Infill_OutputSeparator" state="changed"></Parameter>
<Parameter ID="Infill_OutputLogFile" state="changed">No</Parameter>
<Parameter ID="Infill_OutputLogFileSpec" state="changed">{self.project_path}{self.project_name}/ExternalData/{group}_log</Parameter>
<Parameter ID="Infill_OutputColData" state="changed">Yes</Parameter>
<Parameter ID="Infill_OutputColDataSpec">
<Entity name="Project">
<Key name="name" state="default">{self.project_name}</Key>
<Entity name="ColumnData">
<Key name="name" state="changed">{group}</Key>
</Entity>
</Entity>
</Parameter>
</Page>
</Page>
</Page>"""
            return string            
        
        res = ''
        for group in self.offsetgroups:
            #res = '\n'.join(make_one_block(group))
            res += make_one_block(group)


        return res

    def generate_job(self, project_path, project_name, filename=None):
        self.project_name = project_name
        self.project_path = project_path
        string1 = self._make_intro_string()
        string2 = self._make_model_string()
        string3 = self._make_vessel_string()
        string4 = self._make_modeling_string()
        string_end = self._make_end_string()
        full_string = string1 + string2 + string3 + string4 + string_end
        #_make_model()
        #_make_vessel()
        #_make_modeling()
        print(full_string)
        if filename:
            with open(filename, 'w') as f:
                f.write(full_string)
                f.close()

















