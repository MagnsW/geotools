import pandas as pd 
import numpy as np 

def twt_avg_to_depth_int(vel_model, timeunit='ms', qc=False):
    if timeunit == 's':
        c = 1
    elif timeunit == 'ms':
        c = 1000
    else:
        print('Wrong time unit given. Defaulting to miliseconds')
        c = 1000
    time_col_name = vel_model.columns[0]
    vel_col_name = vel_model.columns[1]
    print('Time column name: ' + time_col_name + '; Velocity column name: ' + vel_col_name)
    vel_model['OWT'] = vel_model[time_col_name] / 2
    vel_model['dOWT'] = vel_model['OWT'].diff()
    vel_model['Depth'] = vel_model[time_col_name] * vel_model[vel_col_name] / (2*c)
    vel_model['dDepth'] = vel_model['Depth'].diff()
    vel_model['Vint'] = c * vel_model['dDepth'] / vel_model['dOWT']
    if qc:
        return vel_model
    else:
        return vel_model[['Depth', 'Vint']]

def twt_int_to_depth_int(vel_model, timeunit='ms', qc=False):
    if timeunit == 's':
        c = 1
    elif timeunit == 'ms':
        c = 1000
    else:
        print('Wrong time unit given. Defaulting to miliseconds')
        c = 1000
    time_col_name = vel_model.columns[0]
    vel_col_name = vel_model.columns[1]
    print('Time column name: ' + time_col_name + '; Velocity column name: ' + vel_col_name)
    vel_model['OWT'] = vel_model[time_col_name] / 2
    vel_model['dOWT'] = vel_model['OWT'].diff()
    vel_model['dDepth'] = vel_model[vel_col_name] * vel_model['dOWT'] / c
    vel_model['Depth'] = vel_model['dDepth'].cumsum()
    if qc:
        return vel_model
    else:
        return vel_model[['Depth', vel_col_name]]

def depth_avg_to_depth_int(vel_model, qc=False):
    depth_col_name = vel_model.columns[0]
    vel_col_name = vel_model.columns[1]
    print('Depth column name: ' + depth_col_name + '; Velocity column name: ' + vel_col_name)
    vel_model['dDepth'] = vel_model[depth_col_name].diff()
    vel_model['OWT'] = vel_model[depth_col_name] / vel_model[vel_col_name]
    vel_model['dOWT'] = vel_model['OWT'].diff()
    vel_model['Vint'] = vel_model['dDepth'] / vel_model['dOWT']
    if qc:
        return vel_model
    else:
        return vel_model[[depth_col_name, 'Vint']]

def dix(vel_model, timeunit='ms', qc=False):
    if timeunit == 's':
        c = 1
    elif timeunit == 'ms':
        c = 1000
    else:
        print('Wrong time unit given. Defaulting to miliseconds')
        c = 1000
    time_col_name = vel_model.columns[0]
    vel_col_name = vel_model.columns[1]
    print('Time column name: ' + time_col_name + '; Velocity column name: ' + vel_col_name)
    vel_model['t*Vrms**2'] = vel_model[time_col_name] * vel_model[vel_col_name]**2
    vel_model['Vint'] = ((vel_model['t*Vrms**2'].diff())/vel_model[time_col_name].diff())**0.5
    if qc:
        return vel_model
    else:
        return vel_model[[time_col_name, 'Vint']]