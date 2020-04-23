import xarray as xr
import numpy as np
import pandas as pd

def make_df_from_columndata(filename, nandefault=-999.25):
    '''This function reads in a Nucleus column data set (.A1X file) and returns
    a pandas dataframe. Default nan value is set to -999.25. To change this,
    use the nandefault parameter when calling the function.
    Is this included????
    '''
    try:
        ds = xr.open_dataset(filename)
    except FileNotFoundError:
        print("File not found")
    else:
        df = ds.to_dataframe()
        df = df.replace(nandefault, np.nan)
        return df

def make_df_from_segy(filename):
    import segyio
    headerword = 'TRACE_SAMPLE_INTERVAL'
    with segyio.open(filename) as f:
        for header in f.header:
            for key in header:
                #print(str(key))
                if str(key) == headerword:
                    #print(key, header[key])
                    samp_int = header[key]
        for trace in f.trace:
            pass
    print(f'Sample interval: {samp_int}')
    dt = np.arange(len(trace)) * int(samp_int/1000)
    df_segy = pd.DataFrame(zip(dt, trace), columns=['dt', 'Amplitude'])
    return df_segy
