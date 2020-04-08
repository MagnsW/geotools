import xarray as xr
import numpy as np

def make_df_from_columndata(filename, nandefault=-999.25):
    '''This function reads in a Nucleus column data set (.A1X file) and returns
    a pandas dataframe. Default nan value is set to -999.25. To change this,
    use the nandefault parameter when calling the function.
    '''
    try:
        ds = xr.open_dataset(filename)
    except FileNotFoundError:
        print("File not found")
    else:
        df = ds.to_dataframe()
        df = df.replace(nandefault, np.nan)
        return df

