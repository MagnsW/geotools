import os
import pandas as pd
from geotools import waterspeed, input_tools, fourD_tools



#help(waterspeed)
#wspmodel_2500 = waterspeed.WSModel(sp_int=25.00, pop_min=4000, pop_max=15000)
wspmodel = waterspeed.WSModel(sp_int=25, pop_min=2000, pop_max=14000)

#wspmodel_2500.setpoplimit(8000, bsplowlimit=3.5)
wspmodel.setpoplimit(6000, bsplowlimit=3.5)

#wspmodel_2500.plotcombspeed(cleanreclength=8000)
wspmodel.plotcombspeed(cleanreclength=6000)

#wspmodel_2500.plotwaterspeed(cleanreclength=8000)
wspmodel.plotwaterspeed(cleanreclength=6000)


#wspmodel_9375.plotwaterspeed()
#wspmodel_9375.setbsplimits([3.0, 4.5])
#wspmodel_9375.setpoplimit(4500, bsplowlimit=4.0)
#wspmodel_9375.plotcombspeed(minpopint=4500)
#print(waterspeed.WSModel.__doc__)
#print(waterspeed.WSModel.plotwaterspeed.__doc__)
#testpath = './testfiles/4D/'
#testfilename = testpath + 'Gap_REFf2-2_on_POLf0-s.A1X'
#testwrongfilename = testpath + 'ap_REFf2-2_on_POLf0-s.A1X'
#print(os.listdir(testpath))

#df = input_tools.make_df_from_columndata(testfilename)
#print(df.columns)
#print(pd.isna(df).sum())

#help(input_tools.make_df_from_columndata)

#testpath = './testfiles/infill/'
#testfilename = testpath + 'NM_i7_100p.A1X'

#df = input_tools.make_df_from_columndata(testfilename)
#print(df.columns)
#print(pd.isna(df).sum())

# path = './testfiles/4D/'

# repeatability_files = {
#     '10x75m - 2deg on 8x75m': path + 'Gap_REFf2-2_on_POLf0-s.A1X',
#     '12x75m - 2deg on 8x75m': path + 'Gap_12x75f2-2_on_POLf0-s.A1X',
#     '19x37.5m - 2deg on 8x75m': path + 'Gap_19x37f2-2_on_POLf0-s.A1X',
#     '19x37.5m - 4deg on 8x75m': path + 'Gap_19x37f4-4_on_POLf0-s.A1X',
#     '19x37.5m - 150perc Fanning - 2deg on 8x75m': path + 'Gap_19x37fanf2-2_POLf0-s.A1X',
#     '19x37.5m - 150perc Fanning - 4deg on 8x75m': path + 'Gap_19x37fanf4-4_POLf0-s.A1X',
# }

# test_4d_data = fourD_tools.RepeatabilityData(repeatability_files)
# #print(test_4d_data.filenames)
# print(test_4d_data.df.columns)
# test_4d_data.offsetsplit(100, 408)
# print(test_4d_data.df.columns)
# print(test_4d_data.make_summary('10x75m - 2deg on 8x75m', 'OffsDiff'))
# print(test_4d_data.scenarios)
# print(test_4d_data.make_full_summary())
# test_4d_data.plot_all_dist()
