from geotools import offset_distribution


path = './testfiles/nearoffset_dist/Eq_Arg/'
filenames = {'Standard Triple Source - 12x75': path + '12x75mx6000m_TS.A1X',
            'Wide Tow Source - 12x75': path + '12x75mx8000m_Equinor.A1X',
            'Standard Triple Source - 12x93.75': path + '12x93,75mx6000m_MW.A1X',
            'Wide Tow Source - 12x93.75': path + '12x93,75mx6000m_WT_MW.A1X'}

my_offset_data = offset_distribution.OffsetData(filenames, maxoffset=600, incoffset=25)

#print(my_offset_data.df)
#print(my_offset_data.offset_binned['Wide Tow Source - 12x75'])
for offset in my_offset_data.offset_planes:
    print(offset)
print(my_offset_data.df_attribs_stats)
print(my_offset_data.df_attribs_stats.columns)
my_offset_data.plot_attrib()