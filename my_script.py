from geotools import waterspeed
#help(waterspeed)
#wspmodel_2500 = waterspeed.WSModel(sp_int=25, maxcurrent=3, pop_min=2000, pop_max=15000)
#wspmodel_9375 = waterspeed.WSModel(9.375, pop_min=2000, pop_max=5000)

#wspmodel_9375.plotwaterspeed()
#wspmodel_9375.setbsplimits([3.0, 4.5])
#wspmodel_9375.setpoplimit(3600, bsplowlimit=4.0)
#wspmodel_9375.plotcombspeed(minpopint=3600)
#print(waterspeed.WSModel.__doc__)
print(waterspeed.WSModel.plotwaterspeed.__doc__)