####################################################################################
#
#    observables.py: script to compute EFPs on two jets from the LHCO dataset
#    Eric M. Metodiev, MIT, 2020
#
####################################################################################

# python library imports
import numpy as np
import matplotlib.pyplot as plt
import energyflow as ef
import time
import os
import sys

sys.path.insert(1, '/Users/raymondwynne/Desktop/FORCE')

file_base = '/Users/raymondwynne/Desktop/FORCE'
filename = '/Users/raymondwynne/Desktop/FORCE/data/LHCO/LHCO_'

# # compute all of the jet transverse momenta for the events
# jet_pts = []
# for k in range(11):
#     events = np.load(filename + 'jets_{}.npz'.format(k))['arr_0']
#     for event in events:
#         p40, p41 = ef.p4s_from_ptyphims(event[0]).sum(0), ef.p4s_from_ptyphims(event[1]).sum(0)
#         jet_pts.append([ef.pts_from_p4s(p40), ef.pts_from_p4s(p41)])
        
#     print("Done with dataset {}!".format(k))
# jet_pts = np.asarray(jet_pts)
# np.savez_compressed(file_base + "/data/LHCO/LHCO_jetpts", jet_pts)

# # compute all of the dijet invariant masses for the events
# dijet_masses = []
# for k in range(11):
#     events = np.load(filename + 'jets_{}.npz'.format(k))['arr_0']
#     for event in events:
#         dijet_masses.append(ef.ms_from_p4s(ef.p4s_from_ptyphims(event[0]).sum(0) + ef.p4s_from_ptyphims(event[1]).sum(0)))
#     print("Done with dataset {}!".format(k))
# dijet_masses = np.asarray(dijet_masses)
# np.savez_compressed(file_base + "/data/LHCO/LHCO_dijetmasses", dijet_masses)

# iterate over the dijets in the dataset, split into chunks
for k in range(0,11):
    start = time.time()
    
    # read in the R&D dataset from the LHC Olympics challenge
    events = np.load(filename + 'jets_{}.npz'.format(k))['arr_0']
    labels = np.load(filename + 'labels{}.npz'.format(k))['arr_0']

    # calculate some jet substructure observables
    efpset = ef.EFPSet('d<=4', measure='hadrdot', beta=1, normed=False, coords='ptyphim')
    
    # array to store the results of the computation
    efps = np.zeros((len(events),2,len(efpset.graphs())))

    # iterate over the events and the jets
    for i,event in enumerate(events):
        for j,jet in enumerate(event):
            # normalize the constituent pTs by 1 TeV
            efps[i,j] = efpset.compute(jet[jet[:,0]>0]/np.asarray([1000,1,1]))

        if i%10000==0: print("Done processing {} jets!".format(i))

    # save the computed efps!
    np.savez_compressed(file_base + '/data/LHCO/LHCO_efps_{}'.format(k), efps)
    
    print("Done with batch {} in {}s.".format(k, time.time()-start))
