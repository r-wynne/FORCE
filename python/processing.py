####################################################################################
#
#    processing.py: script to process the LHCO dataset into the two  leading jets
#    Eric M. Metodiev, MIT, 2020
#
####################################################################################

# python library imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import pyjet

file_base = '/Users/raymondwynne/Desktop/FORCE'

# use a generator to loop over the LHCO data file


def generator(filename, i=0, chunksize=100000, totalsize=1100000):
    while True:
        yield pd.read_hdf(filename, start=i*chunksize, stop=(i+1)*chunksize).values
        i += 1
        if (i+1)*chunksize > totalsize:
            print("End of file!")
            i = 0


# point the generator to the LHCO file
filename = file_base + '/data/LHCO/events_anomalydetection_v2.h5'
chunksize, totalsize = 100000, 1100000

# counter how many events have been processed
counter = 0
num = counter * chunksize

# get a generator to get chunks of events
gen = generator(filename, i=counter, chunksize=chunksize, totalsize=totalsize)

while num < totalsize:

    # get a chunk of data
    file = next(gen)
    start = time.time()

    # get events as list of [pT, y, phi] for 700 (zero-padded) particles
    events = file[:, :-1].reshape((len(file), 700, 3))

    # get the signal/background label (used for validation only)
    labels = file[:, -1]

    # prepare a (nevents, njets, nconsts, nkinematics) array
    events_proc = np.zeros((len(events), 2, 300, 3))

    # loop over the events in the chunk
    for i, event in enumerate(events):

        # write the particles in the event as pseudojets
        pseudojets_input = np.zeros(
            len([x for x in event[:, 0] if x > 0]), dtype=pyjet.DTYPE_PTEPM)
        for j in range(700):
            if (event[j, 0] > 0):
                pseudojets_input[j]['pT'] = event[j, 0]
                pseudojets_input[j]['eta'] = event[j, 1]
                pseudojets_input[j]['phi'] = event[j, 2]

        # cluster R=1.0 jets with the anti-kT algorithm
        sequence = pyjet.cluster(pseudojets_input, R=1.0, p=-1)

        # get the leading two jets
        jets = sequence.inclusive_jets(ptmin=250)[:2]

        # ensure that there are two jets
        if len(jets) < 2:
            print("Minimum pT cut too high!")

        # randomly shuffle the two jets
        jetorder = [[0, 1], [1,0]][np.random.randint(2)]
        for j, k in enumerate(jetorder):
            jetconstituents = np.asarray(
                [[pjet.pt, pjet.eta, pjet.phi] for pjet in jets[k].constituents()])

            # store the jet constituents in the processed events array
            for l, constituent in enumerate(jetconstituents):
                events_proc[i, j, l] = constituent

        if i % 10000 == 0:
            print("Done with {} events".format(i))

    # save the files
    np.savez_compressed(
        file_base + "/data/LHCO/LHCO_jets_"+str(counter), events_proc)
    np.savez_compressed(
        file_base + "/data/LHCO/LHCO_labels"+str(counter), labels)

    num += chunksize
    counter += 1
    print("Done with chunk {} in {}s!".format(counter-1, time.time() - start))
