'''
Module for reading in various data types related to electrical stimulation 
data acquired from the Old LabVIEW data acquisition software.

NOTE: these functions assume SINGLE electrode stimulation.

@author Alex Gogliettino
@date 2020-05-09
'''

import os
import numpy as np

def get_oldlabview_pp_data(analysis_path, pattern_no, movie_ind):
    '''
    Function: get_oldlabview_pp_data
    Usage: get_oldlabview_pp_data(analysis_path,pattern_no,movie_ind)
    -----------------------------------------------------------------
    @param analysis_path Path to 1-el scan analysis.
           example: '/Volumes/Analysis/2017-11-20-9/data001/' (str).
    @param pattern_no: Number of pattern application
           example: 172 (int or str).
    @param movie_ind: Index of the amplitude application. Typically, at least 37
           is the max, but some electrodes will have 38/39.
    @return T x E x 100 dimensional tensor, where t is the number of trials, 
            E is the number of electrodes, and 100 is the number of samples.
    @throw OSError if the data don't exist
           ValueError if index is invalid.

    Function to load in raw data from an Old LabVIEW single electrode scan.
    Based of a function written by bhaishahster.
    '''
    
    # Get the movie numbers.
    pattern_no = 'p' + str(pattern_no)
    pattern_path = os.path.join(analysis_path,pattern_no)

    if not os.path.isdir(pattern_path):
        raise OSError("Data for %s not written"%pattern_no)

    movienos = []

    for filename in os.listdir(pattern_path):
        if filename[len(pattern_no)+2:].isdigit():
            movienos.append(int(filename[len(pattern_no)+2:])) # Because of '_m'
    movienos = sorted(movienos)

    # invalid current index
    if movie_ind < 0:
        raise ValueError("Invalid movie index %d"%movie_ind)

    # empty current index (no data for this current index)
    if movie_ind not in movienos:
        return None
    
    # Get the filename requested by the user.
    src_file = pattern_no +  '_m' + str(movie_ind)
    filepath = os.path.join(pattern_path, src_file)

    # check that the file exists
    if not os.path.isfile(filepath):
        raise OSError("File %s doesn't exit" % src_file)

    # i/o. 
    b = np.fromfile(filepath, dtype='<h')
    b0 = b[:1000]
    b1 = b[1000:]
    data_traces = np.reshape(b1, [b0[0],b0[1],b0[2]], order='f')
    return data_traces
