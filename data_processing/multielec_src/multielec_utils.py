import os
from scipy.io import loadmat


def get_stim_amps_newlv(analysis_path, pattern):
    """
    Read a newlv pattern files directory and get the stimulation 
    amplitudes.
    
    Parameters:
    analysis_path: Path to preprocessed data
    pattern: Pattern for which to get stimulation smplitudes

    Returns:
    amplitudes (np.ndarray): Stimulation amplitudes for the pattern
    """
    patternStruct = loadmat(os.path.join(analysis_path, "pattern_files/p" + str(pattern) + ".mat"), struct_as_record=False, squeeze_me=True)['patternStruct']
    return patternStruct.amplitudes
