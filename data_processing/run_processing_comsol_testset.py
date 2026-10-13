# standard imports
import os
import re
from dataclasses import replace

# project imports
from data_processing.pipeline import RunConfig, run_pipeline
from data_processing.configurations import ElectrodeConfiguration, CalibrationMeasurements, CalibrationMethods, \
	VoltageExtractionMethods, DecodingTool, GroundTruthTypes

# add full paths before running!

# # this is for LE
# input_folder = (r".../simulated_data")
# root_folder = (r".../aggregated_long_evans_data")
# indices_to_process = [2000]
# user_prefix = 'testset_8000'

# this is for RCS
input_folder = (r".../simulated_data")
root_folder = (r".../aggregated_rcs_data")
indices_to_process = [4000]
user_prefix = 'testset_8000'

# input file name pattern
filename_re = re.compile(
	r"^comsol_testset_\d+(?:\.\d+)?_NFL_GCL_(?P<GCL>[\d.]+)_IPL_(?P<IPL>[\d.]+)"
	r"_nu_(?P<nu>[\d.]+)_PR_(?P<PR>[\d.]+)_saline_(?P<saline>[\d.]+)$"
)

#################################### EDIT THIS SECTION TO CONFIGURE YOUR RUN ####################################

base_cfg = RunConfig(

	#####################################
	### Frequently changed parameters ###
	#####################################

	# root folder where the data is stored, which files to process, and output folder prefix (set above)
	root_folder = root_folder,
	indices_to_process = indices_to_process,
	user_prefix = user_prefix,

	# set debug mode True for more detailed output
	debug_mode = False,

	# which settings to use for calibration
	dilutions_for_calibration = [
		CalibrationMeasurements.X1.name,
		CalibrationMeasurements.X10.name,
		CalibrationMeasurements.X30.name,
		CalibrationMeasurements.X50.name,
		CalibrationMeasurements.X100.name,
	],
	saline_folder_suffix = '',

	# which data to use as the sample
	use_saline_as_sample = None,
	generate_ideal_data = False,
	use_simulated_data_as_sample = True,
	simulated_file_name = "",  # replaced by each test file name in the loop at the bottom

	##########################################
	### Less frequently changed parameters ###
	##########################################

	# calibration parameters
	coefficients_threshold_in_percents = 50,
	ground_truth_type = GroundTruthTypes.ANALYTICAL,
	calibration_method = CalibrationMethods.COEFFICIENTS_VECTOR,
	voltage_extraction_method = VoltageExtractionMethods.MAX,
	add_auto_potential_electrode_res2dinv = False,
	scale_auto_potential_electrode = True,
	currents_to_process = None,
	currents_to_process_for_autopotentials = None,
	currents_to_invert = None,
	use_positive_currents = True,
	use_positive_phase = True,
	correct_gains = True,
	correct_currents = True,

	# decoding parameters
	filter_distant_elecs = 600,
	output_resistivity = False,#True
	decoding_tools = [DecodingTool.ML_DECODER, DecodingTool.RES2DINV],
	electrode_configuration = ElectrodeConfiguration.ALL_COMBS_MINIMAL,
	fix_saline_resistivity = False,
	fix_boundaries = False,
)

###############################################################################################################
# filter out files that have already been processed, to avoid re-running the same inversion on the same input data

# output folder
output_folder = (r".../testset_8000_res2dinv_sample_no_calib")

# output file name pattern
output_re = re.compile(
	r"^sim_sample_comsol_testset_\d+(?:\.\d+)?_NFL_GCL_(?P<GCL>[\d.]+)_IPL_(?P<IPL>[\d.]+)"
	r"_nu_(?P<nu>[\d.]+)_PR_(?P<PR>[\d.]+)_saline_(?P<saline>[\d.]+)_no_ap_all_combs_minimal\.DAT$"
)

# function to extract params
def _param_key(match):
	return (match.group("GCL"), match.group("IPL"), match.group("nu"), match.group("PR"), match.group("saline"))


def find_files_to_process():
	# collect the GCL/IPL/nu/PR/saline combinations that already have a processed output file
	processed_param_keys = set()
	if os.path.isdir(output_folder):
		for out_name in os.listdir(output_folder):
			out_match = output_re.match(out_name)
			if out_match:
				processed_param_keys.add(_param_key(out_match))

	# skip files that have already been processed
	files_to_process = []
	skipped_already_processed = 0

	for name in sorted(os.listdir(input_folder)):
		# only search xlsx files
		if not name.endswith(".xlsx"):
			continue

		# extract the file name
		stem = name[:-len(".xlsx")]

		# check if matches the expected pattern
		match = filename_re.fullmatch(stem)

		# skip if does not match the expected pattern
		if not match:
			continue

		# skip files that already have a matching output file
		if _param_key(match) in processed_param_keys:
			skipped_already_processed += 1
			continue

		files_to_process.append(stem)

	return files_to_process

######################################## DO NOT EDIT BELOW THIS LINE ###########################################

if __name__ == '__main__':
	files_to_process = find_files_to_process()
	print(len(files_to_process))

	# run the shared pipeline once per simulated test file
	for test_file_name in files_to_process:
		run_pipeline(replace(base_cfg, simulated_file_name=test_file_name))
