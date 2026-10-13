# project imports
from data_processing.pipeline import RunConfig, run_pipeline
from data_processing.configurations import ElectrodeConfiguration, CalibrationMeasurements, CalibrationMethods, \
	VoltageExtractionMethods, DecodingTool, GroundTruthTypes


#################################### EDIT THIS SECTION TO CONFIGURE YOUR RUN ####################################

cfg = RunConfig(

	#####################################
	### Frequently changed parameters ###
	#####################################

	# set the root folder where the data is stored
	root_folder = (r".../processed_long_evans_data"),

	# define which files to process
	indices_to_process = [1094],

	# define custom prefix for output folder
	user_prefix = '600um',

	# set debug mode True for more detailed output
	debug_mode = True,

	# define which settings to use for calibration
	dilutions_for_calibration = [
		CalibrationMeasurements.X1.name,
		CalibrationMeasurements.X10.name,
		CalibrationMeasurements.X30.name,
		CalibrationMeasurements.X50.name,
		CalibrationMeasurements.X100.name,
	],
	saline_folder_suffix = 'after',

	# define which data to use as the sample
	use_saline_as_sample = None,
	generate_ideal_data = False,
	use_simulated_data_as_sample = False,
	simulated_file_name = "",

	##########################################
	### Less frequently changed parameters ###
	##########################################

	# set calibration parameters
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

	# set decoding parameters
	filter_distant_elecs = 600,
	output_resistivity = False,
	decoding_tools = [DecodingTool.ML_DECODER, DecodingTool.RES2DINV],
	electrode_configuration = ElectrodeConfiguration.ALL_COMBS_MINIMAL,
	fix_saline_resistivity = False,
	fix_boundaries = False,
)

######################################## DO NOT EDIT BELOW THIS LINE ###########################################

if __name__ == '__main__':
	run_pipeline(cfg)
