# standard imports
import os
import gc
import traceback
from datetime import datetime
from dataclasses import dataclass

# project imports
from data_processing.common_utils import CommonUtils
from data_processing.saline_processor import SalineDataProcessor
from data_processing.sample_processor import SampleDataProcessor
from data_processing.experiment_manager import ExperimentManager
from data_processing.decoding_manager import DecodingManager
from data_processing.configurations import ElectrodeConfiguration, CalibrationMethods, VoltageExtractionMethods, GroundTruthTypes


@dataclass
class RunConfig:
	"""All the settings of a single run. There are no defaults: every execution script must set every field."""

	#####################################
	### Frequently changed parameters ###
	#####################################

	# root folder where the data is stored
	root_folder: str

	# which files to process
	indices_to_process: list

	# custom prefix for output folder
	user_prefix: str

	# set debug mode True for more detailed output
	debug_mode: bool

	# which settings to use for calibration
	dilutions_for_calibration: list
	saline_folder_suffix: str

	# which data to use as the sample
	use_saline_as_sample: None
	generate_ideal_data: bool
	use_simulated_data_as_sample: bool
	simulated_file_name: str

	##########################################
	### Less frequently changed parameters ###
	##########################################

	# calibration parameters
	coefficients_threshold_in_percents: float
	ground_truth_type: GroundTruthTypes
	calibration_method: CalibrationMethods
	voltage_extraction_method: VoltageExtractionMethods
	add_auto_potential_electrode_res2dinv: bool
	scale_auto_potential_electrode: bool
	currents_to_process: None
	currents_to_process_for_autopotentials: None
	currents_to_invert: None
	use_positive_currents: bool
	use_positive_phase: bool
	correct_gains: bool
	correct_currents: bool

	# decoding parameters
	filter_distant_elecs: float
	output_resistivity: bool
	decoding_tools: list
	electrode_configuration: ElectrodeConfiguration
	fix_saline_resistivity: bool
	fix_boundaries: bool


def run_pipeline(cfg: RunConfig):
	"""Process every header file found in cfg.root_folder that matches the run settings."""

	# simulated ML / COMSOL data is stored in megaohms
	simulated_data_in_megaohms = True if ("ML" in cfg.simulated_file_name or 'comsol' in cfg.simulated_file_name) else False

	# find all header files in the given root folder
	header_files = CommonUtils.find_header_files(cfg.root_folder, keyword='header')

	# iterate over the identified files
	for header_file_path in header_files:

		# start timer
		start_time = datetime.now()

		try:
			# parse header file to extract experiment parameters
			print('Parsing experiment parameters from the header file:', os.path.basename(header_file_path))
			experiment_manager = ExperimentManager(header_file_path,
												   calibration_method=cfg.calibration_method,
												   voltage_extraction_method=cfg.voltage_extraction_method,
												   dilutions_to_use_for_calibration=cfg.dilutions_for_calibration,
												   generate_ideal_data=cfg.generate_ideal_data,
												   ground_truth_type=cfg.ground_truth_type,
												   use_saline_as_sample=cfg.use_saline_as_sample,
												   use_simulated_data_as_sample=cfg.use_simulated_data_as_sample,
												   simulated_file_name=cfg.simulated_file_name,
												   indices_to_process=cfg.indices_to_process,
												   debug_mode=cfg.debug_mode,
												   currents_to_process=cfg.currents_to_process,
												   currents_to_process_for_shunt=cfg.currents_to_process_for_autopotentials,
												   use_positive_currents=cfg.use_positive_currents,
												   use_positive_phase=cfg.use_positive_phase,
												   currents_to_invert=cfg.currents_to_invert,
												   add_auto_potential_electrode_res2dinv=cfg.add_auto_potential_electrode_res2dinv,
												   scale_auto_potential_electrode=cfg.scale_auto_potential_electrode,
												   correct_gains=cfg.correct_gains,
												   correct_currents=cfg.correct_currents,
												   coefficients_threshold_in_percents=cfg.coefficients_threshold_in_percents,
												   saline_folder_suffix=cfg.saline_folder_suffix,
												   filter_distant_elecs_um=cfg.filter_distant_elecs,
												   simulated_data_in_megaohms=simulated_data_in_megaohms)

			# filter the files based on the requested indices and spacing
			if experiment_manager.skip_file:
				print('Skipping this file, as it doesn\'t match current run requirements...\n')
				continue

			# create the output folder if it does not exist
			if not os.path.exists(experiment_manager.output_folder):
				print('Creating output folder:', experiment_manager.output_folder)
				os.makedirs(experiment_manager.output_folder)

			# complete runtime parameters
			experiment_manager.complete_runtime_parameters()

			# plot the electrode positions
			if cfg.debug_mode:
				experiment_manager.plot_electrode_positions()

			# run calibration using the saline data; will do nothing if no calibration is needed
			saline_processor = SalineDataProcessor(experiment_manager)
			(gain_coefficients, gain_coefficients_per_dilution, auto_potential_xy_functions, xy_ratio_fit_function,
			 xy_gain_fit_function) = saline_processor.process_saline_data()

			# process the sample data
			sample_processor = SampleDataProcessor(experiment_manager, gain_coefficients, gain_coefficients_per_dilution,
												   auto_potential_xy_functions, xy_ratio_fit_function,
												   xy_gain_fit_function)
			calibrated_resistances, resistance_sem, autopotential_distances, original_autopotential_distances = sample_processor.process_sample_data()

			# define fixed resistivity regions
			fixed_resistivities = list()
			if cfg.fix_saline_resistivity:
				fixed_resistivities.append(experiment_manager.saline_fixed_resistivity)

			# define fixed boundary regions
			fixed_boundaries = list()
			if cfg.fix_boundaries and not (cfg.use_saline_as_sample or cfg.generate_ideal_data):
				# this is hard-coded for a 200um LE
				fixed_boundaries.extend([18.75, 66.56, 145.25, 200])

			# generate inversion input files
			print('Generating input files for the decoders')
			decoding_manager = DecodingManager(experiment_manager, calibrated_resistances,
											   resistance_sem,
											   autopotential_distances,
											   original_autopotential_distances,
											   electrode_configuration=cfg.electrode_configuration,
											   output_resistivity=cfg.output_resistivity,
											   fixed_resistivity_regions=fixed_resistivities,
											   fixed_boundary_regions=fixed_boundaries,
											   user_prefix=cfg.user_prefix)

			# generate the relevant input file for each of the requested decoders
			for decoding_tool in cfg.decoding_tools:
				decoding_manager.run_decoder(decoding_tool=decoding_tool)

		# ensure that any error in processing one file does not stop the entire batch
		except Exception as e:
			print(f"An error occurred while processing the file {os.path.basename(header_file_path)}: {e}")
			traceback.print_exc()

		# end timer and report run time
		finally:
			runtime_minutes = round((datetime.now() - start_time).total_seconds() / 60, 1)
			print(f'Run time: {runtime_minutes} minutes')

			# clear variables, figures, and memory
			gc.collect()
			del experiment_manager
			import matplotlib.pyplot as plt
			plt.close('all')
