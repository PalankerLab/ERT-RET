# standard imports
import os
import glob
import re as regex
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# project-specific imports
from data_processing.common_utils import CommonUtils
import data_processing.multielec_src.multielec_utils as mutils
from data_processing.configurations import ExperimentTypes, CalibrationMeasurements, VoltageExtractionMethods, ArrayTypes, GroundTruthTypes


class ExperimentManager:

	def __init__(self, header_file,
				 calibration_method=None,
				 voltage_extraction_method=None,
				 dilutions_to_use_for_calibration=None,
				 generate_ideal_data=False,
				 ground_truth_type=GroundTruthTypes.ANALYTICAL,
				 use_saline_as_sample=None,
				 use_simulated_data_as_sample=False,
				 simulated_file_name=None,
				 coefficients_threshold_in_percents=5,
				 indices_to_process=None,
				 debug_mode=True,
				 currents_to_process=None,
				 currents_to_process_for_shunt=None,
				 use_positive_currents=False,
				 use_positive_phase=False,
				 currents_to_invert=None, add_noise=False,
				 add_auto_potential_electrode_res2dinv=False,
				 scale_auto_potential_electrode=True,
				 correct_gains=True,
				 correct_currents=True,
				 saline_folder_suffix=None,
				 filter_distant_elecs_um=None,
				 simulated_data_in_megaohms=True):

		# initialize the skip file flag
		self.skip_file = False

		# initialize debug flag
		self.debug_mode = debug_mode

		# set hard-coded values
		self.infinite_thickness_um = 1000
		self.auto_potential_distance_um = 5

		# define system saturation voltage
		self.system_saturation_voltage_uv = 1600
		self.system_saturation_voltage_uv_ap = 1600

		# store given parameters
		self.correct_currents = correct_currents
		self.correct_gains = correct_gains
		self.calibration_method = calibration_method
		self.voltage_extraction_method = voltage_extraction_method
		self.generate_data = generate_ideal_data
		self.dilutions_to_use_for_calibration = dilutions_to_use_for_calibration if dilutions_to_use_for_calibration else CalibrationMeasurements.get_all_dilutions()
		self.ground_truth_type = ground_truth_type
		self.use_saline_as_sample = use_saline_as_sample
		self.use_simulated_data_as_sample = use_simulated_data_as_sample
		self.simulated_file_name = simulated_file_name
		self.currents_to_process = currents_to_process
		self.currents_to_process_for_autopotentials = currents_to_process_for_shunt
		self.use_positive_currents = use_positive_currents
		self.use_positive_phase = use_positive_phase
		self.currents_to_invert = currents_to_invert
		self.add_noise = add_noise
		self.add_auto_potential_electrode_res2dinv = add_auto_potential_electrode_res2dinv
		self.scale_auto_potential_electrode = scale_auto_potential_electrode
		self.simulated_data_in_megaohms = simulated_data_in_megaohms
		self.saline_folder_suffix = saline_folder_suffix
		self.filter_distant_elecs_um = filter_distant_elecs_um
		self.common_coefficients_threshold = coefficients_threshold_in_percents

		# extract file name from the path
		file_name = os.path.basename(header_file)

		# remove the extension
		file_name = file_name.split('.')[: -1]

		# connect the name string back together
		file_name = '.'.join(file_name)

		# split the name using the '_' character
		parts = file_name.split('_')

		# remove the first part which is the word 'header'
		parts = parts[1:]

		# set default values; these will be updated from the header file
		self.index = None
		self.species = None
		self.thickness_um = None
		self.array_size_um = None
		self.array_spacing_um = None
		self.injection_line_spacing_um = None
		self.injection_line_num = None
		self.saline_dilutions = {}
		self.saline_dilutions_for_calibration = {}
		self.num_dilutions_for_calibration = None
		self.currents_nanoamps = None
		self.stimulation_electrodes = None
		self.start_offset = None
		self.rep_stim_electrode = None
		self.linear_stim_electrodes = None
		self.recording_indices = None
		self.num_recording_electrodes = None
		self.electrode_specific_noise_uv = None

		# check if we are running simulated data as the sample
		if self.use_simulated_data_as_sample:
			self.experiment_type = ExperimentTypes.COMSOL
			self.generate_data = True
			self.num_calibrations = 0

			# current is 1uA for simulated data
			self.currents_nanoamps = np.array([1000])

		else:
			self.experiment_type = ExperimentTypes.MEA

		# parse non-resistivity values by scanning all parts
		for p in parts:
			if self.index is None and p.replace('.', '', 1).isdigit():
				index = float(p)

				# check if the index is in the list of indices to process, if not return
				if indices_to_process and index not in indices_to_process:
					self.skip_file = True
					return

				# if the index is valid, set it
				self.index = index
				continue

			# species: first non-numeric string after index
			if self.species is None and not any(c.isdigit() for c in p):
				self.species = p
				continue

			# thickness, array size, and array spacing (all ending with 'um')
			if p.endswith('um'):
				val = int(p.replace('um', ''))

				# array_spacing_um: < 100 µm
				if val < 100 and self.array_spacing_um is None:
					self.array_spacing_um = val

				# thickness_um: 100–400 µm range
				elif 100 < val < 400 and self.thickness_um is None:
					self.thickness_um = val

				# array_size_um: ≥ 400 µm
				elif val >= 400 and self.array_size_um is None:
					self.array_size_um = val

				continue

			# start offset
			if 'offset' in p and self.start_offset is None:
				self.start_offset = int(p.replace('offset', ''))
				continue

			# representative stimulation electrode
			if 'rstim' in p and self.rep_stim_electrode is None:
				self.rep_stim_electrode = int(p.replace('rstim', ''))
				continue

			# injection line number
			if 'line' in p and self.injection_line_num is None:
				self.injection_line_num = int(p.replace('line', ''))
				continue

		# extract all resistivity values ending with 'Ocm'
		resistivity_values = [float(p.replace('Ocm', '')) for p in parts if p.endswith('Ocm')]

		# assign each resistivity value to the closest enum dilution
		base_resistivity = 50.0
		dilution_factors = {member.name: int(member.name[1:]) for member in CalibrationMeasurements}
		for r in resistivity_values:
			ratio = r / base_resistivity
			closest_enum_name = min(dilution_factors, key=lambda k: abs(dilution_factors[k] - ratio))
			self.saline_dilutions[closest_enum_name] = r

		# if only certain dilutions are requested for calibration, filter the resistivities
		self.num_dilutions_for_calibration = 0
		if self.dilutions_to_use_for_calibration and not self.generate_data:
			self.saline_dilutions_for_calibration = {k: v for k, v in self.saline_dilutions.items() if k in self.dilutions_to_use_for_calibration}
			self.num_dilutions_for_calibration = len(self.saline_dilutions_for_calibration)

		# if rstim is still None, set it to 448
		if self.rep_stim_electrode is None:
			self.rep_stim_electrode = 448

		# if offset is still None, set it to 49
		if self.start_offset is None:
			self.start_offset = 49

		# get the position files that match the given spacing
		positions_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MEA_layout_files')
		prefix = 'MEA{}'.format(self.array_spacing_um)

		# see if there is a file for that specific spacing and array size
		matching_files = glob.glob(os.path.join(positions_folder, f'{prefix}*.csv'))

		# check if any of the retrieved files match the array size
		electrode_positions = None
		if matching_files:
			for file in matching_files:
				if str(self.array_size_um) in file:
					electrode_positions = np.loadtxt(file, delimiter=',')
					break

		# obtain the number of electrodes in this MEA from the electrode positions
		self.electrode_positions = electrode_positions
		self.num_electrodes = self.electrode_positions.shape[0]

		# add concentrated saline resistivity as the resistivity used in the sample measurements
		self.sample_saline_resistivity_ohm_um = None
		if self.saline_dilutions[CalibrationMeasurements.X1.name]:
			self.sample_saline_resistivity_ohm_um = round(self.saline_dilutions[CalibrationMeasurements.X1.name] * 1E4, 4)

		# get the array type for this experiment
		self.array_type = ArrayTypes.find_array_type(self.array_spacing_um, self.experiment_type)

		# add path to data files to the dictionary
		parent_folder = os.path.dirname(header_file)
		self.input_folder = os.path.join(parent_folder, "raw_data")
		self.output_folder = os.path.join(parent_folder, "processed_data")

		# complete input path for the sample data
		self.sample_data_folder = self.use_saline_as_sample if self.use_saline_as_sample else 'sample'
		if self.use_saline_as_sample and self.saline_folder_suffix:
			self.sample_data_folder = self.sample_data_folder + "_" + self.saline_folder_suffix
		self.sample_type = self.sample_data_folder
		self.sample_data_path = os.path.join(self.input_folder, self.sample_data_folder)

		# add simulated data parameters
		self.simulated_data_folder = os.path.join(parent_folder, "simulated_data")
		self.simulated_data = None

		# if only one line of recording electrodes is used, extract the indices of the electrodes and the spacing
		self.recording_indices = range(self.num_electrodes)
		self.injection_line_spacing_um = self.array_spacing_um

		if self.injection_line_num is not None:

			# extract the recording electrodes for this injection line
			rec_nums = self.array_type.injection_line_electrodes[self.injection_line_num]
			self.recording_indices = [rec_num - 1 for rec_num in rec_nums]

			# calculate the spacing in this line
			line_positions = self.electrode_positions[self.recording_indices]
			diffs = np.diff(line_positions, axis=0)
			self.injection_line_spacing_um = np.linalg.norm(diffs, axis=1).min()

		# store the number of recording electrodes based on the extracted indices
		self.num_recording_electrodes = len(self.recording_indices)

		# try to load the simulated data file, if we are running simulated data as the sample
		if self.use_simulated_data_as_sample:

			# verify that the simulated data folder exists
			if not os.path.exists(self.simulated_data_folder):
				raise ValueError('Simulated data folder not found in the experiment directory.')

			# verify the needed file exists
			simulated_file_path = os.path.join(self.simulated_data_folder, self.simulated_file_name + '.xlsx')
			if not os.path.exists(simulated_file_path):
				raise ValueError(f'Simulated data file {self.simulated_file_name} not found in the simulated data folder.')

			# load the simulated data dictionary of distance → voltage
			self.simulated_data = self._load_simulated_resistances_as_a_func_of_distance(simulated_file_path)

		# system noise floor
		self.system_noise_floor_uv = 10 if not self.use_simulated_data_as_sample else None

		# add experimental setup data
		self.sampling_interval_usec = 50 # this is the sample rate of the system
		self.sampling_interval_msec = self.sampling_interval_usec * 1E-3
		self.switch_between_phases_usec = 500 # there are two phases in the biphasic pulse, each lasts 0.5 ms

		# this is the signal length we want to analyze (the full recording is much longer than the stimulation pulse)
		self.signal_length = 20
		self.signal_length_one_phase = 11

		# create an indicator for the sign of the current (during the two phases of the biphasic pulse)
		self.phase_sign_mask = np.ones(self.signal_length)
		self.phase_sign_mask[self.signal_length // 2:] = -1 if not self.use_positive_phase else 1

		# generate a time vector for the signal
		self.time_vector_msec = np.arange(self.signal_length) * self.sampling_interval_msec

		# add values to be filled in during execution
		self.num_stimulation_electrodes = None
		self.num_currents = None
		self.current_sign_mask = None
		self.num_positive_currents = None
		self.positive_currents_nanoamps = None

		# initialize the valid current parameters
		self.valid_currents_nanoamps = None
		self.num_valid_currents = None
		self.valid_current_indices = None
		self.current_index_map = {}
		self.valid_current_indices_autopotentials = None
		self.valid_currents_nanoamps_autopotentials = None

		# add default saline fixed resistivity region
		self.saline_fixed_resistivity = {"z1": self.thickness_um,
										 "z2": self.thickness_um + self.infinite_thickness_um,
										 "resistivity": self.sample_saline_resistivity_ohm_um,
										 "weight": 10.0}

	def _determine_valid_current_levels(self, run_for_autopotentials=False):
		# initialize the valid current indices to all currents
		valid_current_indices = np.ones(self.num_currents, dtype=bool)

		# intersect these valid current indices with the ones we want to process in this run (user request)
		if run_for_autopotentials:
			if self.currents_to_process_for_autopotentials is not None:
				# create a mask of the required currents
				mask = np.zeros_like(valid_current_indices, dtype=bool)
				mask[self.currents_to_process_for_autopotentials] = 1

				# apply the mask to the valid current indices to remove entries that are valid, but we don't want to run
				valid_current_indices = valid_current_indices * mask
		else:
			if self.currents_to_process is not None:
				# create a mask of the required currents
				mask = np.zeros_like(valid_current_indices, dtype=bool)
				mask[self.currents_to_process] = 1

				# apply the mask to the valid current indices to remove entries that are valid, but we don't want to run
				valid_current_indices = valid_current_indices * mask

		# do the same for positive currents (another user request)
		if self.use_positive_currents:
			# create a mask of the positive currents
			mask = np.zeros_like(valid_current_indices, dtype=bool)
			mask[self.currents_nanoamps > 0] = 1

			# apply the mask to the valid current indices to remove entries that are valid, but we don't want to run
			valid_current_indices = valid_current_indices * mask

		# if all currents are invalid, terminate the run
		if not np.any(valid_current_indices):
			raise ValueError('All currents are invalid. Please review the configuration and the data.')

		return valid_current_indices

	def _load_simulated_resistances_as_a_func_of_distance(self, excel_file):
		# load file
		df = pd.read_excel(excel_file)

		# extract only columns starting with "V_"
		v_columns = [col for col in df.columns if col.startswith("V_")]

		# read the first (or only) row of V values
		r_values = df[v_columns].iloc[0].values

		# convert from MOhm to kOhm
		if self.simulated_data_in_megaohms:
			r_values = np.round(r_values * 1E3, 4)
		else:
			r_values = np.round(r_values, 4)

		# build distance array
		distances = [self.auto_potential_distance_um] + [round(i * self.injection_line_spacing_um, 2) for i in range(
			1, len(r_values))]

		# Create distance → voltage dictionary
		dist_resis_dict = dict(zip(distances, r_values))

		return dist_resis_dict

	def complete_runtime_parameters(self):

		# extract the stimulation electrodes for this experiment
		if self.stimulation_electrodes is None:

			# get all p folders in the input folder
			self.stimulation_electrodes = self.get_list_of_stimulating_electrodes(self.sample_data_path)

		# extract current amplitude values from run files, for experimental data
		if self.currents_nanoamps is None:

			# extract the current amplitudes in microamperes from the recorded data
			self.rep_stim_electrode = self.rep_stim_electrode if self.rep_stim_electrode else self.stimulation_electrodes[0]
			currents_microamps = mutils.get_stim_amps_newlv(self.sample_data_path, self.rep_stim_electrode)

			# convert to nanoamperes
			currents_nanoamps = currents_microamps * 1E3

			# remove zero current entries
			currents_nanoamps = currents_nanoamps[currents_nanoamps != 0]

			# store processed currents list
			self.currents_nanoamps = np.array(currents_nanoamps)

		self.num_currents = self.currents_nanoamps.size
		self.num_stimulation_electrodes = len(self.stimulation_electrodes)

		# create an indicator for the sign of the initial phase of the current;
		# this doesn't change whether the current is valid or not because we want the reading indices to match
		self.current_sign_mask = np.ones(self.num_currents)
		self.current_sign_mask[self.currents_nanoamps < 0] = -1

		# initialize valid current to all currents
		self.valid_currents_nanoamps = self.currents_nanoamps

		# now, determine the valid current indices based on the user requirements
		valid_current_indices = self._determine_valid_current_levels()

		# update list of valid current indices
		self.valid_current_indices = np.where(valid_current_indices == 1)[0]
		self.valid_currents_nanoamps = self.currents_nanoamps[valid_current_indices]
		self.num_valid_currents = len(self.valid_current_indices)

		# reconcile between currents to process and currents to invert, if both are provided
		if self.currents_to_invert is not None and self.valid_current_indices is not None:

			# if the lists of currents are different we need to find the intersection, if exists
			if len(self.valid_current_indices) != len(self.currents_to_invert) or (len(self.valid_current_indices) == len(self.currents_to_invert) != 1):
				self.currents_to_invert = self.valid_current_indices[np.isin(self.valid_current_indices, self.currents_to_invert)]

				# raise error if the intersection is empty
				if self.currents_to_invert is None or len(self.currents_to_invert) == 0:
					raise ValueError("Provided currents to invert do not match the valid current indices")

			# if we only have one value in each list, they have to match
			if len(self.valid_current_indices) == len(self.currents_to_invert) == 1 and self.valid_current_indices != self.currents_to_invert:
				raise ValueError("Provided currents to invert do not match the valid current indices")

			# translate the currents to invert to the running indices of the valid currents
			current_index_map = {val: idx for idx, val in enumerate(self.valid_current_indices)}
			self.currents_to_invert = [current_index_map[i] for i in self.currents_to_invert]

		# identify positive currents (synthetic data has only positive currents, experimental has both)
		self.positive_currents_nanoamps = self.valid_currents_nanoamps[self.valid_currents_nanoamps > 0]
		self.num_positive_currents = self.positive_currents_nanoamps.size

		# determine the valid currents for the shunt calibration using autopotentials
		valid_current_indices_autopotentials = self._determine_valid_current_levels(run_for_autopotentials=True)
		self.valid_current_indices_autopotentials = np.where(valid_current_indices_autopotentials == 1)[0]
		self.valid_currents_nanoamps_autopotentials = self.currents_nanoamps[self.valid_current_indices_autopotentials]

		# and then turn all currents to positive for the rest of the processing
		self.currents_nanoamps = abs(self.currents_nanoamps)
		self.valid_currents_nanoamps = abs(self.valid_currents_nanoamps)

	def plot_electrode_positions(self):
		# plot the electrode positions
		plt.figure(figsize=(12, 12))

		# change the background color of the axes
		plt.gca().set_facecolor('white')

		# change the background color of the figure
		plt.gcf().set_facecolor('white')

		# plot all electrodes
		plt.scatter(self.electrode_positions[:, 0], self.electrode_positions[:, 1], c='black', s=17)

		# add labels
		plt.xlabel('X (um)', fontsize=16)
		plt.ylabel('Y (um)', fontsize=16)

		# calculate the size of the array in micrometers
		array_size = int(self.electrode_positions[:, 0].max() - self.electrode_positions[:, 0].min())
		plt.title(f'{array_size}um-wide array with {self.array_spacing_um}um spacing', fontsize=18)

		# extract stimulation electrodes
		injection_line_0_center_30um = list({455, 454, 456, 453, 457, 452, 458, 451, 459, 450, 460, 449, 461, 448, 189, 201, 190, 200, 191, 199, 192,198, 193, 197, 194, 196, 195})
		injection_line_1_top_left_bottom_right_30um = list({271, 276, 281, 284, 288, 287, 295, 292, 300, 291, 309, 202, 448, 50, 49, 31, 40, 32, 35, 27, 28, 24, 21,16, 11})
		injection_line_2_top_right_bottom_left_30um = list({379, 374, 369, 366, 362, 363, 355, 358, 350, 359, 341, 340, 448, 188, 81, 99, 90, 98, 95, 103, 102, 106, 109, 114, 119})
		stimulation_nums = injection_line_0_center_30um + injection_line_1_top_left_bottom_right_30um + injection_line_2_top_right_bottom_left_30um
		stimulation_electrodes = self.electrode_positions[np.array(stimulation_nums) - 1]

		# add stimulation electrodes in a different color
		plt.scatter(stimulation_electrodes[:, 0], stimulation_electrodes[:, 1], c='#E377C3', s=17)

		# add label to each electrode
		for i in range(self.num_electrodes):
			plt.annotate(i + 1, (self.electrode_positions[i, 0], self.electrode_positions[i, 1]),textcoords="offset points", xytext=(0, 0), ha='center', fontsize=6, color='black')

		# show figure
		plt.show()

		# create an additional figure
		plt.figure(figsize=(15, 15))
		plt.gca().set_facecolor('white')
		plt.gcf().set_facecolor('white')

		# scatter plot for all electrodes
		plt.scatter(self.electrode_positions[:, 0], self.electrode_positions[:, 1], c='#C0C0C0', s=10)

		# add stimulation electrodes in a different color
		plt.scatter(stimulation_electrodes[:, 0], stimulation_electrodes[:, 1], c='#6CA965', s=10)

		# extract coordinates by given electrode numbers
		if self.array_spacing_um == 30:
			electrode_1 = self.electrode_positions[50 - 1]
			electrode_2 = self.electrode_positions[449 - 1]
			electrode_3 = self.electrode_positions[461 - 1]
			adjacent_electrodes = np.vstack((electrode_1, electrode_2, electrode_3))
		else:
			electrode_1 = self.electrode_positions[300 - 1]
			electrode_2 = self.electrode_positions[81 - 1]
			electrode_3 = self.electrode_positions[89 - 1]
			adjacent_electrodes = np.vstack((electrode_1, electrode_2, electrode_3))

		# scatter plot for the chosen electrodes
		plt.scatter(adjacent_electrodes[:, 0], adjacent_electrodes[:, 1], c=['#FF5733', '#FFC300', '#33C7FF'], s=20)
		plt.xlabel('X (um)')
		plt.ylabel('Y (um)')
		plt.title('Sample Stimulation Electrodes')

		plt.annotate(
			f'({electrode_1[0]}, {electrode_1[1]})',
			(electrode_1[0], electrode_1[1]),
			textcoords="offset points",
			xytext=(30, 15),  # Adjust position here
			ha='center',
			fontsize=8,
			color='#FF5733'
		)

		# annotation for Electrode 2
		plt.annotate(
			f'({electrode_2[0]}, {electrode_2[1]})',
			(electrode_2[0], electrode_2[1]),
			textcoords="offset points",
			xytext=(40, -15),  # Adjust position here
			ha='center',
			fontsize=8,
			color='#FFC300'
		)

		# annotation for Electrode 3
		plt.annotate(
			f'({electrode_3[0]}, {electrode_3[1]})',
			(electrode_3[0], electrode_3[1]),
			textcoords="offset points",
			xytext=(-30, 20),  # Adjust position here
			ha='center',
			fontsize=8,
			color='#33C7FF'
		)

		# show figure
		plt.show(block=False)

	def extract_auto_potential_slopes(self, file_path, stim_idx):
		# extract the stimulation electrode using the index
		stim_elec = self.stimulation_electrodes[stim_idx]

		# read the raw recording waveforms (recorded voltages) for this stimulation electrode
		all_recordings = CommonUtils.read_all_recordings(file_path, signal_length=self.signal_length,
														 stimulation_electrode=stim_elec,
														 start_offset=self.start_offset,
														 num_electrodes=self.num_electrodes,
														 sign_mask=self.current_sign_mask,
														 valid_current_levels=self.valid_current_indices_autopotentials,
														 signal_length_positive_phase=self.signal_length_one_phase
														 )

		# extract the auto recording for this stimulation electrode
		auto_potential_signals = all_recordings[:, stim_elec - 1, :].reshape((self.signal_length, -1), order='F')

		# process the signal according to the chosen voltage extraction method to extract one number per current
		if self.voltage_extraction_method == VoltageExtractionMethods.MAX:
			auto_potentials = np.nanmax(auto_potential_signals, axis=0)

		elif self.voltage_extraction_method == VoltageExtractionMethods.PCA:
			first_pc = CommonUtils.calculate_first_principle_component(auto_potential_signals)
			projected_data = first_pc @ auto_potential_signals
			v_min, v_max = np.nanmin(first_pc), np.nanmax(first_pc)
			scaling_factor = (v_max - v_min) / 2
			auto_potentials = projected_data * scaling_factor

		# remove current indices that are above the saturation voltage
		valid_mask = auto_potentials <= self.system_saturation_voltage_uv_ap
		auto_potentials = auto_potentials[valid_mask]
		auto_potential_signals = auto_potential_signals[:, valid_mask]
		currents_nanoamps_autopotentials = self.valid_currents_nanoamps_autopotentials[valid_mask]

		# remove nan indices before fitting
		valid_indices = ~np.isnan(auto_potentials)
		currents_nanoamps_autopotentials = currents_nanoamps_autopotentials[valid_indices]
		auto_potentials = auto_potentials[valid_indices]

		# get highest remaining current
		highest_current = np.nanmax(currents_nanoamps_autopotentials)

		# add a (0,0) point to the currents and voltages for fitting through the origin
		currents_nanoamps_autopotentials = np.insert(currents_nanoamps_autopotentials, 0, 0)
		auto_potentials = np.insert(auto_potentials, 0, 0)

		# fit the non-saturated auto potentials to a line
		fit = CommonUtils.fit_line(currents_nanoamps_autopotentials, auto_potentials, through_origin=True)

		# if the fit is reasonable, save the fitted values
		output_slope = np.nan
		if fit["R2"] >= 0.94 and fit["slope"] >= 0.5 and fit["intercept"] <= 50:
			output_slope = fit["slope"]

		# plot the auto potentials as a function of current and the fitted line
		if self.debug_mode:
			plt.figure()
			plt.plot(currents_nanoamps_autopotentials, auto_potentials, marker='x', linestyle='')
			plt.plot(fit["I_fit"], fit["V_fit"], linestyle='--', color='red')
			plt.text(0.05, 0.95, f'Slope: {output_slope:.2f} KOhm', transform=plt.gca().transAxes, fontsize=10)
			plt.xlabel('Current (nA)')
			plt.ylabel('Voltage ($\mu V$)')
			plt.title(f'Auto Potentials - stim_num={stim_elec}')
			plt.grid()
			plt.show()

			# plot the auto potentials as a function of time
			plt.figure()
			plt.plot(self.time_vector_msec, auto_potential_signals, marker='x', linestyle='')
			plt.xlabel('Time (ms)')
			plt.ylabel('Voltage ($\mu V$)')
			plt.title(f'Auto Potentials - stim_num={stim_elec}')
			plt.grid()
			plt.show()

		return output_slope, highest_current

	@staticmethod
	def get_list_of_stimulating_electrodes(path):
		# define pattern regex
		pattern = regex.compile(r'p\d+')

		# initialize output
		stimulation_electrodes_list = list()

		# get all directories in the path (data000/data001 style folder) that start with a "p" followed by a number
		for folder_name in os.listdir(path):
			if pattern.match(folder_name):
				stimulation_electrodes_list.append(int(folder_name[1:]))

		return np.array(stimulation_electrodes_list, dtype=int)
