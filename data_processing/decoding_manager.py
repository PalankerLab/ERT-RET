# standard imports
import os
import re
import math
import itertools
import numpy as np
import pandas as pd
from collections import OrderedDict
from matplotlib import pyplot as plt
from collections import defaultdict
from openpyxl.workbook import Workbook
from openpyxl.styles import PatternFill
from matplotlib.ticker import LogLocator, FuncFormatter

# project-specific imports
from data_processing.configurations import ElectrodeConfiguration, DecodingTool, GroundTruthTypes


class DecodingManager:

	def __init__(self, experiment_manager,
				 calibrated_resistances,
				 resistance_sem,
				 autopotential_distances,
				 original_autopotential_distances,
				 electrode_configuration=ElectrodeConfiguration.ALL_COMBS,
				 output_resistivity=False,
				 fixed_resistivity_regions=None,
				 fixed_boundary_regions=None,
				 user_prefix=""):

		# store provided parameters
		self.experiment = experiment_manager
		self.calibrated_resistances = calibrated_resistances
		self.resistances_sem = resistance_sem
		self.autopotential_distances = autopotential_distances
		self.original_autopotential_distances = original_autopotential_distances
		self.electrode_configuration = electrode_configuration
		self.synthetic_data = experiment_manager.use_simulated_data_as_sample
		self.output_resistivity = output_resistivity
		self.user_prefix = user_prefix
		self.fixed_resistivity_regions = fixed_resistivity_regions
		self.fixed_boundary_regions = fixed_boundary_regions

		# add constant parameters
		self.skip_rows = 7
		self.res2dinv_file_suffix = '.DAT'
		self.output_string_template = "{num_elec:<7d} {inject_x:<12} {inject_z:<12} {sense_1_x:<12} {sense_1_z:<12} {sense_2_x:<12} {sense_2_z:<12} {res:<20.8f}\n" if self.electrode_configuration.number_of_array == 11 else ""

		# add debug parameters
		self.generate_output_files = True
		self.visualize_results = False

		# get electrode locations along the injection line
		self.electrodes_grid_x = self.experiment.array_type.electrodes_grid_x

		# initialize dynamic parameters that will be updated when we know the decoding tool
		self.input_folder = None
		self.output_file_full = None

	def run_decoder(self, decoding_tool: DecodingTool = DecodingTool.RES2DINV):
		# populate dynamic tool-specific variables
		self._setup_tool_specific_params(decoding_tool)

		# generate the relevant input file for the decoding process
		if decoding_tool == DecodingTool.RES2DINV:
			self._generate_inversion_input_file()

		if decoding_tool == DecodingTool.ML_DECODER:
			self._generate_ml_input_file()

	def _setup_tool_specific_params(self, decoding_tool):
		# create the input path
		self.input_folder = self._build_input_path(decoding_tool)

		# create the input folders if they do not exist
		if not os.path.exists(self.input_folder):
			os.makedirs(self.input_folder)

		# create the output path
		self.output_file_full = self._build_output_path(decoding_tool)

	def _build_input_path(self, decoding_tool):
		# start with the tool type
		software_prefix = decoding_tool.tool_name

		# add the sample type
		sample_type = self.experiment.sample_type.lower().split('_')[0]

		# indicate the number of calibrations used
		if self.experiment.num_dilutions_for_calibration:
			num_calibrations_suffix = f"_{self.experiment.num_dilutions_for_calibration}calib"
		else:
			num_calibrations_suffix = "_no_calib"

		# concat to form suffix
		suffix = f"{sample_type}{num_calibrations_suffix}"

		# add the user's prefix
		user_prefix = f"{self.user_prefix}_" if (self.user_prefix != "" and self.user_prefix) else ""

		# construct the full input path
		return os.path.join(self.experiment.output_folder, f'{user_prefix}{software_prefix}_' + suffix)

	def _build_output_path(self, decoding_tool):
		# start with the data type prefix
		if self.experiment.use_simulated_data_as_sample:
			output_name = "sim_"
		elif self.experiment.generate_data:
			output_name = "ideal_"
		else:
			output_name = ""

		# add the sample type (saline vs. retina)
		output_name += f"{self.experiment.sample_type.lower().split('_')[0]}"

		# add the simulated file name
		if not self.experiment.use_simulated_data_as_sample:
			output_name += ""
		else:
			# output_name += f"_{self.experiment.simulated_file_name.split('.')[0]}"
			simulated_file_prefix = re.sub(r'\.[A-Za-z][A-Za-z0-9]*$', '', self.experiment.simulated_file_name)
			output_name += f"_{simulated_file_prefix}"

		# add groud truth used for calibration
		if self.experiment.num_dilutions_for_calibration == 0:
			output_name += ""
		elif self.experiment.ground_truth_type == GroundTruthTypes.ANALYTICAL:
			output_name += "_analygt"
		else:
			output_name += "_simulgt"

		# indicate the presence of auto-potential measurements
		if decoding_tool == DecodingTool.ML_DECODER:
			output_name += "_with_ap"
		elif decoding_tool == DecodingTool.RES2DINV:
			output_name += "_with_ap" if self.experiment.add_auto_potential_electrode_res2dinv else "_no_ap"
		else:
			output_name += "_no_ap"

		# get fixed regions
		self.fixed_resistivity_regions = self.fixed_resistivity_regions
		self.fixed_boundary_regions = self.fixed_boundary_regions

		# get the number of fixed regions
		num_fixed_with_resistivity = len(self.fixed_resistivity_regions) if self.fixed_resistivity_regions else 0
		num_fixed_with_boundaries = len(self.fixed_boundary_regions) if self.fixed_boundary_regions else 0

		# indicate the presence of fixed regions for Res2DInv files
		if decoding_tool == DecodingTool.RES2DINV:
			output_name += "_{}".format(self.electrode_configuration.name.lower())
			output_name += "_{}fixed_bound".format(num_fixed_with_boundaries) if num_fixed_with_boundaries > 0 else ""
			output_name += "_{}fixed_resis".format(num_fixed_with_resistivity) if num_fixed_with_resistivity > 0 else ""

		# add the number of dilutions for used for calibration
		if self.experiment.num_dilutions_for_calibration > 0:
			output_name += "_{}calib".format(self.experiment.num_dilutions_for_calibration)

		# indicate the type of saline used (for samples with before and after calibration)
		output_name += "_{}".format(self.experiment.saline_folder_suffix) if self.experiment.saline_folder_suffix else ""

		# construct the full output file name
		return os.path.join(self.input_folder, f"{output_name}{self.res2dinv_file_suffix}")

	def _generate_ml_input_file(self):
		# start excel writer
		excel_name = f'{self.experiment.index}_{os.path.basename(self.input_folder)}_distance_voltage_sem_all_stim'
		excel_name += '_cal_{}'.format(self.experiment.saline_folder_suffix) if self.experiment.saline_folder_suffix else ''
		excel_name += '.xlsx'
		excel_path = os.path.join(self.input_folder, excel_name)
		excel_writer = pd.ExcelWriter(excel_path, engine='openpyxl')

		# start a combined plot
		plt.figure()
		combined_ax = plt.gca()

		# define columns
		columns = ['stim_idx', 'read_idx', 'distance (um)', 'resistance_mean (kOhm)', 'resistance_sem (kOhm)', 'stim_x (um)', 'stim_y (um)']

		# iterate over all stimulation electrodes and generate output
		for stim_idx, stim_num in enumerate(self.experiment.stimulation_electrodes):

			# get electrode x-y coordinates
			stim_coordinates = self.experiment.electrode_positions[stim_num - 1]
			stim_x, stim_y = stim_coordinates

			# initialize rows for the data itself
			rows = []

			# iterate over all read electrodes for this stim electrode
			for read_idx in self.experiment.recording_indices:

				# extract distance, slope and sem for this stim-read pair
				distance_um, slope_kohm, slope_sem_kohm = self.resistances_sem[stim_idx, read_idx]

				# initialize distance for file entry
				distance = distance_um

				# if distance is zero, replace it by the value from the auto-potential distance
				if distance_um == 0:
					distance = self.autopotential_distances[stim_idx]

				# append this result to the output rows
				rows.append([stim_idx, read_idx, round(distance, 2), slope_kohm, slope_sem_kohm, stim_x, stim_y])

			# group by distance and create average values for each distance
			grouped = defaultdict(list)
			for r in rows:
				grouped[r[2]].append(r)

			merged_rows = []
			for dist, group in grouped.items():
				mean_resistance = np.nanmean([g[3] for g in group])
				mean_sem = np.nanmean([g[4] for g in group])
				read_indices = [g[1] for g in group]
				merged_rows.append([
					stim_idx,
					','.join(map(str, read_indices)),
					dist,
					round(mean_resistance, 2),
					round(mean_sem, 2),
					stim_x,
					stim_y
				])

			# sort by distance before printing
			merged_rows.sort(key=lambda x: (math.isnan(x[2]), x[2]))

			# output to excel
			df = pd.DataFrame(merged_rows, columns=columns)
			sheet_name = f"stim_{stim_idx}"
			df.to_excel(excel_writer, sheet_name=sheet_name, index=False)

			# name the sheet based on the stimulation electrode number
			ws = excel_writer.sheets[sheet_name]

			# find the first empty row after the table
			start_row = ws.max_row + 2

			# add metadata rows below
			ws.cell(row=start_row, column=1).value = 'resistivity (ohm*cm)'
			ws.cell(row=start_row, column=2).value = (
				f'{self.experiment.saline_dilutions[self.experiment.use_saline_as_sample]}'
				if self.experiment.use_saline_as_sample else 'retinal sample'
			)

			if self.experiment.add_auto_potential_electrode_res2dinv:
				ws.cell(row=start_row + 1, column=1).value = 'original ap distance (um)'
				ws.cell(row=start_row + 1, column=2).value = round(self.original_autopotential_distances[stim_idx], 2)

				ws.cell(row=start_row + 2, column=1).value = 'scaled ap distance (um)'
				ws.cell(row=start_row + 2, column=2).value = f'{self.experiment.auto_potential_distance_um}'

			# generate log-log plot for this stim
			if self.experiment.debug_mode:
				try:
					plt.figure(figsize=(12, 10))
					plt.plot(df['distance (um)'].iloc[1:], df['resistance_mean (kOhm)'].iloc[1:], linestyle='None',
							 marker='o', color='darkblue', markersize=12)
					plt.grid()
					plt.xscale('log')
					plt.yscale('log')

					ax = plt.gca()

					# more ticks, still log-spaced (3,6,10 within each decade)
					ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[3, 6, 10]))
					ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[3, 6, 10]))

					# format labels as plain floats, not 10^x
					ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f}'))
					ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{y:.1f}'))
					plt.tick_params(axis='both', labelsize=20)

					plt.xlabel('Distance (µm)', fontsize=24, labelpad=15)
					plt.ylabel('Average resistance (kΩ)', fontsize=24, labelpad=15)
					plt.title(f'Stim {stim_idx} - Resistance vs Distance', fontsize=18, pad=15)
					plt.tight_layout()
					plt.show()

				except Exception as e:
					print(f"Could not generate plot for stim {stim_idx}: {e}")

			# add to combined plot
			combined_ax.errorbar(
				df['distance (um)'], df['resistance_mean (kOhm)'],
				yerr=df['resistance_sem (kOhm)'], fmt='o-', capsize=3, label=f'Stim {stim_idx}'
			)

		# finalized combined plot
		try:
			combined_ax.set_xscale('log')
			combined_ax.set_yscale('log')
			combined_ax.set_xlabel('Distance (µm)')
			combined_ax.set_ylabel('Resistance Mean (kΩ)')
			combined_ax.set_title('All Stim Electrodes - Resistance vs Distance')
			combined_ax.grid(True, which="both", ls="--", lw=0.5)
			combined_ax.legend()
			combined_plot_file = os.path.join(self.experiment.output_folder, 'all_stims_loglog.png')
			plt.savefig(combined_plot_file, dpi=300, bbox_inches='tight')
			plt.close()
		except Exception as e:
			print(f"Could not generate combined plot: {e}")

		# close the excel writer and file
		excel_writer.close()

	def _generate_inversion_input_file(self):

		# extract data
		experimental_data, coords_mapping = self._extract_experimental_data()

		# extract minimal spacing between electrodes
		coords = np.array(sorted(coords_mapping.keys()))
		diffs = np.diff(coords, axis=0)
		electrode_spacing = np.linalg.norm(diffs, axis=1).min()

		# if we have auto-potential distances, and we should use them, take the min of that as the spacing
		if self.autopotential_distances is not None and self.experiment.add_auto_potential_electrode_res2dinv:
			autopotential_spacing = min(self.autopotential_distances)
			electrode_spacing = min(electrode_spacing, autopotential_spacing)

		# split output file name
		output_file_parts = self.output_file_full.split('.')
		output_file_suffix = output_file_parts[-1]
		output_file_prefix = ".".join(output_file_parts[:-1])

		# get the stim electrodes
		stim_elecs = experimental_data.keys()

		# start a new output file
		output_file = output_file_prefix + "." + output_file_suffix
		print('Generating file:', os.path.basename(output_file))

		# initialize output across al stims and currents
		output_content = list()
		output_file_contents = OrderedDict()
		excel_output_file_contents = list()

		# iterate the stim electrodes
		for _, stim_elec in enumerate(stim_elecs):

			# extract stim electrode coordinates
			stim_coordinates = self.experiment.electrode_positions[stim_elec - 1]

			# map the original coordinates to the projected coordinates
			projected_stim_coordinates = coords_mapping[tuple(stim_coordinates.tolist())]

			# extract recorded data
			recorded_data = experimental_data[stim_elec]

			# if no data is recorded for this stim electrode, skip it
			if recorded_data is None or len(recorded_data) == 0:
				continue

			# remove auto-potential measurements if requested
			if not self.experiment.add_auto_potential_electrode_res2dinv:
				recorded_data = recorded_data[recorded_data[:, 2] >= electrode_spacing]

			# average resistance across same distances
			if not self.experiment.use_simulated_data_as_sample:
				recorded_data = self._average_resistance_same_distance(recorded_data, projected_stim_coordinates)

			# calculate resistance values for this stim electrode
			points = self._resistance_data_points_constructor(self.electrode_configuration, recorded_data, projected_stim_coordinates[0])

			# update content
			output_content.extend(points)

		# sort the output data points by injection, first sensing, and second sensing electrode locations
		output_content = sorted(output_content, key=lambda x: (x[0], x[1], x[2]))

		# iterate over the output contents and format electrode sets into strings
		for data_point in output_content:

			# extract data from the data point tuple
			(injecting_electrode_x, first_sensing_x_location, second_sensing_x_location, output_res,
			 first_measured_kohm, second_measured_kohm, am_distance, an_distance, geometric_factor_um) = data_point

			# format the output string
			output_string, is_multiple_strings = self._populate_output_string(first_sensing_x_location,
																			  injecting_electrode_x,
																			  output_res,
																			  second_sensing_x_location)

			# add to output file contents
			if is_multiple_strings:
				for string in output_string:
					output_file_contents[string.strip() + "\n"] = None
			else:
				output_file_contents[output_string.strip() + "\n"] = None

			# also append to excel output (for debug only)
			excel_output_file_contents.append(data_point)

		# convert the output to list
		output_file_contents = list(output_file_contents.keys())

		# count the number of data points
		num_data_points_in_file = len(output_file_contents)

		# report number of data points
		print(f"Number of data points in file: {num_data_points_in_file}")

		# add fixed regions
		footer = self._add_fixed_regions()

		# output this configuration to the inversion's input file
		content = output_file_contents + footer
		self._write_to_dat_file(num_data_points_in_file, content, output_file, electrode_spacing, noise=False)

		# also output to an excel file for debug
		if self.experiment.debug_mode:
			output_file_excel = output_file_prefix + "_aggr" + '.xlsx'
			self._write_to_excel_file(excel_output_file_contents, output_file_excel)

		# for pure non-processed experimental data only
		if not self.experiment.use_simulated_data_as_sample:

			# create a distance-averaged file combining all stim electrodes
			all_resistances_by_distance = {}

			# reiterate through the stim electrodes to collect resistance data by distance
			for _, stim_elec in enumerate(stim_elecs):

				# extract stim electrode coordinates
				stim_coordinates = self.experiment.electrode_positions[stim_elec - 1]
				projected_stim_coordinates = coords_mapping[tuple(stim_coordinates.tolist())]

				# extract recorded data
				recorded_data = experimental_data[stim_elec]

				# if no data is recorded for this stim electrode, skip it
				if recorded_data is None or len(recorded_data) == 0:
					continue

				# calculate the distances between the stim electrode and recording electrodes
				rec_x_coords = recorded_data[:, 2]
				distances = np.abs(rec_x_coords - projected_stim_coordinates[0])

				# group the resistance values by distance
				for i, distance in enumerate(distances):
					distance_rounded = np.round(distance, 2)
					resistance_value = recorded_data[i, 4]

					if distance_rounded not in all_resistances_by_distance:
						all_resistances_by_distance[distance_rounded] = []
					all_resistances_by_distance[distance_rounded].append(resistance_value)

			# calculate the average resistance for each distance
			distance_averaged_voltages = {}
			for distance, resistances in all_resistances_by_distance.items():
				distance_averaged_voltages[distance] = np.nanmean(resistances)

			# initialize new output content with distance-averaged values
			output_content_averaged = list()
			output_file_contents_averaged = OrderedDict()

			# reprocess all the stim electrodes with averaged resistance values (replacing original with averaged)
			for _, stim_elec in enumerate(stim_elecs):

				# extract stim electrode coordinates
				stim_coordinates = self.experiment.electrode_positions[stim_elec - 1]
				projected_stim_coordinates = coords_mapping[tuple(stim_coordinates.tolist())]

				# extract recorded data
				recorded_data = experimental_data[stim_elec]

				if recorded_data is None or len(recorded_data) == 0:
					continue

				# create new recorded data with averaged resistance values
				recorded_data_averaged = recorded_data.copy()

				# iterate the recorded data points (the rows)
				for i in range(recorded_data_averaged.shape[0]):

					# calculate the distance of the stim-rec pair
					distance = abs(projected_stim_coordinates[0] - recorded_data_averaged[i, 2])
					distance_rounded = np.round(distance, 2)

					# replace the resistance value with the distance-averaged resistance value
					if distance_rounded in distance_averaged_voltages:
						recorded_data_averaged[i, 4] = distance_averaged_voltages[distance_rounded]

				# calculate resistance values for this stim electrode and current level
				points = self._resistance_data_points_constructor(self.electrode_configuration, recorded_data_averaged, projected_stim_coordinates[0])

				# update content
				output_content_averaged.extend(points)

			# sort the output data points by injection, first sensing, and second sensing electrode locations
			output_content_averaged = sorted(output_content_averaged, key=lambda x: (x[0], x[1], x[2]))

			# iterate over the unique output contents and format them into strings
			for data_point in output_content_averaged:

				# extract data from the data point tuple
				(injecting_electrode_x, first_sensing_x_location, second_sensing_x_location, output_res,
				 first_measured_kohm, second_measured_kohm, am_distance, an_distance, geometric_factor_um) = data_point

				output_string, is_multiple_strings = self._populate_output_string(first_sensing_x_location,
																			  injecting_electrode_x,
																			  output_res,
																			  second_sensing_x_location)

				# add to output file contents
				if is_multiple_strings:
					for string in output_string:
						output_file_contents_averaged[string.strip() + "\n"] = None
				else:
					output_file_contents_averaged[output_string.strip() + "\n"] = None

			# convert to list
			output_file_contents_averaged = list(output_file_contents_averaged.keys())

			# count the number of data points in the averaged file
			num_data_points_averaged = len(output_file_contents_averaged)

			# report the number of data points in the averaged file
			print(f"Number of data points in distance-averaged file: {num_data_points_averaged}")

			output_file_averaged = output_file_prefix + "_distance_averaged" + "." + output_file_suffix
			print('Generating distance-averaged file:', os.path.basename(output_file_averaged))

			# add fixed regions for the averaged file
			footer_averaged = self._add_fixed_regions()

			# output distance-averaged configuration to file
			content_averaged = output_file_contents_averaged + footer_averaged
			self._write_to_dat_file(num_data_points_averaged, content_averaged, output_file_averaged, electrode_spacing, noise=False)

			# write averaged data to an excel file for debug
			if self.experiment.debug_mode:
				output_file_excel_averaged = output_file_prefix + "_distance_averaged" + '.xlsx'
				self._write_to_excel_file(output_file_contents_averaged, output_file_excel_averaged)

	def _extract_experimental_data(self):
		# initialize output data dictionary
		output_data_dict = dict()

		# if one specific injection line is used in this experiment, get this line
		if self.experiment.injection_line_num is not None:
			injection_line = self.experiment.array_type.injection_line_electrodes[self.experiment.injection_line_num]

		# else, use the first injection line
		else:
			injection_line = self.experiment.array_type.injection_line_electrodes[0]

		# get the injection indices
		injection_line_indices = [x - 1 for x in injection_line]

		# get the coordinates of the electrodes in the injection line
		injection_line_coordinates = self.experiment.electrode_positions[injection_line_indices, :]

		# ensure no duplicate coordinates
		injection_line_coordinates, unique_idx = np.unique(injection_line_coordinates, axis=0, return_index=True)

		# filter indices to match unique coordinates
		injection_line_indices = [injection_line_indices[i] for i in unique_idx]

		# project the original coordinates to a 2d line where x is the distance along the line and y is 0
		projected_coordinates, mapping = self._project_coordinates(injection_line_coordinates)

		# iterate over the stimulating electrodes and store the coordinates and resistances of the recording electrodes
		for stim_idx, stim_num in enumerate(self.experiment.stimulation_electrodes):

			# make sure stim_num is an integer
			stim_num = int(stim_num)

			# initialize dict for this stimulating electrode
			output_data_dict[stim_num] = dict()

			# copy coordinates and indices
			temp_injection_line_coordinates = np.copy(injection_line_coordinates)
			temp_injection_line_indices = np.copy(injection_line_indices)

			# extract the stim electrode coordinates
			stim_coord = self.experiment.electrode_positions[stim_num - 1]

			# find the index of the stim electrode coordinates in the injection line coordinates
			match_index = np.where(np.all(temp_injection_line_coordinates == stim_coord, axis=1))[0]

			# remove the stim electrode coordinates from the injection line coordinates
			if ((self.autopotential_distances is None or not self.experiment.add_auto_potential_electrode_res2dinv) and
					match_index.size > 0):
				temp_injection_line_coordinates = np.delete(temp_injection_line_coordinates, match_index[0], axis=0)
				temp_injection_line_indices = np.delete(temp_injection_line_indices, match_index[0], axis=0)

			# get the final coordinates
			injection_line_coordinates_minus_stim = temp_injection_line_coordinates

			# initialize the output structure for this stimulation electrode
			measured_resistances_and_coordinates = np.full((len(injection_line_coordinates_minus_stim), 5), np.nan)

			# assign the recording coordinates
			measured_resistances_and_coordinates[:, 0:2] = injection_line_coordinates_minus_stim

			# define the projected coordinates
			projected_coordinates = np.array([mapping[tuple(pt)] for pt in injection_line_coordinates_minus_stim])

			# replace the x-coordinate of the injection electrode if we need to use auto-potentials
			if ((self.autopotential_distances is not None and self.experiment.add_auto_potential_electrode_res2dinv) and
					~np.isnan(self.autopotential_distances[stim_idx])):
				projected_coordinates[match_index, 0] = projected_coordinates[match_index, 0] + round(self.autopotential_distances[stim_idx], 2)

			# assign the projected coordinates
			measured_resistances_and_coordinates[:, 2:4] = projected_coordinates

			# extract the measured resistances
			resistances = self.calibrated_resistances[stim_idx, temp_injection_line_indices]

			# if all values are nan for this current, skip
			if np.all(np.isnan(resistances)):
				continue

			# assign the measured resistances for this stimulating electrode
			measured_resistances_and_coordinates[:, 4] = resistances

			# add the array to the output data dict
			output_data_dict[stim_num] = measured_resistances_and_coordinates

		return output_data_dict, mapping

	@staticmethod
	def _project_coordinates(original_coordinates):
		# choose two extreme points to define the direction
		start = original_coordinates[np.argmin(np.sum(original_coordinates, axis=1))]
		end = original_coordinates[np.argmax(np.sum(original_coordinates, axis=1))]

		# define the direction vector of the line
		direction = end - start
		direction = direction / np.linalg.norm(direction)

		# project the original coordinates onto the direction vector
		relative_positions = (original_coordinates - start) @ direction

		# sort based on scalar projection
		sort_idx = np.argsort(relative_positions)

		# project onto x-axis with proper spacing
		projected = np.stack((relative_positions, np.zeros_like(relative_positions)), axis=1)

		# build mapping from original coordinates to projected coordinate
		mapping = {}
		for orig_idx, sorted_idx in enumerate(sort_idx):
			key = tuple(original_coordinates[sorted_idx])
			mapping[key] = projected[orig_idx]

		return projected, mapping

	def _resistance_data_points_constructor(self, electrode_configuration, *args, **kwargs):
		if electrode_configuration == ElectrodeConfiguration.ALL_COMBS:
			return self._all_combinations(*args, **kwargs)
		if electrode_configuration == ElectrodeConfiguration.ALL_COMBS_MINIMAL:
			return self._all_combinations_minimal(*args, **kwargs)

		raise ValueError(f"Inversion type {electrode_configuration} is not supported")

	def _all_combinations(self, measured_voltage_and_coordinates, stim_x_location):

		# initialize output
		output_data_points = list()

		# generate all combinations of recording electrodes
		unique_combinations = set()
		number_of_sensing_electrodes = measured_voltage_and_coordinates.shape[0]

		# sort the combination tuples to ensure (1,2) and (2,1) are treated as the same
		for combination in itertools.combinations(range(number_of_sensing_electrodes), 2):
			sorted_combination = tuple(sorted(combination))
			unique_combinations.add(sorted_combination)

		# iterate over all combinations and calculate their resistance values
		for combination in unique_combinations:

			# extract indices
			first_index, second_index = combination

			# first sensing electrode data
			first_sensing_x_location = measured_voltage_and_coordinates[first_index, 2]
			first_measured_kohm = measured_voltage_and_coordinates[first_index, 4]

			# second sensing electrode data
			second_sensing_x_location = measured_voltage_and_coordinates[second_index, 2]
			second_measured_kohm = measured_voltage_and_coordinates[second_index, 4]

			# skip if any of the values is nan
			if math.isnan(first_measured_kohm) or math.isnan(second_measured_kohm) or math.isnan(
					first_sensing_x_location) or math.isnan(second_sensing_x_location):
				continue

			# skip if the injection electrode is in between the two sensing electrodes
			if first_sensing_x_location < stim_x_location < second_sensing_x_location or second_sensing_x_location < stim_x_location < first_sensing_x_location:
				continue

			# fix resistivity sign, if needed or convert to apparent resistivity
			output_value, injecting_electrode_x_new, first_sensing_x_location_new, second_sensing_x_location_new = (self._calculate_resistance_or_resistivity(stim_x_location, first_sensing_x_location,
														  second_sensing_x_location, first_measured_kohm, second_measured_kohm))

			# skip negative or zero resistivity values
			if output_value is None:
				continue

			# round value to 2 decimal places
			output_value = round(output_value, 2)

			# and add to output
			output_data_points.append((injecting_electrode_x_new, first_sensing_x_location_new, second_sensing_x_location_new, output_value, first_measured_kohm, second_measured_kohm))

		return output_data_points

	def _all_combinations_minimal(self, measured_voltage_and_coordinates, stim_x_location):
		# initialize output
		output_data_points = list()

		# generate all combinations of sensing electrodes
		unique_combinations = set()
		number_of_sensing_electrodes = measured_voltage_and_coordinates.shape[0]

		# build a minimal set of n-1 pairs in a chain
		for i in range(number_of_sensing_electrodes - 1):
			unique_combinations.add((i, i + 1))

		# iterate over the combinations and calculate their resistance values
		for combination in unique_combinations:

			# extract indices
			first_index, second_index = combination

			# first sensing electrode data
			first_sensing_x_location = measured_voltage_and_coordinates[first_index, 2]
			first_measured_kohm = measured_voltage_and_coordinates[first_index, 4]

			# second sensing electrode data
			second_sensing_x_location = measured_voltage_and_coordinates[second_index, 2]
			second_measured_kohm = measured_voltage_and_coordinates[second_index, 4]

			# skip if any of the values is nan
			if math.isnan(first_measured_kohm) or math.isnan(second_measured_kohm) or math.isnan(
					first_sensing_x_location) or math.isnan(second_sensing_x_location):
				continue

			# skip if the injection electrode is in between the two sensing electrodes
			if first_sensing_x_location < stim_x_location < second_sensing_x_location or second_sensing_x_location < stim_x_location < first_sensing_x_location:
				continue

			# fix resistivity sign, if needed or convert to apparent resistivity
			(output_value, injecting_electrode_x_new, first_sensing_x_location_new, second_sensing_x_location_new,
			 am_distance, an_distance, geometric_factor_um) = (
				self._calculate_resistance_or_resistivity(stim_x_location, first_sensing_x_location,
														second_sensing_x_location, first_measured_kohm, second_measured_kohm))

			# skip negative or zero resistivity values
			if output_value is None:
				continue

			# round value to 2 decimal places
			output_value = round(output_value, 2)

			# and add to output
			output_data_points.append((injecting_electrode_x_new, first_sensing_x_location_new,
									   second_sensing_x_location_new, output_value, first_measured_kohm,
									   second_measured_kohm, am_distance, an_distance, geometric_factor_um))

		return output_data_points

	def _calculate_resistance_or_resistivity(self, injection_x, first_sensing_x, second_sensing_x, first_measured_kohm,
											second_measured_kohm):
		# store local copies of the input values
		injection_x_inner = round(injection_x, 2)
		first_sensing_x_inner = round(first_sensing_x, 2)
		second_sensing_x_inner = round(second_sensing_x, 2)
		first_measured_kohm_inner = first_measured_kohm
		second_measured_kohm_inner = second_measured_kohm

		# calculate the distances between the injection electrode to the first and second recording electrodes
		am_distance = round(abs(injection_x_inner - first_sensing_x_inner), 2)
		an_distance = round(abs(injection_x_inner - second_sensing_x_inner), 2)

		# if the first sensing electrode is closer to the injection electrode than the second sensing electrode,
		# swap them to avoid negative geometric factor values
		if am_distance > an_distance:

			# swap recording electrodes coordinates
			new_first_sensing_x = second_sensing_x_inner
			second_sensing_x_inner = first_sensing_x_inner
			first_sensing_x_inner = new_first_sensing_x

			# swap measured resistances
			new_first_measured_kohm = second_measured_kohm_inner
			second_measured_kohm_inner = first_measured_kohm_inner
			first_measured_kohm_inner = new_first_measured_kohm

			# recalculate distances
			am_distance = round(abs(injection_x_inner - first_sensing_x_inner), 2)
			an_distance = round(abs(injection_x_inner - second_sensing_x_inner), 2)

		# calculate the resistance between two recording electrodes
		delta_resistance_in_kohm = first_measured_kohm_inner - second_measured_kohm_inner

		# scale for better inversion results with Res2DInv (bigger and smaller values are less stable)
		resistance_ohm = delta_resistance_in_kohm / 10

		# calculate the geometric factor
		geometric_factor_um = round(2 * np.pi * (1 / am_distance - 1 / an_distance) ** -1, 2)

		# calculate apparent resistivity in ohm*micro-meter
		app_resistivity_ohm_micrometer = round(resistance_ohm * geometric_factor_um)

		# convert to ohm*cm for debugging
		if self.experiment.debug_mode:
			app_resistivity_ohm_centimeter = round(app_resistivity_ohm_micrometer * 1E-4)
			print(app_resistivity_ohm_centimeter)

		# set return values
		app_resistivity = app_resistivity_ohm_micrometer
		resistance = resistance_ohm

		# remove negative resistivity or resistance values as the injected current is positive
		if (first_measured_kohm_inner < 0 or second_measured_kohm_inner < 0) or app_resistivity <= 0:
			resistance = None
			app_resistivity = None

		# return the calculated resistance or resistivity
		if self.output_resistivity:
			return app_resistivity, injection_x_inner, first_sensing_x_inner, second_sensing_x_inner, am_distance, an_distance, geometric_factor_um

		return resistance, injection_x_inner, first_sensing_x_inner, second_sensing_x_inner, am_distance, an_distance, geometric_factor_um

	def _populate_output_string(self, first_sensing_x_location, injecting_electrode_x, output_res,
								second_sensing_x_location):
		# define the number of electrodes in each set in the output file
		num_electrodes = self.electrode_configuration.number_of_electrodes

		# initialize multiple output string flag
		is_multiple_strings = False

		# get the z-coordinate for this array (should be zero for all now)
		z_value = self.experiment.array_type.z_location

		# format output string
		output_string = self.output_string_template.format(
			num_elec=num_electrodes,
			inject_x=injecting_electrode_x, inject_z=z_value,
			sense_1_x=first_sensing_x_location, sense_1_z=z_value,
			sense_2_x=second_sensing_x_location, sense_2_z=z_value,
			res=output_res)

		return output_string, is_multiple_strings

	def _add_fixed_regions(self):

		# initialize footer
		footer = list()

		# add fixed resistivities
		footer += self._add_fixed_resistivity_regions()

		# add fixed boundaries
		footer += self._add_fixed_boundary_regions()

		return footer

	def _add_fixed_boundary_regions(self):

		# initialize output
		output = list()

		# skip if no fixed regions
		if len(self.fixed_boundary_regions) == 0:
			return output

		# get the number of boundaries to fix
		number_of_boundary_boundaries = len(self.fixed_boundary_regions)

		# topography flag, 0 for no topography, 1 for topography
		output.append("0\n")

		# Number of regions to fix, put 0 if none
		output.append("0\n")

		# no aquatic survey
		output.append("0\n")

		# start assembling the footer
		output.append("Sharp layers present\n")
		output.append("Number of boundaries\n")
		output.append("{}\n".format(number_of_boundary_boundaries))

		# iterate over the boundaries to fix
		for index, boundary in enumerate(self.fixed_boundary_regions):

			# add default parameters
			output.append("Boundary {} parameters\n".format(index + 1))
			output.append("Boundary type\n")
			output.append("Layer\n")

			# don't fix resistivity above the fixed region
			output.append("Group Weight\n")
			output.append("1.0\n")

			# continue by defining the boundary of the region
			output.append("Number of points on boundary\n")
			output.append("3\n")
			output.append("x and z coordinates of points\n")

			# extract x boundaries for this region
			x_start = self.experiment.array_type.x1_of_fixed_region
			x_end = self.experiment.array_type.x2_of_fixed_region

			# add the middle of the x range
			x_middle = (x_start + x_end) / 2

			# extract z boundary for this region
			z = boundary

			# x,z start (far-left corner)
			output.append("{}, {}\n".format(x_start, z))

			# x,z middle
			output.append("{}, {}\n".format(x_middle, z))

			# x,z end (far-right corner)
			output.append("{}, {}\n".format(x_end, z))

		return output

	def _add_fixed_resistivity_regions(self):
		# initialize output
		output = list()

		# extract fixed regions
		fixed_regions = self.fixed_resistivity_regions

		# get the number of regions to fix
		number_of_fixed_regions = len(fixed_regions)

		# add fixed regions at the end of the file
		# topography flag, 0 for no topography, 1 for topography
		output.append("0\n")

		# number of regions to fix, put 0 if none
		output.append("{}\n".format(number_of_fixed_regions))

		# iterate over the regions to fix
		for region in range(number_of_fixed_regions):

			# extract fixation parameters for this region
			x1_of_fixed_region = fixed_regions[region]["x1"] if "x1" in fixed_regions[region] else self.experiment.array_type.x1_of_fixed_region
			x2_of_fixed_region = fixed_regions[region]["x2"] if "x2" in fixed_regions[region] else self.experiment.array_type.x2_of_fixed_region

			y1_of_fixed_region = fixed_regions[region]["y1"] if "y1" in fixed_regions[region] else None
			y2_of_fixed_region = fixed_regions[region]["y2"] if "y2" in fixed_regions[region] else None

			region_resistivity = fixed_regions[region]["resistivity"] if fixed_regions[region]["resistivity"] else 0.68
			damping_factor = fixed_regions[region]["weight"] if fixed_regions[region]["weight"] else 10.0

			# adding the fixed region
			# type of region, R for rectangular
			output.append("R\n")

			# x,y,z top-left corner
			y1_of_fixed_region = "{},".format(y1_of_fixed_region) if y1_of_fixed_region else ""
			output.append("{},{}{}\n".format(x1_of_fixed_region, y1_of_fixed_region, fixed_regions[region]["z1"]))

			# x,z bottom-right corner
			y2_of_fixed_region = "{},".format(y2_of_fixed_region) if y2_of_fixed_region else ""
			output.append("{},{}{}\n".format(x2_of_fixed_region, y2_of_fixed_region, fixed_regions[region]["z2"]))

			# resistivity value of region
			output.append("{}\n".format(region_resistivity))

			# damping factor weight
			output.append("{}\n".format(damping_factor))

		return output

	def _write_to_dat_file(self, number_of_data_points, output_file_contents, output_file_full, electrode_spacing):
		# create the file's header
		output_name = os.path.basename(output_file_full).split('.')[0]
		header_lines = self._generate_file_header(number_of_data_points, output_name, electrode_spacing)

		# put everything together
		output = header_lines + output_file_contents

		# add zeros at the very end of the file
		trailing_zeros = "0,0,0,0,0"
		output.append(trailing_zeros)

		# write generated output lines to the output file
		with open(output_file_full, "w") as text_file:
			for line in output:
				text_file.write(line)

	def _generate_file_header(self, number_of_data_points, output_name, electrode_spacing=None):
		# initialize output structure
		header_lines = list()

		# create the file's header
		header_lines.append(f"{output_name}\n")

		# electrode spacing in the x-direction
		electrode_spacing = electrode_spacing if electrode_spacing is not None else self.experiment.array_type.effective_electrode_spacing_um
		header_lines.append("{}\n".format(str(electrode_spacing)))

		# array number
		header_lines.append("{}\n".format(self.electrode_configuration.number_of_array))

		# sub-array number
		header_lines.append("0\n")

		# measurement type (0=app. resistivity,1=resistance)
		header_lines.append("Type of measurement (0=app. resistivity,1=resistance)\n")
		header_lines.append("{}\n".format("0" if self.output_resistivity else "1"))

		# declare the number of data points
		header_lines.append(f"{number_of_data_points}\n")

		# 1 for actual horizontal distances
		header_lines.append("1\n")
		header_lines.append("0\n")

		return header_lines

	def _write_to_excel_file(self, output_file_contents, output_file_full):
		# create a workbook and a sheet
		wb = Workbook()
		ws = wb.active
		ws.title = "data"

		# define the headers
		headers = [
			"N_elec", "Ax", "Ay", "Mx", "My", "Nx", "Ny",  "Res_1 [kOhm]", "Res_2 [kOhm]", "Res [kOhm]", "Res [Ohm]",
			"AM", "AN", "Geometric [um]", "Apparent resistivity"
		]
		ws.append(headers)

		# define the headers' colors
		orange_fill = PatternFill(start_color="FFA500", end_color="FFA500", fill_type="solid")
		green_fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")

		# apply colors to headers
		for col, header in enumerate(headers, start=1):
			cell = ws.cell(row=1, column=col)
			if header in ["AM", "AN", "Geometric [um]", "Apparent resistivity"]:
				cell.fill = green_fill
			else:
				cell.fill = orange_fill

		# initialize params
		N_elec = self.electrode_configuration.number_of_electrodes
		Ay = self.experiment.array_type.z_location
		My = self.experiment.array_type.z_location
		Ny = self.experiment.array_type.z_location

		# iterate over the data points
		for point in output_file_contents:

			# unpack values
			Ax, Mx, Nx, resis, res_kohm_1, res_kohm_2, AM, AN, geometric = map(float, point)

			# calculate the apparent resistivity
			if self.output_resistivity:
				apparent = resis
				Res_ohm = resis / geometric if not math.isnan(geometric) else float('nan')
			else:
				Res_ohm = resis
				apparent = Res_ohm * geometric if not math.isnan(geometric) else float('nan')

			# add resistance in kOhm
			Res_kohm = Res_ohm / 1000

			# append row
			ws.append([N_elec, Ax, Ay, Mx, My, Nx, Ny, res_kohm_1, res_kohm_2, Res_kohm, Res_ohm, AM, AN, geometric,
					   apparent])

		# save the excel file
		wb.save(output_file_full)

	@staticmethod
	def _average_resistance_same_distance(recorded_data, stim_xy):
		# copy recorded data
		output_recorded_data = recorded_data.copy()

		# get the stimulation coordinates
		xs, ys = stim_xy

		# get the recording coordinates
		rx, ry = recorded_data[:, 2], recorded_data[:, 3]

		# calculate the distances between the stimulation electrode and all recording electrode
		dist = np.sqrt((rx - xs) ** 2 + (ry - ys) ** 2)

		# round the distance for precision
		dist_round = np.round(dist, 1)

		# for each distance, calculate the average of the recorded values
		for d in np.unique(dist_round):
			idx = np.where(dist_round == d)[0]

			# if there are two values for the same distance, average them
			if len(idx) == 2:

				# calculate the average of the two recorded values
				avg = (recorded_data[idx[0], 4] + recorded_data[idx[1], 4]) / 2

				# assign the average resistance value to each of the two electrodes
				output_recorded_data[idx[0], 4] = avg
				output_recorded_data[idx[1], 4] = avg

		return output_recorded_data


