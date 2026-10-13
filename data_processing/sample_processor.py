# standard imports
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
from matplotlib.ticker import LogLocator, FuncFormatter

# project-specific imports
from data_processing.common_utils import CommonUtils
from data_processing.configurations import VoltageExtractionMethods


class SampleDataProcessor:

	def __init__(self, experiment_manager, gain_coefficients, gain_coefficients_per_dilution,
				 auto_potential_xy_functions, xy_ratio_fit_function, xy_gain_values):
		# save given parameters
		self.experiment = experiment_manager
		self.common_gains = gain_coefficients
		self.raw_gains = gain_coefficients_per_dilution
		self.auto_potential_xy_functions = auto_potential_xy_functions
		self.xy_ratio_fit_function = xy_ratio_fit_function
		self.xy_gain_values = xy_gain_values

		# add calculated parameters
		self.input_folder = self.experiment.sample_data_path
		self.use_saline_as_sample = self.experiment.use_saline_as_sample
		self.use_simulated_as_sample = self.experiment.use_simulated_data_as_sample

		# define a file for recording and reading out slopes scatter for noise simulations
		self.slopes_spread_file = os.path.join(self.experiment.output_folder, "slopes_spread.csv")

	def process_sample_data(self, input_folder=None):

		# report progress
		print('Processing sample data')

		# if the input folder is not provided, use the default one
		input_folder = input_folder if input_folder else self.input_folder

		# calculate adjusted currents using auto-potentials and current ratios
		current_ratios, adjusted_currents, resistivities = self._calculate_adjusted_currents()

		# use adjusted currents to extract stim-rec slopes
		measured_slopes_kohm, measured_slopes_kohm_sem, output_autopotential_distances, original_autopotential_distances_um = self._extract_stim_rec_slopes(input_folder, adjusted_currents, resistivities)

		# correct the slopes using the gains
		if self.experiment.correct_gains:
			measured_slopes_kohm, measured_slopes_kohm_sem = self._calibrate_measured_slopes(measured_slopes_kohm,
																							 measured_slopes_kohm_sem, resistivities, adjusted_currents)

		# plot the slopes as a function of distance
		self._plot_slopes_vs_distance(measured_slopes_kohm)

		# return calibrated_voltages_uv, measured_slopes_kohm, measured_slopes_kohm_sem, output_autopotential_distances, original_autopotential_distances_um
		return measured_slopes_kohm, measured_slopes_kohm_sem, output_autopotential_distances, original_autopotential_distances_um

	def _calculate_adjusted_currents(self):

		# initialize an output array
		adjusted_currents = np.full((self.experiment.num_stimulation_electrodes, self.experiment.num_valid_currents), np.nan)
		current_ratios = np.full(self.experiment.num_stimulation_electrodes, np.nan)
		resistivities = np.full(self.experiment.num_stimulation_electrodes, np.nan)

		# iterate over the stimulation electrodes
		for stim_idx, stim_num in enumerate(self.experiment.stimulation_electrodes):

			# if we are asked to correct the currents and this is experimental data, proceed
			if (not (self.experiment.generate_data and (self.experiment.use_saline_as_sample or self.experiment.use_simulated_data_as_sample)) and self.experiment.correct_currents):

				# extract the x-y function for this stimulation electrode
				auto_potential_fit_x = self.auto_potential_xy_functions[stim_idx, 0]
				auto_potential_fit_y = self.auto_potential_xy_functions[stim_idx, 1]

				# extract the ratio fit function
				ratio_fit_rho_values = self.xy_ratio_fit_function[stim_idx, 0]
				ratio_fit_ratio_values = self.xy_ratio_fit_function[stim_idx, 1]

				# if the auto-potential fit is not available, skip this stimulation electrode
				if np.all(np.isnan(auto_potential_fit_x)) or np.all(np.isnan(auto_potential_fit_y)):
					continue

				# extract auto-potential slopes
				auto_potential_slope, highest_current_nA = self.experiment.extract_auto_potential_slopes(self.input_folder, stim_idx)

				# initialize resistivity and current ratio
				resistivity_ohm_cm = np.nan
				ratio = np.nan

				# get the matching resistivity for the auto-potential slope
				if auto_potential_slope != np.nan:

					# find the index of the y value closest to the auto-potential slope
					closest_index = np.argmin(np.abs(auto_potential_fit_y - auto_potential_slope))

					# get the autopotential slope at this index
					y_at_closest_index = auto_potential_fit_y[closest_index]

					# calculate the distance to the measured auto-potential slope
					distance_between_slopes_kohm = np.abs(y_at_closest_index - auto_potential_slope)
					distance_between_slopes_uV = distance_between_slopes_kohm * highest_current_nA

					# if distance in uV is twice the noise, skip
					if distance_between_slopes_uV > 2 * self.experiment.system_noise_floor_uv:
						continue

					# extract the corresponding resistivity
					resistivity_ohm_cm = auto_potential_fit_x[closest_index]

				# find the current ratio for this resistivity
				if resistivity_ohm_cm != np.nan:

					# plot on the fit function graph
					if self.experiment.debug_mode:

						plt.figure(figsize=(12, 10))
						ax = plt.gca()

						# plot saved auto-potential fit function
						plt.plot(auto_potential_fit_x, auto_potential_fit_y, label="Calibration Curve", linestyle='-',
								 linewidth=3)

						# add vertical and horizontal lines
						plt.axhline(y=auto_potential_slope, color='r', linestyle='--')
						plt.axvline(x=resistivity_ohm_cm, color='purple', linestyle='--')

						# add text annotations
						ax.text(resistivity_ohm_cm, 0.95, f"Resistivity: {resistivity_ohm_cm:.0f} Ω·cm",
								transform=ax.get_xaxis_transform(), fontsize=18, color='purple', rotation=90, va='top',
								ha='right')
						ax.text(0.98, auto_potential_slope, f"Resistance: {auto_potential_slope:.0f} kΩ",
								transform=ax.get_yaxis_transform(), fontsize=18, color='r', va='bottom', ha='right')
						plt.xlabel("Resistivity (Ω·cm)", fontsize=24, labelpad=15)
						plt.ylabel("Auto-Resistance (kΩ)", fontsize=24, labelpad=15)
						plt.title(f"Auto-Resistance Extraction (Stim {stim_num})", fontsize=18, pad=15)
						plt.tick_params(axis='both', which='major', labelsize=18)
						plt.legend(fontsize=20)
						plt.grid()
						plt.tight_layout()
						plt.show()

					# store the resistivity for this stimulation electrode
					resistivities[stim_idx] = resistivity_ohm_cm

					# find the closest resistivity in the ratio_fit function
					closest_index = np.argmin(np.abs(ratio_fit_rho_values - resistivity_ohm_cm))

					# check the distance to the measured resistivity
					rho_at_closest_index = ratio_fit_rho_values[closest_index]
					if np.abs(rho_at_closest_index - resistivity_ohm_cm) > 20:
						continue

					# extract the corresponding ratio
					ratio = ratio_fit_ratio_values[closest_index]

				# if ratio is not nan or zero, calculate the adjusted currents
				if ratio is not np.nan and ratio != 0:

					# plot on the fit function graph
					if self.experiment.debug_mode:
						plt.figure(figsize=(12, 10))
						ax = plt.gca()

						# plot saved ratio fit function
						plt.plot(ratio_fit_rho_values, ratio_fit_ratio_values, linewidth=3, label="Calibration Curve")

						# add vertical and horizontal lines
						plt.axvline(x=resistivity_ohm_cm, color='purple', linestyle='--')
						plt.axhline(y=ratio, color='magenta', linestyle='--')

						# add text annotations
						ax.text(0.98, ratio, f"Ratio: {ratio:.2f}",
								transform=ax.get_yaxis_transform(), fontsize=18, color='magenta',
								va='bottom', ha='right')

						ax.text(resistivity_ohm_cm, 0.95, "",
								transform=ax.get_xaxis_transform(), fontsize=18, color='purple',
								rotation=90, va='top', ha='right')

						plt.xlabel("Resistivity (Ω·cm)", fontsize=24, labelpad=15)
						plt.ylabel("Current Ratio", fontsize=24, labelpad=15)
						plt.title(f"Ratio Extraction (Stim {stim_num})", fontsize=18, pad=15)
						plt.tick_params(axis='both', which='major', labelsize=20)
						plt.legend(fontsize=20)
						plt.grid(True)
						plt.tight_layout()
						plt.show()

					# save the current ratio and adjusted currents
					current_ratios[stim_idx] = ratio
					adjusted_currents[stim_idx, :] = self.experiment.valid_currents_nanoamps * ratio

			# for ideal data, the ratio is 1, and the adjusted currents are the same as original valid currents
			else:
				current_ratios[stim_idx] = 1
				adjusted_currents[stim_idx, :] = self.experiment.valid_currents_nanoamps
				resistivities[stim_idx] = self.experiment.saline_dilutions[self.experiment.use_saline_as_sample] if self.experiment.use_saline_as_sample else np.nan

		# for experimental data, plot the resistivity values as a function of stimulation electrode location
		if self.experiment.debug_mode and not (self.experiment.generate_data or
											   self.experiment.use_saline_as_sample or self.experiment.use_simulated_data_as_sample):

			# get the coordinates for stimulation electrodes with valid resistivity values
			valid_stim_indices = np.where(~np.isnan(resistivities))[0]

			# if there are valid stimulation electrodes, proceed to create plots
			if len(valid_stim_indices) > 0:
				stim_coordinates_list = []
				valid_resistivity_values = []

				for idx in valid_stim_indices:
					stim_elec = self.experiment.stimulation_electrodes[idx]
					stim_coordinates = self.experiment.electrode_positions[stim_elec - 1]
					stim_coordinates_list.append(stim_coordinates)
					valid_resistivity_values.append(resistivities[idx])

				stim_coordinates_array = np.array(stim_coordinates_list)

				# create the colormap plot
				if self.experiment.debug_mode:
					plt.figure(figsize=(10, 8))

					# plot all electrode positions in light gray
					plt.scatter(self.experiment.electrode_positions[:, 0],
							   self.experiment.electrode_positions[:, 1],
							   c='lightgray', s=10, alpha=0.5, label='All electrodes')

					# create a scatter plot with resistivity colormap
					scatter = plt.scatter(stim_coordinates_array[:, 0],
										 stim_coordinates_array[:, 1],
										 c=valid_resistivity_values,
										 cmap='viridis',
										 s=50,
										 edgecolors='black',
										 linewidth=0.5)

					# add colorbar
					colorbar = plt.colorbar(scatter)
					colorbar.set_label('Resistivity (Ω·cm)', fontsize=12)

					# add labels and title
					plt.xlabel('X Position (μm)', fontsize=12)
					plt.ylabel('Y Position (μm)', fontsize=12)
					plt.title('Decoded Resistivity Values by Stimulation Electrode Location', fontsize=14)
					plt.legend()
					plt.grid(True, alpha=0.3)
					plt.tight_layout()
					plt.show()

		# return the adjusted currents
		return current_ratios, adjusted_currents, resistivities

	def _extract_stim_rec_slopes(self, input_folder, adjusted_currents_all_stim=None, resistivities=None):
		# initialize output arrays
		output_slopes = np.full((self.experiment.num_stimulation_electrodes, self.experiment.num_electrodes), np.nan)
		output_slopes_sem = np.full((self.experiment.num_stimulation_electrodes, self.experiment.num_electrodes, 3), np.nan)
		output_autopotential_distances_um = np.full(self.experiment.num_stimulation_electrodes, np.nan)
		original_autopotential_distances_um = np.full(self.experiment.num_stimulation_electrodes, np.nan)

		# set figure limit per stim
		max_figure_per_stim = 5

		# iterate over stimulation electrodes
		for stim_idx, stim_num in enumerate(self.experiment.stimulation_electrodes):

			# extract the adjusted currents for this stimulation electrode
			adjusted_currents = self.experiment.valid_currents_nanoamps
			if adjusted_currents_all_stim is not None:
				adjusted_currents = adjusted_currents_all_stim[stim_idx, :]

			# if all currents are nan, skip this stimulation electrode
			if np.all(np.isnan(adjusted_currents)):
				continue

			# extract the resistivity sensed by this stimulation electrode
			resistivity = resistivities[stim_idx]

			# if resistivity is nan and this is experimental data, skip this stimulation electrode
			if np.isnan(resistivity) and not self.experiment.use_simulated_data_as_sample:
				continue

			# initialize a plotting counter
			stim_plotted_counter = 0

			# calculate the distance between this stimulation electrode and all recording electrodes
			distances_um = la.norm(self.experiment.electrode_positions - self.experiment.electrode_positions[stim_num - 1, :], axis=1)

			# round distances to 2 decimals
			distances_um = np.round(distances_um, 2)

			# read experimental data
			if not self.experiment.use_simulated_data_as_sample:

				# read the raw recording waveforms (recorded voltages) for this injection
				all_recordings = CommonUtils.read_all_recordings(input_folder,
																 signal_length=self.experiment.signal_length,
																 stimulation_electrode=stim_num,
																 start_offset=self.experiment.start_offset,
																 num_electrodes=self.experiment.num_electrodes,
																 sign_mask=self.experiment.current_sign_mask,
																 valid_current_levels=self.experiment.valid_current_indices,
																 signal_length_positive_phase=self.experiment.signal_length_one_phase
																 )

				# if all recordings are zero, skip this stimulation electrode
				if np.all(all_recordings == 0):
					continue

			# for each stimulation electrode, iterate over all recording electrodes
			for rec_idx in self.experiment.recording_indices:

				# print progress
				print(f'{stim_num}: {rec_idx + 1}')

				# extract the distance
				distance = distances_um[rec_idx]

				# initialize output values and plotting values
				slope = np.nan
				slope_sem = np.nan
				valid_currents_nanoamps = np.array([np.nan])

				# skip noisy distant electrodes, if requested
				if self.experiment.filter_distant_elecs_um and distance > self.experiment.filter_distant_elecs_um:
					output_slopes_sem[stim_idx, rec_idx] = (distance, np.nan, np.nan)
					continue

				# this section is for handling simulated or analytical data
				if self.experiment.generate_data:

					# initialize voltages
					ideal_voltage_all_currents_uv = None

					# auto-potential
					if stim_num - 1 == rec_idx:
						distance = self.experiment.auto_potential_distance_um
						output_autopotential_distances_um[stim_idx] = distance

					# analytical: calculate the ideal analytical voltage for all currents for this resistivity and
					# distance
					if self.experiment.use_saline_as_sample:
						saline_resistivity = self.experiment.saline_dilutions[self.experiment.use_saline_as_sample]
						ideal_voltage_all_currents_uv = CommonUtils.calculate_theoretical_voltage(saline_resistivity,
																									  adjusted_currents,
																									  distance)
					# simulated: extract the simulated voltage for this distance
					if self.experiment.use_simulated_data_as_sample:
						slope = self.experiment.simulated_data.get(distance, np.nan)
						slope_sem = 0.0

					# set the output voltage to the ideal voltage
					extracted_voltage_per_current_uv = ideal_voltage_all_currents_uv
					measured_voltage_all_currents_uv = None
					measured_voltage_all_currents_uv_for_plotting = None

				# this section is for experimental data
				else:

					# extract measured voltage from this rec electrode. This array is of shape (signal_length, N_currents)
					measured_voltage_all_currents_uv = all_recordings[:, rec_idx, :].reshape((self.experiment.signal_length, -1), order='F')

					# if the measured voltage is all zeros, record the distance and skip this pair
					if not measured_voltage_all_currents_uv.any() or np.all(np.isnan(measured_voltage_all_currents_uv)):
						output_slopes_sem[stim_idx, rec_idx] = (distance, np.nan, np.nan)
						continue

					# save full signal for plotting
					measured_voltage_all_currents_uv_for_plotting = measured_voltage_all_currents_uv

					# if we are only using the positive phase, cut the signals
					if self.experiment.use_positive_phase:
						measured_voltage_all_currents_uv = measured_voltage_all_currents_uv[:self.experiment.signal_length_one_phase]

					# if the max value in the signal is saturated, set the signal to nan
					max_per_current = np.nanmax(measured_voltage_all_currents_uv, axis=0)
					mask = (max_per_current > self.experiment.system_saturation_voltage_uv)
					measured_voltage_all_currents_uv[:, mask] = np.nan

					# extract a single voltage value for each current level
					if self.experiment.voltage_extraction_method == VoltageExtractionMethods.MAX:
						extracted_voltage_per_current_uv = np.nanmax(measured_voltage_all_currents_uv, axis=0)
					else:
						raise ValueError("Invalid voltage extraction method.")

					# calculate the distance of the auto-potential recording
					if distance == 0:

						# calculate the AP electrode location arising from the measured data
						original_autopotential_distances_um[stim_idx] = np.nanmean(adjusted_currents_all_stim[stim_idx] * resistivity * 1E1 / (2 * np.pi * extracted_voltage_per_current_uv))
						output_autopotential_distances_um[stim_idx] = original_autopotential_distances_um[stim_idx]

						# find scaling factor compared to the hard-coded auto-potential distance
						if self.experiment.scale_auto_potential_electrode and not self.experiment.use_simulated_data_as_sample:
							if self.experiment.auto_potential_distance_um > 0 and original_autopotential_distances_um[stim_idx] < self.experiment.auto_potential_distance_um:
								output_autopotential_distances_um[stim_idx] = self.experiment.auto_potential_distance_um
								scaling_factor = self.experiment.auto_potential_distance_um / original_autopotential_distances_um[stim_idx]
								extracted_voltage_per_current_uv /= scaling_factor

				# this will handle experimental data where the slope was not yet defined
				if slope is np.nan and slope_sem is np.nan:

					# find indices where the extracted voltage is not nan
					valid_indices = ~np.isnan(extracted_voltage_per_current_uv)

					# check if there are any valid indices
					if np.any(valid_indices):
						measured_voltage_all_currents_uv = measured_voltage_all_currents_uv[:, valid_indices] if measured_voltage_all_currents_uv is not None else None
						extracted_voltage_per_current_uv = extracted_voltage_per_current_uv[valid_indices]
						valid_currents_nanoamps = adjusted_currents[valid_indices]

					# no valid indices left
					else:
						measured_voltage_all_currents_uv = np.array([np.nan]) if measured_voltage_all_currents_uv is not None else None
						extracted_voltage_per_current_uv = np.array([np.nan])
						valid_currents_nanoamps = np.array([np.nan])

					# if the remaining extracted voltages are all nan, skip this pair
					if np.all(np.isnan(extracted_voltage_per_current_uv)) or len(extracted_voltage_per_current_uv) == 0:
						continue

					# fit data to a line
					fit = CommonUtils.fit_line(valid_currents_nanoamps, extracted_voltage_per_current_uv, through_origin=False)

					# extract slope
					slope = fit["slope"]
					slope_sem = fit["slope_sem"]

					# add the slope and the distance to the csv file
					if self.experiment.use_saline_as_sample:
						CommonUtils.append_slope_measurement_to_file(self.slopes_spread_file, distance, slope)

					# if the intercept is too high, skip this pair
					if fit["intercept"] >= 50:
						output_slopes_sem[stim_idx, rec_idx] = (distance, np.nan, np.nan)
						continue

				# save the slope
				output_slopes[stim_idx, rec_idx] = slope
				output_slopes_sem[stim_idx, rec_idx] = (distance, slope, slope_sem)

				# plot the recordings for selected electrodes
				if self.experiment.debug_mode:

					# if we have plotted enough figures for this stim electrode, skip plotting
					if stim_plotted_counter >= max_figure_per_stim:
						continue

					if (round(distance) == 0
							or round(distance) == 30
							or round(distance) == 90
							or round(distance) == 330
							or round(distance) == 450
							or round(distance) == 34
							or round(distance) == 805
							or round(distance) == 210
							or round(distance) == 60
							or round(distance) == 270):

						# define a color cycle
						colors = plt.cm.tab10.colors

						# set the color cycle for the plots to maintain consistency across all plots
						plt.rc('axes', prop_cycle=plt.cycler('color', colors))

						# plot the original measured voltages
						if isinstance(measured_voltage_all_currents_uv, np.ndarray) and ~np.isnan(np.all(
								measured_voltage_all_currents_uv)) or measured_voltage_all_currents_uv:

							# here we can select which currents to plot
							selected_currents = CommonUtils.select_current_indices(
								measured_voltage_all_currents_uv_for_plotting.shape[1], n_to_select=None)

							plt.figure(figsize=(12, 10))
							plt.plot(self.experiment.time_vector_msec,
									 measured_voltage_all_currents_uv_for_plotting[:, selected_currents], linewidth=3,
									 marker='x', markersize=5)
							plt.xlabel('Time (ms)', fontsize=30)
							plt.ylabel('Voltage ($\mu V$)', fontsize=30)
							plt.title(f'stim_num={stim_num}, rec_num={rec_idx + 1}, dist={round(distance, 1)}', fontsize=30)
							plt.tick_params(axis='both', labelsize=20)
							plt.grid()

							# set black border
							ax = plt.gca()
							for spine in ax.spines.values():
								spine.set_edgecolor('black')
								spine.set_linewidth(2)

							plt.legend()
							plt.show()

						# plot the extracted single voltage per current points
						if not self.experiment.use_simulated_data_as_sample:
							plt.figure()
							extraction = 'MAX'
							plt.plot(valid_currents_nanoamps, extracted_voltage_per_current_uv, marker='x', linestyle='')
							plt.plot(fit["I_fit"], fit["V_fit"], linestyle='--', color='red', label=f'Fit: R2'
																									f'={round(fit["R2"], 2)}, Slope={round(fit["slope"], 3)}')
							plt.xlabel('Current (nA)')
							plt.ylabel('Voltage ($\mu V$) - ' + extraction)
							plt.title(f'stim_num={stim_num}, rec_num={rec_idx + 1}, dist={round(distance, 1)}')
							plt.xlim(0, np.nanmax(self.experiment.valid_currents_nanoamps) + 5)
							plt.ylim(0, np.nanmax(extracted_voltage_per_current_uv) + 10)
							plt.grid()
							plt.legend(fontsize=10)
							plt.show()

						stim_plotted_counter += 1

		return output_slopes, output_slopes_sem, output_autopotential_distances_um, original_autopotential_distances_um

	def _calibrate_measured_slopes(self, measured_slopes, measured_slopes_sem, resistivities, adjusted_currents):
		# no need to calibrate slopes if running ideal data or simulated data
		if (self.experiment.use_saline_as_sample or self.experiment.use_simulated_data_as_sample) and self.experiment.generate_data:
			return measured_slopes, measured_slopes_sem

		# initialize the output array
		calibrated_slopes = np.full_like(measured_slopes, np.nan)

		# iterate over stimulation electrodes
		for stim_idx, stim_num in enumerate(self.experiment.stimulation_electrodes):

			# extract the resistivity sensed by this stimulation electrode
			resistivity = resistivities[stim_idx]

			# if resistivity is nan, skip this stimulation electrode
			if np.isnan(resistivity):
				continue

			# if all measured slopes for this stimulation electrode are nan, skip it
			if np.all(np.isnan(measured_slopes[stim_idx, :])):
				continue

			# initialize plot counter
			counter = 2

			# iterate over recording electrodes
			for rec_idx in self.experiment.recording_indices:

				# if auto-potential, take the theoretical slope at the scaled distance
				if stim_num - 1 == rec_idx:

					# calculate the theoretical voltage for this auto-potential at the scaled distance
					analytical_voltages_all_currents_uv = CommonUtils.calculate_theoretical_voltage(
						resistivity,
						adjusted_currents[stim_idx],
						self.experiment.auto_potential_distance_um)

					# extract the analytical slope for this pair
					calibrated_slopes[stim_idx, rec_idx] = analytical_voltages_all_currents_uv[0] / adjusted_currents[stim_idx, 0]

				# else, interpolate between the two closest gain values
				else:
					# extract the xy_gain_values for this pair of stimulation and recording electrodes
					x_resistivity = self.xy_gain_values[stim_idx, rec_idx, 0]
					y_gain = self.xy_gain_values[stim_idx, rec_idx, 1]

					# if this pair didn't pass the calibration threshold test (saline_processor), the value would
					# remain nan, and we should skip it
					if np.isnan(self.common_gains[stim_idx, rec_idx]):
						calibrated_slopes[stim_idx, rec_idx] = np.nan
					else:
						# otherwise, extract the gain by interpolating between the closest values
						interpolated_gain = self._find_and_plot_interpolation(x_resistivity, y_gain, resistivity, stim_num, rec_idx + 1, plot_output=counter)

						# decrease plot counter
						counter -= 1

						# apply the gain to the measured slopes
						calibrated_slopes[stim_idx, rec_idx] = measured_slopes[stim_idx, rec_idx] * interpolated_gain

				# adjust the calibrated slope in the saved array
				old_entry = measured_slopes_sem[stim_idx, rec_idx]
				new_entry = (old_entry[0], calibrated_slopes[stim_idx, rec_idx], old_entry[2])
				measured_slopes_sem[stim_idx, rec_idx] = new_entry

		return calibrated_slopes, measured_slopes_sem

	def _plot_slopes_vs_distance(self, slopes):

		# iterate over the stimulation electrodes
		for stim_idx, stim_num in enumerate(self.experiment.stimulation_electrodes):

			# extract distances for this stimulation electrode
			distances = la.norm(self.experiment.electrode_positions - self.experiment.electrode_positions[stim_num - 1], axis=1)

			# extract the slopes for this stimulation electrode
			these_slopes = slopes[stim_idx, :]

			# if all nan, skip this stimulation electrode
			if np.all(np.isnan(these_slopes)):
				continue

			if self.experiment.debug_mode:

				# this is for analytical saline data
				if self.experiment.use_saline_as_sample:

					# fit a line (in log-log space)
					valid = (distances > 0) & (these_slopes > 0)
					p = np.polyfit(np.log10(distances[valid]), np.log10(these_slopes[valid]), 1)
					fit_x = np.linspace(min(distances[valid]), max(distances[valid]), 100)
					fit_y = 10 ** (p[1]) * fit_x ** (p[0])

					plt.figure(figsize=(12, 8))
					plt.scatter(distances, these_slopes, marker='x', s=50)
					plt.plot(fit_x, fit_y, 'r-', label=f'Fit: slope={p[0]:.2f}')
					plt.xlabel('Distance (μm)', fontsize=24, labelpad=15)
					plt.ylabel('Resistance (kOhm)', fontsize=24, labelpad=15)
					plt.title(f'Stimulation Electrode {stim_num}', fontsize=18, pad=15)
					plt.tick_params(axis='both', which='major', labelsize=18)
					plt.legend(fontsize=18)
					plt.xscale('log')
					plt.yscale('log')
					plt.grid()
					plt.legend()
					plt.show()

				# this is for experimental data
				else:
					# plot the slopes as a function of distance
					plt.figure(figsize=(12, 10))
					plt.scatter(distances, these_slopes, marker='o', s=18, color='black')
					plt.xlabel('Distance (μm)', fontsize=24, labelpad=15)
					plt.ylabel('Resistance (kΩ)', fontsize=24, labelpad=15)
					plt.title(f'Stimulation Electrode {stim_num}', fontsize=18, pad=15)

					ax = plt.gca()

					# more ticks, still log-spaced (3,6,10 within each decade)
					ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[3, 6, 10]))
					ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[3, 6, 10]))

					# format labels as plain floats, not 10^x
					ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}'))
					ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{y:.1f}'))

					plt.tick_params(axis='both', labelsize=20)

					plt.grid()
					plt.xscale('log')
					plt.yscale('log')
					plt.tight_layout()
					plt.show()

	def _find_and_plot_interpolation(self, resistivities, coefficients, new_resistivity, stim_num=None, rec_num=None,
									 plot_output=True):
		# find the closest bounding values from the calibration data
		pair = self._find_bounding_values(resistivities, coefficients, new_resistivity)

		# if the measured resistivity is out of bounds, the gain is None
		if pair is None:
			return None

		# if the measured resistivity matches an existing value, return that gain
		if isinstance(pair, tuple) and len(pair) == 2 and not isinstance(pair[0], tuple) and pair[0] == new_resistivity:
			x1, y1 = pair
			x2, y2 = pair
			interpolated_gain = y1

		# else, extract the bounding points and interpolate
		else:
			(x1, y1), (x2, y2) = pair
			interpolated_gain = self._interpolate_gain_from_resistivity(x1, y1, x2, y2, new_resistivity)

		# plot the interpolation
		if self.experiment.debug_mode and plot_output:

			# plot original data and interpolation
			plt.figure(figsize=(12, 10))
			plt.plot(resistivities, coefficients, 'ko', markersize=11, label='Calibration Points')

			plt.plot([x1, x2], [y1, y2], linestyle='--', linewidth=3.5, color='lightblue', label='Interpolated Segment')
			plt.plot(new_resistivity, interpolated_gain, marker='o', markersize=12, color='blue',
					 label=f'Extracted Coefficient {round(interpolated_gain, 2)}')

			ax = plt.gca()

			# add dotted guidelines
			plt.axvline(new_resistivity, color='purple', linestyle=':', linewidth=2.5)
			plt.axhline(interpolated_gain, color='gray', linestyle=':', linewidth=2.5)

			# add annotations
			ax.text(new_resistivity, 0.95, f"Resistivity: {new_resistivity:.0f} Ω·cm",
					transform=ax.get_xaxis_transform(), fontsize=18, color='purple', rotation=90, va='top',
					ha='right')

			plt.tick_params(axis='both', which='major', labelsize=18)

			# labels and legend
			plt.xlabel('Resistivity (Ω·cm)', fontsize=24, labelpad=15)
			plt.ylabel('Correction Coefficient', fontsize=24, labelpad=15)
			plt.title(f'stim_num={stim_num} rec_num={rec_num} resistivity {round(new_resistivity, 2)} Ω·cm',
					  fontsize=18, pad=15)
			plt.legend(fontsize=18)
			plt.grid()
			plt.show()

		return interpolated_gain

	@staticmethod
	def _find_bounding_values(resistivities, coefficients, new_resistivity):
		# ensure the inputs are numpy arrays
		resistivities = np.array(resistivities)
		coefficients = np.array(coefficients)

		# sort the arrays
		sort_idx = np.argsort(resistivities)
		resistivities = resistivities[sort_idx]
		coefficients = coefficients[sort_idx]

		# check if new resistivity is within bounds
		if new_resistivity < resistivities[0] or new_resistivity > resistivities[-1]:
			return None

		# find where the new resistivity fits
		idx = np.searchsorted(resistivities, new_resistivity)

		# check if the new resistivity matches an existing value
		if resistivities[idx] == new_resistivity:
			return (resistivities[idx], coefficients[idx])

		# otherwise, extract the bounding values
		x1, y1 = resistivities[idx - 1], coefficients[idx - 1]
		x2, y2 = resistivities[idx], coefficients[idx]

		return (x1, y1), (x2, y2)

	@staticmethod
	def _interpolate_gain_from_resistivity(x1, y1, x2, y2, x):
		# interpolate the value between the two bounding points
		if x1 == x2:
			return None
		return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

