# standard imports
import os
import math
import itertools
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from matplotlib.ticker import LogLocator, FuncFormatter

# project-specific imports
from data_processing.common_utils import CommonUtils
from data_processing.configurations import VoltageExtractionMethods


class SalineDataProcessor:

	def __init__(self, experiment_manager):
		# store the given experiment parameters
		self.experiment = experiment_manager
		self.calibration_method = self.experiment.calibration_method
		self.common_coefficients_threshold = self.experiment.common_coefficients_threshold

		# check if cross-calibration is requested
		self.run_cross_calibration = True if self.experiment.num_dilutions_for_calibration > 1 else False

		# initialize the adjusted currents arrays
		self.adjusted_currents = np.full((self.experiment.num_stimulation_electrodes, self.experiment.num_dilutions_for_calibration, self.experiment.num_valid_currents), np.nan)

	def process_saline_data(self):
		# if no calibration needed, return empty arrays
		if self.experiment.num_dilutions_for_calibration == 0:
			return [], [], [], [], []

		# report progress
		print('\nProcessing saline data for calibration')

		# process saline data
		gain_coefficients, gain_coefficients_per_dilution, auto_potential_xy_functions, xy_ratio_fit_function, xy_gain_fit_function = self._run_calibration()

		# if all gain coefficients are nan, raise an error
		if np.all(np.isnan(gain_coefficients)):
			raise ValueError("No common gain coefficients to proceed with. Please check the calibration procedure.")

		# otherwise, return the calibration coefficients
		return gain_coefficients, gain_coefficients_per_dilution, auto_potential_xy_functions, xy_ratio_fit_function, xy_gain_fit_function

	def _run_calibration(self):
		# process all saline measurements sequentially to extract currents ratios
		injected_to_actual_current_ratios,  stim_rec_slopes, stim_rec_slopes_analytical, _ = self._extract_currents_ratio_from_all_salines_sequentially()

		# fit a line to the auto-potential data of each stim electrode across all dilutions
		auto_potential_xy_functions = self._fit_auto_potential_across_dilutions(stim_rec_slopes)

		# fit the best shunt model across all dilutions
		xy_ratio_fit_function = self._fit_current_ratio_across_dilutions(injected_to_actual_current_ratios)

		# determine gain corrections
		gain_coefficients_per_dilution = self._determine_gain_corrections(stim_rec_slopes, stim_rec_slopes_analytical, xy_ratio_fit_function)

		# if there is more than one dilution, we need to filter the coefficients
		gain_coefficients = gain_coefficients_per_dilution
		xy_gain_fit_function = np.full((self.experiment.num_stimulation_electrodes, self.experiment.num_electrodes, 2, self.experiment.num_dilutions_for_calibration), np.nan)

		if self.run_cross_calibration:
			gain_coefficients, xy_gain_fit_function = self._filter_calibration_results(gain_coefficients_per_dilution)

		return gain_coefficients, gain_coefficients_per_dilution, auto_potential_xy_functions, xy_ratio_fit_function, xy_gain_fit_function

	def _extract_currents_ratio_from_all_salines_sequentially(self):

		# initialize output and intermediate arrays
		stim_rec_slopes = np.full((self.experiment.num_dilutions_for_calibration, self.experiment.num_stimulation_electrodes, self.experiment.num_electrodes), np.nan)
		stim_rec_slopes_analytical = np.full((self.experiment.num_dilutions_for_calibration, self.experiment.num_stimulation_electrodes, self.experiment.num_electrodes), np.nan)
		currents_ratio = np.full((self.experiment.num_dilutions_for_calibration, self.experiment.num_stimulation_electrodes), np.nan)
		output_fit_data = np.full((self.experiment.num_dilutions_for_calibration, self.experiment.num_stimulation_electrodes, 3), np.nan)

		# iterate over the measurements and calculate the calibration coefficients for each
		for saline_num, (saline_type, saline_resistivity) in enumerate(self.experiment.saline_dilutions_for_calibration.items()):

			# report progress
			print(f'Processing {saline_type.lower()}:')

			# extract the data file path for this saline measurement
			saline_data_path = os.path.join(self.experiment.input_folder, saline_type.lower())

			# add saline folder suffix if it exists (for measurements with bookend saline - "before" and "after")
			if self.experiment.saline_folder_suffix is not None and self.experiment.saline_folder_suffix != "":
				saline_data_path = saline_data_path + "_" + self.experiment.saline_folder_suffix

			# if the file does not exist, report an error and skip this measurement
			if not os.path.exists(saline_data_path):
				print(f"Saline measurement not found: {saline_data_path}")
				continue

			# calculate the calibration coefficients for this saline measurement
			stim_rec_slopes[saline_num], stim_rec_slopes_analytical[saline_num] = self._extract_stim_rec_slopes_from_single_saline_measurement(saline_data_path, saline_resistivity)

			# for each stim elec, fit the slopes to a line in log scale and find the ratio
			currents_ratio[saline_num], output_fit_data[saline_num] = self._calculate_currents_ratio(stim_rec_slopes[saline_num], stim_rec_slopes_analytical[saline_num], saline_resistivity)

		return currents_ratio, stim_rec_slopes, stim_rec_slopes_analytical, output_fit_data

	def _extract_stim_rec_slopes_from_single_saline_measurement(self, file_path, saline_resistivity):

		# initialize output arrays
		output_slopes_measured = np.full((self.experiment.num_stimulation_electrodes, self.experiment.num_electrodes), np.nan)
		output_slopes_analytical = np.full((self.experiment.num_stimulation_electrodes, self.experiment.num_electrodes), np.nan)

		# set figure limit per stim
		max_figure_per_stim = 5

		# iterate over stimulation electrodes
		for stim_idx, stim_num in enumerate(self.experiment.stimulation_electrodes):

			# initialize a plotting counter
			stim_plotted_counter = 0

			# calculate the distance between this stimulation electrode and all recording electrodes
			distances_um = la.norm(self.experiment.electrode_positions - self.experiment.electrode_positions[stim_num - 1, :], axis=1)

			# read the raw recording waveforms (recorded voltages) for this injection
			all_recordings = CommonUtils.read_all_recordings(file_path, signal_length=self.experiment.signal_length,
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
			for read_idx in self.experiment.recording_indices:

				# if same as stim index, extract the auto potential slope and skip the rest
				if read_idx == stim_num - 1:
					auto_potential_slope, _ = self.experiment.extract_auto_potential_slopes(file_path, stim_idx)
					output_slopes_measured[stim_idx, read_idx] = auto_potential_slope
					output_slopes_analytical[stim_idx, read_idx] = np.nan
					continue

				# extract measured voltage from this rec electrode. This array is of shape (signal_length, N_currents)
				measured_voltage_all_currents_uv = all_recordings[:, read_idx, :].reshape((self.experiment.signal_length, -1), order='F')

				# if the measured voltage is all zeros, skip this pair of stim-record
				if not measured_voltage_all_currents_uv.any() or np.all(np.isnan(measured_voltage_all_currents_uv)):
					continue

				# save full signal for plotting
				measured_voltage_all_currents_uv_for_plotting = measured_voltage_all_currents_uv

				# if we are only using the positive phase, cut the signals
				if self.experiment.use_positive_phase:
					measured_voltage_all_currents_uv = measured_voltage_all_currents_uv[:self.experiment.signal_length_one_phase]

				# print progress
				print(f'{stim_num}: {read_idx + 1}')

				# if the max value in the signal is saturated, set the signal to nan
				max_per_current = np.nanmax(measured_voltage_all_currents_uv, axis=0)
				mask = (max_per_current > self.experiment.system_saturation_voltage_uv)
				measured_voltage_all_currents_uv[:, mask] = np.nan

				# extract a single voltage value for each current level
				if self.experiment.voltage_extraction_method == VoltageExtractionMethods.MAX:
					extracted_voltage_per_current_uv = np.nanmax(measured_voltage_all_currents_uv, axis=0)
				else:
					raise ValueError("Invalid voltage extraction method.")

				# remove indices where the extracted voltage is nan
				valid_indices = ~np.isnan(extracted_voltage_per_current_uv)
				extracted_voltage_per_current_uv = extracted_voltage_per_current_uv[valid_indices]
				valid_currents_nanoamps = self.experiment.valid_currents_nanoamps[valid_indices]

				# if the remaining extracted voltages are all nan, skip this pair
				if np.all(np.isnan(extracted_voltage_per_current_uv)) or len(extracted_voltage_per_current_uv) == 0:
					continue

				# fit the data to a line
				fit = CommonUtils.fit_line(valid_currents_nanoamps, extracted_voltage_per_current_uv,
										   through_origin=False)

				# TODO: uncomment below for paper figure
				# if read_idx == 198:
				# 	fit = CommonUtils.fit_line(valid_currents_nanoamps[0:33], extracted_voltage_per_current_uv[0:33],
				# 						   through_origin=False)

				# extract slope
				slope = fit["slope"]

				# skip this pair if this is a problematic measurement
				if fit["intercept"] >= 50:
					continue

				# otherwise, save the slope
				output_slopes_measured[stim_idx, read_idx] = slope

				# calculate the expected theoretical voltage for this recording
				if distances_um[read_idx] == 0:
					# for auto-potential, we set the theoretical voltage to nan
					analytical_slope = np.nan

				else:
					# otherwise, we calculate it based on the 1/r relationship
					analytical_voltages_all_currents_uv = CommonUtils.calculate_theoretical_voltage(
						saline_resistivity,
						self.experiment.valid_currents_nanoamps,
						distances_um[read_idx])

					# extract the analytical slope for this pair
					analytical_slope = analytical_voltages_all_currents_uv[0] / self.experiment.valid_currents_nanoamps[0]

				# save extracted analytical slope
				output_slopes_analytical[stim_idx, read_idx] = analytical_slope

				# plot the recordings for selected electrodes
				if self.experiment.debug_mode:

					# if we plotted enough rec electrodes for this stim electrode, skip plotting
					if stim_plotted_counter >= max_figure_per_stim:
						continue

					# otherwise, plot the recordings if the distance is one of the selected distances
					if (round(distances_um[read_idx]) == 0
							or round(distances_um[read_idx]) == 30
							or round(distances_um[read_idx]) == 330
							or round(distances_um[read_idx]) == 335
							or round(distances_um[read_idx]) == 600
							or round(distances_um[read_idx]) == 34
							or round(distances_um[read_idx]) == 805) \
							:

						# define a color cycle
						colors = plt.cm.tab10.colors

						# set the color cycle for the plots to maintain consistency across all plots
						plt.rc('axes', prop_cycle=plt.cycler('color', colors))

						# select up to 10 currents from the bottom, middle, and high ranges to plot
						selected_currents = CommonUtils.select_current_indices(
							measured_voltage_all_currents_uv_for_plotting.shape[1], n_to_select=6)

						# remove the first index for improved visibility of the selected currents
						selected_currents = np.delete(selected_currents, 1)

						# add labels for the selected currents
						labels = [f"{c:.1f} nA" for c in self.experiment.valid_currents_nanoamps[selected_currents]]

						# plot
						plt.figure(figsize=(12, 10))
						plt.plot(self.experiment.time_vector_msec,
								 measured_voltage_all_currents_uv_for_plotting[:, selected_currents], linewidth=3,
								 marker='x', markersize=5, label=labels)
						plt.xlabel('Time (ms)', fontsize=30, labelpad=15)
						plt.ylabel('Voltage ($\mu V$)', fontsize=30, labelpad=15)
						plt.title(
							f'{saline_resistivity}: stim_num={stim_num}, rec_num={read_idx + 1}, dist={round(distances_um[read_idx], 1)}',
							fontsize=25, pad=15)
						plt.tick_params(axis='both', labelsize=26)
						plt.grid()
						plt.legend(fontsize=20)

						# set black border
						ax = plt.gca()
						for spine in ax.spines.values():
							spine.set_edgecolor('black')
							spine.set_linewidth(2)

						plt.tight_layout()
						plt.show()

						# extract max and plot voltage as function of current as well
						plt.figure(figsize=(12, 10))
						extraction = 'Max'
						plt.plot(valid_currents_nanoamps, extracted_voltage_per_current_uv, marker='x', linestyle='',
								 markersize=14, color='black')

						# TODO: uncomment below for paper figure
						# x_line = self.experiment.valid_currents_nanoamps
						# y_line = fit["slope"] * x_line + fit["intercept"]
						# plt.plot(x_line, y_line, linestyle='-',
						# 		 label=f'Fit: R2={round(fit["R2"], 2)}, Resistance={round(fit["slope"], 2)} kΩ',
						# 		 color='blue', linewidth=2)

						# plot fit
						plt.plot(fit["I_fit"], fit["V_fit"], linestyle='-',
								 label=f'Fit: R2={round(fit["R2"], 2)}, Resistance={round(fit["slope"], 2)} kΩ',
								 color="orange", linewidth=2)
						plt.xlabel('Current (nA)', fontsize=30, labelpad=15)
						plt.ylabel(extraction + ' Voltage ($\mu V$)', fontsize=30, labelpad=15)
						plt.title(
							f'{saline_resistivity}: stim_num={stim_num}, rec_num={read_idx + 1}, dist='
							f'{round(distances_um[read_idx], 1)}', fontsize=25, pad=15)
						plt.tick_params(axis='both', labelsize=26)
						plt.xlim(0, np.nanmax(self.experiment.valid_currents_nanoamps) + 5)
						plt.ylim(0, np.nanmax(extracted_voltage_per_current_uv) + 10)
						plt.grid()

						# set black border
						ax = plt.gca()
						for spine in ax.spines.values():
							spine.set_edgecolor('black')
							spine.set_linewidth(2)

						plt.tight_layout()
						plt.show()

						stim_plotted_counter += 1

		return output_slopes_measured, output_slopes_analytical

	def _calculate_currents_ratio(self, slopes_measured, slopes_analytical, saline_resistivity):
		# initialize an output array
		currents_ratio = np.full(self.experiment.num_stimulation_electrodes, np.nan)
		output_fit_data = np.full((self.experiment.num_stimulation_electrodes, 3), np.nan)

		# iterate over all stimulation electrodes
		for stim_idx, stim_num in enumerate(self.experiment.stimulation_electrodes):

			# extract the slopes for this stimulation electrode
			measured_slopes = slopes_measured[stim_idx, :]
			analytical_slopes = slopes_analytical[stim_idx, :]

			# if all slopes are nan, skip this stim electrode
			if np.all(np.isnan(measured_slopes)):
				continue

			# calculate the distances between this stim to all rec electrodes
			distances = la.norm(self.experiment.electrode_positions - self.experiment.electrode_positions[stim_num - 1, :], axis=1)

			# sort the distances and slopes by distance
			sorted_indices = np.argsort(distances)
			sorted_distances = distances[sorted_indices]
			sorted_measured_slopes = measured_slopes[sorted_indices]
			sorted_analytical_slopes = analytical_slopes[sorted_indices]

			# remove clear outliers from the measured slopes (more than 3 stds away)
			measured_slopes_filtered = CommonUtils.remove_high_outliers(sorted_measured_slopes[1:], std_thresh=3)

			# add back the first element to the measured slopes (also gets removed by the outlier filter)
			sorted_measured_slopes = np.insert(measured_slopes_filtered, 0, sorted_measured_slopes[0])

			# remove negative or zero values
			filtered_distances = sorted_distances[sorted_measured_slopes > 0]
			filtered_measured_slopes = sorted_measured_slopes[sorted_measured_slopes > 0]
			filtered_analytical_slopes = sorted_analytical_slopes[sorted_measured_slopes > 0]

			# remove nan values
			valid_indices = ~np.isnan(filtered_analytical_slopes) & ~np.isnan(filtered_measured_slopes)
			filtered_distances = filtered_distances[valid_indices]
			filtered_analytical_slopes = filtered_analytical_slopes[valid_indices]
			filtered_measured_slopes = filtered_measured_slopes[valid_indices]

			# apply log transformation to fit the data to a line
			log_distances = np.log10(filtered_distances)
			log_analytical_slopes = np.log10(filtered_analytical_slopes)
			log_measured_slopes = np.log10(filtered_measured_slopes)

			# fit a line to the log-transformed data
			fit_measured, trace_measured = CommonUtils.fit_line_fixed_slope_bayesian(log_distances, log_measured_slopes)
			fit_analytical = CommonUtils.fit_line(log_distances, log_analytical_slopes, through_origin=False)

			# construct the fit functions
			fitted_distances = 10 ** log_distances
			fitted_measured_slopes = 10 ** (fit_measured["slope"] * log_distances + fit_measured["intercept"])
			fitted_analytical_slopes = 10 ** (fit_analytical["slope"] * log_distances + fit_analytical["intercept"])

			# calculate the ratio that minimizes the difference between the analytical and measured slopes
			ratio = (fitted_analytical_slopes @ fitted_measured_slopes) / (la.norm(fitted_analytical_slopes) ** 2)

			# the measured data cannot exceed the analytical data, so we limit the ratio to 1
			if ratio > 1:
				ratio = 1

			# extract the auto-potential
			log_auto_potential = np.log10(sorted_measured_slopes[0])

			# store the result
			currents_ratio[stim_idx] = ratio
			output_fit_data[stim_idx] = (log_auto_potential, fit_measured["slope"], fit_measured["intercept"])

			# plot the data
			if self.experiment.debug_mode:
				plt.figure(figsize=(12, 10))

				# plot analytical data
				plt.plot(fitted_distances, fitted_analytical_slopes, linestyle='-', linewidth=2.3,
						 label=f'Analytical', color='green')

				# plot measured data and fitted line
				plt.plot(filtered_distances, filtered_measured_slopes, marker='o', linestyle='', markersize=8,
						 color='black', label=f'Experimental')
				plt.plot(fitted_distances, fitted_measured_slopes, linestyle='-', linewidth=2.3, label=f'1/L Fit',
						 color='black')

				# add text label with the current ratio
				plt.text(0.3, 0.9, f'ratio: {currents_ratio[stim_idx]:.3f}', transform=plt.gca().transAxes, fontsize=22,
						 color='blue')

				# plot settings
				plt.xscale('log')
				plt.yscale('log')

				# add labels and title
				plt.title(f'{saline_resistivity} Ω·cm: stim_num={stim_num}, all rec electrodes',
						  fontsize=18, pad=15)
				plt.xlabel('Distance ($\mu m$)', fontsize=24, labelpad=15)
				plt.ylabel('Resistance (kΩ)', fontsize=24, labelpad=15)

				# set up axis
				plt.tick_params(axis='both', labelsize=14)

				ax = plt.gca()

				# more ticks, still log-spaced (3,6,10 within each decade)
				ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[3, 6, 10]))
				ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[3, 6, 10]))

				# format labels as plain floats, not 10^x
				ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f}'))
				ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{y:.0f}'))

				plt.grid()

				# add legend
				plt.legend(fontsize=16)

				# show the plot
				plt.show()

		# return the current ratios for this saline measurement
		return currents_ratio, output_fit_data

	def _fit_auto_potential_across_dilutions(self, stim_rec_slopes):
		# define interpolation size
		interpolation_size = 1000

		# initialize an output array
		auto_potential_fits = np.full((self.experiment.num_stimulation_electrodes, 2, interpolation_size), np.nan)

		# iterate stim electrodes
		for stim_idx, stim_num in enumerate(self.experiment.stimulation_electrodes):

			# extract auto-potential slopes for this stim electrode across all dilutions
			auto_potential_slopes = stim_rec_slopes[:, stim_idx, stim_num - 1]

			# sort dilutions by value
			sorted_indices = np.argsort([float(v) for v in self.experiment.saline_dilutions_for_calibration.values()])
			auto_potential_slopes = auto_potential_slopes[sorted_indices]
			resistivities = np.array(list(self.experiment.saline_dilutions_for_calibration.values()))[sorted_indices]

			# if the extracted values are not monotonically increasing, skip this electrode
			if np.any(np.diff(auto_potential_slopes) <= 0):
				continue

			# if not all dilutions have a value, find the common values
			if len(resistivities) != len(auto_potential_slopes):
				valid_mask = ~np.isnan(auto_potential_slopes)
				resistivities = resistivities[valid_mask]
				auto_potential_slopes = auto_potential_slopes[valid_mask]

			# remove nan values before fitting
			valid_mask = ~np.isnan(auto_potential_slopes)
			x_valid = resistivities[valid_mask]
			y_valid = auto_potential_slopes[valid_mask]

			# ensure x is strictly increasing for PCHIP
			if np.all(np.diff(x_valid) > 0):
				interpolator = PchipInterpolator(x_valid, y_valid)

				# generate smooth fitted values
				x_smooth = np.linspace(x_valid.min(), x_valid.max(), interpolation_size)
				y_smooth = interpolator(x_smooth)
			else:
				# fallback in case data is invalid
				x_smooth = y_smooth = np.full(interpolation_size, np.nan)

			# plot as a function of resistivity
			if self.experiment.debug_mode:
				plt.figure(figsize=(12, 10))
				plt.plot(resistivities, auto_potential_slopes, marker='o', markersize=10, linestyle='',
						 label='Extracted values', color='black')
				if x_smooth is not None:
					plt.plot(x_smooth, y_smooth, linestyle='--', color='red', label='Interpolation', linewidth=2)
				plt.xlabel('Resistivity (Ω·cm)', fontsize=24, labelpad=15)
				plt.ylabel('Auto-Resistance (kΩ)', fontsize=24, labelpad=15)
				plt.title(f'Stimulation Electrode {stim_num}', fontsize=16, pad=15)
				plt.tick_params(axis='both', labelsize=16)
				plt.legend(fontsize=18)
				plt.grid()
				plt.show()

			# store the auto-potential fit data
			auto_potential_fits[stim_idx, 0] = x_smooth
			auto_potential_fits[stim_idx, 1] = y_smooth

		return auto_potential_fits

	def _fit_current_ratio_across_dilutions(self, injected_to_actual_current_ratios):
		# define interpolation size
		interpolation_size = 1000

		# initialize output array
		xy_ratio_fit_function = np.full((self.experiment.num_stimulation_electrodes, 2, interpolation_size), np.nan)

		# iterate over all stimulation electrodes
		for stim_idx, stim_num in enumerate(self.experiment.stimulation_electrodes):

			# extract the current ratios for this stimulation electrode across all dilutions
			measured = injected_to_actual_current_ratios[:, stim_idx]

			# if all ratios are nan, skip this stim electrode
			if np.all(np.isnan(measured)):
				continue

			# extract the current ratios for this stimulation electrode across all dilutions
			measured = injected_to_actual_current_ratios[:, stim_idx]

			# if all ratios are nan, skip this stim electrode
			if np.all(np.isnan(measured)):
				continue

			# create an array of resistivities corresponding to the used dilutions
			rho_ohm_cm = np.array([self.experiment.saline_dilutions_for_calibration[saline_type] for saline_type in self.experiment.saline_dilutions_for_calibration.keys()])

			# remove indices where measured is nan
			valid_indices = ~np.isnan(measured)
			rho_ohm_cm = rho_ohm_cm[valid_indices].flatten()
			measured = measured[valid_indices].flatten()

			# sort by increasing resistivity (important for monotonic splines)
			sorted_indices = np.argsort(rho_ohm_cm)
			rho_ohm_cm = rho_ohm_cm[sorted_indices]
			measured = measured[sorted_indices]

			# create a smooth x range for plotting or evaluation
			rho_smooth = np.linspace(rho_ohm_cm.min(), rho_ohm_cm.max(), interpolation_size)

			# use linear piece-wise interpolation
			fit_smooth = np.interp(rho_smooth, rho_ohm_cm, measured)

			# plot to visualize
			if self.experiment.debug_mode:
				plt.figure(figsize=(12, 10))
				plt.plot(rho_ohm_cm, measured, 'o', label='Extracted values', markersize=10, color='black')
				plt.plot(rho_smooth, fit_smooth, '--', label='Interpolation', linewidth=2, color='magenta')
				plt.xlabel('Resistivity (Ω·cm)', fontsize=24, labelpad=15)
				plt.ylabel('Current Ratio', fontsize=24, labelpad=15)
				plt.title(f'Stimulation Electrode {stim_num}', fontsize=18, pad=15)
				plt.tick_params(axis='both', labelsize=16)
				plt.grid(True)
				plt.legend(fontsize=18)
				plt.show()

			# store the results
			xy_ratio_fit_function[stim_idx, 0] = rho_smooth
			xy_ratio_fit_function[stim_idx, 1] = fit_smooth

		return xy_ratio_fit_function

	def _determine_gain_corrections(self, stim_rec_slopes, stim_rec_slopes_analytical, xy_ratio_fit_function):
		# initialize an output array
		gain_coefficients = np.full((self.experiment.num_stimulation_electrodes, self.experiment.num_electrodes, self.experiment.num_dilutions_for_calibration), np.nan)

		# iterate over all stimulation electrodes
		for stim_idx, stim_num in enumerate(self.experiment.stimulation_electrodes):

			# extract the distances between this stim electrode and all rec electrodes
			distances_um = la.norm(self.experiment.electrode_positions - self.experiment.electrode_positions[stim_num - 1, :], axis=1)

			# iterate over all dilutions
			for saline_idx, (saline_type, saline_resistivity) in enumerate(self.experiment.saline_dilutions_for_calibration.items()):

				# extract the adjusted currents for this stimulation electrode and saline type
				adjustment_term = 1.0

				if xy_ratio_fit_function[stim_idx] is not None:
					# extract the fitted rho and ratio for this stim
					rho_fit = xy_ratio_fit_function[stim_idx, 0]
					ratio_fit = xy_ratio_fit_function[stim_idx, 1]

					# find the closest resistivity in the fitted data
					closest_idx = np.argmin(np.abs(rho_fit - saline_resistivity))

					# extract the ratio for this resistivity
					adjustment_term = ratio_fit[closest_idx]

				# extract all analytical and measured slopes for this stim
				analytical_slopes = stim_rec_slopes_analytical[saline_idx, stim_idx, :]
				measured_slopes = stim_rec_slopes[saline_idx, stim_idx, :]

				# if all measured slopes are nan, skip this saline
				if np.all(np.isnan(measured_slopes)):
					continue

				# modify the slopes to reflect the adjusted currents
				adjusted_analytical_slopes = stim_rec_slopes_analytical[saline_idx, stim_idx, :] * adjustment_term

				# plot the adjusted slopes
				if self.experiment.debug_mode:
					plt.figure(figsize=(12, 10))
					plt.plot(distances_um, analytical_slopes, marker='', linestyle='-', markersize=10,
							 label='analytical fit', color='magenta', linewidth=2)
					plt.plot(distances_um, adjusted_analytical_slopes, marker='', linestyle='-', markersize=10,
							 label='adjusted fit', color='blue', linewidth=2)
					plt.plot(distances_um, measured_slopes, marker='o', linestyle='', markersize=10,
							 label='measured data', color='black')

					plt.xscale('log')
					plt.yscale('log')
					ax = plt.gca()

					# more ticks, still log-spaced (3,6,10 within each decade)
					ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[3, 6, 10]))
					ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[3, 6, 10]))

					# format labels as plain floats, not 10^x
					ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}'))
					ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{y:.1f}'))

					plt.tick_params(axis='both', labelsize=20)
					plt.grid()

					plt.xlabel('Distance ($\mu m$)', fontsize=24, labelpad=15)
					plt.ylabel('Resistance (kΩ)', fontsize=24, labelpad=15)
					plt.title(f'{saline_type}: Stimulation Electrode {stim_num}', fontsize=18, pad=15)
					plt.legend(fontsize=18)
					plt.show()

				# calculate the gain coefficient needed to bring each measured slope to its analytical counterpart
				gain_coefficients[stim_idx, :, saline_idx] = np.divide(adjusted_analytical_slopes, measured_slopes, out=np.full_like(measured_slopes, np.nan), where=adjusted_analytical_slopes != 0)

				# adjust the measured slopes using the gain coefficients (for plotting purposes only)
				adjusted_measured_slopes = measured_slopes * gain_coefficients[stim_idx, :, saline_idx]

				# plot the adjusted measured slopes
				if self.experiment.debug_mode:
					plt.figure(figsize=(12, 10))
					plt.plot(distances_um, measured_slopes, marker='o', linestyle='', markersize=10,
							 label='measured data', color='black')
					plt.plot(distances_um, adjusted_measured_slopes, marker='*', linestyle='--', markersize=15,
							 label='corrected data', color='blue')
					plt.xlabel('Distance ($\mu m$)', fontsize=24, labelpad=15)
					plt.ylabel('Resistance (kΩ)', fontsize=24, labelpad=15)
					plt.title(f'{saline_type}: Stimulation Electrode {stim_num}', fontsize=18, pad=15)
					plt.legend(fontsize=20)
					plt.xscale('log')
					plt.yscale('log')
					ax = plt.gca()

					# more ticks, still log-spaced (3,6,10 within each decade)
					ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[3, 6, 10]))
					ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[3, 6, 10]))

					# format labels as plain floats, not 10^x
					ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}'))
					ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{y:.1f}'))
					plt.tick_params(axis='both', labelsize=18)
					plt.grid()
					plt.show()

		# return the gain coefficients
		return gain_coefficients

	def _filter_calibration_results(self, gain_coefficients_per_dilution):
		# initialize output dictionary
		cross_calibration_results = np.full((self.experiment.num_stimulation_electrodes, self.experiment.num_electrodes), np.nan)

		# initialize fit function
		xy_calibration_fit = np.full((self.experiment.num_stimulation_electrodes, self.experiment.num_electrodes, 2, self.experiment.num_dilutions_for_calibration), np.nan)

		# get min and max across all electrodes and all dilutions
		min_gain = np.nanmin(gain_coefficients_per_dilution)
		max_gain = np.nanmax(gain_coefficients_per_dilution)

		# collect the coefficients across all saline dilutions
		gain_matrices = [gain_coefficients_per_dilution[:, self.experiment.recording_indices, i] for i in range(self.experiment.num_dilutions_for_calibration)]

		# iterate the dilutions to plot the stim-rec grid for each one
		if self.experiment.debug_mode:

			for saline_idx, (saline_type, saline_resistivity) in enumerate(self.experiment.saline_dilutions_for_calibration.items()):

				# extract the gain coefficients for this saline type
				gain_coefficients = gain_matrices[saline_idx]

				# plot the gain coefficients for this saline type
				plt.figure()
				plt.imshow(gain_coefficients, aspect='auto', cmap='viridis', interpolation='nearest')
				plt.colorbar(label='Gain Coefficients')
				plt.clim(min_gain, max_gain)
				plt.title(f'Gain Coefficients for {saline_type}')
				plt.xlabel('Recording Electrode')
				plt.ylabel('Stimulation Electrode')

				plt.xticks(
					ticks=np.arange(len(self.experiment.recording_indices)),
					labels=np.array(self.experiment.recording_indices) + 1,
					fontsize=5
				)
				plt.yticks(
					ticks=np.arange(len(self.experiment.stimulation_electrodes)),
					labels=self.experiment.stimulation_electrodes,
					fontsize=10
				)
				plt.grid(False)
				plt.show()

		# plot the difference between the matrix of coefficients between every two dilutions
		if self.experiment.debug_mode:

			# get saline names
			saline_keys = list(self.experiment.saline_dilutions_for_calibration.keys())

			# get unique unordered pairs
			pairs = list(itertools.combinations(range(len(saline_keys)), 2))

			# grid layout
			n_plots = len(pairs)
			n_cols = math.ceil(math.sqrt(n_plots))
			n_rows = math.ceil(n_plots / n_cols)

			# === figure 1: difference ===
			fig_diff, axes_diff = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
			axes_diff = np.atleast_1d(axes_diff)
			axes_diff = axes_diff.flatten()

			for idx, (i, j) in enumerate(pairs):
				diff = gain_matrices[i] - gain_matrices[j]
				im = axes_diff[idx].imshow(diff, aspect='auto', cmap='viridis')

				label_i = saline_keys[i]
				label_j = saline_keys[j]

				axes_diff[idx].set_title(f'{label_i} - {label_j}')
				axes_diff[idx].set_xlabel('Recording Electrodes')
				axes_diff[idx].set_ylabel('Stimulation Electrodes')
				plt.colorbar(im, ax=axes_diff[idx])

			# hide unused axes
			for ax in axes_diff[len(pairs):]:
				ax.axis('off')

			fig_diff.suptitle('Gain Matrix Differences', fontsize=16)
			plt.tight_layout()
			plt.show()

			# === figure 2: division ===
			fig_div, axes_div = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
			axes_div = np.atleast_1d(axes_div)
			axes_div = axes_div.flatten()

			for idx, (i, j) in enumerate(pairs):
				with np.errstate(divide='ignore', invalid='ignore'):
					div = np.true_divide(gain_matrices[i], gain_matrices[j])
					div[~np.isfinite(div)] = np.nan  # Replace inf and nan with nan

				im = axes_div[idx].imshow(div, aspect='auto', cmap='viridis')

				label_i = saline_keys[i]
				label_j = saline_keys[j]

				axes_div[idx].set_title(f'{label_i} / {label_j}')
				axes_div[idx].set_xlabel('Recording Electrodes')
				axes_div[idx].set_ylabel('Stimulation Electrodes')
				plt.colorbar(im, ax=axes_div[idx])

			# hide unused axes
			for ax in axes_div[len(pairs):]:
				ax.axis('off')

			fig_div.suptitle('Gain Matrix Division', fontsize=16)
			plt.tight_layout()
			plt.show()

		# for each pair of stim-record, calculate the mean gain coefficient if they are all within the threshold
		for stim_idx, stim_num in enumerate(self.experiment.stimulation_electrodes):
			for rec_idx in self.experiment.recording_indices:

				# extract the gain coefficients for this pair of electrodes across all dilutions
				coefficients = gain_coefficients_per_dilution[stim_idx, rec_idx, :]
				resistivities = list(self.experiment.saline_dilutions_for_calibration.values())

				# store resistivity-gain values for this pair
				xy_calibration_fit[stim_idx, rec_idx, 0] = np.array(resistivities)
				xy_calibration_fit[stim_idx, rec_idx, 1] = coefficients

				# sort the coefficients and resistivities by resistivity
				sorted_indices = np.argsort(resistivities)
				coefficients = coefficients[sorted_indices]

				# if all coefficients are nan, skip this pair
				if np.all(np.isnan(coefficients)):
					continue

				# otherwise, check the extent of their scatter
				min_gain = np.nanmin(coefficients)
				max_gain = np.nanmax(coefficients)

				threshold_fraction = self.experiment.common_coefficients_threshold / 100

				# if the difference between the min and max is less than the threshold, take the mean
				# this is only used to filter out inconsistent measurements; we are not using the mean for calibration.
				if min_gain > 0:
					if max_gain / min_gain <= 1 + threshold_fraction:
						cross_calibration_results[stim_idx, rec_idx] = np.nanmean(coefficients)

		# plot the remaining common coefficients
		if self.experiment.debug_mode:
			plt.figure()
			plt.imshow(cross_calibration_results[:, self.experiment.recording_indices], aspect='auto', cmap='viridis', interpolation='nearest')
			plt.colorbar(label='Gain Coefficients')
			plt.title(f'Common Gain Coefficients, threshold={self.experiment.common_coefficients_threshold}%')
			plt.xlabel('Recording Electrode')
			plt.ylabel('Stimulation Electrode')

			plt.xticks(
				ticks=np.arange(len(self.experiment.recording_indices)),
				labels=np.array(self.experiment.recording_indices) + 1,
				fontsize=5
			)
			plt.yticks(
				ticks=np.arange(len(self.experiment.stimulation_electrodes)),
				labels=self.experiment.stimulation_electrodes,
				fontsize=10
			)
			plt.grid(False)
			plt.show()

		# return the cross-calibration results
		return cross_calibration_results, xy_calibration_fit
