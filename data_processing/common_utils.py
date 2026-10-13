# standard imports
import os
import csv
import platform
import pytensor
import pymc as pm
import numpy as np
from scipy import stats

# project imports
from data_processing.multielec_src import old_labview_data_reader as oldlv

# if Apple Silicon (ARM64) Mac, run tensor with numpy
if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
	pytensor.config.cxx = ""


class CommonUtils:

	@staticmethod
	def find_header_files(root_dir, keyword='header', search_depth=1, sort_by_index=False):
		# initialize a list to store the header files
		header_files = []

		# list the immediate contents of the root folder
		for subdir in os.listdir(root_dir):
			subdir_path = os.path.join(root_dir, subdir)

			# check if the path is a directory and we are asked to search within it (search_depth > 0)
			if os.path.isdir(subdir_path) and search_depth > 0:

				# loop through all files in the subdirectory
				for file_name in os.listdir(subdir_path):
					file_path = os.path.join(subdir_path, file_name)

					# if search_depth > 1 and the file is a directory, search within it
					if search_depth > 1 and os.path.isdir(file_path):
						for deeper_file_name in os.listdir(file_path):
							if keyword in deeper_file_name:
								header_files.append(os.path.join(file_path, deeper_file_name))

					# else, check if the file name contains the needed keyword
					elif keyword in file_name:
						header_files.append(file_path)

			# flat layout: this entry is a file directly under root_dir, not a subdir
			elif keyword in subdir:
				header_files.append(subdir_path)

		# sort by index number if requested
		if sort_by_index:
			header_files.sort(key=lambda f: float(''.join(
				c for c in os.path.basename(f).split(keyword)[1] if c.isdigit())) if keyword in os.path.basename(
				f) else float('inf'))

		# return the list of files
		return header_files

	@staticmethod
	def filter_by_keyword_combinations(header_files, keyword_combinations):
		if not keyword_combinations:
			filtered = header_files
		else:
			filtered = [f for f in header_files
						if any(all(kw in f for kw in combo)
							   for combo in keyword_combinations)]

		# guarantee deterministic order regardless of filesystem traversal order
		return sorted(filtered, key=lambda f: float(os.path.basename(f).split("inversion_result_")[1].split("_")[0]))

	@staticmethod
	def read_all_recordings(path, stimulation_electrode, start_offset=2, signal_length=22, num_electrodes=512,
							sign_mask=None, valid_current_levels=None, signal_length_positive_phase=11):
		"""
		This function reads all the voltage recordings for all current amplitudes for a given stimulation electrode.
		path: directory to dataxxx level
		stimulation_electrode: index of the current injecting electrode
		delay: data point start analysing
		num_electrodes: total # of electrodes on the MEA
		num_current_levels: number of current amplitudes in the trial
		start_at_zero: if True subtracts the amplitude of the first data point to start the trace at zero
		"""
		# initialize output array
		num_valid_current_levels = len(valid_current_levels)
		voltage_recordings = np.full((signal_length, num_electrodes, num_valid_current_levels), np.nan)

		# set the start and end index to extract the data
		start_idx_pulse = start_offset + 1
		end_idx_pulse = start_idx_pulse + signal_length

		# set the start and end index to extract the baseline
		start_index_baseline = start_offset - 10
		end_index_baseline = start_offset - 1

		# read the data for each current amplitude, out of the valid levels for this run
		for reduced_index, original_index in enumerate(valid_current_levels):

			# read the raw data for this current amplitude (the file names start at 1, so we add 1 to the index)
			raw_data = oldlv.get_oldlabview_pp_data(path, stimulation_electrode, original_index + 1)

			# if this current amplitude was not applied with this stimulation electrode, skip it
			if raw_data is None:
				continue

			# average over trials (there are usually 20 trials)
			data = np.mean(raw_data, axis=0)

			# convert adc units to uV by subtracting the DC offset and multiplying by the conversion factor
			data = data * 3.6

			# extract the DC offset from the part that precedes the stimulus
			dc_offset = np.mean(data[:, start_index_baseline:end_index_baseline], axis=1)

			# extract the pulse part of the recording
			pulse_data = data[:, start_idx_pulse:end_idx_pulse]

			# remove the DC offset
			pulse_data = pulse_data - dc_offset.reshape((-1, 1))

			# apply the sign mask if provided
			if sign_mask is not None and sign_mask[original_index] == -1:
				pulse_data = -pulse_data

			# # the positive phase should not contain negative values
			# pulse_data[:, 1:signal_length_positive_phase][pulse_data[:, 1:signal_length_positive_phase] < 0] = np.nan

			# store the processed voltage recording for this current amplitude with the signal length as the first dimension
			voltage_recordings[:, :, reduced_index] = pulse_data.T

		# return the voltage recordings
		return voltage_recordings

	@staticmethod
	def calculate_first_principle_component(input_matrix, divide_by_std=True):
		# remove columns with all zeros as this is non-existing data
		input_matrix = input_matrix[:, ~(input_matrix == 0).all(axis=0)]

		# calculate the mean and std along each current level (columns)
		mean = np.nanmean(input_matrix, axis=0).reshape((1, -1))
		std_dev = np.nanstd(input_matrix, axis=0).reshape((1, -1))

		# standardize the data
		standardized_data = input_matrix - mean
		if divide_by_std:
			standardized_data /= std_dev

		# compute the covariance matrix
		cov_matrix = standardized_data @ standardized_data.T

		# get eigenvalues and eigenvectors
		eigen_values, eigen_vectors = np.linalg.eig(np.round(cov_matrix, 3))

		# sort eigenvalues in descending order
		sorted_indices = np.argsort(np.abs(eigen_values))[::-1]
		eigen_vectors = eigen_vectors[:, sorted_indices]

		# extract the first principal component (PC1)
		pc1 = eigen_vectors[:, 0]

		# ensure waveform is positive-first, otherwise flip
		if pc1[0] < 0 or pc1[3] < 0 or pc1[5] < 0:
			pc1 = -pc1

		# ensure the returned vector is real
		pc1 = np.real_if_close(pc1)

		return pc1

	@staticmethod
	def calculate_theoretical_voltage(resistivity_ohmcm, current_nanoamps, distance_to_stim_elec):

		# generate the common term
		common_term = (resistivity_ohmcm * current_nanoamps) / (2 * np.pi)

		# convert to uV
		common_term *= 10

		# calculate the main term of the theoretical voltage
		if distance_to_stim_elec == 0:
			theoretical_voltage_uv = np.full(len(current_nanoamps), np.nan)
		else:
			theoretical_voltage_uv = common_term * (1 / distance_to_stim_elec)

		return theoretical_voltage_uv

	@staticmethod
	def fit_line(I, V, through_origin=False):
		"""Fit linear model V = a * I + b (or V = a * I if through_origin is True)"""
		I = np.asarray(I)
		V = np.asarray(V)

		# fit V = a * I with no intercept
		if through_origin:
			slope, _, _, _ = np.linalg.lstsq(I[:, np.newaxis], V, rcond=None)
			slope = slope[0]
			intercept = 0.0
			V_fit = slope * I

			# calculate R² manually
			ss_res = np.sum((V - V_fit) ** 2)
			ss_tot = np.sum((V - np.mean(V)) ** 2)
			r_squared = 1 - ss_res / ss_tot

		# fit V = a * I + b with intercept
		else:
			slope, intercept, r_value, _, _ = stats.linregress(I, V)
			V_fit = slope * I + intercept
			r_squared = r_value ** 2

		# calculate standard deviation using the fitted slope as the mean
		resistances = V / I
		mean_resistances = V_fit / I
		slope_population_std = np.sqrt((np.nansum((resistances-mean_resistances)**2)) / (np.sum(~np.isnan(resistances)) - 1))

		# standard error of the mean (in kΩ)
		slope_sem = slope_population_std / np.sum(~np.isnan(resistances))

		return {
			'slope': float(slope),
			'intercept': float(intercept),
			'R2': float(r_squared),
			'V_fit': V_fit,
			'I_fit': I,
			'scatter_std': float(slope_population_std),
			'slope_sem': float(slope_sem)
		}

	@staticmethod
	def fit_line_fixed_slope_bayesian(log_x, log_y, fixed_slope=-1, n_samples=500, tune=100, prior_sigma=10,
									  obs_sigma_prior=1):
		"""
		Bayesian fit of a line with fixed slope to log-log data:
		log_y = fixed_slope * log_x + b + noise

		Returns same dict format as CommonUtils.fit_line:
		{
			'slope': float,
			'intercept': float,
			'R2': float,
			'V_fit': array of fitted log_y,
			'I_fit': array of log_x (identity)
		}
		"""
		with pm.Model() as model:
			# prior on intercept b
			b = pm.Normal("b", mu=0, sigma=prior_sigma)

			# likelihood
			mu = fixed_slope * log_x + b
			sigma = pm.HalfNormal("sigma", sigma=obs_sigma_prior)

			y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=log_y)

			# sampling
			trace = pm.sample(n_samples, tune=tune, chains=2, cores=1, progressbar=False)

		# extract posterior mean of b
		b_mean = trace.posterior["b"].values.flatten().mean()

		# predict fitted values
		V_fit = fixed_slope * log_x + b_mean
		I_fit = log_x.copy()

		# compute R² manually
		ss_res = np.sum((log_y - V_fit) ** 2)
		ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
		r_squared = 1 - ss_res / ss_tot

		# return result in the same format
		return {
			'slope': float(fixed_slope),
			'intercept': float(b_mean),
			'R2': float(r_squared),
			'V_fit': V_fit,
			'I_fit': I_fit
		}, trace

	@staticmethod
	def remove_high_outliers(values, std_thresh=3.0):
		values = np.asarray(values, dtype=float)
		mean = np.nanmean(values)
		std = np.nanstd(values)

		outlier_mask = values > (mean + std_thresh * std)
		values[outlier_mask] = np.nan
		return values

	@staticmethod
	def append_slope_measurement_to_file(file, distance_um, slope_kOhm):
		with open(file, "a", newline="") as f:
			writer = csv.writer(f)
			writer.writerow([distance_um, slope_kOhm])

	@staticmethod
	def select_current_indices(n_currents, n_to_select=10):
		# we select all currents if n_to_select is 'all/None' or if n_to_select is greater than the number of currents
		if n_to_select is None or n_to_select == 'all' or n_to_select >= n_currents:
			return np.arange(n_currents)

		# otherwise we select n_to_select evenly spaced indices
		return CommonUtils.pick_evenly(np.arange(n_currents), n_to_select)

	@staticmethod
	def pick_evenly(index_range, count):
		# return an empty array if index_range is empty or count is 0
		if len(index_range) == 0 or count == 0:
			return np.array([], dtype=int)

		# otherwise, pick indices evenly spaced within the range
		picks = np.linspace(0, len(index_range) - 1, min(count, len(index_range)))
		picks = np.unique(picks.round().astype(int))
		return index_range[picks]
