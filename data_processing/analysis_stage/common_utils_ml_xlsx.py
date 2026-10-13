import os.path

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter, LogLocator, FuncFormatter

distance_column_name = "distance (um)"
resistance_column_name = "resistance_mean (kOhm)"
sheet_name_prefix = "stim_"


class MLExcelProcessingUtils:

	@staticmethod
	def _create_spacing_array(spacing, max_distance, start_distance=5, floor_spacing=True):
		# create an array of distances, starting from start_distance to max_distance and spaced by spacing
		if floor_spacing:
			return np.floor(np.concatenate([[start_distance], np.arange(spacing, max_distance, spacing)]))
		else:
			return np.concatenate([[start_distance], np.arange(spacing, max_distance, spacing)])

	@staticmethod
	def _plot_single_trace(distances, resistances, label=None, xlims=None, ylim=None):
		# start new figure
		plt.figure()

		# plot the trace
		plt.loglog(distances, resistances, 'o-', markersize=4)

		# add labels and title
		plt.xlabel("Distance ($\mu$m)")
		plt.ylabel("Resistance (kOhm)")
		plt.title(f"{label}")

		# set up the grid
		plt.grid(True, which="both", ls="--")
		plt.xlim(xlims)
		plt.ylim(ylim)

		# add legend
		plt.tight_layout()

		# show figure
		plt.show()

	@staticmethod
	def _create_average_trace(all_traces, distance_col=distance_column_name,
							  resistance_col=resistance_column_name):
		# initialize output variables
		distance_lists = []
		spacings = []

		# iterate over all traces
		for trace_name, df in all_traces.items():
			if distance_col in df.columns:

				# store distance array
				dist = df[distance_col].values
				distance_lists.append(dist)

				# compute spacing
				spacings.append(np.nanmedian(np.diff(dist)))

		# find the widest spacing
		common_spacing = round(max(spacings), 2)

		# find max distance across all traces
		max_distance = max(dist.max() for dist in distance_lists)

		# create reference distance array and spacing arrays for mapping
		distances = MLExcelProcessingUtils._create_spacing_array(common_spacing, max_distance, floor_spacing=False)
		floored_distances = MLExcelProcessingUtils._create_spacing_array(common_spacing, max_distance, floor_spacing=True)
		floored_spacing_arrays = {np.floor(s): MLExcelProcessingUtils._create_spacing_array(s, max_distance) for s in set(
			spacings)}

		# floor common spacing after creating the reference grid to avoid issues with spacing calculation
		floored_common_spacing = np.floor(common_spacing)

		# create an array to store all resistance values
		resistance_array = np.full((len(all_traces), len(floored_distances)), np.nan)

		# fill the array using interpolation or direct mapping
		for i, (trace_name, df) in enumerate(all_traces.items()):
			if distance_col in df.columns and resistance_col in df.columns:

				# extract trace data
				trace_distances = df[distance_col].values
				trace_resistance = df[resistance_col].values

				# calculate current spacing
				spacing = np.floor(np.nanmedian(np.diff(trace_distances[1:])))

				# floor trace distances here after spacing to avoid issues with spacing calculation
				trace_distances = np.floor(trace_distances)

				# interpolate to reference grid if different from common spacing
				if spacing != floored_common_spacing:

					# get trace spacing array for mapping
					reference_spacing_array = floored_spacing_arrays[spacing]

					# identify which reference spacing points are missing in this current trace
					missing_mask = ~np.isin(reference_spacing_array, trace_distances)
					missing_points = reference_spacing_array[missing_mask]

					# insert nan for missing reference points
					if len(missing_points) > 0:
						insert_indices = np.searchsorted(trace_distances, missing_points)
						trace_distances = np.insert(trace_distances, insert_indices, missing_points)
						trace_resistance = np.insert(trace_resistance, insert_indices, np.nan)

					# remove entries where distance is nan for interpolation
					valid_distance_mask = ~np.isnan(trace_distances)
					valid_trace_distances = trace_distances[valid_distance_mask]
					valid_trace_resistance = trace_resistance[valid_distance_mask]

					# interpolate all reference distances using valid trace data
					interpolated = np.interp(distances, valid_trace_distances, valid_trace_resistance,
											 left=np.nan, right=np.nan)

					# assign interpolated values directly
					resistance_array[i, :] = interpolated

				# map existing values to matching distances, leaving nan for missing ones
				else:
					mask = np.isin(floored_distances, trace_distances)
					indices = np.searchsorted(trace_distances, floored_distances[mask])
					resistance_array[i, mask] = trace_resistance[indices]

		# compute statistics
		average_resistance_trace = np.nanmean(resistance_array, axis=0)
		std_resistance_trace = np.nanstd(resistance_array, axis=0)

		return distances, average_resistance_trace, std_resistance_trace, resistance_array

	@staticmethod
	def _plot_average_trace(distances, average_trace, xlim=None, ylim=None, horizontal_lines=[], vertical_lines=[],
							std_trace=None, units='kOhm', leave_open=False, legend_label='Average Trace',
							trace_color='black', legend_loc="upper right"):
		# start a new plot
		fig, ax = plt.subplots(figsize=(12, 8))

		# add average trace
		ax.plot(distances, average_trace, 'o-', linewidth=2, label=legend_label, color=trace_color)

		# add horizontal lines, if provided
		for value in horizontal_lines:
			ax.axhline(value, linestyle='--', color='black', label=f'{value} {units}')

		# add vertical lines, if provided
		for value in vertical_lines:
			ax.axvline(value, linestyle='-', color='black', label=f'{value} um')

		# add avg+std and avg-std
		if std_trace is not None:
			# calculate the standard deviation area
			average_plus_std = average_trace + std_trace
			average_minus_std = average_trace - std_trace

			# ensure there are no negative values
			average_minus_std[average_minus_std < 0] = 1E-14

			# create the std area
			ax.fill_between(distances, average_plus_std, average_minus_std, alpha=0.2, color=trace_color)

		# set log-log scale
		ax.loglog()

		# set y-axis log labels at 1, 10, 100, 1000
		y_formatter = ScalarFormatter()
		y_formatter.set_scientific(False)
		ax.set_yticks([1, 10, 100, 1000])
		ax.yaxis.set_major_formatter(y_formatter)

		# add labels on the x-axis labeling minor ticks
		x_formatter = ScalarFormatter()
		x_formatter.set_scientific(False)
		ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 3, 6)))
		ax.xaxis.set_major_formatter(x_formatter)
		ax.xaxis.set_minor_formatter(x_formatter)

		ax.tick_params(axis='both', which='both', labelsize=18)

		# enable major and minor gridlines
		ax.grid(True, which='both', alpha=0.7)
		if xlim is not None:
			ax.set_xlim(xlim)
		if ylim is not None:
			ax.set_ylim(ylim)

		# add titles
		ax.set_xlabel('Distance [µm]', fontsize=18, labelpad=10)
		ax.set_ylabel(f'Resistance [{units}]', fontsize=18, labelpad=10)

		# add legend
		if legend_loc:
			plt.legend(loc=legend_loc, fontsize=18)
		else:
			plt.legend(loc='best', fontsize=18)
		plt.tight_layout()

		# show plot, unless the caller wants to keep adding to the figure
		if not leave_open:
			plt.show()

	@staticmethod
	def load_traces_from_excel_files(xlsx_paths):
		# initialize output
		all_traces = {}

		# iterate over all xlsx files
		for xlsx_path in xlsx_paths:

			try:
				# get all sheet names
				xl_file = pd.ExcelFile(xlsx_path)
				sheet_names = xl_file.sheet_names

				# iterate over all sheets
				for sheet_name in sheet_names:

					# only process sheets that start with "stim_"
					if not sheet_name.startswith(sheet_name_prefix):
						continue

					# read the sheet
					df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

					# drop incomplete rows (same as ml_excel_files)
					df = df.dropna(subset=[distance_column_name, resistance_column_name], how="any")

					# skip empty sheets
					if df.empty:
						print(f"skipping {sheet_name}: no valid data after cleaning")
						continue

					# store data with a unique key
					file_key = f"{Path(xlsx_path).stem}_{sheet_name}"
					all_traces[file_key] = df

			# if there's an error, skip this file
			except Exception as e:
				print(f"error loading {xlsx_path}: {e}")
				continue

		return all_traces

	@staticmethod
	def plot_traces_separately_and_together(excel_path, xlim=None, ylim=None):
		# load all traces from the given Excel files
		all_traces = MLExcelProcessingUtils.load_traces_from_excel_files([excel_path])

		# iterate over the traces and plot each one
		for trace_name, df in all_traces.items():
			distances = df[distance_column_name].values
			resistances = df[resistance_column_name].values
			MLExcelProcessingUtils._plot_single_trace(distances, resistances, label=trace_name, xlims=xlim, ylim=ylim)

		# plot all traces combined
		plt.figure()
		for trace_name, df in all_traces.items():

			# extract data
			distances = df[distance_column_name].values
			resistances = df[resistance_column_name].values

			# build name
			stim_num = "stim_" + trace_name.split("_")[-1]

			# plot
			plt.loglog(distances, resistances, 'o-', label=stim_num, markersize=4)

		plt.xlabel("Distance ($\mu$m)")
		plt.ylabel("Resistance (kΩ·cm)")
		plt.grid(True, which="both", ls="--")
		plt.legend(fontsize="small")
		plt.tight_layout()
		plt.xlim(xlim)
		plt.ylim(ylim)
		plt.show()

	@staticmethod
	def create_and_plot_average_trace(xlsx_paths, distance_column=distance_column_name,
									  resistance_column=resistance_column_name, xlim=None, ylim=None,
									  horizontal_lines=[], vertical_lines=[], convert_to_megaohms=False,
									  leave_open=False, legend_label=None, trace_color=None, legend_loc=None):
		# initialize return values
		distances = None
		average_resistance_trace = None
		std_resistance_trace = None

		# load all traces from the given Excel files
		all_traces = MLExcelProcessingUtils.load_traces_from_excel_files(xlsx_paths)

		# if no traces were loaded, exit
		if not all_traces:
			print("no traces loaded. check file paths and column names.")
			return distances, average_resistance_trace, std_resistance_trace

		# combine and average all traces
		(distances, average_resistance_trace, std_resistance_trace, resistance_array) = (
			MLExcelProcessingUtils._create_average_trace(all_traces, distance_column, resistance_column))

		# convert resistance values from kOhm to MOhm if needed
		if convert_to_megaohms:
			average_resistance_trace = np.array(average_resistance_trace) / 1000.0
			std_resistance_trace = np.array(std_resistance_trace) / 1000.0

		# report results
		print(f"\nsuccessfully loaded {len(all_traces)} traces")
		print(f"final averaged trace has {len(distances)} points")
		print(f"distance range: {np.min(distances):.2f} to {np.max(distances):.2f}")

		# plot results
		units = 'MΩ' if convert_to_megaohms else 'kΩ'
		MLExcelProcessingUtils._plot_average_trace(distances, average_resistance_trace, xlim, ylim, horizontal_lines,
												   vertical_lines, std_resistance_trace, units=units,
												   leave_open=leave_open, legend_label=legend_label,
												   trace_color=trace_color, legend_loc=legend_loc)

		return distances, average_resistance_trace, std_resistance_trace

	@staticmethod
	def compute_log_space_rmse(experimental_trace, forward_model_trace, label=None):
		"""
		Estimate the distance between an experimental resistance trace and a forward-model resistance trace,
		using the root mean squared error (RMSE) computed in log space (log10 of the resistance values).

		If the two traces have different lengths, only the overlapping first N points are
		compared, where N = min(len(experimental_trace), len(forward_model_trace)).
		"""
		# ensure inputs are numpy arrays
		experimental_values = np.asarray(experimental_trace, dtype=float)
		forward_model_values = np.asarray(forward_model_trace, dtype=float)

		# truncate to the overlapping length
		n_points = min(len(experimental_values), len(forward_model_values))
		experimental_values = experimental_values[:n_points]
		forward_model_values = forward_model_values[:n_points]

		# compute log-space RMSE
		log_experimental = np.log10(experimental_values)
		log_forward_model = np.log10(forward_model_values)
		mse_log_space = np.nanmean((log_experimental - log_forward_model) ** 2)
		rmse_log_space = np.sqrt(mse_log_space)

		# report the result
		label_str = f" ({label})" if label else ""
		fold_error = 10 ** rmse_log_space
		print(f"Log-space RMSE{label_str}: {rmse_log_space:.6f} decades (~{fold_error:.2f}x typical fold-error)")

		return rmse_log_space

	@staticmethod
	def save_average_trace_to_excel(average_resistance_trace, output_path, first_four_columns=None,
									convert_to_megaohms=True, resistance_std=None, ML_format=False):
		# convert resistance values from kOhm to MOhm if needed
		if convert_to_megaohms:
			resistance_values = np.array(average_resistance_trace) / 1000.0
			std_values = np.array(resistance_std) / 1000.0 if resistance_std is not None else None
		else:
			resistance_values = average_resistance_trace
			std_values = resistance_std if resistance_std is not None else None

		# create the data dictionary
		data = {}

		# if the values of the layers are not provided, set them to None
		if first_four_columns is None:
			num_blanks = 5 if ML_format else 4
			first_four_columns = {f'blank_{i}': None for i in range(num_blanks, 0, -1)}

		for col_name, col_value in first_four_columns.items():
			data[col_name] = [col_value] if col_value is not None else [None]

		# add voltage columns (V_0, V_1, V_2, etc.) with resistance values
		for i, resistance in enumerate(resistance_values):
			column_name = f'V_{i}' if not ML_format else f'{i}'
			data[column_name] = [resistance]

		# add ID column for ML Excel files
		if ML_format:
			data['id'] = ['Avg']

		# create DataFrame
		df = pd.DataFrame(data)

		# add std rows if provided
		if std_values is not None:

			# add plus std
			avg_plus_std = {col_name: [col_value] if col_value is not None else [None]
							for col_name, col_value in first_four_columns.items()}

			for i in range(len(resistance_values)):
				column_name = f'V_{i}' if not ML_format else f'{i}'
				avg_plus_std[column_name] = [resistance_values[i] + std_values[i]]

			# add label
			if ML_format:
				avg_plus_std['id'] = ['Avg+STD']

			# add minus std
			avg_minus_std = {col_name: [col_value] if col_value is not None else [None]
							 for col_name, col_value in first_four_columns.items()}

			for i in range(len(resistance_values)):
				column_name = f'V_{i}' if not ML_format else f'{i}'
				avg_minus_std[column_name] = [resistance_values[i] - std_values[i]]

			# add label
			if ML_format:
				avg_minus_std['id'] = ['Avg-STD']

			# add to DataFrame
			df = pd.concat([df, pd.DataFrame(avg_plus_std), pd.DataFrame(avg_minus_std)], ignore_index=True)

		# save DataFrame to Excel
		df.to_excel(output_path, index=False, engine='openpyxl')
		print(f"\nAverage trace saved to: {output_path}")
		print(f"Saved {len(average_resistance_trace)} resistance values")

		return df

	@staticmethod
	def plot_list_of_average_traces_and_stds(distances_list, average_traces_list, std_list, labels, alpha=0.25,
											 xlim=None, ylim=None, colors=None):
		# start a new plot
		fig, ax = plt.subplots(figsize=(12, 10))

		# iterate over the average traces and plot them
		for i in range(len(average_traces_list)):

			# calculate the standard deviation area
			average_plus_std = average_traces_list[i] + std_list[i]
			average_minus_std = average_traces_list[i] - std_list[i]

			# ensure there are no negative values
			average_minus_std[average_minus_std < 0] = 1E-14

			if colors is not None:
				# plot the average trace
				ax.plot(distances_list[i], average_traces_list[i], 'o-', linewidth=2, label=labels[i], color=colors[i])

				# plot the std area
				ax.fill_between(distances_list[i], average_plus_std, average_minus_std, alpha=alpha, color=colors[i])
			else:
				# plot the average trace
				ax.plot(distances_list[i], average_traces_list[i], 'o-', linewidth=2, label=labels[i])

				# plot the std area
				ax.fill_between(distances_list[i], average_plus_std, average_minus_std, alpha=alpha)

		# set log-log scale
		ax.loglog()

		# set y-axis log labels at 1, 10, 100, 1000
		y_formatter = ScalarFormatter()
		y_formatter.set_scientific(False)
		ax.set_yticks([0.1, 1, 10, 100, 1000])
		ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{val:g}'))

		# add labels on the x-axis labeling minor ticks
		x_formatter = ScalarFormatter()
		x_formatter.set_scientific(False)
		ax.xaxis.set_major_formatter(x_formatter)
		ax.set_xticks([33, 100, 200, 300, 400, 500, 600])

		ax.tick_params(axis='both', which='both', labelsize=15, pad=10)

		# enable major and minor gridlines
		ax.grid(True, which='both', alpha=0.3)
		ax.set_ylim(0.1, 1000) if ylim is None else ax.set_ylim(ylim)
		ax.set_xlim(4, 1000) if xlim is None else ax.set_xlim(xlim)
		plt.legend()
		plt.xlabel('Distance [$\mu$m]', fontsize=20, labelpad=15)
		plt.ylabel('Resistance [kOhm]', fontsize=20, labelpad=15)
		plt.show()

	@ staticmethod
	def plot_sample_trace_with_equivalent_saline(xlsx_paths):
		# load all traces from the given Excel files
		all_traces = MLExcelProcessingUtils.load_traces_from_excel_files(xlsx_paths)

		# initialize the longest sheet and its max distance
		longest_sheet = None
		longest_max_distance = -np.inf

		# iterate over the sheets and find the one with the longest distance range
		for sheet_name, df in all_traces.items():

				# if no data, skip this sheet
				if df.empty:
					continue

				# extract current distance range
				max_dist = df[distance_column_name].max()

				# if this distance is longer than the longest distance, update the longest sheet
				if max_dist > longest_max_distance:
					longest_max_distance = max_dist
					longest_sheet = sheet_name

		# if the counter wasn't updated, there is no valid information
		if longest_sheet is None:
			print("No valid stim sheets")
			return

		# copy the longest sheet data
		df_single = all_traces[longest_sheet].copy()

		# sort by distance
		df_single = df_single.sort_values(distance_column_name)

		# extract column values
		distances_um = df_single[distance_column_name].values
		resistances = df_single[resistance_column_name].values

		# if there is more than one point, compute the theoretical curve
		if len(distances_um) > 1:

			# convert micro-meters to meters
			distances_m = distances_um * 1e-6

			# extract the first point
			d0 = distances_m[0]
			V0 = resistances[0]

			# compute the theoretical curve
			theoretical = (V0 * d0) / distances_m

			# convert kOhm to Ohm
			V0_ohm = V0 * 1e3

			# compute the equivalent saline resistivity using V = I * rho_saline / 2*pi*r
			rho_saline = 2 * np.pi * V0_ohm * d0
			rho_saline_cm = rho_saline * 100

			# plot the results
			plt.figure()

			# plot the measured trace
			plt.loglog(
				distances_um,
				resistances,
				'o-',
				markersize=4,
				linewidth=3,
				label=f"{longest_sheet} measured",
				color='#256080'
			)

			# plot the theoretical trace
			plt.loglog(
				distances_um,
				theoretical,
				'--',
				linewidth=3,
				label=f"equivalent saline ({rho_saline_cm:.1f} Ω·cm)",
				color='#2E7D32'
			)

			# configure the plot
			plt.xlabel("Distance (um)")
			plt.ylabel("Resistance (kOhm)")
			plt.title(f"Measured vs Equivalent Saline ({rho_saline_cm:.1f} Ω·cm)")
			plt.grid(True, which="both", ls="--")
			plt.legend()
			plt.tight_layout()
			plt.show()

	@ staticmethod
	def create_resistance_coverage_plot(excel_path):
		# load Excel file
		xls = pd.ExcelFile(excel_path)
		stim_sheets = [sheet for sheet in xls.sheet_names if sheet.startswith(sheet_name_prefix)]

		# if there are no stim sheets, exit
		if not stim_sheets:
			print(f"No stim sheets found")
			return

		# initialize variables
		stim_data = {}
		max_distances_electrode = None
		max_distance_count = 0
		reference_distances = None

		# iterate over all electrodes and find the one with the longest distance range
		for sheet_name in stim_sheets:

			# read file
			df = pd.read_excel(xls, sheet_name=sheet_name)

			# drop rows with NaN resistance values
			df_clean = df.dropna(subset=[resistance_column_name], how="any")

			# if the current sheet has data, extract the distance points and update the max distance count
			if not df_clean.empty:
				stim_idx = df_clean["stim_idx"].iloc[0] if "stim_idx" in df_clean.columns else int(sheet_name.split("_")[1])
				distances = sorted(df_clean[distance_column_name].values)
				stim_data[stim_idx] = distances

				# update the reference electrode if this electrode has the longest distance range
				if len(distances) > max_distance_count:
					max_distance_count = len(distances)
					max_distances_electrode = stim_idx
					reference_distances = distances

		# if there is no valid data, exit
		if not stim_data or reference_distances is None:
			print(f"No valid data found in {excel_path}")
			return

		# sort electrodes by their maximum distance coverage (shortest to longest)
		sorted_stims = sorted(stim_data.keys(), key=lambda x: max(stim_data[x]))

		# create the plot matrix based on reference distances
		fig, ax = plt.subplots(figsize=(max(12, len(reference_distances) * 0.3), len(sorted_stims) * 0.4 + 2))

		# iterate over the stimulation electrodes
		for row_idx, stim_idx in enumerate(sorted_stims):
			electrode_distances = stim_data[stim_idx]
			max_electrode_distance = max(electrode_distances)

			# iterate over reference distances
			for col_idx, ref_distance in enumerate(reference_distances):

				# check if this electrode has data at this specific distance
				has_data = ref_distance in electrode_distances

				# this is valid data - blue color
				if has_data:
					color = 'blue'
					alpha = 0.8

				# this is beyond the electrode's range - grey out
				elif ref_distance > max_electrode_distance:
					color = 'grey'
					alpha = 0.3

				# this is no data
				else:
					continue

				# draw a rectangle for this distance point
				rect = plt.Rectangle(
					(col_idx - 0.4, row_idx - 0.4),
					0.8,
					0.8,
					facecolor=color,
					alpha=alpha,
					edgecolor='black',
					linewidth=0.5
				)
				ax.add_patch(rect)

		# customize the plot
		ax.set_xlim(-0.5, len(reference_distances) - 0.5)
		ax.set_ylim(-0.5, len(sorted_stims) - 0.5)

		# set y-axis labels (stim indices sorted by coverage)
		ax.set_yticks(range(len(sorted_stims)))
		y_labels = [f"Stim {idx} ({max(stim_data[idx]):.0f}μm)" for idx in sorted_stims]
		ax.set_yticklabels(y_labels)

		# set x-axis labels (actual distances from reference electrode, every nth distance label to avoid overcrowding)
		step = max(1, len(reference_distances) // 15)  # Show ~15 labels max
		x_ticks = range(0, len(reference_distances), step)
		x_labels = [f"{int(reference_distances[i])}" for i in x_ticks]

		ax.set_xticks(x_ticks)
		ax.set_xticklabels(x_labels, rotation=45)

		ax.set_xlabel("Distance (μm)")
		ax.set_ylabel("Stimulation Electrodes (sorted by max distance)")
		ax.set_title(f"{os.path.basename(excel_path)}\n"
					 f"Reference: Electrode {max_distances_electrode} ({len(reference_distances)} points, "
					 f"max: {reference_distances[-1]:.1f} μm)")

		# add grid
		ax.grid(True, alpha=0.3)

		# add legend
		legend_elements = [
			Patch(facecolor='blue', alpha=0.8, label='Valid resistance data'),
			Patch(facecolor='grey', alpha=0.3, label='Beyond electrode range')
		]
		ax.legend(handles=legend_elements, loc='upper right')

		plt.tight_layout()
		plt.show()

