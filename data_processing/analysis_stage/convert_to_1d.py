import os
import re
import pandas as pd
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.signal import find_peaks
from collections import OrderedDict
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D


# number of header lines in the inversion result file
NUM_HEADER_LINES = 48

# a fixed list of colors to cycle through
colors = cycle(['brown', 'green', 'yellow', 'purple', 'orange', 'magenta', 'red'])


class Res2DInvOutputProcessor:

	@staticmethod
	def _get_column_names_and_widths(file_path, column_names_line):
		with open(file_path, 'r') as file:

			# skip lines up to the column names line
			for _ in range(column_names_line):
				file.readline()

			# read the column names line
			column_line = file.readline().replace("/", "").strip()
			columns = column_line.split()

			# read the next line for column widths
			width_line = file.readline()

		# find all column names and their leading spaces
		matches = re.findall(r'(\s*)(\S+)', width_line)

		# for all remaining columns, calculate the width (leading space + value length)
		widths = [len(match[0]) + len(match[1]) for match in matches]

		return columns, widths

	@staticmethod
	def _custom_depth_value_converter(value):
		# convert to numeric representation
		value = float(value)

		# if non-zero, return negative
		if value > 0:
			return -value

		# if zero, return as is
		return value

	@staticmethod
	def _extended_ground_truth_by_layer(z_axis, extended_ground_truth, layers, z_range_to_use):
		# rebuild the ground truth trace so each layer's value spans its full depth boundary
		keys = list(layers.keys())
		z_stair, gt_stair = [], []

		for i, key in enumerate(keys):

			depth1 = float(key)
			depth2 = float(keys[i + 1]) if i < len(keys) - 1 else z_range_to_use[0]
			lo, hi = sorted([depth1, depth2])
			value = next((v for z, v in zip(z_axis, extended_ground_truth) if v and lo <= z <= hi), None)

			if value is None:
				continue

			z_stair.extend([depth1, depth2])
			gt_stair.extend([value, value])

		return gt_stair, z_stair

	@staticmethod
	def _layer_boundaries(layers, z_range_to_use):
		"""
		Build the (depth1, depth2) boundary pair for each layer in `layers`, using dict insertion order (shallow to
		# deep). The last layer's open end is closed with z_range_to_use[0].
		"""
		keys = list(layers.keys())
		boundaries = []
		for i, key in enumerate(keys):
			depth1 = float(key)
			depth2 = float(keys[i + 1]) if i < len(keys) - 1 else z_range_to_use[0]
			boundaries.append((depth1, depth2))
		return boundaries

	@staticmethod
	def _find_extremum_index(values, min_prominence=0.0):
		"""
		Locate a single clear peak or valley inside `values`
		"""
		# ensure values are a numpy array of floats
		values = np.asarray(values, dtype=float)

		# ensure we have at least 3 values
		if len(values) < 3:
			return None

		# find the peak and valley indices and values
		peak_idx, peak_props = find_peaks(values, prominence=0)
		valley_idx, valley_props = find_peaks(-values, prominence=0)

		# create tuples of (index, prominence) for both peaks and valleys
		candidates = (list(zip(peak_idx, peak_props['prominences'])) +
					  list(zip(valley_idx, valley_props['prominences'])))

		# if no candidates, return None
		if not candidates:
			return None

		# find extremum index and prominence
		best_idx, best_prominence = max(candidates, key=lambda c: c[1])
		if best_prominence < min_prominence:
			return None

		return int(best_idx)

	def _ground_truth_by_label_from_extended(self, z_axis, extended_ground_truth, layers, z_range_to_use, labels):
		"""
		Extract a per-label ground truth value from extended_ground_truth, using each layer's own depth boundaries.
		"""
		boundaries = self._layer_boundaries(layers, z_range_to_use)
		ground_truth_by_label = OrderedDict()
		for label, (depth1, depth2) in zip(labels, boundaries):
			lo, hi = sorted([depth1, depth2])
			value = next((v for z, v in zip(z_axis, extended_ground_truth) if v and lo <= z <= hi), None)
			if value is not None:
				ground_truth_by_label[label] = float(value)
		return ground_truth_by_label

	@staticmethod
	def _resolve_from_points(entry, points, method, inspected=None):
		"""
		Set the entry's value and marker_z fields to the mean of the given points.
		"""
		entry['value'] = float(np.mean([r for _, r in points]))
		entry['marker_z'] = float(np.mean([z for z, _ in points]))
		entry['method'] = method
		entry['used_points'] = points
		entry['inspected_points'] = inspected if inspected is not None else points

	@staticmethod
	def _plot_trace_analysis(z_axis, mean_resistivity, entries, ground_truth_by_label=None, z_range_to_use=None,
							 left_side_labels=None):
		"""
		Plot the extracted resistivity values on the 1D trace, marking the extraction method.
		"""
		left_side_labels = set(left_side_labels) if left_side_labels else set()
		fig, ax = plt.subplots(figsize=(10, 12))
		ax.plot(mean_resistivity, z_axis, 'o-', color='black', linewidth=1.5,
				markersize=4, label='Resistivity trace', zorder=2)

		layer_colors = {
			"NFL+GCL": "navy",
			"IPL": "magenta",
			"INL+OPL+ONL": "green",
			"INL": "green",
			"IS+OS": "purple",
			"PBS": "teal",
		}
		default_color_cycle = cycle(["red", "brown", "pink", "olive", "cyan", "magenta"])
		default_colors = {}

		def get_color(label):
			if label in layer_colors:
				return layer_colors[label]
			if label not in default_colors:
				default_colors[label] = next(default_color_cycle)
			return default_colors[label]

		method_markers = {
			'extremum': 'o',
			'first_point': 's',
			'neighbor_average': 'x',
			'window_average': 'x',
			'single_point': 'v',
			'layer_average': 'x',
		}
		default_marker = 'x'
		excluded_color = 'lightgray'

		drawn_depths = set()
		for entry in entries:
			for depth in (entry['depth1'], entry['depth2']):
				if depth in drawn_depths:
					continue
				ax.axhline(depth, color='gray', linewidth=2, linestyle='--', zorder=1)
				drawn_depths.add(depth)

		label_targets = []
		for entry in entries:
			if entry['label'] == 'PBS':
				continue

			color = get_color(entry['label'])
			marker = method_markers.get(entry['method'], default_marker)
			inspected = set(entry['inspected_points'])

			for wz, wr in entry['window_points']:
				point_color = color if (wz, wr) in inspected else excluded_color
				ax.plot(wr, wz, 'o', color=point_color, markersize=10, markerfacecolor='none',
						markeredgewidth=1.5, alpha=0.2, zorder=3)

			if ground_truth_by_label is not None:
				gt_value = ground_truth_by_label.get(entry['label'])
				if gt_value is not None:
					ax.plot([gt_value, gt_value], [entry['depth1'], entry['depth2']],
							color=color, linestyle='dotted', linewidth=2, zorder=1)

			value = entry['value']
			if value is None:
				continue

			label, depth1, depth2, marker_z = (entry['label'], entry['depth1'],
											   entry['depth2'], entry['marker_z'])

			ax.plot([value, value], [depth1, depth2], color=color, linestyle='-',
					linewidth=2, zorder=1)

			used_points = entry['used_points']
			for used_z, used_r in used_points:
				is_extremum_point = (entry['method'] == 'extremum' and (used_z, used_r) == (marker_z, value))
				if not is_extremum_point:
					ax.plot(used_r, used_z, 'x', color=color, markersize=7, markeredgewidth=2, zorder=4)
				if len(used_points) > 1 or (used_z, used_r) != (marker_z, value):
					ax.plot([used_r, value], [used_z, marker_z], color=color,
							linestyle=':', linewidth=1, zorder=2)

			if entry['method'] == 'extremum':
				ax.plot(value, marker_z, marker='o', markerfacecolor='none',
						markeredgecolor=color, markersize=10, markeredgewidth=1.5, alpha=0.2, zorder=5)
			else:
				ax.plot(value, marker_z, marker=marker, color=color, markersize=9,
						markeredgecolor='black', markeredgewidth=0.5, zorder=5)

			label_targets.append((marker_z, label, value, color, entry['method']))

		x_lo, x_hi = ax.get_xlim()
		x_span = x_hi - x_lo
		z_limit_lo, z_limit_hi = sorted(z_range_to_use) if z_range_to_use is not None else (None, None)
		drawn_layer_labels = set()
		for entry in entries:
			key = (entry['label'], entry['depth1'], entry['depth2'])
			if key in drawn_layer_labels:
				continue
			drawn_layer_labels.add(key)

			# skip layers that fall entirely outside the requested depth range
			if z_range_to_use is not None:
				layer_lo, layer_hi = sorted([entry['depth1'], entry['depth2']])
				if layer_hi <= z_limit_lo or layer_lo >= z_limit_hi:
					continue

			on_left = entry['label'] in left_side_labels
			text_x = x_lo + 0.02 * x_span if on_left else x_hi - 0.02 * x_span
			text_y = (entry['depth1'] + entry['depth2']) / 2
			ax.text(text_x, text_y, entry['label'], ha='left' if on_left else 'right', va='center',
					fontsize=16, weight='bold', color='black', zorder=6)

		def method_tag(method):
			if method == 'extremum':
				return 'Ext'
			if 'average' in method:
				return 'Avg'
			return method.replace('_', ' ').title()

		# List each layer's extracted value (with how it was extracted) in the legend
		value_handles = [
			plt.Line2D([0], [0], marker='None', linestyle='none',
					   label=f'{label}: {value:.0f} ({method_tag(method)})')
			for marker_z, label, value, color, method in label_targets
		]
		if value_handles:
			value_box = ax.legend(handles=value_handles, loc='best', fontsize=18,
								  handlelength=0, handletextpad=0, frameon=True,
								  facecolor='white', framealpha=0.85)
			for text, (marker_z, label, value, color, method) in zip(value_box.get_texts(), label_targets):
				text.set_color(color)
			ax.add_artist(value_box)

		# configure axis
		ax.set_xlabel('Resistivity (Ω·cm)', fontsize=24, labelpad=15)
		ax.set_ylabel('Depth ($\mu$m)', fontsize=24, labelpad=15)
		ax.tick_params(axis='both', labelsize=16)

		# add title
		if ground_truth_by_label:
			title = ', '.join(f'{label}: {value:.2f}' for label, value in ground_truth_by_label.items())
		else:
			title = 'Layer resistivity extraction'
		ax.set_title(title, fontsize=12, pad=20)

		ax.grid(True, alpha=0.2, linestyle='--')
		if z_range_to_use is not None:
			ax.set_ylim(z_range_to_use)

		plt.tight_layout()
		plt.show()

		return fig, ax

	@staticmethod
	def convert_to_1d_trace(file_path, x_range_to_use=[-300, 500], ground_truth=None, output_folder=None):
		# extract file name
		file_name = os.path.basename(file_path)

		# get column names from the file
		columns, widths = Res2DInvOutputProcessor._get_column_names_and_widths(file_path, NUM_HEADER_LINES - 1)

		# extract the number of horizontal and vertical blocks from the file header
		with open(file_path, 'r') as file:
			header = file.readlines()[:6]
			num_z_cells = int(header[4].split(":")[1].strip())
			num_x_cells = int(header[5].split(":")[1].strip())

		# load file and read into pandas dataframe
		inversion_data = pd.read_fwf(file_path,
									 header=None,
									 index=None,
									 skiprows=NUM_HEADER_LINES,
									 names=columns,
									 widths=widths,
									 converters={'z1': Res2DInvOutputProcessor._custom_depth_value_converter,
																'z2': Res2DInvOutputProcessor._custom_depth_value_converter,
																'z3': Res2DInvOutputProcessor._custom_depth_value_converter,
																'z4': Res2DInvOutputProcessor._custom_depth_value_converter,
																'zc': Res2DInvOutputProcessor._custom_depth_value_converter})

		# get the start of the grid
		x_min = inversion_data['x1'].min()

		# extract the x from the first row of the inversion data
		x_spacings = []
		for row in inversion_data[0:num_x_cells].iterrows():
			x_spacings.append(row[1]['x4'] - row[1]['x1'])

		# extract the z spacing from all the rows of the inversion data, jumping by num_x_cells
		z_spacings = []
		for row in inversion_data[0:num_x_cells * num_z_cells:num_x_cells].iterrows():
			z_spacings.append(round(row[1]['z2'] - row[1]['z1'], 2))

		# reconstruct the axis based on the spacings
		x_axis = x_min + np.cumsum(np.insert(x_spacings, 0, 0))
		z_axis = np.cumsum(z_spacings)

		# given ground truth values, extrapolate the given dictionary into an array
		extended_ground_truth = []
		if ground_truth:
			for z in z_axis:
				for key, ground_truth_value in sorted(ground_truth.items(), key=lambda x: float(x[0])):
					if z <= float(key) and ground_truth_value:
						extended_ground_truth.append(float(ground_truth_value))
						break

		# if extended_ground_truth was not populated, then initialize it with None values
		if len(extended_ground_truth) == 0:
			extended_ground_truth = [None] * len(z_axis)

		# initialize an empty resistivity matrix
		resistivity_matrix = np.zeros((num_z_cells, num_x_cells))

		# loop through the blocks and populate the resistivity matrix
		for inv_index, row in inversion_data.iterrows():

			# calculate the row and column indices of the block
			col_index = inv_index % num_x_cells
			row_index = inv_index // num_x_cells

			# extract the resistivity value of the block
			res_ohm_cm = row['res']

			# populate the resistivity matrix
			resistivity_matrix[row_index, col_index] = res_ohm_cm

		# convert the x_range_to_use to indices
		start_index, end_index = [np.argmin(np.abs(x_axis - x_range_to_use[0])), np.argmin(np.abs(x_axis - x_range_to_use[1]))]

		# initialize output variables
		mean_resistivity = list()
		std_resistivity = list()
		negative_std_resistivity = list()
		positive_std_resistivity = list()

		# iterate over the z_axis, and for each row (depth value), calculate the mean resistivity in the center
		for row_index in range(num_z_cells):

			# extract the resistivity values of the blocks_to_use in the center
			data = resistivity_matrix[row_index, start_index:end_index + 1]

			# calculate the mean resistivity of the blocks_to_use in the center
			mean = np.nanmean(data)
			mean_resistivity.append(mean)

			# calculate the std of the blocks_to_use in the center
			std = np.nanstd(data)
			std_resistivity.append(std)

			# calculate negative and positive std values
			negative_std = np.std(data[data < mean]) if np.any(data < mean) else 0
			positive_std = np.std(data[data > mean]) if np.any(data > mean) else 0
			negative_std_resistivity.append(negative_std)
			positive_std_resistivity.append(positive_std)

		# save the data (mean and std) to a csv file
		data_file = pd.DataFrame({"z_axis (um)": z_axis, "mean_resistivity (Ohm-cm)": mean_resistivity,
								  "std_resistivity": std_resistivity, "ground_truth_resistivity (Ohm-cm)":
									  extended_ground_truth})
		data_file.to_csv(os.path.join(output_folder, f"1d_trace_{file_name}.csv"), index=False)

		return mean_resistivity, std_resistivity, z_axis, extended_ground_truth

	@staticmethod
	def plot_1d_trace(file_name, ground_truth, extended_ground_truth, mean_resistivity, labels, std_resistivity,
					  z_axis, z_range_to_use, x_range_to_plot=None, kohm_labels=False, left_side_labels=None,
					  legend_label=None, legend_position=None):
		# plot the mean resistivity as a function of depth with error bars
		plt.figure(figsize=(12, 10))

		# add ground truth line, if the values are provided
		if any(extended_ground_truth):
			gt_label = f'{legend_label}_GT' if legend_label else 'Ground Truth'
			plt.plot(extended_ground_truth, z_axis, color='green', linestyle='-', linewidth=2, label=gt_label)

		# add resistivity with error bars plot
		plt.errorbar(mean_resistivity, z_axis, xerr=std_resistivity, fmt='o', capthick=0.4, capsize=0.4, elinewidth=2,
					 ecolor='black', color='black', label='Mean Resistivity', linestyle='--', linewidth=2)

		# common plot parameters
		max_ground_truth = max(extended_ground_truth) if any(extended_ground_truth) else 0
		max_resistivity = max_ground_truth + max_ground_truth * 0.5
		if max_resistivity == 0:
			max_resistivity = max(mean_resistivity) + max(mean_resistivity) * 0.5
		xticks_font_size = 14
		yticks_font_size = 14
		label_font_size = 16

		# add layers boundaries
		Res2DInvOutputProcessor.add_layer_bounds_and_labels_to_plot(ground_truth, width_limit=max_resistivity,
																	depth_limit=z_range_to_use, labels=labels,
																	left_side_labels=left_side_labels)

		# calculate the number of x-ticks to make sure we can see the text
		desired_num_ticks = 12
		tick_interval = round(max_resistivity / (desired_num_ticks - 1))
		x_ticks = np.arange(0, max_resistivity + tick_interval, tick_interval)
		y_ticks = np.arange(z_range_to_use[1], z_range_to_use[0], -20)

		# add limits, labels, etc.
		if kohm_labels:
			plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x / 1000:g}'))
			plt.xlabel('Resistivity (kΩ·cm)', fontsize=label_font_size, labelpad=15)
		else:
			plt.xlabel('Resistivity (Ω·cm)', fontsize=label_font_size, labelpad=15)

		plt.ylabel('Depth ($\mu$m)', fontsize=label_font_size, labelpad=15)
		plt.title(file_name, fontsize=16, pad=20)
		plt.xticks(x_ticks, size=xticks_font_size)
		plt.yticks(y_ticks, size=yticks_font_size)
		plt.xlim([0, max_resistivity])
		plt.ylim(z_range_to_use)

		if x_range_to_plot:
			plt.xlim(x_range_to_plot)

		plt.grid()

		if legend_position:
			plt.legend(fontsize=16, framealpha=0.9, loc=legend_position)
		else:
			plt.legend(fontsize=16, framealpha=0.9, loc='center left', bbox_to_anchor=(0, 0.6))

		# add borders
		for spine in plt.gca().spines.values():
			spine.set_edgecolor('black')
			spine.set_linewidth(2)

		# display figure
		plt.show()

	@staticmethod
	def plot_1d_traces_combined(file_data_list, layers, labels, z_range_to_use, x_range_to_plot=None,
								legend_labels=None, colors=None, layer_bound_ground_truth=False, point_format=None,
								kohm_labels=False, left_side_labels=None, legend_position=None, ax=None,
								layer_label_font_size=16, orientation='horizontal'):
		# create a figure, unless axes to draw into were provided
		standalone = ax is None
		if standalone:
			fig, ax = plt.subplots(figsize=(12, 10))
		else:
			plt.sca(ax)

		# color cycle for different traces
		colors = plt.cm.tab10(np.linspace(0, 1, len(file_data_list))) if colors is None else colors

		# get max resistivity for plot limits
		max_resistivity = 0
		for file_name, mean_resistivity, std_resistivity, z_axis, extended_ground_truth in file_data_list:
			max_ground_truth = max(extended_ground_truth) if any(extended_ground_truth) else 0
			max_res = max_ground_truth + max_ground_truth * 0.5
			if max_res == 0:
				max_res = max(mean_resistivity) + max(mean_resistivity) * 0.5
			max_resistivity = max(max_resistivity, max_res)

		# plot each trace in a different color
		for idx, (file_name, mean_resistivity, std_resistivity, z_axis, extended_ground_truth) in enumerate(
				file_data_list):
			label = legend_labels[idx] if legend_labels and idx < len(legend_labels) else file_name

			if orientation == 'vertical':
				if np.any(extended_ground_truth):
					ax.errorbar(z_axis, mean_resistivity, yerr=std_resistivity, marker='None',
								capthick=0.5, capsize=0.4,
								elinewidth=2, color=colors[idx], label=label, linestyle='--', linewidth=3)
					if layer_bound_ground_truth:
						gt_values, gt_z = Res2DInvOutputProcessor._extended_ground_truth_by_layer(z_axis,
																								  extended_ground_truth,
																								  layers,
																								  z_range_to_use)
					else:
						gt_values, gt_z = extended_ground_truth, z_axis
					ax.plot(gt_z, gt_values, color=colors[idx], linestyle='-', linewidth=2, label=f'{label} GT')
				else:
					ax.errorbar(z_axis, mean_resistivity, yerr=std_resistivity, marker='None',
								capthick=0.5, capsize=0.4,
								elinewidth=2, color=colors[idx], label=label, linestyle='-', linewidth=2)
			else:
				if np.any(extended_ground_truth):
					ax.errorbar(mean_resistivity, z_axis, xerr=std_resistivity, marker='None',
								capthick=1, capsize=0.4,
								elinewidth=2, color=colors[idx], label=label, linestyle='--', linewidth=2)
					if layer_bound_ground_truth:
						gt_values, gt_z = Res2DInvOutputProcessor._extended_ground_truth_by_layer(z_axis,
																								  extended_ground_truth,
																								  layers,
																								  z_range_to_use)
					else:
						gt_values, gt_z = extended_ground_truth, z_axis
					ax.plot(gt_values, gt_z, color=colors[idx], linestyle='-', linewidth=2, label=f'{label} GT')
				else:
					ax.errorbar(mean_resistivity, z_axis, xerr=std_resistivity, marker='None',
								capthick=1, capsize=0.4,
								elinewidth=2, color=colors[idx], label=label, linestyle='-', linewidth=2)
		# add layer boundaries
		value_limit = x_range_to_plot[1] if x_range_to_plot else max_resistivity
		Res2DInvOutputProcessor.add_layer_bounds_and_labels_to_plot(layers, width_limit=value_limit,
																	depth_limit=z_range_to_use, labels=labels, left_side_labels=left_side_labels,
																	label_font_size=layer_label_font_size, orientation=orientation)

		# add labels, etc.
		label_font_size = 18

		# vertical plot
		if orientation == 'vertical':

			ax.set_xlabel('Depth ($\mu$m)', fontsize=label_font_size, labelpad=10)
			if kohm_labels:
				ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x / 1000:g}'))
				ax.set_ylabel('Resistivity (kΩ·cm)', fontsize=label_font_size, labelpad=10)
			else:
				ax.set_ylabel('Resistivity (Ω·cm)', fontsize=label_font_size, labelpad=10)

			# set ticks
			ticks_font_size = 18
			x_ticks = np.arange(z_range_to_use[1], z_range_to_use[0], -20)
			ax.set_xticks(x_ticks)
			ax.tick_params(axis='both', labelsize=ticks_font_size)

			# set limits
			ax.set_ylim([0, max_resistivity])
			ax.set_xlim(z_range_to_use)
			if x_range_to_plot:
				ax.set_ylim(x_range_to_plot)

		# horizontal plot
		else:

			ax.set_ylabel('Depth ($\mu$m)', fontsize=label_font_size, labelpad=10)
			if kohm_labels:
				ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x / 1000:g}'))
				ax.set_xlabel('Resistivity (kΩ·cm)', fontsize=label_font_size, labelpad=10)
			else:
				ax.set_xlabel('Resistivity (Ω·cm)', fontsize=label_font_size, labelpad=10)

			# set ticks
			ticks_font_size = 18
			y_ticks = np.arange(z_range_to_use[1], z_range_to_use[0], -20)
			ax.set_yticks(y_ticks)
			ax.tick_params(axis='both', labelsize=ticks_font_size)

			# set limits
			ax.set_xlim([0, max_resistivity])
			ax.set_ylim(z_range_to_use)
			if x_range_to_plot:
				ax.set_xlim(x_range_to_plot)

		# add grid
		ax.grid()

		# sort legend alphabetically
		handles, plotted_labels = ax.get_legend_handles_labels()
		label_to_handle = dict(zip(plotted_labels, handles))
		ordered_labels = sorted(label_to_handle)
		ordered_handles = [label_to_handle[l] for l in ordered_labels]

		clean_handles = []
		for h in ordered_handles:
			if hasattr(h, 'lines'):  # ErrorbarContainer
				data_line = h.lines[0]
				clean_handles.append(Line2D([], [], color=data_line.get_color(),
											linestyle=data_line.get_linestyle(),
											linewidth=data_line.get_linewidth()))
			else:
				clean_handles.append(h)

		# add legend
		if legend_position:
			ax.legend(clean_handles, ordered_labels, fontsize=14, framealpha=0.9, loc=legend_position)
		else:
			ax.legend(clean_handles, ordered_labels, fontsize=14, framealpha=0.9, loc='center left', bbox_to_anchor=(0, 0.6))

		# add borders
		for spine in ax.spines.values():
			spine.set_edgecolor('black')
			spine.set_linewidth(1)

		# only auto-show when we created the figure ourselves
		if standalone:
			plt.show()

	@staticmethod
	def run_1d_conversion(input_files, indices_to_process, x_range, depth_range,
						  labels, layers, ground_truth, plot=True, plot_all_together=False,
						  x_range_to_plot=None, calculate_average_resistivity=False, legend_labels=None, colors=None,
						  layer_bound_ground_truth=False, point_format=None, kohm_labels=False, left_side_labels=None,
						  legend_position=None, ax=None, layer_label_font_size=16, orientation='horizontal'):
		# initialize output
		output_data = []

		# collect data for a combined plot if requested
		file_data_list = [] if plot_all_together else None

		# initialize processed file index
		file_index = 0

		# iterate over the files
		for input_file_path in input_files:

			# only proceed with the requested indices
			if indices_to_process:

				# extract index
				index = os.path.basename(input_file_path).split("inversion_result_")[1].split("_")[0]

				# compare with the requested indices
				if float(index) not in indices_to_process:
					continue

			# define output folder
			output_folder = os.path.join(os.path.dirname(input_file_path), "1d_data")

			# create the output folder if it does not exist
			if not os.path.exists(output_folder):
				print('creating output folder:', output_folder)
				os.makedirs(output_folder)

			# extract the ground truth values for this file
			this_ground_truth = ground_truth[file_index]

			# convert the file to a 1d trace
			mean_resistivity, std_resistivity, z_axis, extended_ground_truth = Res2DInvOutputProcessor.convert_to_1d_trace(
				input_file_path,
				x_range_to_use=x_range,
				ground_truth=this_ground_truth,
				output_folder=output_folder)

			# save output
			output_data.append((input_file_path, mean_resistivity, std_resistivity, z_axis, extended_ground_truth))

			# plot, if requested
			if plot:
				file_name = os.path.basename(input_file_path)
				if plot_all_together:
					file_data_list.append((file_name, mean_resistivity, std_resistivity, z_axis, extended_ground_truth))
				else:
					Res2DInvOutputProcessor.plot_1d_trace(file_name,
														  this_ground_truth,
														  extended_ground_truth,
														  mean_resistivity,
														  labels,
														  std_resistivity,
														  z_axis,
														  depth_range,
														  x_range_to_plot,
														  kohm_labels,
														  left_side_labels,
														  legend_position)

			# if calculate average resistivity
			if calculate_average_resistivity:
				thickness = float(list(layers.keys())[-1])
				average_resistivity = Res2DInvOutputProcessor.compute_average_one_layer_resistivity(z_axis, mean_resistivity, thickness)
				print(f"Average Resistivity: {average_resistivity} Ω·cm")

			# increment file index
			file_index += 1

		# create a combined plot if requested
		if plot_all_together and file_data_list:
			Res2DInvOutputProcessor.plot_1d_traces_combined(
				file_data_list, layers, labels, depth_range, x_range_to_plot,
				legend_labels, colors=colors,
				layer_bound_ground_truth=layer_bound_ground_truth,
				point_format=point_format, kohm_labels=kohm_labels,
				left_side_labels=left_side_labels, legend_position=legend_position,
				ax=ax, layer_label_font_size=layer_label_font_size, orientation=orientation)

		# return the output data
		return output_data

	@staticmethod
	def add_layer_bounds_and_labels_to_plot(ground_truth_dict, width_limit, depth_limit, handle=None,
											add_boundaries=True, add_labels=True, labels=None,
											left_side_labels=None, label_font_size=16, orientation='horizontal'):
		# color map for each known layer label
		layer_colors = {
			"NFL+GCL": "lightblue",
			"IPL": "orange",
			"INL+OPL+ONL": "lightgreen",
			"IS+OS": "purple",
			"PBS": "gray",
		}

		# labels that should be drawn on the left instead of the right
		left_side_labels = set(left_side_labels) if left_side_labels else set()

		# default colors for any labels not in the map above
		default_cycle = cycle(["red", "brown", "pink", "gray", "olive", "cyan", "magenta"])
		default_colors = {}

		def get_color(label):
			if label in layer_colors:
				return layer_colors[label]
			if label not in default_colors:
				default_colors[label] = next(default_cycle)
			return default_colors[label]

		# extract the keys from the ground truth dictionary
		keys = list(ground_truth_dict.keys())
		handle = plt.gca() if not handle else handle

		# track depths whose horizontal line has already been drawn
		drawn_depths = set()

		# iterate the layers (keys)
		for i, key in enumerate(keys):

			# determine the true label for this layer (used for both coloring and text)
			true_label = str(ground_truth_dict[key]) if not labels else labels[i]

			# determine this layer's depth boundary (shallow to deep)
			depth1 = float(key)
			depth2 = float(keys[i + 1]) if i < len(keys) - 1 else depth_limit[0]

			if orientation == 'vertical':
				# depth on the x-axis, value (resistivity) on the y-axis --
				x0, x1 = sorted([depth1, depth2])
				y0, y1 = 0, width_limit

				if add_boundaries:
					color = get_color(true_label)
					face = to_rgba(color, alpha=0) if color else 'none'
					rectangle = patches.Rectangle(
						(x0, y0),
						x1 - x0,
						y1 - y0,
						edgecolor='none',
						facecolor=face,
					)
					handle.add_patch(rectangle)

					# value-axis bounds (always drawn; adjacent layers tile, they don't overlap)
					handle.plot([x0, x1], [y0, y0], color='gray', linewidth=1.5, linestyle='--')
					handle.plot([x0, x1], [y1, y1], color='gray', linewidth=2, linestyle='--')

					# depth boundaries — skip any already drawn by the previous patch
					for hx in (depth1, depth2):
						if hx not in drawn_depths:
							handle.plot([hx, hx], [y0, y1], color='gray', linewidth=2, linestyle='--')
							drawn_depths.add(hx)

				if add_labels:
					if true_label:
						near_start = true_label in left_side_labels
						text_x = (x0 + x1) / 2
						text_y = y0 + (y1 - y0) * 0.05 if near_start else y1 - (y1 - y0) * 0.05
						handle.text(
							text_x,
							text_y,
							true_label,
							ha='center',
							va='bottom' if near_start else 'top',
							fontsize=label_font_size,
							weight='bold',
							color='black'
						)

				continue

			# determine the height of the rectangle
			height = -abs(depth2 - depth1)

			# determine the width of the rectangle
			width = width_limit

			# determine the starting coordinate of the rectangle
			x = 0
			y = max(depth1, depth2)

			# create the rectangle
			if add_boundaries:
				# look up (or assign a default) fill color; apply high transparency to the fill only
				color = get_color(true_label)
				face = to_rgba(color, alpha=0) if color else 'none'
				# fill only; borders drawn as lines below so shared horizontals aren't drawn twice
				rectangle = patches.Rectangle(
					(x, y),
					width,
					height,
					edgecolor='none',
					facecolor=face,
				)
				handle.add_patch(rectangle)

				# horizontal sides
				for hy in (depth1, depth2):
					if hy in depth_limit:
						continue
					if hy not in drawn_depths:
						handle.plot([x, x + width], [hy, hy], color='gray', linewidth=2, linestyle='--')
						drawn_depths.add(hy)

			# add true label text to the patch
			if add_labels:

				# if the true label is provided, add it to the plot
				if true_label:
					on_left = true_label in left_side_labels
					text_x = width * 0.05 if on_left else width * 0.95
					text_y = y + height / 2
					handle.text(
						text_x,
						text_y,
						true_label,
						ha='left' if on_left else 'right',
						va='center',
						fontsize=label_font_size,
						weight='bold',
						color='black'
					)

	@staticmethod
	def compute_average_one_layer_resistivity(z_axis, mean_resistivity, thickness):
		# ensure thickness is numeric and arrays are numpy arrays
		thickness = float(thickness)
		z_axis = np.array(z_axis)
		mean_resistivity = np.array(mean_resistivity)

		# find indices within the specified thickness
		mask = z_axis >= thickness

		# extract data up to the thickness
		z_subset = z_axis[mask]
		resistivity_subset = mean_resistivity[mask]

		# compute integral and divide by thickness
		integral = np.trapz(resistivity_subset, z_subset)

		return integral / thickness

	def extract_resistivity_values_from_set_of_1d_traces(self, output_data, layers, labels, z_range_to_use, min_prominence=0.0, plot=False,
														 layer_methods=None, max_plots=None, left_side_labels=None):
		"""
		Run resistivity extraction on a set of 1D traces
		"""
		# initialize the output
		results = OrderedDict()

		# initialize the plots counter
		plots_shown = 0

		# iterate the traces
		for input_file_path, mean_resistivity, _std_resistivity, z_axis, extended_ground_truth in output_data:

			# initialize the ground truth label
			gt_by_label = None

			# if ground truth is available, use it
			if extended_ground_truth is not None and any(extended_ground_truth):
				gt_by_label = self._ground_truth_by_label_from_extended(
					z_axis, extended_ground_truth, layers, z_range_to_use, labels
				)

			# set plot flag
			should_plot = plot and (max_plots is None or plots_shown < max_plots)

			# increment plots counter
			if should_plot:
				plots_shown += 1

			# analyze the trace and add the result to the output
			results[input_file_path] = self.extract_resistivity_values_from_1d_trace(
				z_axis, mean_resistivity, layers, labels, z_range_to_use,
				min_prominence=min_prominence, plot=should_plot, ground_truth_by_label=gt_by_label,
				layer_methods=layer_methods, left_side_labels=left_side_labels
			)

		return results

	def extract_resistivity_values_from_1d_trace(self, z_axis, mean_resistivity, layers, labels, z_range_to_use, min_prominence=0.0, plot=False,
												 ground_truth=None, ground_truth_by_label=None, layer_methods=None, left_side_labels=None):
		"""
		Within each layer bounds, resolve the resistivity value. A point sitting
		exactly on the boundary between two layers is shared with the
		neighboring layer and is excluded from analysis entirely, regardless of
		method. layer_methods optionally maps a label to a specific extraction
		method; any label not listed defaults to 'extremum':
		  'first_point'      -- the first sample in the layer's own window
		  'layer_average'     -- the mean of every sample in the layer's window
		  'window_average'    -- the average of the layer's own first and last samples
		  'neighbor_average'  -- the average of the previous and next layer's resolved values
		  'extremum' (default) -- search for a clear peak or valley using this layer's own data plus
					   one sample borrowed from just beyond its deep boundary
		"""
		# ensure depth and resistivity values are numpy arrays
		z_axis = np.asarray(z_axis, dtype=float)
		mean_resistivity = np.asarray(mean_resistivity, dtype=float)

		# identify boundaries between layers
		boundaries = self._layer_boundaries(layers, z_range_to_use)
		if len(boundaries) != len(labels):
			raise ValueError(
				f"expected {len(labels)} layer windows from `layers`, got {len(boundaries)}"
			)

		# assign given methods
		layer_methods = layer_methods or {}

		# this is the number of layers in the model
		n = len(labels)

		# iterate the layers
		entries = []
		for i, (label, (depth1, depth2)) in enumerate(zip(labels, boundaries)):

			# these are the layer's boundaries
			lo, hi = sorted([depth1, depth2])
			mask = (z_axis >= lo) & (z_axis <= hi)
			window_z = z_axis[mask]
			window_r = mean_resistivity[mask]

			# a point within 1um of this layer's boundary is effectively shared
			# with the neighboring layer and is never eligible for analysis here,
			# regardless of method
			boundary_mask = (np.abs(window_z - lo) <= 1.0) | (np.abs(window_z - hi) <= 1.0)
			window_z = window_z[~boundary_mask]
			window_r = window_r[~boundary_mask]

			# these are the points that will be analyzed
			window_points = list(zip(window_z.tolist(), window_r.tolist()))

			# initialize the output
			entry = {'label': label, 'depth1': depth1, 'depth2': depth2,
					 'value': None, 'marker_z': None, 'method': 'none',
					 'used_points': [], 'inspected_points': [], 'window_points': window_points}

			if len(window_r) == 0:
				entries.append(entry)
				continue

			window_first = window_points[0]
			window_last = window_points[-1]

			method = layer_methods.get(label, 'extremum')

			# determine teh resistivity value for this layer based on the requested method
			if method == 'layer_average':
				self._resolve_from_points(entry, window_points, 'layer_average')
			elif method == 'first_point':
				self._resolve_from_points(entry, [window_first], 'first_point')
			elif method == 'window_average':
				self._resolve_from_points(entry, [window_first, window_last], 'window_average')
			elif method == 'neighbor_average':
				entry['method'] = 'pending'
				entry['window_first'] = window_first
				entry['window_last'] = window_last
			elif method == 'extremum':
				if len(window_r) == 1:
					self._resolve_from_points(entry, [window_first], 'single_point', inspected=window_points)
				else:
					mask_indices = np.flatnonzero(mask)
					prev_full_idx = mask_indices[0] - 1
					next_full_idx = mask_indices[-1] + 1

					# first, search using only this layer's own data (plus the
					# deep-side borrowed context point, so a genuine turn at the
					# layer's own last sample can still be confirmed against its
					# true next neighbor)
					primary_r = window_r
					if next_full_idx < len(z_axis):
						primary_r = np.append(primary_r, mean_resistivity[next_full_idx])

					idx_search = self._find_extremum_index(primary_r, min_prominence=min_prominence)
					mapped_idx = idx_search if idx_search is not None and idx_search < len(window_r) else None

					# only reach for the shallow-side context if nothing was found
					# without it -- a clean extremum found entirely within the
					# layer's own data always takes priority over one that needs
					# borrowed context to be confirmed
					if mapped_idx is None and prev_full_idx >= 0:
						extended_r = np.insert(primary_r, 0, mean_resistivity[prev_full_idx])
						idx_search = self._find_extremum_index(extended_r, min_prominence=min_prominence)
						mapped_idx = (idx_search - 1) if idx_search is not None and 1 <= idx_search <= len(
							window_r) else None

					if mapped_idx is not None:
						self._resolve_from_points(entry, [window_points[mapped_idx]], 'extremum',
												  inspected=window_points)
					else:
						self._resolve_from_points(entry, window_points, 'layer_average', inspected=window_points)
			else:
				raise ValueError(f"unknown layer_methods value {method!r} for label {label!r}")

			entries.append(entry)

		# second pass: fill in any remaining pending layers using the average of
		# the previous and next layer's resolved values; fall back further to
		# this window's own first/last average if a neighbor isn't available
		for i, entry in enumerate(entries):
			if entry['method'] != 'pending':
				continue

			prev_entry = entries[i - 1] if i > 0 else None
			next_entry = entries[i + 1] if i < n - 1 else None
			prev_point = (prev_entry['marker_z'], prev_entry['value']) if prev_entry and prev_entry[
				'value'] is not None else None
			next_point = (next_entry['marker_z'], next_entry['value']) if next_entry and next_entry[
				'value'] is not None else None

			neighbor_points = [p for p in (prev_point, next_point) if p is not None]
			if neighbor_points:
				self._resolve_from_points(entry, neighbor_points, 'neighbor_average')
			else:
				self._resolve_from_points(entry, [entry['window_first'], entry['window_last']], 'window_average')

		resolved = OrderedDict((entry['label'], entry['value']) for entry in entries)

		# plot the resolved values
		if plot:
			gt_by_label = ground_truth_by_label
			if gt_by_label is None and ground_truth is not None:
				sorted_items = sorted(ground_truth.items(), key=lambda x: int(x[0]), reverse=True)
				gt_by_label = OrderedDict(
					(label, float(raw_value))
					for label, (key, raw_value) in zip(labels, sorted_items)
					if raw_value not in (None, '')
				)
			self._plot_trace_analysis(z_axis, mean_resistivity, entries, ground_truth_by_label=gt_by_label,
									  z_range_to_use=z_range_to_use, left_side_labels=left_side_labels)

		return resolved

	def average_error_by_layer(self, all_results, ground_truth, labels):
		"""
		For each layer, calculate the average percent deviation from ground truth across
		every trace in all_results
		"""
		# get all input file paths
		file_paths = list(all_results.keys())

		# get all ground truth values
		ground_truths = ground_truth if isinstance(ground_truth, list) else [ground_truth] * len(file_paths)

		# their number should be the same
		if len(ground_truths) != len(file_paths):
			raise ValueError(
				f"expected {len(file_paths)} ground truth entries (one per file), got {len(ground_truths)}"
			)

		# initialize a dictionary to hold errors for each layer
		per_layer_errors = OrderedDict((label, []) for label in labels)

		# iterate the files and their corresponding ground truths
		for file_path, gt in zip(file_paths, ground_truths):

			# compute the errors for this file against its ground truth
			file_errors = self.compare_to_ground_truth(all_results[file_path], gt, labels)

			# append each layer's error to the corresponding entry in per_layer_errors
			for label, error in file_errors.items():
				if error is not None:
					per_layer_errors[label].append(error)

		# return the average error for each layer, or None if no data was recorded for that layer
		return OrderedDict(
			(label, float(np.mean(errors)) if errors else None)
			for label, errors in per_layer_errors.items()
		)

	@staticmethod
	def compare_to_ground_truth(resolved, ground_truth, labels):
		"""
		Compare the extracted resistivities against the ground truth, returning the percent error for
		each layer.
		"""
		# sort ground truth values in descending order
		sorted_items = sorted(ground_truth.items(), key=lambda x: float(x[0]), reverse=True)

		# the number of resistivities should be the same as the number of ground truth values
		if len(sorted_items) != len(labels):
			raise ValueError(
				f"expected {len(labels)} ground truth values, got {len(sorted_items)}"
			)

		# initialize an error dictionary
		errors = OrderedDict()

		# iterate through the ground truth values and compare them to the resolved resistivities
		for label, (key, raw_value) in zip(labels, sorted_items):

			# get the extracted resistivity value for the current layer
			extracted = resolved.get(label)

			# get the ground truth value for the current layer
			true_value = float(raw_value) if raw_value not in (None, '') else None

			# skip missing values
			if extracted is None or true_value is None or true_value <= 0:
				errors[label] = None
				continue

			# otherwise, compute the percent error
			errors[label] = abs(extracted - true_value) / true_value * 100.0

		return errors