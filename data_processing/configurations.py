# standard imports
import numpy as np
from enum import Enum


class GroundTruthTypes(Enum):
	ANALYTICAL = (1, "analytical")
	SIMULATED = (2, "simulated")


class VoltageExtractionMethods(Enum):
	MAX = (1, "max")


class CalibrationMethods(Enum):
	COEFFICIENTS_VECTOR = (1, "vector_of_coefficients")
	SINGLE_COEFFICIENT = (2, "single_coefficient")


class CalibrationMeasurements(Enum):
	X1 = 1
	X10 = 2
	X15 = 3
	X26 = 4
	X30 = 5
	X50 = 6
	X100 = 7

	@classmethod
	def get_all_dilutions(cls):
		return [member.name for member in cls]


class ExperimentTypes(Enum):
	MEA = (1, 'MEA')
	COMSOL = (2, 'COMSOL')

	def __init__(self, number, experiment_type):
		self.number = number
		self.experiment_type = experiment_type


class DecodingTool(Enum):
	RES2DINV = (1, "res2dinv")
	ML_DECODER = (2, "ml")

	def __init__(self, number, software_name):
		self.number = number
		self.tool_name = software_name


class ElectrodeConfiguration(Enum):
	ALL_COMBS = (2, 3, 11)
	ALL_COMBS_MINIMAL = (3, 3, 11)

	def __init__(self, number, number_of_electrodes, number_of_array):
		self.number = number
		self.number_of_electrodes = number_of_electrodes
		self.number_of_array = number_of_array


# MEA 30um
injection_line_0_center_30um = list({455, 454, 456, 453, 457, 452, 458, 451, 459, 450, 460, 449, 461, 448, 189, 201, 190, 200, 191, 199, 192, 198, 193, 197, 194, 196, 195})
injection_line_1_top_left_bottom_right_30um = list({271, 276, 281, 284, 288, 287, 295, 292, 300, 291, 309, 202, 448, 50, 49, 31, 40, 32, 35, 27, 28, 24, 21, 16, 11})
injection_line_2_top_right_bottom_left_30um = list({379, 374, 369, 366, 362, 363, 355, 358, 350, 359, 341, 340, 448, 188, 81, 99, 90, 98, 95, 103, 102, 106, 109, 114, 119})
injection_line_3_vertical_minus_30um = list({445, 444, 443, 442, 441, 440, 439, 438, 437, 436, 446, 447, 340, 202, 203, 204, 214, 213, 212, 211, 210, 209, 208, 207, 206, 205})

# COMSOL 30um
injection_line_0_center_30um = list({189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 461, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 448})
injection_line_1_diagonal_30um = list({50, 72, 77, 80, 86, 89, 93, 94, 97, 100, 105, 110, 364, 367, 368, 371, 372, 375, 376, 378, 383, 386, 416, 447, 461})


# COMSOL 60um
injection_electrodes_60um = []


class ArrayTypes(Enum):
	MEA_30_UM = (1, ExperimentTypes.MEA, 30, 30, 5, 0.0, 0, 780, 15000, [injection_line_0_center_30um, injection_line_1_top_left_bottom_right_30um, injection_line_2_top_right_bottom_left_30um, injection_line_3_vertical_minus_30um])
	MEA_60_UM = (2, ExperimentTypes.MEA, 60, 30, 5, 0.0, 0, 2000, 15000, [])
	COMSOL_30_UM = (3, ExperimentTypes.COMSOL, 30, 30, 5, 0.0, 0, 780, 15000, [injection_line_0_center_30um, injection_line_1_top_left_bottom_right_30um, injection_line_2_top_right_bottom_left_30um, injection_line_3_vertical_minus_30um])
	COMSOL_60_UM = (4, ExperimentTypes.COMSOL, 60, 60, 11, 0.0, -1000.00, 1000.00, 15000, [])
	COMSOL_16_UM = (5, ExperimentTypes.COMSOL, 16, 16, 0, 0.0, 0, 448, 15000)

	def __init__(self, number, experiment_type, electrode_spacing, effective_electrode_spacing, electrode_diameter,
				 z_location, x1_of_fixed_region=None, x2_of_fixed_region=None, distance_to_ground_electrode=None, injection_line_electrodes=None):
		self.number = number
		self.experiment_type = experiment_type
		self.electrode_spacing_um = electrode_spacing
		self.effective_electrode_spacing_um = effective_electrode_spacing
		self.electrode_diameter_um = electrode_diameter
		self.electrode_radius_um = self.electrode_diameter_um / 2
		self.electrode_cross_section_area_um2 = round(np.pi * (self.electrode_radius_um) ** 2, 3)
		self.electrode_resistivity_ohm_m = 0.11
		self.z_location = z_location
		self.z_location_long = 3
		self.x1_of_fixed_region = x1_of_fixed_region
		self.x2_of_fixed_region = x2_of_fixed_region
		self.distance_to_ground_electrode_um = distance_to_ground_electrode
		self.electrodes_grid_x = np.arange(x1_of_fixed_region, x2_of_fixed_region + electrode_spacing, electrode_spacing).tolist()
		self.electrodes_grid_y = np.zeros(len(self.electrodes_grid_x)).tolist()

		# ensure the provided values are unique
		self.injection_line_electrodes = injection_line_electrodes

	@staticmethod
	def find_array_type(electrode_spacing, experiment_type):
		for array_type in ArrayTypes:
			if array_type.electrode_spacing_um == electrode_spacing and array_type.experiment_type == experiment_type:
				return array_type
		return None