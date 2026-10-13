<div align="center">

# Decoding Retinal Resistivity using a Multi-Electrode Array and Electrical Resistance Tomography

![Python](https://img.shields.io/badge/python-3.11-a5c8e4?style=flat-square)
![COMSOL](https://img.shields.io/badge/COMSOL-5.6+%20·%20LiveLink-cdbfe6?style=flat-square)
![Res2DInv](https://img.shields.io/badge/inversion-Res2DInv-f5d0b5?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-b9dfb4?style=flat-square)

[![DOI](https://zenodo.org/badge/1417932154.svg)](https://zenodo.org/badge/latestdoi/1417932154)

</div>

<br>

The toolbox can be used to decode the resistivity of the retinal layers using data obtained using Electrical 
Resistance Tomography (ERT) with a multi-electrode array (MEA). In this procedure, current is injected into an 
ex-vivo retinal sample through one electrode on the MEA and the voltage is recorded on all other electrodes. 
The recordings are calibrated against saline dilutions of known resistivity, converted into resistance-vs-distance 
traces, and decoded in two independent ways: a geophysical inversion in Res2DInv, and a machine-learning decoder (1D CNN) trained on COMSOL Multiphysics forward simulations. 
Data processing and ML decoding are implemented in Python, the forward model is executed from MATLAB via COMSOL 
LiveLink, and the analysis is done in Jupyter notebooks.

This software was developed by **Anna Kochnev Goldstein** and 
**Mohammad Asif Zaman** in the [**Palanker Lab**](https://web.stanford.edu/~palanker/lab/index.html) at Stanford 
University. It can be adapted to other layered tissues or MEA geometries.

<br>

## Software Overview

The software consists of the following key components:

| Path                 | Contents                                     |
|----------------------|----------------------------------------------|
| `comsol_model/`      | 3-layer and 4-layer COMSOL models            |
| `data_processing/`   | The data calibration and processing pipeline |
| `experimental_data/` | Input averaged experimental traces           |
| `ml_decoder/`        | The ML decoder train and test files          |
| `res2dinv/`          | Res2DInv inversion settings file             |

<br>

### Data Processing Pipeline

The data processing pipeline is configured and launched from `run_processing.py`, and it runs in five stages:

1. **Experiment parsing** · `experiment_manager.py`<br>
   Reads the experiment parameters from the header file name, loads the electrode layout, finds the stimulating electrodes and reads the current amplitudes.
2. **Saline calibration** · `saline_processor.py`<br>
   Calibrates every utilized MEA electrode for the day of the measurement. Each stimulation–recording resistance 
   (voltage vs. current) is compared with the expected analytical solution, yielding the delivered-to-nominal current 
   ratio for each stimulating electrode and the amplifier correction coefficient for each stimulating-recording pair.
3. **Sample processing** · `sample_processor.py`<br>
   Applies the calibration to the retinal data and returns calibrated resistance-vs-distance traces.
4. **Decoding** · `decoding_manager.py`<br>
   Writes an input `.DAT` file for Res2DInv and an input `.xlsx` file for the ML decoder. Either decoder is then 
   executed separately to produce the resistivity profile of the sample.
5. **Analysis** · `analysis_stage/`<br>
   All the notebooks used to analyze the data and generate figures.

All stages run for each matching header file, and an error in one experiment does not stop the batch.

`run_processing_comsol_testset.py` runs a folder of COMSOL-simulated traces through the same pipeline to 
validate the decoders against known ground truth.

<br>

#### Input Folder Structure

Each retinal sample and injection line measured on that sample have their own folder under a common root folder. 
The folder is structured as follows:

```
<root_folder>/
└── YYYY-MM-DD-<sample_number>-<injection_line_number>/           one sample × one injection line
    ├── header_<index>_<species>_…                                empty file with metadata encoded in the file name
    ├── raw_data/
    │   ├── sample/                                               retinal data, one folder per electrode
    │   └── x1_before/                                            saline dilutions, obtained before and after the sample
    |   ....
    |   └── x100_after/                        
    ├── simulated_data/                                           optional COMSOL traces used as a synthetic data
    └── processed_data/                                           output folder created by the pipeline
        ├── <user_prefix>_res2dinv_sample_<num_dilutions>calib/   .DAT  → Res2DInv's input file
        └── <user_prefix>_ml_sample_<num_dilutions>calib/         .xlsx → ML decoder's input file                                              
```

<br>

#### Header File Format

The header file is an empty file that serves for storing the experimental metadata and is included in each 
experimental folder. Its name is split on `_`, and each token is recognized by its format to define the experiment:

| Token | Meaning                                                                               | Example     |
|---|---------------------------------------------------------------------------------------|-------------|
| first number | A running experimental index (configured by `indices_to_process`)                     | `1094`      |
| first word | Species                                                                               | `LE`, `RCS` |
| `<N>um` | The thickness of the retina                                                           | `200um`     |
| `<N>um` | The pitch of the array                                                                | `30um`      |
| `<N>um` | The width of the array                                                                | `720um`     |
| `<ρ>Ocm` | Saline dilution resistivities (Ω·cm), matched to ×1…×100                              | `50Ocm`     |
| `offset<N>` | The number of signal samples that precede the injection pulse (default 49)            | `offset49`  |
| `rstim<N>` | Representative electrode number used to read the used current amplitudes (default 448) | `rstim448`  |
| `line<N>` | Injection line number (see `configurations.py`)                                       | `line0`     |


Example: `header_1094_LE_200um_30um_720um_60Ocm_50Ocm_150Ocm_500Ocm_5000Ocm_mea_offset49_rstim448_line0`

<br>


### COMSOL Forward Model

The `comsol_model/` folder holds an axisymmetric COMSOL model of a center electrode located below a layered retina 
surrounded by saline. The model files `3_layer_model.mph` and `4_layer_model.mph` differ in their layer thicknesses.

The model is run from the MATLAB script `run_model.m` via LiveLink. The invocation script reads a CSV of layer resistivities (Ω·cm, one set per row), 
solves each set, and writes the resulting probe voltages at 30 µm and 33 µm spacing to `comsol_output_forward_traces/`. 
Each output row holds the `rho_*` values followed by `V_0 … V_n`, and the probe coordinates (µm) are saved alongside. 
These output files serve as the ML training and test sets.

<br>

### Experimental Data

The `experimental_data/` folder holds the average experimental traces obtained from LE (average of 30 traces) and 
RCS (average of 36 traces) samples. These traces serve as the input for the ML decoder and the Res2DInv inversion.


<br>

### ML Decoder

The `ml_decoder/` folder holds a 1D CNN decoder that maps a resistance-vs-distance trace to four layer or three layer
resistivities. The train and test sets for the 3-layer and 4-layer models are held in `data_3_layer_model/` and 
`data_4_layer_model/`, respectively. 
Each holds 12,000 single-layer and 30,000 multi-layer training samples and an 8,000-sample test set (with 33 µm probe 
spacing).
The code components are:

```
preprocess.py   # standardize data; saves train and test sets
train.py        # train the decoder on the train set
test.py         # evaluate the decoder on the test set and analyze experimental data
```

<br>


### Analysis Notebooks

All the notebooks used to analyze the output data from the pipeline and the decoders are located in 
`data_processing/analysis_stage/`:

| Notebook                      | Purpose                                                         |
|-------------------------------|-----------------------------------------------------------------|
| `a_copy_experimental_data`    | Map raw acquisition folders into the input structure            |
| `b_analyze_noise_floor`       | Estimate the noise floor of the traces                          |
| `c_reformat_comsol_files`     | Convert COMSOL outputs into `.xlsx` traces                      |
| `d_plot_resistance_traces`    | Average and plot traces per sample and species                  |
| `e_process_res2dinv_output`   | Convert Res2DInv models to 1D profiles; obtain decoding accuracy |
| `f_plot_decoded_results`      | Visualize Res2DInv and ML decoded resistivities                 |
| `g_find_closet_forward_model` | Find the closest synthetic COMSOL trace to each experiment      |

<br>


## Feedback

Please do not hesitate to send us your feedback!

We are an academic research group, and made this tool available for a collaborative community of scientific research. 
If you found a bug, have improvement suggestions, or simply want to connect - please create a new issue or pull request, or contact our lab via the [provided website](https://web.stanford.edu/~palanker/lab/index.html)!

<br>

## License

The software is distributed under the MIT License.
If you use this software in your research, please cite the DOI provided at the top of this README file.

<br>

## Disclaimer

This package was created for our research purposes and distributed as open source for the benefit of the scientific community. 
It is NOT intended for medical treatment or diagnosis and DOES NOT come with any kind of warranty. We hereby disclaim any liability related to the direct or indirect usage of this software.
