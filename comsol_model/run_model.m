% Mohammad Asif Zaman 2026
% - Matlab appears to take Ohm.m as inputs 
% - The distances are exported from Comsol in mm (this is due to how it was
%   defined in the model settings

clear all;
close all;
clc;
clf;

tic

% define the path to the csv with the resitivities to simulate. 
% The csv should have four values for each run. For the 3-layered model
% set the last value to saline (rho_PR=60 Ohm.cm)
Mdata_in = csvread('.../decoded_4layer_rcs_values_ml_and_res2dinv.csv');

% do the following scaling if the resistivities in the csv are Ohm.cm
rho_NFL_GCL_set = Mdata_in(:,1)*1e-2;
rho_IPL_set = Mdata_in(:,2)*1e-2;
rho_nu_set = Mdata_in(:,3)*1e-2;
rho_PR_set = Mdata_in(:,4)*1e-2;

% define run parameters
print_step = 4;
total_runs = length(rho_PR_set);
rho_saline = 60*1e-2;  % in Ohm.m

% define the name of the output folder
dir_data_out = 'comsol_output_forward_traces';

% define output files
fname_r_30_cord = ['LV_r_coordinate_30_N='  num2str(total_runs)  '.csv'];
fname_r_33_cord = ['LV_r_coordinate_33_N='  num2str(total_runs)  '.csv'];

fname_data_30_full = ['LV_data_out_30_N='  num2str(total_runs)  '.csv'];
fname_data_30_voltage = ['LV_voltage_out_30_N='  num2str(total_runs)  '.csv'];

fname_data_33_full = ['LV_data_out_33_N='  num2str(total_runs)  '.csv'];
fname_data_33_voltage = ['LV_voltage_out_33_N='  num2str(total_runs)  '.csv'];

% define model file name (the 4-layered and 3-layered models have different
% layer thicknesses)
model_name = '3_layer_model.mph';

addpath('/Applications/COMSOL56/Multiphysics/mli')

import com.comsol.model.*
import com.comsol.model.util.*

% load model
model = mphload(model_name);

% initilize outputs
voltage_matrix_30 = [];
data_matrix_30 = [];

voltage_matrix_33 = [];
data_matrix_33 = [];

% start counter
counter = 0;

% run all provided resitivity sets
for m1 = 1:total_runs
                
                % print progress
                if mod(counter,print_step) == 0
                   percentage = floor(100*counter/total_runs);
                   fprintf('Counter = %d% \n', counter);
                   fprintf(',  Completed %d%% \n', percentage);
                   fprintf('\n')
                end

                % advance counter
                counter = counter + 1;
                
                % extract the predefined resistivity values for this run
                rho_NFL_GCL = rho_NFL_GCL_set(m1);
                rho_IPL = rho_IPL_set(m1);
                rho_nu = rho_nu_set(m1);
                rho_PR = rho_PR_set(m1);
                
                % feed the Ohm.m resistivity values into COMSOL
                model.param.set('rho_NFL_GCL', num2str(rho_NFL_GCL));
                model.param.set('rho_IPL', num2str(rho_IPL));
                model.param.set('rho_nu', num2str(rho_nu));
                model.param.set('rho_PR', num2str(rho_PR));
                model.param.set('rho_saline', num2str(rho_saline));
                
                % run the model with these resisitivity values
                model.study('std1').run;
                
                % extract simulated voltages from the parametric probe line 
                % (tag = 'pc1', verify with "model.result.dataset.tags")
                data30 = mphinterp(model, 'V', 'dataset', 'pc1');
                data33 = mphinterp(model, 'V', 'dataset', 'pc2');
                
                % store extracted voltages for this run 
                voltage_matrix_30(counter, :) = data30;  
                voltage_matrix_33(counter, :) = data33;  
                
                % store voltages and resistivities
                % resistivities are stored in the matrix in Ohm.cm value
                data_matrix_30(counter, :) = [rho_NFL_GCL*1e2 rho_IPL*1e2 rho_nu*1e2 rho_PR*1e2 rho_saline*1e2 data30];  
                data_matrix_33(counter, :) = [rho_NFL_GCL*1e2 rho_IPL*1e2 rho_nu*1e2 rho_PR*1e2 rho_saline*1e2 data33];  

                % clear solution to save memory
                model.sol('sol1').clearSolution;  

end

% write to output csv files
num_points = size(voltage_matrix_30, 2);
header = [{'rho_NFL_GCL'},{'rho_IPL'},{'rho_nu'},{'rho_PR'},{'rho_saline'}, arrayfun(@(i) sprintf('V_%d', i), 0:num_points-1, 'UniformOutput', false)];

% write voltage files
csvwrite([dir_data_out '/' fname_data_30_voltage], voltage_matrix_30);
csvwrite([dir_data_out '/' fname_data_33_voltage], voltage_matrix_33);

% write data (resistivity and voltage) files
output_cell = [header; num2cell(data_matrix_30)];
writecell(output_cell, [dir_data_out '/' fname_data_30_full]);

output_cell2 = [header; num2cell(data_matrix_33)];
writecell(output_cell2, [dir_data_out '/' fname_data_33_full]);

% extract probe coordinates in um
model.study('std1').run;
r_coord = mphinterp(model, 'r', 'dataset', 'pc1')*1000;  
r_coord2 = mphinterp(model, 'r', 'dataset', 'pc2')*1000;  

% write coordinates files
writematrix(r_coord, [dir_data_out '/' fname_r_30_cord]);
writematrix(r_coord2, [dir_data_out '/' fname_r_33_cord]);

% plot one of the output matrices
loglog(voltage_matrix_30(1,:),'r.-');
hold on;
loglog(voltage_matrix_30(2,:),'bo');

toc
