% run_tissue_3d_generator: Calls tissue_3d_generator and saves outputs
% as a .mat file for use in Python (via h5py / scipy.io.loadmat).

%% === Set parameters (adjust as needed) ===
x_max = 10;      % Maximum extent in x/y [physical units]
x_stp = 0.1;      % Sampling step size in x/y [physical units]
z_max = 5;        % Maximum extent in z [physical units]
z_stp = 0.5;       % Sampling step size in z [physical units]
sigt = 1;          % Total scattering coefficient
lambda = 0.532;    % Wavelength [physical units]
is_cyclic = 1;     % Boundary condition flag (cyclic; keeps this grid consistent with getPropConvKerFreq3D.m's kernel sizing)

spr_params.rad_rng = 0.5;   % Radius range
spr_params.rad_min = 0.2;   % Minimum radius
spr_params.ref_rng = 0.05;  % Refractive index range
spr_params.ref_min = 0.0;   % Minimum refractive index
spr_params.od = 1;          % Optical density
spr_params.scl = 1;         % User scaling factor

%% === Run generator ===
[mask_f, mask_f0, z_grid1] = tissue_3d_generator( ...
    x_max, x_stp, z_max, z_stp, sigt, spr_params, lambda, is_cyclic);

%% === Gather from GPU and save for Python ===
mask_f  = gather(mask_f);
mask_f0 = gather(mask_f0);
z_grid1 = gather(z_grid1);

out_file = 'tissue_output.mat';
save(out_file, 'mask_f', 'mask_f0', 'z_grid1', '-v7.3');

fprintf('Saved outputs to %s\n', out_file);
