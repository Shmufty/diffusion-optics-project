% run_tissue_3d_generator_2planes: Calls tissue_3d_generator to produce
% exactly M=2 aberration planes (sim step 1 of the project guidelines),
% and saves outputs as a .mat file for use in Python.
%
% Tissue dimensions (x_max, x_stp, z_max) are unchanged from the baseline
% run_tissue_3d_generator.m. Only z_stp is modified: the number of coarse
% z-planes Nzs produced by tissue_3d_generator.m is set purely by z_max
% and z_stp, so setting z_stp = z_max collapses the z-grid to exactly 2
% planes.
%
% Propagation convention: light enters the tissue at z=0 and propagates
% towards z=Z_tissue (Z_tissue = 2*z_max, the full tissue width -- the
% generator's z_max is a half-width about its own internal z=0). With
% M planes and spacing Delta = Z_tissue/M, plane k must sit at z=k*Delta,
% so the LAST plane lands exactly on the tissue exit face (z=Z_tissue)
% and the first free-space hop (source -> plane 1) is also Delta. This
% does NOT match tissue_3d_generator's internal bin centers, which are
% symmetric about its own z=0 (i.e. at Delta/2, 3*Delta/2, ... in forward
% coordinates -- for M=2 that's Z_tissue/4 and 3*Z_tissue/4). That's fine:
% the bin center only says which slab of tissue a plane's texture was
% averaged from, not where the propagation model is allowed to place the
% resulting thin phase screen. We therefore compute and save the forward
% (z=0 at entrance) plane positions explicitly, rather than relying on
% z_grid1/the implicit bin centers, so the Python propagation code uses
% the correct Delta and plane_z regardless of the generator's internal
% averaging convention.

%% === Set parameters (adjust as needed) ===
x_max = 10;      % Maximum extent in x/y [physical units] (unchanged)
x_stp = 0.1;      % Sampling step size in x/y [physical units] (unchanged)
z_max = 5;        % Tissue half-width in z [physical units] (unchanged)
M = 2;             % Number of aberration planes
Z_tissue = 2*z_max;   % Full tissue width, entrance (z=0) to exit (z=Z_tissue)
Delta = Z_tissue/M;   % Spacing between planes == z_stp passed to the generator
z_stp = Delta;         % Sampling step size in z -> yields exactly M planes
sigt = 1;          % Total scattering coefficient
lambda = 0.532;    % Wavelength [physical units]
is_cyclic = 0;     % Boundary condition flag

spr_params.rad_rng = 0.5;   % Radius range
spr_params.rad_min = 0.2;   % Minimum radius
spr_params.ref_rng = 0.05;  % Refractive index range
spr_params.ref_min = 0.0;   % Minimum refractive index
spr_params.od = 1;          % Optical density
spr_params.scl = 1;         % User scaling factor

%% === Run generator ===
[mask_f, mask_f0, z_grid1] = tissue_3d_generator( ...
    x_max, x_stp, z_max, z_stp, sigt, spr_params, lambda, is_cyclic);

assert(size(mask_f, 3) == M, 'Expected %d aberration planes, got %d', M, size(mask_f, 3));

% Forward-convention plane positions (z=0 at tissue entrance): plane k is
% placed at k*Delta, so plane M sits exactly at the exit face z=Z_tissue.
plane_z = (1:M)*Delta;   % [1 x M], e.g. [Z_tissue/2, Z_tissue] for M=2

%% === Gather from GPU and save for Python ===
mask_f  = gather(mask_f);
mask_f0 = gather(mask_f0);
z_grid1 = gather(z_grid1);

out_file = 'tissue_output_2planes.mat';
save(out_file, 'mask_f', 'mask_f0', 'z_grid1', ...
    'x_max', 'x_stp', 'z_max', 'M', 'Z_tissue', 'Delta', 'plane_z', '-v7.3');

fprintf('Saved outputs to %s (Nplanes = %d, Delta = %.3f, plane_z = %s)\n', ...
    out_file, size(mask_f, 3), Delta, mat2str(plane_z));
