% run_tissue_3d_generator_2planes_50x50x10: Same as
% run_tissue_3d_generator_2planes.m (M=2 aberration planes, sim step 1),
% but with a larger lateral tissue extent: 50x50x10 um instead of
% 20x20x10 um. Only x_max changes (10 -> 25); x_stp, z_max, M, sigt,
% lambda, spr_params are identical, so Delta and plane_z (both derived
% from z_max/M, not x_max) come out the same: Delta=5, plane_z=[5,10].
%
% See run_tissue_3d_generator_2planes.m for the full explanation of the
% forward propagation convention (z=0 at tissue entrance) and why
% plane_z is computed explicitly rather than taken from the generator's
% internal (symmetric, bin-centered) z-grid.

%% === Pick the GPU with the most free memory ===
% The 50x50x10 working grid (~1001x1001x200, non-cyclic) is ~6x the
% 20x20x10 case's array sizes and can exceed a GPU's free memory if it's
% already partly occupied by another process; select the emptiest device.
n_gpus = gpuDeviceCount();
free_mem = zeros(1, n_gpus);
for gi = 1:n_gpus
    d = gpuDevice(gi);
    free_mem(gi) = d.AvailableMemory;
end
[~, best_gpu] = max(free_mem);
gpuDevice(best_gpu);
fprintf('Using GPU %d (%.1f GB free)\n', best_gpu, free_mem(best_gpu)/1e9);

%% === Set parameters (adjust as needed) ===
x_max = 25;        % Maximum extent in x/y [physical units] -> lateral width 50
x_stp = 0.1;        % Sampling step size in x/y [physical units] (unchanged)
z_max = 5;          % Tissue half-width in z [physical units] (unchanged)
M = 2;               % Number of aberration planes
Z_tissue = 2*z_max;     % Full tissue width, entrance (z=0) to exit (z=Z_tissue)
Delta = Z_tissue/M;     % Spacing between planes == z_stp passed to the generator
z_stp = Delta;           % Sampling step size in z -> yields exactly M planes
sigt = 1;            % Total scattering coefficient
lambda = 0.532;      % Wavelength [physical units]
% Cyclic (not the 20x20x10 case's non-cyclic): non-cyclic doubles the
% working domain in x/y (bdr_fact=2), giving a ~1001x1001x200 working
% grid here that exceeds a single 24GB GPU's memory during the FFT
% filtering steps even with the whole GPU free. Cyclic halves it back to
% ~501x501x200, which fits (matches the 20x20x10 case's working-grid
% size). Tissue now wraps periodically at the lateral edges instead of
% being padded; for a point source imaged near the center this trades
% away edge-padding accuracy we don't need in exchange for fitting in
% memory.
is_cyclic = 1;       % Boundary condition flag

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

% Full simulated tissue dimensions [width_x, width_y, depth_z], physical units (um).
tissue_dims_um = [2*x_max, 2*x_max, Z_tissue];

%% === Gather from GPU and save for Python ===
mask_f  = gather(mask_f);
mask_f0 = gather(mask_f0);
z_grid1 = gather(z_grid1);

out_file = sprintf('tissue_output_2planes_%gx%gx%g.mat', tissue_dims_um(1), tissue_dims_um(2), tissue_dims_um(3));
save(out_file, 'mask_f', 'mask_f0', 'z_grid1', ...
    'x_max', 'x_stp', 'z_max', 'M', 'Z_tissue', 'Delta', 'plane_z', 'tissue_dims_um', '-v7.3');

fprintf('Saved outputs to %s (Nplanes = %d, Delta = %.3f, plane_z = %s, tissue_dims_um = %s)\n', ...
    out_file, size(mask_f, 3), Delta, mat2str(plane_z), mat2str(tissue_dims_um));
